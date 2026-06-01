import torch
import utils as utils
import torch.utils.data.dataset as Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import os
import random
import numpy as np
import copy
import pickle
from decord import VideoReader, cpu
import json
import pathlib
from torchvision import transforms
from config import rgb_dirs, pose_dirs


class _Numpy2CompatUnpickler(pickle.Unpickler):
    """Compat unpickler for artifacts produced by NumPy 2.x.

    Some pre-extracted pose PKLs were generated in environments where NumPy 2.x
    stores ndarray-related symbols under the `numpy._core.*` namespace.
    Older NumPy versions (1.x) don't expose `numpy._core`, so unpickling will
    fail with `ModuleNotFoundError: No module named 'numpy._core'`.

    We remap `numpy._core.*` -> `numpy.core.*` to keep the existing environment
    (torch/deepspeed stack) stable.
    """

    def find_class(self, module, name):
        if isinstance(module, str) and module.startswith("numpy._core"):
            module = "numpy.core" + module[len("numpy._core"):]
        return super().find_class(module, name)


def _safe_pickle_load(path: str):
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except ModuleNotFoundError as e:
            # Retry with NumPy 2.x namespace remapping.
            if "numpy._core" not in str(e):
                raise
    with open(path, "rb") as f:
        return _Numpy2CompatUnpickler(f).load()

# Lazy pose extraction (for datasets that only provide raw videos)
_WHOLEBODY_SINGLETON = None


def _get_wholebody(device: str = "cuda"):
    global _WHOLEBODY_SINGLETON
    if _WHOLEBODY_SINGLETON is None:
        # Local import to avoid heavy dependency at module import time
        from rtmlib import Wholebody
        _WHOLEBODY_SINGLETON = Wholebody(
            to_openpose=False,
            mode="lightweight",
            backend="onnxruntime",
            device=device,
        )
    return _WHOLEBODY_SINGLETON


def _extract_pose_to_pkl(
    video_path: str,
    output_pkl: str,
    max_workers: int = 16,
    overwrite: bool = False,
    max_frames: int = 0,
):
    """Extract whole-body pose for a single mp4 and save to pickle.

    Saved format matches `demo/online_inference.py` / `demo/pose_extraction.py`:
    {"keypoints": [ (1,133,2), ... ], "scores": [ (1,133), ... ]}
    with keypoints normalized by image width/height.
    """
    import cv2
    import pickle
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor

    if (not overwrite) and os.path.exists(output_pkl):
        return
    os.makedirs(os.path.dirname(output_pkl), exist_ok=True)

    wholebody = _get_wholebody(device="cuda" if torch.cuda.is_available() else "cpu")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"fail to open: {video_path}")

    frame_indices = None
    if max_frames and max_frames > 0:
        try:
            frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        except Exception:
            frame_cnt = 0
        if frame_cnt and frame_cnt > max_frames:
            frame_indices = set(np.linspace(0, frame_cnt - 1, max_frames).round().astype(int).tolist())

    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if (frame_indices is None) or (frame_idx in frame_indices):
            frames.append(frame)
        frame_idx += 1
    cap.release()

    def _process(frame):
        frame = np.uint8(frame)
        keypoints, scores = wholebody(frame)
        h, w, _ = frame.shape
        return keypoints, scores, (w, h)

    data = {"keypoints": [], "scores": []}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for keypoints, scores, (w, h) in ex.map(_process, frames):
            data["keypoints"].append(keypoints / np.array([w, h])[None, None])
            data["scores"].append(scores)

    with open(output_pkl, "wb") as f:
        pickle.dump(data, f)

# load sub-pose
def load_part_kp(skeletons, confs, force_ok=False):
    thr = 0.3
    kps_with_scores = {}
    scale = None
    
    for part in ['body', 'left', 'right', 'face_all']:
        kps = []
        confidences = []
        
        for skeleton, conf in zip(skeletons, confs):
            skeleton = skeleton[0]
            conf = conf[0]
            
            if part == 'body':
                hand_kp2d = skeleton[[0] + [i for i in range(3, 11)], :]
                confidence = conf[[0] + [i for i in range(3, 11)]]
            elif part == 'left':
                hand_kp2d = skeleton[91:112, :]
                hand_kp2d = hand_kp2d - hand_kp2d[0, :]
                confidence = conf[91:112]
            elif part == 'right':
                hand_kp2d = skeleton[112:133, :]
                hand_kp2d = hand_kp2d - hand_kp2d[0, :]
                confidence = conf[112:133]
            elif part == 'face_all':
                hand_kp2d = skeleton[[i for i in list(range(23,23+17))[::2]] + [i for i in range(83, 83 + 8)] + [53], :]
                hand_kp2d = hand_kp2d - hand_kp2d[-1, :]
                confidence = conf[[i for i in list(range(23,23+17))[::2]] + [i for i in range(83, 83 + 8)] + [53]]

            else:
                raise NotImplementedError
            
            kps.append(hand_kp2d)
            confidences.append(confidence)
            
        kps = np.stack(kps, axis=0)
        confidences = np.stack(confidences, axis=0)
        
        if part == 'body':
            if force_ok:
                result, scale, _ = crop_scale(np.concatenate([kps, confidences[...,None]], axis=-1), thr)

            else:
                result, scale, _ = crop_scale(np.concatenate([kps, confidences[...,None]], axis=-1), thr)
        else:
            assert not scale is None
            result = np.concatenate([kps, confidences[...,None]], axis=-1)
            if scale==0:
                result = np.zeros(result.shape)
            else:
                result[...,:2] = (result[..., :2]) / scale
                result = np.clip(result, -1, 1)
                # mask useless kp
                result[result[...,2]<=thr] = 0
            
        kps_with_scores[part] = torch.tensor(result)
        
    return kps_with_scores


# input: T, N, 3
# input is un-normed joints
def crop_scale(motion, thr):
    '''
        Motion: [(M), T, 17, 3].
        Normalize to [-1, 1]
    '''
    result = copy.deepcopy(motion)
    valid_coords = motion[motion[..., 2]>thr][:,:2]
    if len(valid_coords) < 4:
        return np.zeros(motion.shape), 0, None
    xmin = min(valid_coords[:,0])
    xmax = max(valid_coords[:,0])
    ymin = min(valid_coords[:,1])
    ymax = max(valid_coords[:,1])
    # ratio = np.random.uniform(low=scale_range[0], high=scale_range[1], size=1)[0]
    ratio = 1
    scale = max(xmax-xmin, ymax-ymin) * ratio
    if scale==0:
        return np.zeros(motion.shape), 0, None
    xs = (xmin+xmax-scale) / 2
    ys = (ymin+ymax-scale) / 2
    result[...,:2] = (motion[..., :2] - [xs,ys]) / scale
    result[...,:2] = (result[..., :2] - 0.5) * 2
    result = np.clip(result, -1, 1)
    # mask useless kp
    result[result[...,2]<=thr] = 0
    return result, scale, [xs,ys]


# bbox of hands
def bbox_4hands(left_keypoints, right_keypoints, hw):
    # keypoints --> T,21,2
    # keypoints --> T,21,2
    
    def compute_bbox(keypoints):
        min_x = np.min(keypoints[..., 0], axis=1)
        min_y = np.min(keypoints[..., 1], axis=1)
        max_x = np.max(keypoints[..., 0], axis=1)
        max_y = np.max(keypoints[..., 1], axis=1)
        
        return (max_x+min_x)/2, (max_y+min_y)/2, (max_x-min_x), (max_y-min_y)
    H,W = hw
    
    if left_keypoints is None:
        left_keypoints = np.zeros([1,21,2])
        
    if right_keypoints is None:
        right_keypoints = np.zeros([1,21,2])
    # [T, 21, 2]
    left_mean_x, left_mean_y, left_diff_x, left_diff_y = compute_bbox(left_keypoints)
    left_mean_x = W*left_mean_x
    left_mean_y = H*left_mean_y
    
    left_diff_x = W*left_diff_x
    left_diff_y = H*left_diff_y
    
    left_diff_x = max(left_diff_x)
    left_diff_y = max(left_diff_y)
    left_box_hw = max(left_diff_x,left_diff_y)
    
    right_mean_x, right_mean_y, right_diff_x, right_diff_y = compute_bbox(right_keypoints)
    right_mean_x = W*right_mean_x
    right_mean_y = H*right_mean_y
    
    right_diff_x = W*right_diff_x
    right_diff_y = H*right_diff_y
    
    right_diff_x = max(right_diff_x)
    right_diff_y = max(right_diff_y)
    right_box_hw = max(right_diff_x,right_diff_y)
    
    box_hw = int(max(left_box_hw, right_box_hw) * 1.2 / 2) * 2
    box_hw = max(box_hw, 0)

    left_new_box = np.stack([left_mean_x - box_hw/2, left_mean_y - box_hw/2, left_mean_x + box_hw/2, left_mean_y + box_hw/2]).astype(np.int16)
    right_new_box = np.stack([right_mean_x - box_hw/2, right_mean_y - box_hw/2, right_mean_x + box_hw/2, right_mean_y + box_hw/2]).astype(np.int16)
    
    return left_new_box.transpose(1,0), right_new_box.transpose(1,0), box_hw

def load_support_rgb_dict(tmp, skeletons, confs, full_path, data_transform, psamp: float = 0.1):
    support_rgb_dict = {}
    
    confs = np.array(confs)
    skeletons = np.array(skeletons) 

    # sample index of low scores
    left_confs_filter = confs[:,0,91:112].mean(-1)
    left_confs_filter_indices = np.where(left_confs_filter > 0.3)[0]

    if len(left_confs_filter_indices) == 0:
        left_sampled_indices = None
        left_skeletons = None
    else:
        
        left_confs = confs[left_confs_filter_indices]
        left_confs = left_confs[:,0,[95,99,103,107,111]].min(-1)
        
        left_weights = np.max(left_confs) - left_confs + 1e-5
        left_probabilities = left_weights / np.sum(left_weights)
        
        psamp = float(psamp) if psamp is not None else 0.1
        psamp = max(0.0, min(1.0, psamp))
        left_sample_size = int(np.ceil(psamp * len(left_confs_filter_indices)))
        left_sample_size = max(left_sample_size, 1)
        
        left_sampled_indices = np.random.choice(left_confs_filter_indices.tolist(), 
                                                size=left_sample_size, 
                                                replace=False, 
                                                p=left_probabilities)
        # left_sampled_indices: values: 0-255(0,max_len)
        # tmp: values: 0-(end-start)
        left_sampled_indices = np.sort(left_sampled_indices)
        
        left_skeletons = skeletons[left_sampled_indices,0,91:112]

    right_confs_filter = confs[:,0,112:].mean(-1)
    right_confs_filter_indices = np.where(right_confs_filter > 0.3)[0]
    if len(right_confs_filter_indices) == 0:
        right_sampled_indices = None
        right_skeletons = None
        
    else:
        right_confs = confs[right_confs_filter_indices]
        right_confs = right_confs[:,0,[95+21,99+21,103+21,107+21,111+21]].min(-1)

        right_weights = np.max(right_confs) - right_confs + 1e-5
        right_probabilities = right_weights / np.sum(right_weights)
        
        right_sample_size = int(np.ceil(psamp * len(right_confs_filter_indices)))
        right_sample_size = max(right_sample_size, 1)
        
        right_sampled_indices = np.random.choice(right_confs_filter_indices.tolist(), 
                                                 size=right_sample_size, 
                                                 replace=False, 
                                                 p=right_probabilities)
        right_sampled_indices = np.sort(right_sampled_indices)
        
        right_skeletons = skeletons[right_sampled_indices,0,112:133]
        
    image_size = 112
    all_indices = []
    if not left_sampled_indices is None:
        all_indices.append(left_sampled_indices)
    if not right_sampled_indices is None:
        all_indices.append(right_sampled_indices)
    if len(all_indices) == 0:
        support_rgb_dict['left_sampled_indices'] = torch.tensor([-1])
        support_rgb_dict['left_hands'] = torch.zeros(1, 3, image_size, image_size)
        support_rgb_dict['left_skeletons_norm'] = torch.zeros(1, 21, 2)
        
        support_rgb_dict['right_sampled_indices'] = torch.tensor([-1])
        support_rgb_dict['right_hands'] = torch.zeros(1, 3, image_size, image_size)
        support_rgb_dict['right_skeletons_norm'] = torch.zeros(1, 21, 2)

        return support_rgb_dict

    sampled_indices = np.concatenate(all_indices)
    sampled_indices = np.unique(sampled_indices)
    sampled_indices_real = tmp[sampled_indices]

    # load image sample
    imgs = load_video_support_rgb(full_path, sampled_indices_real)

    # get hand bbox
    left_new_box, right_new_box, box_hw = bbox_4hands(left_skeletons,
                                                        right_skeletons,
                                                        imgs[0].shape[:2])
    
    # crop left and right hand
    image_size = 112
    if box_hw == 0:
        support_rgb_dict['left_sampled_indices'] = torch.tensor([-1])
        support_rgb_dict['left_hands'] = torch.zeros(1, 3, image_size, image_size)
        support_rgb_dict['left_skeletons_norm'] = torch.zeros(1, 21, 2)
        
        support_rgb_dict['right_sampled_indices'] = torch.tensor([-1])
        support_rgb_dict['right_hands'] = torch.zeros(1, 3, image_size, image_size)
        support_rgb_dict['right_skeletons_norm'] = torch.zeros(1, 21, 2)

        return support_rgb_dict

    factor = image_size / box_hw
    
    if left_sampled_indices is None:
        left_hands = torch.zeros(1, 3, image_size, image_size)
        left_skeletons_norm = torch.zeros(1, 21, 2)
        
    else:
        left_hands = torch.zeros(len(left_sampled_indices), 3, image_size, image_size)
            
        left_skeletons_norm = left_skeletons * imgs[0].shape[:2][::-1] - left_new_box[:, None, [0,1]]
        left_skeletons_norm = left_skeletons_norm / box_hw
        left_skeletons_norm = left_skeletons_norm.clip(0,1)

    if right_sampled_indices is None:
        right_hands = torch.zeros(1, 3, image_size, image_size)
        right_skeletons_norm = torch.zeros(1, 21, 2)
        
    else:
        right_hands = torch.zeros(len(right_sampled_indices), 3, image_size, image_size)
        
        right_skeletons_norm = right_skeletons * imgs[0].shape[:2][::-1] - right_new_box[:, None, [0,1]]
        right_skeletons_norm = right_skeletons_norm / box_hw
        right_skeletons_norm = right_skeletons_norm.clip(0,1)
    left_idx = 0
    right_idx = 0

    for idx, img in enumerate(imgs):
        mapping_idx = sampled_indices[idx]
        if not left_sampled_indices is None and left_idx < len(left_sampled_indices) and mapping_idx == left_sampled_indices[left_idx]:
            box = left_new_box[left_idx]
            
            img_draw = np.uint8(copy.deepcopy(img))[box[1]:box[3],box[0]:box[2],:]
            img_draw = np.pad(img_draw, ((0, max(0, box_hw-img_draw.shape[0])), (0, max(0, box_hw-img_draw.shape[1])), (0, 0)), mode='constant', constant_values=0)
            
            f_img = Image.fromarray(img_draw).convert('RGB').resize((image_size, image_size))
            f_img = data_transform(f_img).unsqueeze(0)
            left_hands[left_idx] = f_img
            left_idx += 1
            
        if not right_sampled_indices is None and right_idx < len(right_sampled_indices) and mapping_idx == right_sampled_indices[right_idx]:
            box = right_new_box[right_idx]
            
            img_draw = np.uint8(copy.deepcopy(img))[box[1]:box[3],box[0]:box[2],:]
            img_draw = np.pad(img_draw, ((0, max(0, box_hw-img_draw.shape[0])), (0, max(0, box_hw-img_draw.shape[1])), (0, 0)), mode='constant', constant_values=0)
            
            f_img = Image.fromarray(img_draw).convert('RGB').resize((image_size, image_size))
            f_img = data_transform(f_img).unsqueeze(0)
            right_hands[right_idx] = f_img
            right_idx += 1
   
    if left_sampled_indices is None:
        left_sampled_indices = np.array([-1])
        
    if right_sampled_indices is None:
        right_sampled_indices = np.array([-1])

    # get index, images and keypoints priors
    support_rgb_dict['left_sampled_indices'] = torch.tensor(left_sampled_indices)
    support_rgb_dict['left_hands'] = left_hands
    support_rgb_dict['left_skeletons_norm'] = torch.tensor(left_skeletons_norm)
    
    support_rgb_dict['right_sampled_indices'] = torch.tensor(right_sampled_indices)
    support_rgb_dict['right_hands'] = right_hands
    support_rgb_dict['right_skeletons_norm'] = torch.tensor(right_skeletons_norm)

    return support_rgb_dict


# use split rgb video for save time
def load_video_support_rgb(path, tmp):
    vr = VideoReader(path, num_threads=1, ctx=cpu(0))
    
    vr.seek(0)
    buffer = vr.get_batch(tmp).asnumpy()
    batch_image = buffer
    del vr

    return batch_image

# build base dataset
class Base_Dataset(Dataset.Dataset):
    def collate_fn(self, batch):
        tgt_batch,src_length_batch,name_batch,pose_tmp,gloss_batch = [],[],[],[],[]
        
        for name_sample, pose_sample, text, gloss, _ in batch:
            name_batch.append(name_sample)
            pose_tmp.append(pose_sample)
            tgt_batch.append(text)
            gloss_batch.append(gloss)

        src_input = {}

        keys = pose_tmp[0].keys()
        for key in keys:
            max_len = max([len(vid[key]) for vid in pose_tmp])
            video_length = torch.LongTensor([len(vid[key]) for vid in pose_tmp])
            
            padded_video = [torch.cat(
                (
                    vid[key],
                    vid[key][-1][None].expand(max_len - len(vid[key]), -1, -1),
                )
                , dim=0)
                for vid in pose_tmp]
            
            img_batch = torch.stack(padded_video,0)
            
            src_input[key] = img_batch
            if 'attention_mask' not in src_input.keys():
                src_length_batch = video_length

                mask_gen = []
                for i in src_length_batch:
                    tmp = torch.ones([i]) + 7
                    mask_gen.append(tmp)
                mask_gen = pad_sequence(mask_gen, padding_value=0,batch_first=True)
                img_padding_mask = (mask_gen != 0).long()
                src_input['attention_mask'] = img_padding_mask

                src_input['name_batch'] = name_batch
                src_input['src_length_batch'] = src_length_batch
                
        if self.rgb_support:
            support_rgb_dicts = {key:[] for key in batch[0][-1].keys()}
            for _, _, _, _, support_rgb_dict in batch:
                for key in support_rgb_dict.keys():
                    support_rgb_dicts[key].append(support_rgb_dict[key])
            
            for part in ['left', 'right']:
                index_key = f'{part}_sampled_indices'
                skeletons_key = f'{part}_skeletons_norm'
                rgb_key = f'{part}_hands'
                len_key = f'{part}_rgb_len'

                index_batch = torch.cat(support_rgb_dicts[index_key], 0)
                skeletons_batch = torch.cat(support_rgb_dicts[skeletons_key], 0)
                img_batch = torch.cat(support_rgb_dicts[rgb_key], 0)
                
                src_input[index_key] = index_batch
                src_input[skeletons_key] = skeletons_batch
                src_input[rgb_key] = img_batch
                src_input[len_key] = [len(index) for index in support_rgb_dicts[index_key]]

        tgt_input = {}
        tgt_input['gt_sentence'] = tgt_batch
        tgt_input['gt_gloss'] = gloss_batch

        return src_input, tgt_input


class S2T_Dataset(Base_Dataset):
    def __init__(self, path, args, phase):
        super(S2T_Dataset, self).__init__()
        self.args = args
        self.rgb_support = self.args.rgb_support
        self.max_length = args.max_length
        self.raw_data = utils.load_dataset_file(path)
        self.phase = phase

        if self.args.dataset == "CSL_Daily":
            self.pose_dir = pose_dirs[args.dataset]
            self.rgb_dir = rgb_dirs[args.dataset]

        elif self.args.dataset == "CE-CSL":
            # video_path already contains split/translator subdirs (e.g. train/A/train-00001.mp4)
            # so we use root dirs directly.
            self.pose_dir = pose_dirs[args.dataset]
            self.rgb_dir = rgb_dirs[args.dataset]
            
        elif "WLASL" in self.args.dataset:
            self.pose_dir = os.path.join(pose_dirs[args.dataset], phase)
            self.rgb_dir = os.path.join(rgb_dirs[args.dataset], phase)

        elif "How2Sign" in self.args.dataset:
            if phase == 'dev':
                raise NotImplementedError("How2Sign dev set is not supported")
            self.pose_dir = pose_dirs[args.dataset].format(phase)
            self.rgb_dir = os.path.join(rgb_dirs[args.dataset], phase)

        elif "OpenASL" in self.args.dataset:
            self.pose_dir = pose_dirs[args.dataset].format(phase)
            self.rgb_dir = os.path.join(rgb_dirs[args.dataset], phase)

        else:
            raise NotImplementedError

        self.list = list(self.raw_data.keys())

        # Optional fast-path for CE-CSL: only keep samples whose pose .pkl already exists.
        # This avoids extremely slow on-the-fly pose extraction when iterating many videos.
        if getattr(self.args, 'ce_csl_existing_pose_only', False) and self.args.dataset == "CE-CSL":
            filtered = []
            for k in self.list:
                sample = self.raw_data[k]
                video_path = sample.get('video_path', '')
                if not video_path:
                    continue
                pose_pkl = os.path.join(self.pose_dir, video_path.replace('.mp4', '.pkl'))
                if os.path.exists(pose_pkl):
                    filtered.append(k)
            self.list = filtered

        self.data_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
                                    ])

    def __len__(self):
        return len(self.list)
    
    def __getitem__(self, index):
        # Robust against a small number of corrupted/empty pose pkls.
        # When it happens, retry a few other samples instead of crashing the whole epoch.
        max_retry = int(getattr(self.args, 'dataset_retry', 10))
        last_err = None
        for _ in range(max_retry):
            key = self.list[index]
            sample = self.raw_data[key]

            text = sample['text']
            if "gloss" in sample.keys():
                gloss = " ".join(sample['gloss'])
            else:
                gloss = ''

            name_sample = sample['name']
            try:
                pose_sample, support_rgb_dict = self.load_pose(sample['video_path'])
                return name_sample, pose_sample, text, gloss, support_rgb_dict
            except Exception as e:
                last_err = e
                # pick another sample
                index = random.randint(0, len(self.list) - 1)
                continue

        raise RuntimeError(f"failed to load sample after {max_retry} retries, last_err={last_err}")
    
    def load_pose(self, path):
        pose_pkl = os.path.join(self.pose_dir, path.replace(".mp4", ".pkl"))
        if (not os.path.exists(pose_pkl)) and self.args.dataset == "CE-CSL":
            # On-demand pose extraction for CE-CSL
            video_full_path = os.path.join(self.rgb_dir, path)
            _extract_pose_to_pkl(
                video_full_path,
                pose_pkl,
                max_workers=getattr(self.args, "ce_csl_pose_max_workers", 16),
                overwrite=False,
                max_frames=getattr(self.args, "ce_csl_pose_max_frames", 128),
            )

        pose = _safe_pickle_load(pose_pkl)
            
        if 'start' in pose.keys():
            assert pose['start'] < pose['end']
            duration = pose['end'] - pose['start']
            start = pose['start']
        else:
            duration = len(pose['scores'])
            start = 0

        if duration <= 0:
            raise ValueError(f"invalid pose duration={duration}: {pose_pkl}")
                
        if duration > self.max_length:
            # Training can benefit from random temporal sampling (augmentation).
            # For eval/dev/test, use deterministic uniform sampling to make BLEU comparable
            # across runs and to better cover the whole sentence.
            if self.phase != 'train':
                tmp = np.linspace(0, duration - 1, self.max_length).round().astype(int).tolist()
            else:
                tmp = sorted(random.sample(range(duration), k=self.max_length))
        else:
            tmp = list(range(duration))
        
        tmp = np.array(tmp) + start
            
        skeletons = pose['keypoints']
        confs = pose['scores']
        skeletons_tmp = []
        confs_tmp = []
        for index in tmp:
            skeletons_tmp.append(skeletons[index])
            confs_tmp.append(confs[index])

        skeletons = skeletons_tmp
        confs = confs_tmp
    
        kps_with_scores = load_part_kp(skeletons, confs, force_ok=True)

        support_rgb_dict = {}
        if self.rgb_support:
            full_path = os.path.join(self.rgb_dir, path)
            support_rgb_dict = load_support_rgb_dict(
                tmp,
                skeletons,
                confs,
                full_path,
                self.data_transform,
                psamp=getattr(self.args, "rgb_psamp", 0.1),
            )
            
        return kps_with_scores, support_rgb_dict

    def __str__(self):
        return f'#total {len(self)}'

class S2T_Dataset_news(Base_Dataset):
    def __init__(self, path, args, phase):
        super(S2T_Dataset_news, self).__init__()
        self.args = args
        self.rgb_support = self.args.rgb_support
        self.phase = phase
        self.max_length = args.max_length

        path = pathlib.Path(path)

        with path.open(encoding='utf-8') as f:
            self.annotation = json.load(f)
       
        if self.args.dataset == "CSL_News":
            self.pose_dir = pose_dirs[args.dataset]
            self.rgb_dir = rgb_dirs[args.dataset]
      
        else:
            raise NotImplementedError
        sum_sample = len(self.annotation)
        self.data_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), 
                                    ])

        if phase == 'train':
            self.start_idx = int(sum_sample * 0.0)
            self.end_idx = int(sum_sample * 0.99)
        else:
            self.start_idx = int(sum_sample * 0.99)
            self.end_idx = int(sum_sample)

        # Optional: only keep samples whose required files exist.
        # This makes training robust while CSL-News is still downloading.
        #
        # When rgb_support is disabled, we only require pose to exist (so users can
        # train pose-only without downloading 400+ RGB archives).
        self._valid_indices = None
        if getattr(self.args, 'news_existing_only', False):
            valid = []
            need_rgb = bool(getattr(self.args, 'rgb_support', False))
            for idx in range(self.start_idx, self.end_idx):
                s = self.annotation[idx]
                pose_rel = s.get('pose', '')
                rgb_rel = s.get('video', '')
                if not pose_rel:
                    continue
                pose_path = os.path.join(self.pose_dir, pose_rel)
                if not os.path.exists(pose_path):
                    continue

                if need_rgb:
                    if (not rgb_rel) or (not os.path.exists(os.path.join(self.rgb_dir, rgb_rel))):
                        continue

                valid.append(idx)
            self._valid_indices = valid
        
    def __len__(self):
        if self._valid_indices is not None:
            return len(self._valid_indices)
        return self.end_idx - self.start_idx
    
    def __getitem__(self, index):
        max_retry = int(getattr(self.args, 'dataset_retry', 10))
        last_err = None
        for _ in range(max_retry):
            # Map to absolute annotation index.
            if self._valid_indices is not None:
                ann_idx = self._valid_indices[index]
            else:
                ann_idx = self.start_idx + index

            sample = self.annotation[ann_idx]
            text = sample['text']
            name_sample = sample['video']

            try:
                pose_sample, support_rgb_dict = self.load_pose(sample['pose'], sample['video'])
                # CSL-News provides text only (gloss may be unavailable)
                return name_sample, pose_sample, text, '', support_rgb_dict
            except Exception as e:
                last_err = e
                # pick another sample
                if self._valid_indices is not None:
                    index = random.randint(0, len(self._valid_indices) - 1)
                else:
                    index = random.randint(0, (self.end_idx - self.start_idx) - 1)
                continue

        raise RuntimeError(f"failed to load CSL_News sample after {max_retry} retries, last_err={last_err}")
    
    def load_pose(self, pose_name, rgb_name):
        pose = _safe_pickle_load(os.path.join(self.pose_dir, pose_name))
        full_path = os.path.join(self.rgb_dir, rgb_name)
        
        duration = len(pose['scores'])

        if duration <= 0:
            raise ValueError(f"invalid pose duration={duration}: {pose_name}")

        if duration > self.max_length:
            # Keep eval deterministic for stable BLEU/ROUGE.
            if self.phase != 'train':
                tmp = np.linspace(0, duration - 1, self.max_length).round().astype(int).tolist()
            else:
                tmp = sorted(random.sample(range(duration), k=self.max_length))
        else:
            tmp = list(range(duration))
        
        tmp = np.array(tmp)
            
        # dict_keys(['keypoints', 'scores'])
        # keypoints (1, 133, 2)
        # scores (1, 133)
        
        skeletons = pose['keypoints']
        confs = pose['scores']
        skeletons_tmp = []
        confs_tmp = []
        
        for index in tmp:
            skeletons_tmp.append(skeletons[index])
            confs_tmp.append(confs[index])

        skeletons = skeletons_tmp
        confs = confs_tmp
                
        # Force safe cropping even if confidence is low, to avoid NaNs.
        kps_with_scores = load_part_kp(skeletons, confs, force_ok=True)
        
        support_rgb_dict = {}
        if self.rgb_support:
            support_rgb_dict = load_support_rgb_dict(
                tmp,
                skeletons,
                confs,
                full_path,
                self.data_transform,
                psamp=getattr(self.args, "rgb_psamp", 0.1),
            )

        return kps_with_scores, support_rgb_dict

    def __str__(self):
        return f'#total {len(self)}'


class S2T_Dataset_combined(Base_Dataset):
    """Combine multiple S2T datasets into a single training set.

    This is mainly used to train a unified model on (CE-CSL + CSL_News).
    It returns the same sample tuple as other datasets.

    Sampling strategy:
    - concat: concatenate all samples (epoch length = sum)
    - balanced: interleave datasets to make each dataset contribute equally per epoch
    """

    def __init__(self, args, phase: str, datasets: str = "CE-CSL,CSL_News", sampling: str = "balanced"):
        super(S2T_Dataset_combined, self).__init__()
        self.args = args
        self.phase = phase
        self.rgb_support = self.args.rgb_support

        ds_names = [d.strip() for d in str(datasets).split(',') if d.strip()]
        if not ds_names:
            raise ValueError("combined datasets is empty")

        self._ds_names = ds_names
        self._datasets = []

        # Build each component dataset with a copied args where `dataset` is the component name.
        for name in self._ds_names:
            sub_args = copy.deepcopy(args)
            sub_args.dataset = name

            if name == 'CSL_News':
                # CSL_News uses a single JSON label file and does split inside dataset.
                # config.py already points to ./data/CSL_News/CSL_News_Labels.json
                from config import train_label_paths, dev_label_paths, test_label_paths
                label_path = {
                    'train': train_label_paths[name],
                    'dev': dev_label_paths[name],
                    'test': test_label_paths[name],
                }[phase]
                ds = S2T_Dataset_news(path=label_path, args=sub_args, phase=phase)
            else:
                from config import train_label_paths, dev_label_paths, test_label_paths
                label_path = {
                    'train': train_label_paths[name],
                    'dev': dev_label_paths[name],
                    'test': test_label_paths[name],
                }[phase]
                ds = S2T_Dataset(path=label_path, args=sub_args, phase=phase)

            self._datasets.append(ds)

        self._sampling = sampling
        self._index_map = []
        self._build_index_map()

    def _build_index_map(self):
        lens = [len(d) for d in self._datasets]
        if any(l <= 0 for l in lens):
            if getattr(self.args, 'combined_allow_empty', False):
                kept = [(n, d) for (n, d, l) in zip(self._ds_names, self._datasets, lens) if l > 0]
                dropped = [(n, l) for (n, l) in zip(self._ds_names, lens) if l <= 0]
                if not kept:
                    raise RuntimeError(f"all datasets are empty in combined: {list(zip(self._ds_names, lens))}")
                if dropped:
                    print(f"[warn] drop empty datasets in combined: {dropped}")
                self._ds_names = [n for n, _ in kept]
                self._datasets = [d for _, d in kept]
                lens = [len(d) for d in self._datasets]
            else:
                raise RuntimeError(f"empty dataset in combined: {list(zip(self._ds_names, lens))}")

        if self._sampling == 'concat':
            # (dataset_id, local_index)
            for ds_id, l in enumerate(lens):
                self._index_map.extend([(ds_id, i) for i in range(l)])
        elif self._sampling == 'balanced':
            # Interleave datasets to equalize contributions per epoch.
            max_len = max(lens)
            for i in range(max_len):
                for ds_id, l in enumerate(lens):
                    self._index_map.append((ds_id, i % l))
        else:
            raise ValueError(f"unknown combined sampling: {self._sampling}")

    def __len__(self):
        return len(self._index_map)

    def __getitem__(self, index):
        ds_id, local_idx = self._index_map[index]
        name = self._ds_names[ds_id]
        name_sample, pose_sample, text, gloss, support_rgb_dict = self._datasets[ds_id][local_idx]
        # Make sample name unique across datasets.
        name_sample = f"{name}::{name_sample}"
        return name_sample, pose_sample, text, gloss, support_rgb_dict

    def __str__(self):
        lens = [len(d) for d in self._datasets]
        return f"#total {len(self)} (sampling={self._sampling}) | " + ", ".join(
            [f"{n}:{l}" for n, l in zip(self._ds_names, lens)]
        )

class S2T_Dataset_online(Base_Dataset):
    def __init__(self, args):
        super(S2T_Dataset_online, self).__init__()
        self.args = args
        self.rgb_support = self.args.rgb_support
        self.max_length = args.max_length

        # place holder
        self.rgb_data = None
        self.pose_data = None

        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return 1

    def __getitem__(self, index):
        text = ''
        gloss = ''
        name_sample = 'online_data'

        pose_sample, support_rgb_dict = self.load_pose()

        return name_sample, pose_sample, text, gloss, support_rgb_dict

    def load_pose(self):
        pose = self.pose_data

        duration = len(pose['scores'])
        start = 0

        if duration > self.max_length:
            tmp = sorted(random.sample(range(duration), k=self.max_length))
        else:
            tmp = list(range(duration))

        tmp = np.array(tmp) + start

        skeletons = pose['keypoints']
        confs = pose['scores']
        skeletons_tmp = []
        confs_tmp = []
        for index in tmp:
            skeletons_tmp.append(skeletons[index])
            confs_tmp.append(confs[index])

        skeletons = skeletons_tmp
        confs = confs_tmp

        kps_with_scores = load_part_kp(skeletons, confs, force_ok=True)

        support_rgb_dict = {}
        if self.rgb_support:
            full_path = self.rgb_data
            support_rgb_dict = load_support_rgb_dict(
                tmp,
                skeletons,
                confs,
                full_path,
                self.data_transform,
                psamp=getattr(self.args, "rgb_psamp", 0.1),
            )

        return kps_with_scores, support_rgb_dict

    def __str__(self):
        return f'#total {len(self)}'
