import os
import sys

# Allow running this script from any working directory.
_UNI_SIGN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _UNI_SIGN_ROOT not in sys.path:
    sys.path.insert(0, _UNI_SIGN_ROOT)

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from models import Uni_Sign
import utils as utils
from datasets import S2T_Dataset_online
from pathlib import Path
from config import *
import argparse
import os
import cv2
import numpy as np
import pickle
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from rtmlib import Wholebody, draw_skeleton


DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def get_runtime_device(args):
    requested = str(getattr(args, "device", "auto")).lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is requested but no NVIDIA GPU is available.")
    if requested not in {"cuda", "cpu"}:
        raise ValueError(f"Unsupported device: {requested}")
    return requested

def main(args):
    print(args)
    utils.set_seed(args.seed)
    model = load_model(args)
    prediction = predict_video(args.online_video, model, args)
    print(f"Prediction result is: {prediction}")
    return prediction


def get_target_dtype(args):
    dtype_name = str(getattr(args, "dtype", "bf16")).lower()
    if dtype_name not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    target_dtype = DTYPE_MAP[dtype_name]
    if get_runtime_device(args) == "cpu" and target_dtype != torch.float32:
        return torch.float32
    return target_dtype


def create_online_dataloader(args, video_path, pose_data=None):
    if pose_data is None:
        pose_data = pose_extraction(
            video_path,
            max_workers=getattr(args, "ce_csl_pose_max_workers", 16),
            max_frames=getattr(args, "ce_csl_pose_max_frames", 128),
            device=get_runtime_device(args),
        )

    if not pose_data or not pose_data.get("scores"):
        raise ValueError(f"No valid pose frames were extracted from video: {video_path}")

    print(f"Creating dataset:")
    online_data = S2T_Dataset_online(args=args)
    print(online_data)
    online_data.rgb_data = video_path
    online_data.pose_data = pose_data

    online_sampler = torch.utils.data.SequentialSampler(online_data)
    return DataLoader(
        online_data,
        batch_size=1,
        collate_fn=online_data.collate_fn,
        sampler=online_sampler,
    )


def load_model(args):
    print(f"Creating model:")
    runtime_device = get_runtime_device(args)
    model = Uni_Sign(args=args)
    model.to(runtime_device)
    model.train()
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    if args.finetune != '':
        if not os.path.exists(args.finetune):
            raise FileNotFoundError(f"Checkpoint not found: {args.finetune}")
        print('***********************************')
        print('Load Checkpoint...')
        print('***********************************')
        state_dict = torch.load(args.finetune, map_location='cpu')['model']

        ret = model.load_state_dict(state_dict, strict=True)
        print('Missing keys: \n', '\n'.join(ret.missing_keys))
        print('Unexpected keys: \n', '\n'.join(ret.unexpected_keys))
    else:
        raise ValueError(
            "A checkpoint is required for inference. Please pass --finetune /path/to/best_checkpoint.pth "
            "or set UNISIGN_CHECKPOINT in your environment."
        )

    model.eval()
    model.to(get_target_dtype(args))
    return model


def predict_video(video_path, model, args, pose_data=None):
    data_loader = create_online_dataloader(args, video_path, pose_data=pose_data)
    return inference(data_loader, model, args)


class _CompatibleNumpyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def load_pose_file(pose_path):
    with open(pose_path, 'rb') as file:
        try:
            return pickle.load(file)
        except ModuleNotFoundError as exc:
            if "numpy._core" not in str(exc):
                raise
        file.seek(0)
        return _CompatibleNumpyUnpickler(file).load()

def process_frame(frame, wholebody):
    frame = np.uint8(frame)
    keypoints, scores = wholebody(frame)
    H, W, C = frame.shape
    return keypoints, scores, [W, H]

def pose_extraction(video_path, max_workers: int = 16, max_frames: int = 0, device: str = "cuda"):
    wholebody = Wholebody(
        to_openpose=False,
        mode="lightweight",
        backend="onnxruntime",
        device=device
    )

    data = {"keypoints": [], "scores": []}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"fail to open: {video_path}")
        return None

    frame_indices = None
    if max_frames and max_frames > 0:
        try:
            frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        except Exception:
            frame_cnt = 0
        if frame_cnt and frame_cnt > max_frames:
            frame_indices = set(np.linspace(0, frame_cnt - 1, max_frames).round().astype(int).tolist())

    vid_data = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if (frame_indices is None) or (frame_idx in frame_indices):
            vid_data.append(frame)
        frame_idx += 1

    cap.release()

    if not vid_data:
        return None

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_frame, frame, wholebody) for frame in vid_data]
        for f in tqdm(futures, desc="Processing frames", total=len(vid_data)):
            results.append(f.result())

    for keypoints, scores, w_h in results:
        data['keypoints'].append(keypoints / np.array(w_h)[None, None])
        data['scores'].append(scores)

    return data

def inference(data_loader, model, args):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    runtime_device = get_runtime_device(args)
    target_dtype = get_target_dtype(args)

    with torch.no_grad():
        tgt_pres = []

        for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, 10, header)):
            if target_dtype != None:
                for key in src_input.keys():
                    if isinstance(src_input[key], torch.Tensor):
                        src_input[key] = src_input[key].to(device=runtime_device, dtype=target_dtype)

            stack_out = model(src_input, tgt_input)

            gen_max_new_tokens = getattr(args, "gen_max_new_tokens", 100)
            gen_num_beams = getattr(args, "gen_num_beams", 4)
            gen_kwargs = {
                "length_penalty": getattr(args, "gen_length_penalty", 1.0),
                "no_repeat_ngram_size": getattr(args, "gen_no_repeat_ngram_size", 0),
                "repetition_penalty": getattr(args, "gen_repetition_penalty", 1.0),
            }
            if not gen_kwargs.get("no_repeat_ngram_size"):
                gen_kwargs.pop("no_repeat_ngram_size", None)

            output = model.generate(
                stack_out,
                max_new_tokens=gen_max_new_tokens,
                num_beams=gen_num_beams,
                **gen_kwargs,
            )

            for i in range(len(output)):
                tgt_pres.append(output[i])

    tokenizer = model.mt5_tokenizer
    padding_value = tokenizer.eos_token_id

    # `pad_sequence` supports variable-length 1D tensors directly.
    # Avoid fixed-length padding which can fail for long generations.
    tgt_pres = pad_sequence(tgt_pres, batch_first=True, padding_value=padding_value)
    tgt_pres = tokenizer.batch_decode(tgt_pres, skip_special_tokens=True)

    return tgt_pres[0] if tgt_pres else ""


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser('Uni-Sign scripts', parents=[utils.get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
