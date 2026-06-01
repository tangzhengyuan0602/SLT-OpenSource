import argparse
import os
import cv2
import glob
import pickle
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from rtmlib import Wholebody, draw_skeleton
from typing import Optional

def process_frame(frame, wholebody):
    frame = np.uint8(frame)
    keypoints, scores = wholebody(frame)
    H, W, C = frame.shape
    return keypoints, scores, [W, H]

def process_video(
    video_path,
    tgt_dir,
    wholebody,
    max_workers=16,
    overwrite=False,
    rel_path: Optional[str] = None,
    max_frames: int = 0,
):
    # Preserve directory structure when rel_path is provided
    if rel_path is None:
        output_path = os.path.join(tgt_dir, os.path.basename(video_path).replace(".mp4", ".pkl"))
    else:
        output_path = os.path.join(tgt_dir, rel_path).replace(".mp4", ".pkl")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path) and not overwrite:
        return

    data = {"keypoints": [], "scores": []}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"fail to open: {video_path}")
        return

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

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_frame, frame, wholebody) for frame in vid_data]
        for f in tqdm(futures, desc="Processing frames", total=len(vid_data)):
            results.append(f.result())

    for keypoints, scores, w_h in results:
        data['keypoints'].append(keypoints / np.array(w_h)[None, None])
        data['scores'].append(scores)

    with open(output_path, 'wb') as file:
        pickle.dump(data, file)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--src_dir", required=True, help="video dir path")
    parser.add_argument("--tgt_dir", required=True, help="pose dir path")

    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--backend", default="onnxruntime", choices=["opencv", "onnxruntime", "openvino"])
    parser.add_argument("--openpose_skeleton", action="store_true", help="use openpose format")
    parser.add_argument("--mode", default="lightweight", choices=["performance", "lightweight", "balanced"],)

    parser.add_argument("--video_extensions", nargs='+', default=["mp4"])
    parser.add_argument("--recursive", action="store_true", help="recursively search videos under src_dir and preserve subdirectories")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process first N videos after optional shuffle. 0 means all.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the discovered video list before applying --limit.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Uniformly sample at most N frames per video before pose extraction. 0 means all frames.",
    )
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.tgt_dir, exist_ok=True)

    wholebody = Wholebody(
        to_openpose=args.openpose_skeleton,
        mode=args.mode,
        backend=args.backend,
        device=args.device
    )

    video_files = []
    if args.recursive:
        for ext in args.video_extensions:
            video_files.extend(glob.glob(os.path.join(args.src_dir, f'**/*.{ext}'), recursive=True))
        # filter out directories just in case
        video_files = [p for p in video_files if os.path.isfile(p)]
    else:
        for ext in args.video_extensions:
            video_files.extend(glob.glob(os.path.join(args.src_dir, f'*.{ext}')))

    print(f"found {len(video_files)} videos")

    if args.shuffle:
        rng = np.random.RandomState(args.seed)
        rng.shuffle(video_files)

    if args.limit and args.limit > 0:
        video_files = video_files[: args.limit]
        print(f"limit enabled: only process {len(video_files)} videos")

    for video_path in tqdm(video_files, desc="Processing"):
        rel_path = None
        if args.recursive:
            rel_path = os.path.relpath(video_path, args.src_dir)
        process_video(
            video_path=video_path,
            tgt_dir=args.tgt_dir,
            wholebody=wholebody,
            max_workers=args.max_workers,
            overwrite=args.overwrite,
            rel_path=rel_path,
            max_frames=args.max_frames,
        )


if __name__ == "__main__":
    main()
