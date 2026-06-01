<h1 align="center">Uni-Sign Demo</h1>

<p align="center">
  A practical demo workspace for <code>Uni-Sign: Toward Unified Sign Language Understanding at Scale</code>,
  covering pose extraction, offline inference, sample playback, and a real-time web demo.
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2501.15187"><img src="https://img.shields.io/badge/arXiv-2501.15187-AD1C18.svg?logo=arXiv" alt="arXiv" /></a>
  <a href="https://huggingface.co/ZechengLi19/Uni-Sign"><img src="https://img.shields.io/badge/HuggingFace-Model%20Checkpoints-purple" alt="checkpoints" /></a>
  <a href="https://huggingface.co/datasets/ZechengLi19/CSL-News"><img src="https://img.shields.io/badge/HuggingFace-CSL--News%20RGB-blue" alt="dataset-rgb" /></a>
  <a href="https://huggingface.co/datasets/ZechengLi19/CSL-News_pose"><img src="https://img.shields.io/badge/HuggingFace-CSL--News%20Pose-yellow" alt="dataset-pose" /></a>
</p>

## Overview

This repository snapshot focuses on making `Uni-Sign` easier to run and demonstrate in practice. It bundles together:

- offline video inference,
- pose extraction utilities,
- demo-oriented online inference scripts,
- and a real-time single-window webcam interface with pose overlay plus asynchronous subtitle decoding.

## Demo Preview

<p align="center">
  <img src="Uni-Sign/demo/20260531-152708.gif" alt="Uni-Sign demo preview" width="960" />
</p>

<p align="center">
  <a href="Uni-Sign/demo/20260516-110153.mp4">Watch Demo Video 1</a>
  ·
  <a href="Uni-Sign/demo/20260516-110350.mp4">Watch Demo Video 2</a>
</p>

> On GitHub, the GIF above is rendered inline in the README. The two MP4 links open their corresponding file pages so readers can inspect the full demo output in full length.

## What You Can Do

| Capability | Description |
| --- | --- |
| Video inference | Upload or run a video and obtain translated text plus pose visualization. |
| Pose extraction | Convert raw sign language videos into pose files for downstream use. |
| Online inference | Run demo inference in pose-only or RGB-pose mode. |
| Real-time demo | Open a webcam, render pose overlay in real time, and trigger subtitle decoding asynchronously. |
| Sample playback | Use built-in sample videos for quick qualitative checks without preparing custom input first. |

## Quick Start

If you only want to launch the real-time demo as quickly as possible:

```bash
conda create --name Uni-Sign python=3.9
conda activate Uni-Sign

cd Uni-Sign
pip install -r requirements.txt
pip install onnxruntime-gpu cuda-toolkit

cd demo/rtmlib-main
pip install -e .
cd ../../

export CKPT_PATH=/path/to/best_checkpoint.pth
python ./demo/web_app.py --host :: --port 9001 --device cpu --finetune "${CKPT_PATH}"
```

Then open `http://[::1]:9001/` in your browser.

> Checkpoints, large datasets, training outputs, and cached artifacts are intentionally not bundled in this open-source export. Please download model weights separately before running inference.

## Installation

We recommend using a dedicated conda environment.

```bash
# create environment
conda create --name Uni-Sign python=3.9
conda activate Uni-Sign

# install project dependencies
cd Uni-Sign
pip install -r requirements.txt

# install demo pose runtime dependencies
pip install onnxruntime-gpu cuda-toolkit
cd demo/rtmlib-main
pip install -e .
cd ../../

# optional: BLEURT evaluation for How2Sign / OpenASL
git clone https://github.com/google-research/bleurt.git
cd bleurt
pip install .
cd ../
wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip
unzip BLEURT-20.zip
```

## Data Preparation

- Follow `Uni-Sign/docs/DATASET.md` for dataset preparation.
- Download the `mt5-base` weights and place them under `Uni-Sign/pretrained_weight/mt5-base`.
- If the `sentence-crop` folder is missing in the CSL-Daily dataset, refer to Issue `#7` in the original Uni-Sign repository.
- For How2Sign and OpenASL pose archives split into multiple files, merge and extract them before use.

```bash
# How2Sign
cat how2sign_pose_format.zip.* > how2sign_pose_format.zip
unzip how2sign_pose_format.zip

# OpenASL
cat openasl_pose_format.zip.* > openasl_pose_format.zip
unzip openasl_pose_format.zip
```

## Checkpoints and Resources

- Model checkpoints: <https://huggingface.co/ZechengLi19/Uni-Sign>
- CSL-News RGB dataset: <https://huggingface.co/datasets/ZechengLi19/CSL-News>
- CSL-News pose dataset: <https://huggingface.co/datasets/ZechengLi19/CSL-News_pose>
- OpenReview paper page: <https://openreview.net/pdf?id=0Xt7uT04cQ>

## Usage

All commands below should be executed from the `Uni-Sign` directory unless otherwise noted.

### 1. Pose Extraction

```bash
cd Uni-Sign

python ./demo/pose_extraction.py \
  --src_dir {video_dir} \
  --tgt_dir {pose_dir}
```

Notes:

- `{video_dir}` should contain the source `.mp4` files.
- `{pose_dir}` is the destination directory for extracted pose files.

### 2. Online Inference

```bash
cd Uni-Sign

# pose-only setting
ckpt_path=/path/to/best_checkpoint.pth
python ./demo/online_inference.py \
  --online_video {video_path} \
  --finetune ${ckpt_path}

# RGB-pose setting
ckpt_path=/path/to/best_checkpoint.pth
python ./demo/online_inference.py \
  --online_video {video_path} \
  --finetune ${ckpt_path} \
  --rgb_support
```

### 3. Real-Time Web Demo

```bash
cd Uni-Sign

python ./demo/web_app.py --host :: --port 9001 --device cpu --finetune /path/to/best_checkpoint.pth
```

After startup:

- open `http://[::1]:9001/`,
- allow browser camera access,
- and use the single-window preview for real-time pose overlay and subtitle updates.

If you do not want to pass a checkpoint path every time, you can also set `UNISIGN_CHECKPOINT=/path/to/best_checkpoint.pth` before launching the demo.

## Training and Evaluation

For the pre-training implementation, refer to Issue `#15` in the original Uni-Sign repository.

```bash
cd Uni-Sign

# stage 1: pose-only pre-training
bash ./script/train_stage1.sh

# stage 2: RGB-pose pre-training
bash ./script/train_stage2.sh

# stage 3: downstream fine-tuning
bash ./script/train_stage3.sh

# evaluation after stage 3 fine-tuning
bash ./script/eval_stage3.sh
```

## Repository Layout

```text
README.md                    # this top-level project overview
Uni-Sign/README.md           # original Uni-Sign project README
Uni-Sign/demo/README.md      # demo-oriented pose extraction and inference guide
Uni-Sign/demo/web_app.py     # real-time web demo entry
Uni-Sign/demo/rtmlib-main/   # lightweight pose runtime dependency
```

## Related Documents

- Original Uni-Sign README: `Uni-Sign/README.md`
- Demo-specific instructions: `Uni-Sign/demo/README.md`
- Dataset preparation guide: `Uni-Sign/docs/DATASET.md`

## Acknowledgement

The Uni-Sign codebase is adapted from `GFSLT-VLP`, while the pose and temporal encoder implementations are derived from `CoSign`. We also acknowledge the following projects that support this work:

- `SSVP-SLT`: <https://github.com/facebookresearch/ssvp_slt>
- `MMPose`: <https://github.com/open-mmlab/mmpose>
- `FUNASR`: <https://github.com/modelscope/FunASR>

## Contact

For questions about the original Uni-Sign project, please contact Zecheng Li at `lizecheng19@gmail.com`.

## Citation

If you find Uni-Sign useful for research or applications, please cite:

```bibtex
@article{li2025uni,
  title={Uni-Sign: Toward Unified Sign Language Understanding at Scale},
  author={Li, Zecheng and Zhou, Wengang and Zhao, Weichao and Wu, Kepeng and Hu, Hezhen and Li, Houqiang},
  journal={arXiv preprint arXiv:2501.15187},
  year={2025}
}
```
