#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# evaluate pretrained weights (Stage 2)
# Need to use the SLT evaluation logic in fine_tuning.py
python fine_tuning.py \
    --dataset CE-CSL,CSL_News \
    --finetune pretrained_weight/uni-sign-hf/csl_stage2_weight.pth \
    --eval \
    --task SLT \
    --batch_size 16 \
    --num_workers 4 \
    --dtype fp32
