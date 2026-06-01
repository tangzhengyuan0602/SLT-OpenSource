output_dir=${OUTPUT_DIR:-out/combined_stage3_csl_daily_rgbpose}

deepspeed_bin=${DEEPSPEED_BIN:-deepspeed}

# RGB-pose setting (continue from combined pre-training stage2)
ckpt_path=${CKPT_PATH:-out/combined_stage2_rgbpose/best_checkpoint.pth}

$deepspeed_bin --include localhost:0 --master_port 29511 fine_tuning.py \
  --batch-size 8 \
  --gradient-accumulation-steps 1 \
  --epochs 20 \
  --opt AdamW \
  --lr 3e-4 \
  --output_dir $output_dir \
  --finetune $ckpt_path \
  --dataset CSL_Daily \
  --task SLT \
  --rgb_support # enable RGB-pose setting

# example of ISLR
# $deepspeed_bin --include localhost:0 --master_port 29511 fine_tuning.py \
#    --batch-size 8 \
#    --gradient-accumulation-steps 1 \
#    --epochs 20 \
#    --opt AdamW \
#    --lr 3e-4 \
#    --output_dir $output_dir \
#    --finetune $ckpt_path \
#    --dataset WLASL \
#    --task ISLR \
#    --max_length 64 \
#    --rgb_support # enable RGB-pose setting

## pose-only setting (run separately if needed)
# output_dir=out/combined_stage3_csl_daily_poseonly
# ckpt_path=out/combined_stage1_poseonly/best_checkpoint.pth
# deepspeed --include localhost:0,1,2,3 --master_port 29511 fine_tuning.py \
#   --batch-size 8 \
#   --gradient-accumulation-steps 1 \
#   --epochs 20 \
#   --opt AdamW \
#   --lr 3e-4 \
#   --output_dir $output_dir \
#   --finetune $ckpt_path \
#   --dataset CSL_Daily \
#   --task SLT
