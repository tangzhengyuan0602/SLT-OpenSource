ckpt_path=${CKPT_PATH:-out/combined_stage3_csl_daily_rgbpose/best_checkpoint.pth}

output_dir=${OUTPUT_DIR:-out/combined_eval_csl_daily_rgbpose}

deepspeed_bin=${DEEPSPEED_BIN:-deepspeed}

# single gpu inference
# RGB-pose setting
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
   --eval \
   --rgb_support

# # pose-only setting
#ckpt_path=out/combined_stage3_csl_daily_poseonly/best_checkpoint.pth
#$deepspeed_bin --include localhost:0 --master_port 29511 fine_tuning.py \
#   --batch-size 8 \
#   --gradient-accumulation-steps 1 \
#   --epochs 20 \
#   --opt AdamW \
#   --lr 3e-4 \
#   --output_dir out/combined_eval_csl_daily_poseonly \
#   --finetune $ckpt_path \
#   --dataset CSL_Daily \
#   --task SLT \
#   --eval \
