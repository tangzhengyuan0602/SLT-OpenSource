output_dir=${OUTPUT_DIR:-out/combined_stage2_rgbpose}

deepspeed_bin=${DEEPSPEED_BIN:-deepspeed}

ckpt_path=${CKPT_PATH:-out/combined_stage1_poseonly/best_checkpoint.pth}

$deepspeed_bin --include localhost:0 --master_port 29511 pre_training.py \
   --batch-size 4 \
   --gradient-accumulation-steps 8 \
   --epochs 5 \
   --opt AdamW \
   --lr 3e-4 \
   --quick_break 2048 \
   --output_dir $output_dir \
   --finetune $ckpt_path \
   --dataset "CE-CSL,CSL_News" \
   --combined_allow_empty \
   --news_existing_only \
   --ce_csl_existing_pose_only \
   --task SLT \
   --rgb_support
