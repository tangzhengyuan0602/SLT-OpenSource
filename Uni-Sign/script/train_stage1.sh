output_dir=${OUTPUT_DIR:-out/combined_stage1_poseonly}

# Override with DEEPSPEED_BIN if needed.
deepspeed_bin=${DEEPSPEED_BIN:-deepspeed}

$deepspeed_bin --include localhost:0 --master_port 29511 pre_training.py \
   --batch-size 16 \
   --gradient-accumulation-steps 8 \
   --epochs 20 \
   --opt AdamW \
   --lr 3e-4 \
   --quick_break 2048 \
   --output_dir $output_dir \
   --dataset "CE-CSL,CSL_News" \
   --combined_allow_empty \
   --news_existing_only \
   --ce_csl_existing_pose_only \
   --task SLT
