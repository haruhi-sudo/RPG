CUDA_VISIBLE_DEVICES=0 python generation/long_form.py \
  --model_path "path" \
  --input_file "data/eval_data/asqa_eval_gtr_top100.json" \
  --output_file "output/asqa.json" \
  --task "asqa"
