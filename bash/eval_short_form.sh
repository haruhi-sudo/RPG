CUDA_VISIBLE_DEVICES=5 python generation/short_form.py \
  --model_path "path" \
  --input_file "data/eval_data/health_claims_processed.jsonl" \
  --output_file "output/hc.json" \
  --task "fever"
