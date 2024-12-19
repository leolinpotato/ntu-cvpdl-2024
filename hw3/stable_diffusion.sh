CUDA_VISIBLE_DEVICES=6 python stable_diffusion.py \
  --model_name runwayml/stable-diffusion-v1-5 \
  --input_file annotations/visualization_200-opt-6.7b-coco.json \
  --output_dir stable_diffusion_w_prompt_200 \
  --prompt generated_text_w_prompt