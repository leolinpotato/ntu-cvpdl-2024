# model_name:
#   masterful/gligen-1-4-generation-text-box
#   masterful/gligen-1-4-inpainting-text-box
# type:
#   text
#   layout
#   image

python gligen.py \
  --model_name masterful/gligen-1-4-generation-text-box \
  --input_file annotations/visualization_200-opt-6.7b-coco.json \
  --output_dir gligen_text \
  --prompt generated_text \
  --type text