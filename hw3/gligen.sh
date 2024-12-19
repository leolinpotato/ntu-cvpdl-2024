# model_name:
#   masterful/gligen-1-4-generation-text-box  -> layout
#   anhnct/Gligen_Text_Image                  -> image 
# type:
#   layout
#   image

CUDA_VISIBLE_DEVICES=0 python gligen.py \
  --model_name masterful/gligen-1-4-generation-text-box \
  --input_file annotations/visualization_200-flan-t5-xl.json \
  --output_dir gligen_layout_simple_background_200_flan \
  --image_dir images_512 \
  --prompt simple_background \
  --type layout