# model_name:
#   masterful/gligen-1-4-generation-text-box  -> layout
#   anhnct/Gligen_Text_Image                  -> image 
# type:
#   layout
#   image

CUDA_VISIBLE_DEVICE=4,6 python gligen.py \
  --model_name masterful/gligen-1-4-generation-text-box \
  --input_file annotations/label-opt-6.7b-coco.json \
  --output_dir gligen_layout \
  --image_dir images \
  --prompt generated_text \
  --type layout