# model_name: 
#   Salesforce/blip2-opt-2.7b
#   Salesforce/blip2-opt-6.7b-coco
#   Salesforce/blip2-opt-6.7b
#   Salesforce/blip2-flan-t5-xl

python blip2.py \
  --model_name Salesforce/blip2-opt-6.7b-coco \
  --input_file visualization_200.json \
  --output_file visualization_200-opt-6.7b-coco.json \
  --image_dir images