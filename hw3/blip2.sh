# model_name: 
#   Salesforce/blip2-opt-2.7b
#   Salesforce/blip2-opt-6.7b-coco
#   Salesforce/blip2-opt-6.7b
#   Salesforce/blip2-flan-t5-xl

python blip2.py \
  --model_name Salesforce/blip2-flan-t5-xl \
  --input_file visualization_200.json \
  --output_file annotations/visualization_200-flan-t5-xl.json \
  --image_dir images