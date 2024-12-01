# NTU-CVPDL2024 HW3

## Installation
- python==3.12.7
- CUDA 12.6

```shell
# Create virtual environment
conda create -n cvpdl_hw3 python=3.12.7

# Activate the virtual environment
conda activate cvpdl_hw3

# Install dependencies
pip install -r requirements.txt
```

## BLIP-2
```shell
bash blip2.sh
```
```shell
python blip2.py \
  --model_name Salesforce/blip2-flan-t5-xl \
  --input_file label.json \
  --output_file annotations/label-flan-t5-xl.json \
  --image_dir images
```

## GLIGEN
```shell
bash gligen.sh
```
```shell
python gligen.py \
  --model_name masterful/gligen-1-4-generation-text-box \
  --input_file annotations/visualization_200-opt-6.7b-coco.json \
  --output_dir gligen_text \
  --prompt generated_text \
  --type text
```

## FID
### Resize
```shell
bash resize.sh
```
```shell
python resize.py \
  --input_dir gligen_text \
  --output_dir gligen_text_512
```

### Calculate FID
```shell
python -m pytorch_fid images_512 gligen_text_512
``` 