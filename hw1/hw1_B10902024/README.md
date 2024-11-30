# NTU-CVPDL2024 HW1

## Installation
- python==3.10.15
- [checkpoint](https://drive.google.com/file/d/1k3lK4Z4tfmXzB3YWxc0SPq9BdOcL68Mk/view?usp=sharing)

```shell
# Move the downloaded checkpoint into checkpoints directory
mv pytorch_model.bin Relation-DETR/checkpoints

# Navigate to the project directory
cd Relation-DETR

# Create virtual environment
conda create -n cvpdl_hw1 python=3.10.15

# Activate the virtual environment
conda activate cvpdl_hw1

# Install dependencies
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt

cd ..
```

## Prepare the dataset
```shell
python convert_data.py
```

## Train
```shell
bash train.sh  # Train with shell script
CUDA_VISIBLE_DEVICES=0 accelerate launch main.py --mixed-precision fp16  # Train with python command
```

## Predict
```shell
bash predict.sh  # Predict with shell script
CUDA_VISIBLE_DEVICES=0 python inference.py \  # Predict with python command
    --image-dir ../test/images \
    --model-config configs/relation_detr/relation_detr_focalnet_large_lrf_fl4_800_1333.py \
    --checkpoint checkpoints/pytorch_model.bin \
    --output-dir . \
```

# Process the predicted file
```shell
python select_conf.py --conf 0.3 --input_json 'Relation-DETR/predictions.json' --output_json 'test_B10902024.json'
```

## Evaluation
```shell
python eval.py valid_B10902024.json valid_target.json
```