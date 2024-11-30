cd Relation-DETR

CUDA_VISIBLE_DEVICES=0 accelerate launch main.py \
    --mixed-precision fp16

cd ..