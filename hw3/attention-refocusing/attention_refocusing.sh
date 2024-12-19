CUDA_VISIBLE_DEVICES=5 python attention_refocusing.py \
    --model_name gligen_checkpoints/diffusion_pytorch_model.bin \
    --input_file ../annotations/visualization_200_altered-opt-6.7b-coco.json \
    --prompt simple_background \
    --output_dir visualization_200_simple_background \
