# cd Relation-DETR

# CUDA_VISIBLE_DEVICES=0 python inference.py \
#     --image-dir ../test/images \
#     --model-config configs/relation_detr/relation_detr_focalnet_large_lrf_fl4_800_1333.py \
#     --checkpoint checkpoints/pytorch_model.bin \
#     --output-dir . \

# cd ..

cd Relation-DETR

CUDA_VISIBLE_DEVICES=0 python inference.py \
    --image-dir /data1/leolin/CVPDL/ntu-cvpdl-2024/hw3/stable_diffusion_200 \
    --model-config configs/relation_detr/relation_detr_focalnet_large_lrf_fl4_800_1333.py \
    --checkpoint checkpoints/pytorch_model.bin \
    --output-file /data1/leolin/CVPDL/ntu-cvpdl-2024/hw3/predicted_visualization_200_stable_diffusion.json \

cd ..