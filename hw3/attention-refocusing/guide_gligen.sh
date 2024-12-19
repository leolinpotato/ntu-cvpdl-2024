CUDA_VISIBLE_DEVICES=2 python guide_gligen.py \
    --ckpt gligen_checkpoints/diffusion_pytorch_model.bin \
    --file_save counting_500 \
    --type counting \
    --box_pickle data_evaluate_LLM/gpt_generated_box/counting.p