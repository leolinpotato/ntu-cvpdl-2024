import torch
from diffusers import StableDiffusionPipeline
import argparse
import os
import json
import ipdb
from tqdm import tqdm
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser("Stable Diffusion", add_help=False)
    parser.add_argument("--model_name", default="", type=str, required=True)
    parser.add_argument("--input_file", default="", type=str, required=True)
    parser.add_argument("--output_dir", default="", type=str, required=True)
    parser.add_argument("--prompt", default="", type=str, required=True)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionPipeline.from_pretrained(args.model_name, torch_dtype=torch.float16)
    pipe = pipe.to(device)

    with open(args.input_file, "r") as input_file:
        input_annotations = json.load(input_file)

    for i, input_annotation in enumerate(tqdm(input_annotations)):
        prompt = input_annotation[args.prompt]
        image = pipe(prompt, width=512, height=512).images[0]
        image.save(os.path.join(args.output_dir, input_annotation['image']))

if __name__ == "__main__":
    main()