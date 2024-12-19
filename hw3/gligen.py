import torch
from diffusers import StableDiffusionGLIGENPipeline, StableDiffusionGLIGENTextImagePipeline
from diffusers.utils import load_image
import argparse
import os
import json
import ipdb
from tqdm import tqdm
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser("GLIGEN", add_help=False)
    parser.add_argument("--model_name", default="", type=str, required=True)
    parser.add_argument("--input_file", default="", type=str, required=True)
    parser.add_argument("--output_dir", default="", type=str, required=True)
    parser.add_argument("--image_dir", default="", type=str, required=True)    
    parser.add_argument("--prompt", default="", type=str, required=True)
    parser.add_argument("--type", default="", type=str, required=True)

    args = parser.parse_args()
    return args

def normalize_bbox(image_width, image_height, bbox):
    x_min, y_min, x_max, y_max = bbox
    normalized_x_min = x_min/image_width
    normalized_y_min = y_min/image_height
    normalized_x_max = x_max/image_width
    normalized_y_max = y_max/image_height
    return [normalized_x_min, normalized_y_min, normalized_x_max, normalized_y_max]

def main():
    args = parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Generate an image described by the prompt and
    # insert objects described by text at the region defined by bounding boxes
    if args.type == 'layout':
        pipe = StableDiffusionGLIGENPipeline.from_pretrained(
            args.model_name, variant="fp16", torch_dtype=torch.float16
        )
    elif args.type == 'image':
        pipe = StableDiffusionGLIGENTextImagePipeline.from_pretrained(
            args.model_name, torch_dtype=torch.float16
        )
    pipe = pipe.to(device)

    with open(args.input_file, "r") as input_file:
        input_annotations = json.load(input_file)

    for input_annotation in tqdm(input_annotations):
        prompt = input_annotation[args.prompt]
        boxes = [normalize_bbox(input_annotation['width'], input_annotation['height'], bbox) for bbox in input_annotation['bboxes']]
        phrases = input_annotation['labels']
        gligen_image = load_image(os.path.join(args.image_dir, input_annotation["image"]))

        if args.type == 'layout':
            image = pipe(
                prompt=prompt,
                gligen_phrases=phrases,
                gligen_boxes=boxes,
                gligen_scheduled_sampling_beta=1,
                output_type="pil",
                num_inference_steps=50,
                width=512,
                height=512
            ).images[0]
        elif args.type == 'image':
            image = pipe(
                prompt=prompt,
                gligen_phrases=phrases,
                gligen_images=[gligen_image],
                gligen_boxes=boxes,
                gligen_scheduled_sampling_beta=1,
                output_type="pil",
                num_inference_steps=50,
                width=512,
                height=512
            ).images[0]

        image.save(os.path.join(args.output_dir, input_annotation['image']))

if __name__ == "__main__":
    main()