from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import argparse
import os
import json
import ipdb
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser("BLIP2 Image Captioning", add_help=False)
    parser.add_argument("--model_name", default="", type=str, required=True)
    parser.add_argument("--input_file", default="", type=str, required=True)
    parser.add_argument("--output_file", default="", type=str, required=True)
    parser.add_argument("--image_dir", default="", type=str, required=True)

    args = parser.parse_args()
    return args

def generate_text(model, processor, image, device, prompt=None):
    if prompt:
        inputs = processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)
    else:
        inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
    generated_text = [text.strip() for text in generated_text]
    return generated_text

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = Blip2Processor.from_pretrained(args.model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        args.model_name, load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16
    )  # doctest: +IGNORE_RESULT

    with open(args.input_file, "r") as input_file:
        input_annotations = json.load(input_file)

    output_annotations = []
    batch_size = 16

    for i in tqdm(range(0, len(input_annotations), batch_size)):
        batch_annotation = input_annotations[i:i+batch_size]
        images = []
        prompts = []

        for input_annotation in batch_annotation:
            image_path = os.path.join(args.image_dir, input_annotation['image'])
            image = Image.open(image_path)
            images.append(image)
            prompts.append('The background looks')

        generated_text = generate_text(model, processor, images, device)
        generated_text_w_prompt = generate_text(model, processor, images, device, prompts)

        '''
        # Prompt Design 1: with label
        prompt_w_label = f"{generated_text}. {', '.join(input_annotation['labels'])}, height: {input_annotation['height']}, width: {input_annotation['width']}"

        # Prompt Design 2: with suffix
        prompt_w_suffix = f"{prompt_w_label}, HD quality, highly detailed"

        # Prompt Design 3: with background
        prompt = 'The background looks'
        background = generate_text(model, processor, image, device, prompt)
        prompt_w_background = f"{prompt_w_suffix}. {background}"

        # Prompt Design 4: Specific prompt image captioning
        num_persons = sum(1 for label in input_annotation['labels'] if label == "Person")

        if num_persons == 1:
            image_type = "single_worker"
        elif num_persons > 1:
            image_type = "multiple_workers"
        else:
            image_type = "general_scene"

        def select_prompt(image_type):
            if image_type == "single_worker":
                return "Describe the person and their safety equipment, focusing on what they are wearing and doing on the construction site."
            elif image_type == "multiple_workers":
                return "Describe the workers, their safety equipment, and their actions, including any tools or equipment they are using."
            elif image_type == "general_scene":
                return "Describe the construction site and the objects or workers in the scene, focusing on safety equipment and tools."
        
        prompt = select_prompt(image_type)
        prompt_w_specific_prompt = generate_text(model, processor, image, device, prompt)
        '''

        for idx, input_annotation in enumerate(batch_annotation):
            output_annotation = input_annotation
            output_annotation['generated_text'] = generated_text[idx]
            output_annotation['prompt_w_label'] = f"{generated_text[idx]}. {', '.join(list(set(input_annotation['labels'])))}, height: {input_annotation['height']}, width: {input_annotation['width']}"
            output_annotation['prompt_w_suffix'] = f"{output_annotation['prompt_w_label']}, HD quality, highly detailed"
            output_annotation['simple_background'] = f"{output_annotation['prompt_w_suffix']}. Background is simple"
            output_annotation['generated_text_w_prompt'] = f"{output_annotation['prompt_w_suffix']}. {generated_text_w_prompt[idx]}"
            output_annotations.append(output_annotation)
    
    with open(args.output_file, "w") as output_file:
        json.dump(output_annotations, output_file, indent=4)

if __name__ == "__main__":
    main()