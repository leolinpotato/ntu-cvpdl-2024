import os
from PIL import Image
import argparse

def parse_args():
    parser = argparse.ArgumentParser("Resize images to 512*512", add_help=False)
    parser.add_argument("--input_dir", default="", type=str, required=True)
    parser.add_argument("--output_dir", default="", type=str, required=True)

    args = parser.parse_args()
    return args

def resize_images(input_dir, output_dir, target_size=(512, 512)):
    """
    Resizes all images in the input directory to the specified target size and saves them to the output directory.

    Args:
        input_dir (str): Path to the input directory containing images.
        output_dir (str): Path to the output directory to save resized images.
        target_size (tuple): Target size for resizing (width, height).
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate over all files in the input directory
    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)
        try:
            # Open the image
            with Image.open(filepath) as img:
                # Resize the image
                img_resized = img.resize(target_size)
                
                # Save the resized image to the output directory
                output_path = os.path.join(output_dir, filename)
                img_resized.save(output_path)
                print(f"Resized and saved: {output_path}")
        except Exception as e:
            print(f"Error processing {filepath}: {e}")

if __name__ == "__main__":
    args = parse_args()
    resize_images(args.input_dir, args.output_dir, target_size=(512, 512))
