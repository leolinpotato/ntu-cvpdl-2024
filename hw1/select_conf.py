import json
import os
import argparse

categories = ['Person', 'Ear', 'Earmuffs', 'Face', 'Face-guard', 'Face-mask-medical', 'Foot', 
 'Tools', 'Glasses', 'Gloves', 'Helmet', 'Hands', 'Head', 'Medical-suit', 
 'Shoes', 'Safety-suit', 'Safety-vest', 'None']

def parse_args():
    parser = argparse.ArgumentParser(description="Confidence Score")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--input_json", type=str, default=None)
    parser.add_argument("--output_json", type=str, default=None)

    args = parser.parse_args()
    return args

def filter_predictions(input_json_path, output_json_path, confidence_score):
    # Load the JSON data from the input file
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    # Initialize a new dictionary to hold the filtered predictions
    filtered_data = {}

    # Iterate over each image in the data
    for image_name, predictions in data.items():
        boxes = predictions.get('boxes', [])
        labels = predictions.get('labels', [])
        scores = predictions.get('scores', [])

        # Check that all lists are of the same length
        if not (len(boxes) == len(labels) == len(scores)):
            print(f"Warning: Mismatched lengths in image {image_name}")
            continue  # Skip this image if lengths do not match

        # Filter out predictions based on the confidence score
        filtered_boxes = []
        filtered_labels = []

        for box, label, score in zip(boxes, labels, scores):
            if score > confidence_score:
                filtered_boxes.append(box)
                filtered_labels.append(label-1)

        # If there are any predictions left after filtering, add them to the filtered_data
        if filtered_boxes:
            filtered_data[image_name] = {
                'boxes': filtered_boxes,
                'labels': [categories[l] for l in filtered_labels],
            }
        else:
            print(f"No predictions above confidence score for image {image_name}")

    # Save the filtered data to the output JSON file
    with open(output_json_path, 'w') as f:
        json.dump(filtered_data, f, indent=4)

    print(f"Filtered predictions saved to {output_json_path}")

# Example usage:
if __name__ == "__main__":
    args = parse_args()
    # Specify the paths to the input and output JSON files
    input_json = args.input_json  # Replace with your input JSON file path
    output_json = args.output_json  # Replace with your desired output JSON file path

    # Specify the confidence score threshold
    confidence_threshold = args.conf  # Set your desired confidence score threshold

    # Call the function to filter predictions
    filter_predictions(input_json, output_json, confidence_threshold)
