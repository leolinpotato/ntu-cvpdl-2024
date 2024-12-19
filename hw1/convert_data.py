import os
import json
import cv2

def convert_bbox_yolo_to_coco(image_width, image_height, bbox):
    x_center, y_center, width, height = bbox
    x_min = (x_center - width / 2) * image_width
    y_min = (y_center - height / 2) * image_height
    width = width * image_width
    height = height * image_height
    return [x_min, y_min, width, height]

def create_coco_json(images_folder, labels_folder, output_json, category_list):
    images = []
    annotations = []
    categories = []

    # Build the category section of COCO
    for i, category_name in enumerate(category_list):
        categories.append({
            "id": i + 1,  # COCO IDs start at 1
            "name": category_name,
            "supercategory": category_name
        })

    annotation_id = 1
    image_id = 1

    for image_file in os.listdir(images_folder):
        if image_file.endswith((".jpeg", ".jpg", ".png")):
            image_path = os.path.join(images_folder, image_file)
            image = cv2.imread(image_path)
            height, width, _ = image.shape

            images.append({
                "id": image_id,
                "file_name": image_file,
                "height": height,
                "width": width
            })

            # Corresponding label file
            label_file = os.path.join(labels_folder, os.path.splitext(image_file)[0] + ".txt")
            if os.path.exists(label_file):
                with open(label_file, 'r') as f:
                    for line in f.readlines():
                        class_id, x_center, y_center, bbox_width, bbox_height = map(float, line.strip().split())

                        # Convert YOLO bbox format to COCO
                        bbox = convert_bbox_yolo_to_coco(width, height, [x_center, y_center, bbox_width, bbox_height])

                        annotations.append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": int(class_id) + 1,  # COCO category IDs start at 1
                            "bbox": bbox,
                            "area": bbox[2] * bbox[3],  # area = width * height
                            "iscrowd": 0
                        })
                        annotation_id += 1

            image_id += 1

    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(output_json, 'w') as f:
        json.dump(coco_format, f, indent=4)

if __name__ == "__main__":
    train_images = "train/images"
    train_labels = "train/labels"
    valid_images = "valid/images"
    valid_labels = "valid/labels"

    categories = ["Person", "Ear", "Earmuffs", "Face", "Face-guard", "Face-mask-medical", "Foot", "Tools", "Glasses", "Gloves", "Helmet", "Hands", "Head", "Medical-suit", "Shoes", "Safefy-suit", "Safefy-vest"]  # Replace with your actual category names

    # Convert train set
    create_coco_json(train_images, train_labels, "annotations/train.json", categories)
    
    # Convert validation set
    create_coco_json(valid_images, valid_labels, "annotations/valid.json", categories)
