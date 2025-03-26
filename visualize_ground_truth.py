import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

# Load COCO JSON file
coco_path = "/home/maruko/tree_delineation_Jan25/tree_delineation/data/train/_annotations.coco.json"

image_folder = "/home/maruko/tree_delineation_Jan25/tree_delineation/data/train/"

with open(coco_path, 'r') as f:
    coco_data = json.load(f)

# Initialize COCO API
coco = COCO(coco_path)


# Extract all image filenames
for i, img in enumerate(coco_data["images"]):
    image_filename = img["file_name"]
    image_path = f"{image_folder}{image_filename}"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_info = coco.imgs[list(coco.imgs.keys())[i]] 
    image_id = image_info["id"]

    # Get annotations for the image
    ann_ids = coco.getAnnIds(imgIds=image_id)
    annotations = coco.loadAnns(ann_ids)

    # Plot image
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.axis("off")

    # Overlay annotations
    for ann in annotations:
        bbox = ann["bbox"]  # [x, y, width, height]
        x, y, w, h = map(int, bbox)
        
        # Draw bounding box
        plt.gca().add_patch(plt.Rectangle((x, y), w, h, edgecolor="red", linewidth=2, fill=False))

        # Draw segmentation mask (if available)
        if "segmentation" in ann and ann["segmentation"]:
            for seg in ann["segmentation"]:
                poly = np.array(seg).reshape((len(seg) // 2, 2))
                plt.plot(poly[:, 0], poly[:, 1], marker='o', linestyle='-', color="cyan", linewidth=1)

        # Display category name
        category_id = ann["category_id"]
        category_name = coco.cats[category_id]["name"]
        plt.text(x, y - 5, category_name, color="white", fontsize=10, bbox=dict(facecolor="red", alpha=0.5))

    plt.savefig(f"./ground_truth/output_gt_{image_filename}.png")
    
    # Load image
    grey_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale

    # Apply Canny edge detection
    edges = cv2.Canny(grey_image, threshold1=300, threshold2=100)  # Adjust thresholds as needed

    # Display results
    plt.figure(figsize=(10, 5))
    
    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(grey_image, cmap="gray")
    plt.title(f"Original: {image_filename}")
    plt.axis("off")

    # Edge Detection Result
    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap="gray")
    plt.title("Canny Edge Detection")
    plt.axis("off")
    plt.savefig(f"./canny/canny_{image_filename}.png")
