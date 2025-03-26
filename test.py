import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from pycocotools.coco import COCO
from torchvision.datasets import CocoDetection

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn

def test_model(model, test_loader, optimizer, scheduler, num_epochs=10, device="cuda"):
    test_losses = []
    # Training loop
    for epoch in range(num_epochs):  # Number of epochs
        model.eval()
        test_loss = 0
        for images, targets in test_loader:
            images = [img.to(device) for img in images]

            processed_targets = []
            for image_targets in targets:
                if not isinstance(image_targets, list):
                    raise TypeError(f"Expected a list of dictionaries, but got {type(image_targets)}")

                valid_boxes = []
                valid_labels = []
                valid_areas = []
                valid_iscrowd = []

                for t in image_targets:
                    if not isinstance(t, dict):
                        raise TypeError(f"Expected dictionary, but got {type(t)}")

                    x, y, w, h = t["bbox"]
                    if w > 0 and h > 0:  # Ensure positive width and height
                        valid_boxes.append([x, y, x + w, y + h])  # Convert to (x_min, y_min, x_max, y_max)
                        valid_labels.append(t["category_id"])
                        valid_areas.append(t["area"])
                        valid_iscrowd.append(t["iscrowd"])

                if len(valid_boxes) == 0:
                    continue  # Skip images with no valid boxes

                image_data = {
                    "boxes": torch.tensor(valid_boxes, dtype=torch.float32).to(device),
                    "labels": torch.tensor(valid_labels, dtype=torch.int64).to(device),
                    "image_id": torch.tensor(image_targets[0]["image_id"], dtype=torch.int64).to(device),
                    "area": torch.tensor(valid_areas, dtype=torch.float32).to(device),
                    "iscrowd": torch.tensor(valid_iscrowd, dtype=torch.int64).to(device),
                }
                processed_targets.append(image_data)

            # Skip batch if no valid targets
            if len(processed_targets) == 0:
                continue

            # Forward pass
            optimizer.zero_grad()
            loss_dict = model(images, processed_targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass and optimization
            losses.backward()
            optimizer.step()
            test_losses.append(losses.item())

        scheduler.step()
        print(f"Epoch {epoch + 1}, Loss: {test_losses[-1]:.4f}")
    return test_losses

