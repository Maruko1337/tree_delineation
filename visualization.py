import matplotlib.pyplot as plt
import torchvision.transforms as T
import cv2
import numpy as np
import torch

def visual_loss(all_train_losses, all_val_losses, num_epochs, test_losses, best_model_fold):

    # Plot the loss curves
    plt.figure(figsize=(10, 5))
    for fold in range(len(all_train_losses)):
        plt.plot(range(1, num_epochs + 1), all_train_losses[fold], label=f'Train Fold {fold+1}')
        plt.plot(range(1, num_epochs + 1), all_val_losses[fold], label=f'Val Fold {fold+1}', linestyle="--")

    plt.plot(range(1, num_epochs + 1), test_losses, label=f'Test Fold {best_model_fold}')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss Over Folds")
    plt.legend()
    plt.grid()

    # Save the plot
    plt.savefig("loss_plot.png")

def visualize_predictions(model, data_loader, device, num_images=5):
    model.eval()
    transform = T.ToPILImage()

    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader):
            print(f"i = {i}")
            if i >= num_images:  # Limit visualization
                break

            images = [img.to(device) for img in images]
            # print(f"image is : {images}")
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

                image_data = {
                    "boxes": torch.tensor(valid_boxes, dtype=torch.float32).to(device),
                    "labels": torch.tensor(valid_labels, dtype=torch.int64).to(device),
                    "image_id": torch.tensor(image_targets[0]["image_id"], dtype=torch.int64).to(device),
                    "area": torch.tensor(valid_areas, dtype=torch.float32).to(device),
                    "iscrowd": torch.tensor(valid_iscrowd, dtype=torch.int64).to(device),
                    "scores":  torch.tensor(1, dtype=torch.int64).to(device)
                }
                processed_targets.append(image_data)

            outputs = model(images)
            outputs = processed_targets
            print(f"output is: {outputs}")

            # Convert images to numpy format
            img = images[0].permute(1, 2, 0).cpu().numpy()
            img = (img * 255).astype(np.uint8)  # Scale for visualization

            # Convert RGB to BGR (for OpenCV)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            print(f"{len(outputs[0]["boxes"])}")


            # Draw bounding boxes
            for j in range(len(outputs[0]["boxes"])):
                box = outputs[0]["boxes"][j].cpu().numpy().astype(int)
                label = outputs[0]["labels"][j].cpu().item()
                score = outputs[0]["scores"].cpu().item()
                print(f"score = {score}")

                if score > 0.5:  # Confidence threshold
                    cv2.rectangle(img, (box[0], box[1]-10), (box[2]-10, box[3]-10), (0, 255, 0), 2)
                    text = f"Class: tree, Score: {score:.2f}"
                    cv2.putText(img, text, (box[0]-100, box[1] +20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Convert back to RGB and plot
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(8, 6))
            plt.imshow(img)
            plt.axis("off")
            plt.savefig(f"output_{i}.png")