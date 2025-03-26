import torch
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from pycocotools.coco import COCO
from torchvision.datasets import CocoDetection

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import matplotlib.pyplot as plt

from train import get_model, train_model  # Import from train.py
from validation import *
from test import *
from visualization import *

# Check if CUDA is available
if torch.cuda.is_available():
    print(f"Number of GPUs: {torch.cuda.device_count()}")
else:
    print("No GPU available.")


from data.dataset import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define transforms for images and targets
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Path to COCO annotations and images
root_dir = "/home/maruko/tree_delineation_Jan25/tree_delineation/data/train"
ann_file = "/home/maruko/tree_delineation_Jan25/tree_delineation/data/train/_annotations.coco.json"

# Store loss values for plotting
all_train_losses = []
all_val_losses = []


# Load the COCO dataset
# coco_dataset = CocoDetection(root=root_dir, annFile=ann_file, transform=image_transform)
coco_dataset = FixedCocoDetection(root=root_dir, annFile=ann_file, transform=image_transform)


train_val_indices, test_indices = train_test_split(
    range(len(coco_dataset)), test_size=0.2, random_state=42
)

test_dataset = Subset(coco_dataset, test_indices)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=lambda batch: tuple(zip(*batch)))



# Set up K-Fold cross-validation
k = 2
kf = KFold(n_splits=k, shuffle=True, random_state=3)
best_model_fold = 2
prev_loss = 99999

# Loop through folds
for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_indices)):
    print(f"Fold {fold + 1}")

    train_subset_indices = [train_val_indices[i] for i in train_idx]
    val_subset_indices = [train_val_indices[i] for i in val_idx]

    # Create train and validation datasets
    train_dataset = Subset(coco_dataset, train_subset_indices)
    val_dataset = Subset(coco_dataset, val_subset_indices)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, collate_fn=lambda batch: tuple(zip(*batch)))
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=lambda batch: tuple(zip(*batch)))

    # Initialize model (e.g., Faster R-CNN pre-trained on COCO)
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    model.to(device)

#     # Define optimizer and scheduler
#     optimizer = SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
#     scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

#     # Loss function (if required explicitly, depends on the model)
#     criterion = nn.CrossEntropyLoss()  # Example for classification models
#     num_epochs = 10
    
#     trian_loss = train_model(model, train_loader, optimizer, scheduler, num_epochs=num_epochs, device=device)
#     val_loss = val_model(model, val_loader, criterion, optimizer, scheduler, num_epochs=num_epochs, device=device)
    
#     # Store loss for plotting
#     all_train_losses.append(trian_loss)
#     all_val_losses.append(val_loss)

#     print(f"Fold {fold + 1}, Epoch {num_epochs + 1}, Validation Loss: {val_loss[-1]:.4f}")
#     if val_loss[-1] <= prev_loss and val_loss[-1] >= 0:
#         prev_loss = val_loss[-1]
#         best_model_fold = fold + 1
#         # Save model for each fold
#         torch.save(model.state_dict(), f"model_fold_{fold + 1}.pth")
#         print(f"Model for fold {fold + 1} saved.")


# # Step 3: Evaluate on the test set
# print("\nEvaluating on the test set...")

# Load the best model (or ensemble models from all folds)
best_model = fasterrcnn_resnet50_fpn(pretrained=True)
best_model.load_state_dict(torch.load(f"model_fold_{best_model_fold}.pth"))  # Change if using ensemble
best_model.to(device)
best_model.eval()

# test_losses = test_model(best_model, test_loader, optimizer, scheduler, num_epochs=10, device="cuda")  # Define test_model similar to train_model/val_model

# print(f"Test Loss: {test_losses[-1]:.4f}")

# visual_loss(all_train_losses, all_val_losses, num_epochs, test_losses, best_model_fold)

visualize_predictions(best_model, val_loader, device, num_images=5)
