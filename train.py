from models.gan import Generator, Discriminator
from models.unet import UNet
from utils.image_processing import image_pyramid
from utils.losses import edge_loss, region_loss
from utils.helpers import train
from configs.config import Config
import torch
import torch.optim as optim

# Initialize model, optimizer, and loss functions
generator = Generator().to(Config.DEVICE)
discriminator = Discriminator().to(Config.DEVICE)
unet = UNet().to(Config.DEVICE)
optimizer = optim.Adam(list(generator.parameters()) + list(discriminator.parameters()), lr=Config.LEARNING_RATE)

# Load the dataset
train_loader = DataLoader(...)

# Train the model
train(generator, train_loader, optimizer, edge_loss)  # Modify based on your actual loss
