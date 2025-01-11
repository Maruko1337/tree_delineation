from models.gan import Generator
from utils.helpers import evaluate
from configs.config import Config
import torch

generator = Generator().to(Config.DEVICE)
# Load trained model weights
generator.load_state_dict(torch.load('generator_model.pth'))

# Run evaluation
evaluate(generator, val_loader)
