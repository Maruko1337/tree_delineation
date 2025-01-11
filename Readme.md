# MOITD (Multi-Scale Occlusion-Aware Individual Tree Delineation)

The **Multi-Scale Occlusion-Aware Individual Tree Delineation (MOITD)** algorithm is designed to estimate tree structural parameters in complex forest environments using depth camera images. The algorithm addresses challenges like brightness variations, terrain undulations, and segmentation difficulties due to tree occlusion, using advanced techniques such as image pyramids, deep learning (CNN, GAN, U-Net), and spatial attention.

## Features

- **Image Pyramid Technology**: Enables feature extraction at multiple scales by resizing input images to various resolutions.
- **Generative Adversarial Network (GAN)**: Used to reconstruct occluded or missing depth information, improving segmentation accuracy in occluded regions.
- **U-Net Architecture**: Integrates multi-scale features with reconstructed depth information for high-quality tree segmentation.
- **Spatial Attention Module**: Enhances the recognition of tree contours, improving edge detection.
- **End-to-End Training**: The model is trained using a comprehensive loss function, including edge loss, region loss, and occlusion-aware loss.
- **Real-World Applications**: Tested on real-world forest environments with robust performance under various conditions.

## Installation

To run this project, you'll need to set up the environment and install the required dependencies. You can do this by running the following commands:

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/MOITD_Project.git
   cd MOITD_Project

2. Set up the environment (preferably in a virtual environment):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
3. Install the dependencies:

    ```bash
    pip install -r requirements.txt

## Project Structure
The project is organized into the following structure:

    MOITD_Project/
    ├── data/
    │   └── dataset.py              # Handles dataset loading, preprocessing, and augmentation
    ├── models/
    │   ├── gan.py                  # Generator and Discriminator definitions
    │   ├── unet.py                 # U-Net model for feature integration
    │   └── attention.py            # Spatial attention module
    ├── utils/
    │   ├── image_processing.py     # Image pyramid generation and any other preprocessing functions
    │   ├── losses.py               # Definitions for edge_loss, region_loss, occlusion_aware_loss
    │   └── helpers.py              # Any utility functions such as training loops, evaluation, etc.
    ├── configs/
    │   └── config.py               # Configurations (hyperparameters, paths, etc.)
    ├── train.py                    # Main training script
    ├── test.py                     # Script for testing and validation of the trained model
    └── requirements.txt            # List of dependencies
# Usage
## Training the Model
To train the model, you can run the train.py script. It will load the dataset, initialize the model, and start the training process.

    python train.py
Make sure to adjust the configuration parameters in configs/config.py for your dataset and training settings (e.g., batch size, learning rate, number of epochs).

## Testing the Model
After training the model, you can test it on a validation set or real-world data by running the test.py script:

    python test.py
Make sure to load the pre-trained model weights before testing.

# Loss Functions
The MOITD algorithm utilizes several custom loss functions to optimize segmentation:

- Edge Loss: Focuses on the accuracy of edge detection.
- Region Loss: Ensures the segmentation regions match the target regions.
- Occlusion-Aware Loss: Addresses occlusion by filling missing or occluded regions with plausible depth data.
These losses are implemented in utils/losses.py.

# Datasets
The model was trained and tested using the publicly available Low-viewpoint Forest Depth Dataset, which includes over 9,700 RGB/depth image pairs captured under various weather and lighting conditions.

You can load and preprocess this dataset using data/dataset.py.

# Results
The MOITD algorithm has shown superior performance in identifying and reconstructing tree structures, particularly in occluded areas. The method improves segmentation accuracy and edge detection, making it ideal for applications in forestry remote sensing and ecological monitoring.

# Sample Outputs:
Example of the segmentation outputs for individual trees in the forest can be visualized using your preferred image viewing software.


