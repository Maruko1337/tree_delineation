class Config:
    IMAGE_SIZE = 256
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    EPOCHS = 20
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TRAIN_DATA_PATH = "path_to_train_data"
    VAL_DATA_PATH = "path_to_val_data"
