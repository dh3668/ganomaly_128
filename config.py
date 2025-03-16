import torch

CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "input_channels": 3,
    "latent_dim": 128,
    "condition_dim": 8,
    "batch_size": 32,
    "epochs": 800,
    "learning_rate": 0.00006,
    "image_size": (224, 224),
    "dataset_path": "./cropped_data_8/train",
    "model_save_path": "./ConditionalGANomaly_64_modified.pth",
}