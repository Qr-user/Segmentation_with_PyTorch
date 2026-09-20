from torch import nn
import segmentation_models_pytorch as smp
import torch
from config import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"




model = smp.create_model(
    arch=f"{Model}",                     # name of the architecture, e.g. 'Unet'/ 'FPN' / etc. Case INsensitive!
    encoder_name="resnet101",
    encoder_weights="imagenet",
    classes=25,
    activation=None,
).to(DEVICE)