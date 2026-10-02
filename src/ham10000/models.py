from torchvision import models
import torch.nn as nn

def build_model(num_classes):
    """Build a ResNet18 pretrained on ImageNet for num_classes outputs.

    Loads IMAGENET1K_V1 weights and replaces the final fully-connected layer
    with a new Linear layer sized for num_classes (transfer learning). All
    layers are trainable (full fine-tuning).
    """
    
    model = models.resnet18(weights="IMAGENET1K_V1")
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

