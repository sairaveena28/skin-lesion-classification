from torchvision import models
import torch.nn as nn

def build_model(num_classes, model_name="resnet18"):
    """Build a pretrained model on ImageNet for num_classes outputs.

    Loads IMAGENET1K_V1 weights and replaces the final fully-connected layer
    with a new Linear layer sized for num_classes (transfer learning). All
    layers are trainable (full fine-tuning).
    """
    
    if model_name == "resnet18":
        model = models.resnet18(weights="IMAGENET1K_V1")
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights="IMAGENET1K_V1")
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    
    return model

