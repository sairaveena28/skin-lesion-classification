import pytest
from ham10000.data import Ham10000Dataset
from torchvision import transforms
import torch
from ham10000.models import build_model
from ham10000.data import build_dataloaders
from ham10000.utils import load_config
import os
DATA_AVAILABLE = os.path.exists("data/ham10000_metadata.csv")

@pytest.fixture
def dataset():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    return Ham10000Dataset(
        csv_path='data/ham10000_metadata.csv',
        image_dir='data',
        transform=transform
    )
    
@pytest.fixture
def config():
    return load_config("configs/baseline.yaml")

@pytest.mark.skipif(not DATA_AVAILABLE, reason="dataset not available in CI")
def test_dataset_length(dataset):
    assert len(dataset) == 10015

@pytest.mark.skipif(not DATA_AVAILABLE, reason="dataset not available in CI")
def test_dataset_shape(dataset):
    img, label = dataset[0]
    assert img.shape == (3, 224, 224)
    assert isinstance(label, int)

@pytest.mark.skipif(not DATA_AVAILABLE, reason="dataset not available in CI")
def test_labels(dataset):
    set(dataset.class_to_idx.values()) == set(range(7))

@pytest.mark.skipif(not DATA_AVAILABLE, reason="dataset not available in CI")
def test_no_lesion_leakage(config):
    train_loader, val_loader, _ = build_dataloaders(config)
    full = train_loader.dataset.dataset.data
    train_lesions = set(full.iloc[list(train_loader.dataset.indices)]['lesion_id'])
    val_lesions = set(full.iloc[list(val_loader.dataset.indices)]['lesion_id'])
    assert train_lesions.isdisjoint(val_lesions)

def test_model_output_shape():
    model = build_model(num_classes=7)
    model.eval()
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (1, 7)