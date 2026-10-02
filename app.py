import gradio as gr
import torch
from PIL import Image
from torchvision import transforms

from ham10000.models import build_model
from ham10000.utils import load_config

CLASS_NAMES = {
    "akiec": "Actinic keratoses / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi (benign moles)",
    "vasc": "Vascular lesions",
}

CLASSES = sorted(CLASS_NAMES.keys())

config = load_config("configs/baseline.yaml")
device = torch.device("cpu")  # Spaces free tier is CPU-only

model = build_model(config.num_classes)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((config.image_size, config.image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def predict(image):
    """Return {class_name: probability} for an uploaded lesion image."""
    img = Image.fromarray(image).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    return {CLASS_NAMES[CLASSES[i]]: float(probs[i]) for i in range(len(CLASSES))}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload a dermatoscopic image"),
    outputs=gr.Label(num_top_classes=3, label="Prediction"),
    title="HAM10000 Skin Lesion Classifier",
    description="Upload a dermatoscopic image to see the model's top predictions. "
                "Educational demo only — not a medical diagnostic tool.",
)

if __name__ == "__main__":
    demo.launch()