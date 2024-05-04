import torch
import json
import torch.nn.functional as F
from models import EnsembleModel, MobileNetV2Model, InceptionV3Model, device
from torchvision import transforms
from PIL import Image


def predict_image(model, image_path, device):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    image = image.to(device)
    with torch.no_grad():
        logits = model(image)
        probabilities = F.softmax(logits, dim=1)  
        predicted_class = torch.argmax(probabilities, dim=1) 
    return probabilities, predicted_class.item()


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = torch.load('./best_ensemble_model.pth', map_location=device)
    model = model.eval()

    with open('food_classes.json', 'r', encoding='utf-8') as file:
        class_names = json.load(file)

    image_path = "./test_imgs/th.jfif"  
    probabilities, predicted_class = predict_image(model, image_path, device)
    print("Class probabilities:")
    for i, prob in enumerate(probabilities[0]):  # Access the first batch of probabilities
        print(f"{class_names[str(i)]}: {prob.item()*100:.2f}%")