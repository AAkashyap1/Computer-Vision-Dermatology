import coremltools as ct
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import transforms

# Define the classes
classes = [
    "Atopic Dermatitis",
    "Basal Cell Carcinoma",
    "Benign Keratosis-like Lesions",
    "Eczema",
    "Melanocytic Nevi",
    "Melanoma",
    "Psoriasis",
    "Seborrheic Keratoses, or other Benign Tumor",
    "Tinea Ringworm Candidiasis, or other Fungal Infection",
    "Warts, Mollescum, or other Viral Infection"
]

# Load the Core ML model
model = ct.models.MLModel('final_skin_condition_model.mlmodel')

# Define the image transformation pipeline


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert('RGB')
    # Convert to numpy array with batch dimension
    image = transform(image).unsqueeze(0).numpy()
    return image


def predict_image(image_path):
    # Preprocess the image
    image = preprocess_image(image_path)

    # Convert the image to the input format of the model
    # Convert from (1, 3, 224, 224) to (1, 224, 224, 3)
    image = np.transpose(image, (0, 2, 3, 1))

    # Perform the prediction
    prediction = model.predict({'input_image': image})

    # Get the predicted class
    predicted_class = prediction['classLabel']
    confidence = prediction['classLabelProbs'][predicted_class]

    print(
        f'Predicted class: {predicted_class}, Confidence: {confidence * 100:.2f}%')


# Example usage
# Replace with your image path
image_path = '/Users/ananthkashyap/Desktop/CVDermatology/C0567936-Erythrodermic_atopic_dermatitis.jpg'
predict_image(image_path)
