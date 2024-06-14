import torch
from torchvision import transforms
from PIL import Image

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Load the model and move it to the appropriate device
model = torch.jit.load('final_skin_condition_model.pt', map_location=device)
model.eval()

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


def predict_image(image_path):
    # Load and transform the image
    image = Image.open(image_path).convert('RGB')
    # Add batch dimension and move to device
    image = transform(image).unsqueeze(0).to(device)

    # Perform the prediction
    with torch.no_grad():
        outputs = model(image)
        print(outputs)
        _, predicted = torch.max(outputs, 1)

    predicted_class = classes[predicted[0].item()]
    print(f'Predicted class: {predicted_class}')


# Example usage
# replace with your image path
image_path = '/Users/ananthkashyap/Desktop/CVDermatology/C0567936-Erythrodermic_atopic_dermatitis.jpg'
predict_image(image_path)
