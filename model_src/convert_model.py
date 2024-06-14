import torch
import coremltools as ct
from torchvision import models
import torch.nn as nn

# Define the PyTorch model class
class EfficientNetB2Model(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB2Model, self).__init__()
        self.model = models.efficientnet_b2(
            weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
        self.model.classifier[1] = nn.Linear(
            self.model.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.model(x)


# Load the TorchScript model directly
device = torch.device("cpu")
model = torch.jit.load('final_skin_condition_model.pt', map_location=device)
model.eval()

# Example input for tracing
example_input = torch.rand(1, 3, 224, 224)

# Define class labels
class_labels = [
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

# Set the image scale and bias for input image preprocessing
scale = 1/255.0
bias = [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225]

image_input = ct.ImageType(name="input_image",
                           shape=example_input.shape,
                           scale=scale, bias=bias)

# Convert to Core ML using the Unified Conversion API
mlmodel = ct.convert(
    model,
    inputs=[image_input],
    classifier_config=ct.ClassifierConfig(class_labels),
    compute_units=ct.ComputeUnit.CPU_ONLY
)

# Save the converted model
mlmodel.save("final_skin_condition_model.mlpackage")
print('Model converted and saved')
