import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.mobile_optimizer import optimize_for_mobile

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Create a mapping for the labels
label_mapping = {
    "Atopic Dermatitis": 0,
    "Basal Cell Carcinoma": 1,
    "Benign Keratosis-like Lesions": 2,
    "Eczema": 3,
    "Melanocytic Nevi": 4,
    "Melanoma": 5,
    "Psoriasis": 6,
    "Seborrheic Keratoses, or other Benign Tumor": 7,
    "Tinea Ringworm Candidiasis, or other Fungal Infection": 8,
    "Warts, Mollescum, or other Viral Infection": 9
}

# Dataset class
class SkinConditionDataset(Dataset):
    def __init__(self, csv_file, label_mapping, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.label_mapping = label_mapping
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = self.annotations.iloc[idx, 0]
        image = Image.open(img_path).convert("RGB")
        label = self.annotations.iloc[idx, 1]
        label = self.label_mapping[label]
        if self.transform:
            image = self.transform(image)
        return image, label


# Data transformation and augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                            saturation=0.2, hue=0.2),
    transforms.RandomAffine(degrees=0, shear=10)
])

"""
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
"""

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# EfficientNet-B2 Model
class EfficientNetB2Model(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB2Model, self).__init__()
        self.model = models.efficientnet_b2(
            weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
        self.model.classifier[1] = nn.Linear(
            self.model.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Training function
def train_model(model, dataloader, criterion, optimizer, device, num_epochs=25):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            print(loss)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

# Evaluation function
def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(
        f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}')

# Save the model
def save_model(model, model_path):
    model.eval()
    example_input = torch.rand(1, 3, 224, 224).to(
        next(model.parameters()).device)
    traced_script_module = torch.jit.trace(model, example_input)
    optimized_model = optimize_for_mobile(traced_script_module)
    optimized_model.save(model_path)
    print(f'Model saved to {model_path}')

# Main script to initialize and start training
def main():
    # Constants
    # Replace with your actual CSV file path
    CSV_FILE = 'skin_conditions_labels.csv'
    BATCH_SIZE = 32
    NUM_CLASSES = len(label_mapping)
    NUM_EPOCHS = 20
    LEARNING_RATE = 5e-4
    WEIGHT_DECAY = 1e-6
    MODEL_PATH = 'skin_condition_model.pt'

    # Split data into train and test sets
    annotations = pd.read_csv(CSV_FILE)
    train_annotations = annotations.sample(frac=0.8, random_state=42)
    test_annotations = annotations.drop(train_annotations.index)

    # Save split datasets to CSV
    train_annotations.to_csv('train_skin_conditions_labels.csv', index=False)
    test_annotations.to_csv('test_skin_conditions_labels.csv', index=False)

    # Prepare datasets and dataloaders
    train_dataset = SkinConditionDataset(
        csv_file='train_skin_conditions_labels.csv', label_mapping=label_mapping, transform=train_transform)
    test_dataset = SkinConditionDataset(
        csv_file='test_skin_conditions_labels.csv', label_mapping=label_mapping, transform=test_transform)

    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize the model, loss function, and optimizer
    device = torch.device(
        'mps' if torch.backends.mps.is_available() else 'cpu')
    model = EfficientNetB2Model(num_classes=NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # Train the model
    train_model(model, train_dataloader, criterion,
                optimizer, device, num_epochs=NUM_EPOCHS)

    # Evaluate the model
    evaluate_model(model, test_dataloader, device)

    # Save the model
    save_model(model, MODEL_PATH)


if __name__ == "__main__":
    main()
