import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from medmnist import PneumoniaMNIST
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Hyperparameters and Configuration ---
# Justification in README.md
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 10
DATA_AUGMENTATION_LEVEL = 0.1 # Strength of rotation, shift etc.

# --- 2. Data Loading and Augmentation ---
print("Loading and preparing data...")

# Transformations to prevent overfitting
data_transform = transforms.Compose([
    transforms.RandomRotation(DATA_AUGMENTATION_LEVEL * 100),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # Inception-V3 requires 3-channel input
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Download and load the datasets
train_dataset = PneumoniaMNIST(split="train", transform=data_transform, download=True)
test_dataset = PneumoniaMNIST(split="test", transform=data_transform, download=True)

# --- 3. Mitigating Class Imbalance ---
# Calculate class weights to handle imbalance
labels = np.array([sample[1][0] for sample in train_dataset])
class_counts = np.bincount(labels)
class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
sample_weights = class_weights[labels]
sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

# Create DataLoaders
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 4. Model Fine-Tuning: Inception-V3 ---
print("Setting up the Inception-V3 model...")
# Load pre-trained Inception-V3
model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)

# Freeze all layers in the network
for param in model.parameters():
    param.requires_grad = False

# Replace the final classifier layer for our binary task
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2) # 2 classes: normal, pneumonia

# Also unfreeze and replace the auxiliary classifier if it exists
if model.AuxLogits is not None:
    num_aux_ftrs = model.AuxLogits.fc.in_features
    model.AuxLogits.fc = nn.Linear(num_aux_ftrs, 2)

# --- 5. Training the Model ---
print("Starting model training...")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
# Adam is a good default optimizer
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.squeeze().long().to(device)

        optimizer.zero_grad()

        # Inception-V3 in training mode returns main and auxiliary outputs
        outputs, aux_outputs = model(inputs)
        loss1 = criterion(outputs, labels)
        loss2 = criterion(aux_outputs, labels)
        loss = loss1 + 0.4 * loss2 # As recommended in Inception-V3 paper

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.4f}")

print("Training finished.")

# --- 6. Model Evaluation ---
print("Evaluating the model...")
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.squeeze().long().to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# --- 7. Reporting Performance ---
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='binary')
recall = recall_score(all_labels, all_preds, average='binary')

print("\n--- Evaluation Metrics ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Pneumonia'],
            yticklabels=['Normal', 'Pneumonia'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
print("\nConfusion matrix saved as 'confusion_matrix.png'")