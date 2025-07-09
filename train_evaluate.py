import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import Inception_V3_Weights # Import for explicit weights
import medmnist
from medmnist import INFO, Evaluator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import os

# --- Configuration ---
DATA_FLAG = 'pneumoniamnist'
DOWNLOAD_PATH = './data' # Directory to download MedMNIST data
BATCH_SIZE = 64
NUM_EPOCHS = 20 # You can adjust this based on your early stopping
LEARNING_RATE = 0.001 # Initial learning rate for new layers
FINE_TUNE_LR = 0.0001 # Lower learning rate for fine-tuning entire model
NUM_CLASSES = 2 # For binary classification (Normal, Pneumonia)

# --- 1. Data Loading and Preprocessing ---
info = INFO[DATA_FLAG]
DataClass = getattr(medmnist, info['python_class'])

# Define transformations
data_transform = transforms.Compose([
    transforms.Resize((299, 299)), # Resize images to 299x299 for InceptionV3
    transforms.Grayscale(num_output_channels=3), # Convert 1-channel to 3-channel
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet normalization
])

# Load datasets
train_dataset = DataClass(split='train', transform=data_transform, download=True, root=DOWNLOAD_PATH)
val_dataset = DataClass(split='val', transform=data_transform, download=True, root=DOWNLOAD_PATH)
test_dataset = DataClass(split='test', transform=data_transform, download=True, root=DOWNLOAD_PATH)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# Create DataLoaders
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 2. Model for Transfer Learning (Inception-V3) ---
# Load pre-trained Inception-V3 model
# IMPORTANT: Use weights=Inception_V3_Weights.IMAGENET1K_V1 or Inception_V3_Weights.DEFAULT
# and keep aux_logits=True as the pre-trained model expects it.
# We will handle the auxiliary output during the forward pass.
model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True) # Changed from pretrained=True

# Freeze all parameters in the feature extractor
for param in model.parameters():
    param.requires_grad = False

# Modify the classifier (fully connected layer) for our specific task
# InceptionV3's main classifier is `fc`
# The input features to the `fc` layer are `model.fc.in_features`
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

# Also modify the auxiliary classifier if aux_logits is True
# This is crucial because if aux_logits is True, the model will output two sets of logits.
# If you don't modify this, it will still have the original 1000 output classes.
model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, NUM_CLASSES)


# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

# --- 3. Evaluation Strategy ---

# b. Detect and mitigate class imbalance
train_labels = np.array(train_dataset.labels).flatten()
class_counts = np.bincount(train_labels)
total_samples = len(train_labels)
# Handle potential division by zero if a class has 0 samples (unlikely for MedMNIST)
class_weights = total_samples / (NUM_CLASSES * class_counts)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

print(f"Class counts (0: Normal, 1: Pneumonia): {class_counts}")
print(f"Calculated class weights: {class_weights_tensor}")

# Use weighted cross-entropy loss to mitigate imbalance
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

# Optimizer for the newly added layers (main classifier and auxiliary classifier)
# We need to include parameters for both `fc` and `AuxLogits.fc`
optimizer = optim.Adam([
    {'params': model.fc.parameters()},
    {'params': model.AuxLogits.fc.parameters()}
], lr=LEARNING_RATE)


# --- Training Loop ---
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, phase_name="Training"):
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_loss = float('inf')
    patience = 5 # Number of epochs to wait for improvement before stopping
    epochs_no_improve = 0

    print(f"\n--- Starting {phase_name} ---")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.squeeze().long().to(device) # Ensure labels are long and squeezed

            optimizer.zero_grad()
            
            # When aux_logits=True, model returns a tuple (main_output, aux_output) during training
            outputs, aux_outputs = model(inputs)
            
            # Calculate total loss (main loss + auxiliary loss)
            loss = criterion(outputs, labels) + 0.4 * criterion(aux_outputs, labels) # Auxiliary loss is typically scaled by 0.4
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.squeeze().long().to(device)

                # During evaluation, InceptionV3 typically only returns the main output
                outputs = model(inputs) 
                
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        epoch_val_accuracy = correct_predictions / total_predictions
        val_accuracies.append(epoch_val_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_accuracy:.4f}")

        # Early stopping
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print(f"Early stopping triggered after {epoch+1} epochs (no improvement for {patience} epochs).")
                break
    return train_losses, val_losses, val_accuracies

# --- Phase 1: Train only the classifier (newly added layers) ---
print("Phase 1: Training only the classifier (feature extractor frozen)")
# For phase 1, we only want to optimize the new layers (fc and AuxLogits.fc)
optimizer_phase1 = optim.Adam([
    {'params': model.fc.parameters()},
    {'params': model.AuxLogits.fc.parameters()}
], lr=LEARNING_RATE)
train_model(model, train_loader, val_loader, criterion, optimizer_phase1, num_epochs=5, phase_name="Phase 1 Training")

# --- Phase 2: Fine-tune the entire model ---
print("\nPhase 2: Fine-tuning the entire model (unfreezing all layers)")

# Unfreeze all parameters
for param in model.parameters():
    param.requires_grad = True

# Set a lower learning rate for fine-tuning
optimizer_ft = optim.Adam(model.parameters(), lr=FINE_TUNE_LR)

# Load the best model from Phase 1 to continue fine-tuning
if os.path.exists('best_model.pth'):
    model.load_state_dict(torch.load('best_model.pth'))
    print("Loaded best model from Phase 1 for fine-tuning.")

train_model(model, train_loader, val_loader, criterion, optimizer_ft, num_epochs=NUM_EPOCHS, phase_name="Phase 2 Fine-tuning")


# --- Evaluation on Test Set ---
print("\n--- Evaluating on Test Set ---")
model.eval() # Set model to evaluation mode
all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.squeeze().long().to(device)

        # During evaluation, InceptionV3 typically only returns the main output
        outputs = model(inputs) 
        
        probabilities = torch.softmax(outputs, dim=1) # Get probabilities for ROC AUC

        _, predicted = torch.max(outputs.data, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probabilities[:, 1].cpu().numpy()) # Probabilities for the positive class (pneumonia)

# Calculate metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='binary')
recall = recall_score(all_labels, all_preds, average='binary')
f1 = f1_score(all_labels, all_preds, average='binary')
roc_auc = roc_auc_score(all_labels, all_probs)
cm = confusion_matrix(all_labels, all_preds)

print(f"\nTest Set Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print("\nConfusion Matrix:")
print(cm)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
