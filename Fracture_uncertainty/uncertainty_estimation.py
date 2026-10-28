import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sure_denoise import SUREDenoise
 

np.random.seed(1000)
torch.manual_seed(1000)
"""changed dataset to not include x-ray's with hardware"""
"""dataset used is shown in GitHub along with citations"""
root = '/Users/alfredthebuttler/Desktop/PHY379/Fracture_uncertainty/Bone fracture dataset/Bone Fracture'
weights_path = '/Users/alfredthebuttler/Desktop/PHY379/Fracture_uncertainty/resnet18_fracture.pt'  # same file for training and inference
dropout = 0.5 
 
""" SUREDenoise estimates the noise level of each X-ray and picks the DCT soft-threshold that minimizes SURE (see sure_denoise.py)"""
train_transform = transforms.Compose([
    SUREDenoise(),
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
 
valid_transform = transforms.Compose([
    SUREDenoise(),   
    transforms.Resize((224, 224)),                   
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
 
train_dataset = datasets.ImageFolder(os.path.join(root, 'Train'), transform=train_transform)
valid_dataset = datasets.ImageFolder(os.path.join(root, 'Valid'), transform=valid_transform)
num_classes = len(train_dataset.classes)
 
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
 
 
def build_model(pretrained=False):
    model = models.resnet18(weights=None)
    #    local_weights = torch.load(
    #   "/path/to/resnet18-f37072fd.pth",
    #    map_location="cpu",
     #   weights_only=True
    #)
    #model.load_state_dict(local_weights)
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(model.fc.in_features, num_classes),
    )

    return model


def set_train_mode(model):
    """Train mode, but the frozen layers keep their batch-norm statistics fixed."""
    model.train()
    
 
 
def main():
    print(train_dataset.class_to_idx)
    print(valid_dataset.class_to_idx)
    print(len(train_dataset), len(valid_dataset), 'device:', device)
 

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    model = build_model(pretrained=False) # Turned off online downloading
    model = model.to(device)
 
    learning_rate = 0.0001
    num_epochs = 15
 
   
    criterion = nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    """adding a scheduler to cut learning rate in half validation progress stalls"""
 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []
 
    best_val_loss = float('inf')
    early_stop_patience = 5
    patience_counter = 0

 
    for epoch in range(num_epochs):
        set_train_mode(model)
        running_loss = 0.0
        correct = 0
        total = 0
 
        for i, (images, labels) in enumerate(train_dataloader):
            images = images.to(device)
            labels = labels.to(device)
 
            # Forward steps
            outputs = model(images)
            loss = criterion(outputs, labels)
 
            # Backward steps
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
 
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
 
        # Training accuracy and loss
        train_accuracy = 100 * correct / total
        train_loss = running_loss / len(train_dataloader)
 
        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0
        class_correct = np.zeros(num_classes)
        class_total = np.zeros(num_classes)
 
        with torch.no_grad():
            for images, labels in valid_dataloader:
                images = images.to(device)
                labels = labels.to(device)
 
                outputs = model(images)
                valid_loss += criterion(outputs, labels).item()
 
                _, predicted = outputs.max(1)
                valid_total += labels.size(0)
                valid_correct += predicted.eq(labels).sum().item()
                for c in range(num_classes):
                    mask = labels == c
                    class_total[c] += mask.sum().item()
                    class_correct[c] += predicted[mask].eq(c).sum().item()
 
        valid_accuracy = 100 * valid_correct / valid_total
        valid_loss /= len(valid_dataloader)
        recalls = class_correct / np.maximum(class_total, 1)
        balanced_acc = float(recalls.mean())
 
       
        recall_strings = [f"{train_dataset.classes[idx]}: {recalls[idx]:.3f}" for idx in range(num_classes)]
        recall_log = ", ".join(recall_strings)
 
        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, "
              f"Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.2f}%, "
              f"Recalls -> {recall_log}, Balanced acc: {balanced_acc:.3f}")
        
        scheduler.step(valid_loss)
        """added checkpoint logic for early stopping"""
 
        train_loss_list.append(train_loss)
        train_acc_list.append(train_accuracy)
        valid_loss_list.append(valid_loss)
        valid_acc_list.append(valid_accuracy)
        # checkpoint on balanced accuracy: with a weighted loss, raw accuracy would favour the epoch that ignores "normal"
        
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            torch.save(model.state_dict(), weights_path)
            print(f" => Validation loss decreased ({best_val_loss:.4f}). Saving model checkpoint...")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f" => No validation loss improvement. Early stopping counter: {patience_counter}/{early_stop_patience}")
 
        if patience_counter >= early_stop_patience:
            print(f"Early stopping triggered => Training stopped at epoch {epoch + 1}.")
            break
 
 
if __name__ == '__main__':
    main()