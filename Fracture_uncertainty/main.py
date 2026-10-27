import torch
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as functor
import torchvision
import numpy as np
import matplotlib.pyplot as pyplot
from torchvision import models

root = '/Users/alfredthebuttler/Desktop/PHY379/Fracture_uncertainty/Bone fracture dataset/Dataset'
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_dataset = datasets.ImageFolder(os.path.join(root, 'Train '), transform=transform)
valid_dataset = datasets.ImageFolder(os.path.join(root, 'Valid'), transform=transform)
num_classes = len(train_dataset.classes)

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

def main():
    print(train_dataset.class_to_idx)  
    print(valid_dataset.class_to_idx)  
    print(len(train_dataset), len(valid_dataset), 'device:', device)


    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=2)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, num_classes),
    )
    model = model.to(device)
    learning_rate = 0.001
    num_epochs = 15
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []
    for epoch in range(num_epochs):
        model.train()
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

        with torch.no_grad():
            for images, labels in valid_dataloader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                valid_loss += criterion(outputs, labels).item()

                _, predicted = outputs.max(1)
                valid_total += labels.size(0)
                valid_correct += predicted.eq(labels).sum().item()

        valid_accuracy = 100 * valid_correct / valid_total
        valid_loss /= len(valid_dataloader)

        print(f"Epoch [{epoch + 1}/{num_epochs}], "
              f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%, "
              f"Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_accuracy:.2f}%")

        train_loss_list.append(train_loss)
        train_acc_list.append(train_accuracy)
        valid_loss_list.append(valid_loss)
        valid_acc_list.append(valid_accuracy)
if __name__ == '__main__':
    main()
 
