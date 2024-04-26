import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torchvision.datasets import MNIST



data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
dataset = MNIST(root='./data/', download=True, transform=data_transform)


data_loader = DataLoader(dataset, batch_size=64, shuffle=False)


model = models.resnet18(pretrained=True)
model.eval()


true_labels = []
predicted_labels = []


def recall(confusion_matrix, class_index):
    true_positive = confusion_matrix[class_index, class_index]
    actual_positives = np.sum(confusion_matrix[class_index, :])
    recall = true_positive / actual_positives
    return recall

def precision(confusion_matrix, class_index):
    true_positive = confusion_matrix[class_index, class_index]
    predicted_positives = np.sum(confusion_matrix[:, class_index])
    precision = true_positive / predicted_positives
    return precision

with torch.no_grad():
    for inputs, labels in data_loader:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        true_labels.extend(labels.numpy())
        predicted_labels.extend(preds.numpy())


cm = confusion_matrix(true_labels, predicted_labels)


print("Confusion Matrix:")
print(cm)


class_index = 0
recall_value = recall(cm, class_index)
precision_value = precision(cm, class_index)

print(f"Recall for class {class_index}: {recall_value}")
print(f"Precision for class {class_index}: {precision_value}")


plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
