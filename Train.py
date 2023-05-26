import torch
import torchvision.models as models
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Set the random seed for PyTorch
torch.manual_seed(42)


def run(model_type: str, epochs: int, lr: float, optim_type: str, num_classes: int,
        images: DataLoader):  # -> (object, dict):
    model = model_type_to_model(model_type, num_classes=num_classes)
    optim = optimiser_type_to_optimiser(optimiser_type=optim_type, model=model, lr=lr)
    model, (train_losses, train_accuracies) = train(epochs=epochs, images=images, model=model, optim=optim)
    torch.save(model, 'out/model.pth')
    print("Model saved to out/model.pth")
    torch.save(model.state_dict(), 'out/model_state_dict.txt')
    print("Model's state dictionary saved to out/model.pth")
    return model, (train_losses, train_accuracies)


def train(epochs: int, images: DataLoader, model, optim):
    print("Training has started.")
    train_losses = []
    train_accuracies = []

    for epoch in range(epochs):
        (model, (train_loss, train_accuracy)) = _epoch_train(images=images, model=model, optim=optim)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

    print("Training has ended.")
    return model, (train_losses, train_accuracies)


def _epoch_train(images: DataLoader, model, optim: object):
    # Loss criterion will always be CrossEntropy
    criterion = nn.CrossEntropyLoss()

    # Performance Data
    correct = 0
    total = 0
    running_loss = 0

    # Let's put our network in training mode
    model.train()

    for data in tqdm(images):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optim.zero_grad()

        # forward + backward + optimise
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optim.step()

        # statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(images)
    train_accuracy = 100. * correct / total
    print("Train Loss: %.3f | Accuracy: %.3f" % (train_loss, train_accuracy))
    return model, (train_loss, train_accuracy)


def optimiser_type_to_optimiser(optimiser_type: str, model, lr: float) -> object:
    match optimiser_type:
        case 'Adam':
            return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        case 'SGD':
            return torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        case bad:
            raise ValueError("Illegal optimiser type: ", bad)


def model_type_to_model(model_type: str, num_classes: int):
    match model_type:
        case 'densenet':
            model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', weights=None)
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, num_classes)

            # Add softmax activation function to the output layer
            model.add_module('softmax', nn.Softmax(dim=1))
            return model
        case 'resnet':
            model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights=None)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)

            # Add softmax activation function to the output layer
            model.add_module('softmax', nn.Softmax(dim=1))
            return model
        case 'efficientnet':
            # Create a new EfficientNet-B0 model with 3 output classes and no pretrained weights
            model = models.efficientnet_b0(num_classes=num_classes, weights=None)

            # Add a softmax activation function to the output layer for better output interpretation
            model.add_module('softmax', nn.Softmax(dim=1))

            return model
        case bad:
            raise ValueError("Illegal model type: ", bad)