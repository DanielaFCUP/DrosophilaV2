import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix  # , roc_curve
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import *
from tqdm import tqdm

# Set the random seed for PyTorch
torch.manual_seed(42)

# Loss criterion will always be CrossEntropy
criterion = nn.CrossEntropyLoss()


def run(epochs: int, train_images: DataLoader, test_images: DataLoader, model, optimiser):  # -> (object, (list, list, list, list):
    model, (train_losses, train_accuracies, test_losses, test_accuracies), f1_epochs = train(epochs=epochs,
                                                                                  train_images=train_images,
                                                                                  test_images=test_images,
                                                                                  model=model,
                                                                                  optimiser=optimiser)
    torch.save(model, 'out/model.pth')
    print("Model saved to out/model.pth")
    torch.save(model.state_dict(), 'out/model_state_dict.txt')
    print("Model's state dictionary saved to out/model.pth")
    return model, (train_losses, train_accuracies, test_losses, test_accuracies), f1_epochs


def train(epochs: int, train_images: DataLoader, test_images: DataLoader, model, optimiser):  # -> (object, (list, list, list, list)):
    print("Training has started.")
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    f1_epochs = []

    for epoch in range(1, epochs+1):
        # Train
        (model, (train_loss, train_accuracy)) = _epoch_train(images=train_images, model=model, optimiser=optimiser)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        # Test
        (test_loss, test_accuracy, f1_score_class) = _epoch_test(images=test_images, model=model)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        f1_epochs.append(f1_score_class)

    return model, (train_losses, train_accuracies, test_losses, test_accuracies), f1_epochs


def _epoch_train(images: DataLoader, model, optimiser: object):  # -> object, (list, list):
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
        optimiser.zero_grad()

        # forward + backward + optimise
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()

        # statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(images)
    train_accuracy = 100. * correct / total
    print("Train Loss: %.3f | Accuracy: %.3f" % (train_loss, train_accuracy))
    return model, (train_loss, train_accuracy)


def _epoch_test(images: DataLoader, model) -> (list, list):
    # Let's put our network in classification mode
    model.eval()

    # Performance Data
    correct = 0
    total = 0
    running_loss = 0
    f1_class = []
    correct_predictions = []
    model_predictions = []

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in tqdm(images):
            inputs, labels = data

            # calculate outputs by running images through the network
            outputs = model(inputs)

            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # statistics
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # F1-score

            correct_predictions += labels.numpy().tolist()
            model_predictions += predicted.numpy().tolist()

        f1_score_class = f1_score(correct_predictions, model_predictions, average=None)
        f1_class.append(f1_score_class)

    test_loss = running_loss / len(images)
    test_accuracy = 100. * correct / total
    print("Test Loss: %.3f | Accuracy: %.3f" % (test_loss, test_accuracy))
    return test_loss, test_accuracy, f1_score_class


def optimiser_type_to_optimiser(optimiser_type: str, model, lr: float) -> object:
    match optimiser_type:
        case 'Adam':
            return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        case 'SGD':
            return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
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


def f1_and_confusion_matrix(images: DataLoader, model, class_names: list) -> str | dict:
    # Let's put our network in classification mode
    model.eval()

    # Performance Data
    correct = 0
    total = 0
    running_loss = 0
    correct_predictions = []  # Ground-truth labels for the test set
    model_predictions = []  # List of the predicted labels for the test set

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in images:
            inputs, labels = data

            # calculate outputs by running images through the network
            outputs = model(inputs)

            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # statistics
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)

            # Build confusion matrix
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += labels.numpy().tolist()
            model_predictions += predicted.tolist()


    # Confusion Matrix
    cf_matrix = confusion_matrix(correct_predictions, model_predictions)
    dataframe = pd.DataFrame(cf_matrix, index=class_names, columns=class_names)
    plt.figure(figsize=(8, 6))
    sns.heatmap(dataframe, annot=True, cbar=None, cmap="YlGnBu", fmt="d")
    plt.title("Test Confusion Matrix"), plt.tight_layout()
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.savefig('out/plots/test_confusion_matrix.png')
    plt.show()

    report = classification_report(correct_predictions, model_predictions, target_names=class_names)

    return report