import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import *
from tqdm import tqdm

# Loss criterion will always be CrossEntropy
criterion = nn.CrossEntropyLoss()


def run(epochs: int, images: DataLoader, model) -> (list, list):
    print("Testing has started.")
    test_losses = []
    test_accuracies = []
    f1_scores = []

    for epoch in range(epochs):
        (test_loss, test_accuracy, correct_prediction, model_prediction) = _epoch_test(images=images, model=model)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        # Calculate F1 score for the epoch
        f1_score_epoch = f1_score(np.array(correct_prediction), np.array(model_prediction), average=None)
        f1_scores.append(f1_score_epoch)

    print("Testing has ended.")
    return test_losses, test_accuracies, np.array(f1_scores)


def _epoch_test(images: DataLoader, model) -> (list, list):
    # Let's put our network in evaluation mode
    model.eval()

    # Performance Data
    correct = 0
    total = 0
    running_loss = 0
    correct_predictions = []  # Ground-truth labels for the test set
    model_predictions = []  # List of the predicted labels for the test set

    # Since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in tqdm(images):
            inputs, labels = data

            # Calculate outputs by running images through the network
            outputs = model(inputs)

            # The class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Statistics
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Store predictions for each batch
            correct_predictions += labels.numpy().tolist()
            model_predictions += predicted.numpy().tolist()

    test_loss = running_loss / len(images)
    test_accuracy = 100. * correct / total
    print("Test Loss: %.3f | Accuracy: %.3f" % (test_loss, test_accuracy))
    return test_loss, test_accuracy, correct_predictions, model_predictions


def matrix_of_confusion(images: DataLoader, model, class_names: list) -> str | dict:
    # Let's put our network in evaluation mode
    model.eval()

    # Performance Data
    correct_predictions = []  # Ground-truth labels for the test set
    model_predictions = []  # List of the predicted labels for the test set

    # Since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in images:
            inputs, labels = data

            # Calculate outputs by running images through the network
            outputs = model(inputs)

            # statistics
            _, predicted = outputs.max(1)
            correct_predictions += labels.numpy().tolist()
            model_predictions += predicted.numpy().tolist()

    # Confusion Matrix
    cf_matrix = confusion_matrix(correct_predictions, model_predictions)

    # Create pandas dataframe
    dataframe = pd.DataFrame(cf_matrix, index=class_names, columns=class_names)

    plt.figure(figsize=(8, 6))
    # Create heatmap
    sns.heatmap(dataframe, annot=True, cbar=None, cmap="YlGnBu", fmt="d")
    plt.title("Test Confusion Matrix")
    plt.tight_layout()
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.savefig('out/plots/test_confusion_matrix.png')
    plt.show()

    report = classification_report(correct_predictions, model_predictions, target_names=class_names)

    return report
