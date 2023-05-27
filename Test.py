import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix  # , roc_curve
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import *
from tqdm import tqdm

import Train

# Loss criterion will always be CrossEntropy
criterion = nn.CrossEntropyLoss()


def run(train_losses: list, train_accuracies: list, epochs: int, images: DataLoader, model, optimiser) -> (list, list):
    print("Testing has started.")
    test_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        (test_loss, test_accuracy) = _epoch_test(images=images, model=model)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        # Iterate training
        (model, (train_loss, train_accuracy)) = Train._epoch_train(images=images, model=model, optimiser=optimiser)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

    print("Testing has ended.")
    return model, (test_losses, test_accuracies, train_losses, train_accuracies)


def _epoch_test(images: DataLoader, model) -> (list, list):
    # Let's put our network in classification mode
    model.eval()

    # Performance Data
    correct = 0
    total = 0
    running_loss = 0

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

    test_loss = running_loss / len(images)
    test_accuracy = 100. * correct / total
    print("Test Loss: %.3f | Accuracy: %.3f" % (test_loss, test_accuracy))
    return test_loss, test_accuracy


def f1_and_confusion_matrix(images: DataLoader, model, class_names: list) -> str | dict:
    # Let's put our network in classification mode
    model.eval()

    # Performance Data
    correct = 0
    total = 0
    running_loss = 0
    correct_predictions = []  # Ground-truth labels for the test set
    model_predictions = []  # List of the predicted labels for the test set
    f1_class = []

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

            # F1 score
            correct_predictions += labels.numpy().tolist()
            model_predictions += predicted.numpy().tolist()

            f1_score_class = f1_score(correct_predictions, model_predictions, average=None)
            f1_class.append(f1_score_class)

            # Build confusion matrix
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += labels.numpy().tolist()
            model_predictions += predicted.tolist()

    # F1 Score
    print(f1_class)
    plt.plot(f1_class, '-o')
    plt.xlabel('Batch')
    plt.ylabel('F1-score')
    plt.legend(class_names)
    plt.title('F1')
    plt.savefig('out/plots/f1-score.png')
    plt.show()

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
