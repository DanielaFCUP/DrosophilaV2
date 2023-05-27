import os

import matplotlib.pyplot as plt


def plots(train_accuracies: list, train_losses: list, test_accuracies: list, test_losses: list, f1_epochs: list, class_names: list):
    try:
        os.makedirs("out/plots/")
    except FileExistsError:
        # directory already exists
        pass

    plt.plot(train_accuracies, '-o')
    plt.plot(test_accuracies, '-o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Test'])
    plt.title('Train vs Test Accuracy')
    plt.savefig('out/plots/train_test_accuracies.png')
    plt.show()

    plt.plot(train_losses, '-o')
    plt.plot(test_losses, '-o')
    plt.xlabel('Epoch')
    plt.ylabel('Losses')
    plt.legend(['Train', 'Test'])
    plt.title('Train vs Test Losses')
    plt.savefig('out/plots/train_test_losses.png')
    plt.show()

    # F1 Score
    print(f1_epochs)
    plt.plot(f1_epochs, '-o')
    plt.xlabel('Epoch')
    plt.ylabel('F1-score')
    plt.legend(class_names)
    plt.title('F1')
    plt.savefig('out/plots/f1-score.png')
    plt.show()
