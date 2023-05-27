import sys

import torch
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder

import Classify
import Config
import Learn
import Performance
import PreProcess

# Set the random seed for PyTorch
torch.manual_seed(42)


def run_preproc(conf: dict) -> ImageFolder:
    preproc_dir = PreProcess.run(mode=conf['preproc'], raw_image_directory=conf['raw'])
    return preproc_dir


def run_learn(images_dir: ImageFolder, conf: dict):
    # Get the number of images in the dataset
    data_size = len(images_dir)

    # Split the dataset into training and testing datasets with a 70-30 ratio

    train_size = int(0.7 * data_size)
    test_size = data_size - train_size
    (train_img_select, test_img_select) = random_split(images_dir, [train_size, test_size])

    # Create data loaders for the training and testing datasets
    train_img_dl = torch.utils.data.DataLoader(train_img_select, batch_size=conf['batch'], shuffle=True, num_workers=0)
    test_img_dl = torch.utils.data.DataLoader(test_img_select, batch_size=conf['batch'], shuffle=True, num_workers=0)

    model = Learn.model_type_to_model(model_type=conf['model'], num_classes=len(images_dir.classes))
    optimiser = Learn.optimiser_type_to_optimiser(optimiser_type=conf['optim'], model=model, lr=conf['lr'])

    # Train and Test a model on the dataset and obtain the model and its statistics
    (model, (train_losses, train_accuracies, test_losses, test_accuracies), f1_epochs) = Learn.run(epochs=conf['epochs'],
                                                                                        train_images=train_img_dl,
                                                                                        test_images=test_img_dl,
                                                                                        model=model,
                                                                                        optimiser=optimiser)

    # Plot the training and testing statistics
    Performance.plots(train_accuracies=train_accuracies,
                      test_accuracies=test_accuracies,
                      train_losses=train_losses,
                      test_losses=test_losses,
                      f1_epochs=f1_epochs,
                    class_names=conf['class_names'])

    # Evaluate the model's F1 score and confusion matrix on the testing dataset
    class_report = Learn.f1_and_confusion_matrix(images=test_img_dl, model=model, class_names=conf['class_names'])
    print(class_report)

    with open("out/outputs.txt", 'a') as f:
        for epoch in range(conf['epochs']):
            # print('epoch', epoch)
            f.writelines('EPOCH' + str(epoch + 1) + '\n' + 'Train Accuracy: \t' + str(
                train_accuracies[epoch]) + '\n Train loss: \t' + str(train_losses[epoch]))
            f.write('\n')
            f.writelines('EPOCH' + str(epoch + 1) + '\n' + 'Test Accuracy: \t' + str(
                test_accuracies[epoch]) + '\n Test loss: \t' + str(test_losses[epoch]))
            f.write('\n \n')

        f.writelines(class_report)


def run_prepare(conf: dict):
    preproc_dir = run_preproc(conf)
    run_learn(preproc_dir, conf)


def run_classify(conf: dict):
    loaded_model = torch.load(conf['model_pth'])
    loaded_model.load_state_dict(torch.load(conf['model_state_dict']))
    Classify.run(model=loaded_model, image_path=conf['image'], class_names=conf['class_names'])


# For any run mode, we want to load the configuration first.
conf = Config.load()

# Execute the desired run mode
match conf['run']:
    case 'preproc':  # Pre_Process
        run_preproc(conf=conf)
        sys.exit(0)
    case 'prepare':  # Pre_Process -> Learn -> Performance
        run_prepare(conf=conf)
        sys.exit(0)
    case 'classify':  # Classify
        run_classify(conf=conf)
        sys.exit(0)
    case 'full':  # Pre_Process -> Learn -> Performance -> Classify
        run_prepare(conf=conf)
        run_classify(conf=conf)
        sys.exit(0)
    case bad:
        raise ValueError("Illegal run mode: ", bad)
