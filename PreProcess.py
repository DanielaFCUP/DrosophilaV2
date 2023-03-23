import os

import cv2
import torchvision
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

pre_processed_directory = "out/preproc_imgs/"

transformation = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def load_image(image_path: str) -> Image:
    global transformation
    image = Image.open(image_path)
    image = transformation(image)
    return image


def run(mode: str, raw_image_directory: str) -> ImageFolder:
    global pre_processed_directory
    print('Pre-processing has started.')

    try:
        os.makedirs(pre_processed_directory)
    except FileExistsError:
        # directory already exists
        pass

    if mode is "skip":
        pre_processed_directory = raw_image_directory
    else:
        for classe in os.listdir(raw_image_directory):
            raw_class_dir_path = os.path.join(raw_image_directory, classe)
            for img_name in os.listdir(raw_class_dir_path):
                path_to_img = os.path.join(raw_class_dir_path, img_name)
                img = cv2.imread(path_to_img)
                match mode:
                    case 'gaussian':
                        new_img = cv2.GaussianBlur(img, (5, 5), 0)
                    # case 'mean':

                    # case 'median':

                    # case 'bilateral':

                    # case 'smooth':
                    # raise NotImplementedError('This type of pre processing wasn\'t implemented yet.')
                    # case 'threshold':
                    # raise NotImplementedError('This type of pre processing wasn\'t implemented yet.')
                    # case 'unsharp':
                    case bad:
                        raise ValueError('Pre-processing found illegal mode: ', bad)
                pre_processed_directory_class = os.path.join(pre_processed_directory, classe)
                try:
                    os.makedirs(pre_processed_directory_class)
                except FileExistsError:
                    # directory already exists
                    pass
                new_img_path = os.path.join(pre_processed_directory_class, img_name)
                cv2.imwrite(new_img_path, new_img)

    print('Pre-processed images are in:', pre_processed_directory)

    torch_images = torchvision.datasets.ImageFolder(root=pre_processed_directory, transform=transformation)

    print('Pre-processing has ended.')

    return torch_images
