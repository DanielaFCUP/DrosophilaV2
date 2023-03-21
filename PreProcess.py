import os

import torchvision
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

transformation = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def load_image(image_path: str) -> Image:
    image = Image.open(image_path)
    image = transformation(image)
    return image


def run(mode: str, raw_image_directory: str) -> ImageFolder:
    print('Pre-processing has started.')

    try:
        os.makedirs("out/preproc_imgs/")
    except FileExistsError:
        # directory already exists
        pass

    match mode:
        case 'skip':
            pre_processed_directory = raw_image_directory
        case 'smooth':
            raise NotImplementedError('This type of pre processing wasn\'t implemented yet.')
        case 'threshold':
            raise NotImplementedError('This type of pre processing wasn\'t implemented yet.')
        case bad:
            raise ValueError('Pre-processing found illegal mode: ', bad)

    print('Pre-processed images are in:', pre_processed_directory)

    torch_images = torchvision.datasets.ImageFolder(root=pre_processed_directory, transform=transformation)

    print('Pre-processing has ended.')

    return torch_images
