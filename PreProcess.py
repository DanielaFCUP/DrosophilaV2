import os

import cv2
import torchvision
from PIL import Image
from rembg import remove
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

    if mode == "skip":
        pre_processed_directory = raw_image_directory
    else:
        for classe in os.listdir(raw_image_directory):
            raw_class_dir_path = os.path.join(raw_image_directory, classe)
            for img_name in os.listdir(raw_class_dir_path):
                path_to_img = os.path.join(raw_class_dir_path, img_name)
                img = cv2.imread(path_to_img)
                match mode:
                    case 'remove_background':
                        new_img = remove(img)
                    case 'gaussian':
                        new_img = cv2.GaussianBlur(img, (5, 5), 1)
                    case 'median':
                        # newImage = img.copy()
                        # new_img = cv2.medianBlur(img, newImage, 5)
                        new_img = cv2.medianBlur(img, 5)
                    case 'bilateral':
                        new_img = cv2.bilateralFilter(img, 10, 100, 100)
                    # case 'gauss_threshold':
                    #    new_img = cv2.GaussianBlur(img, (5, 5), 0)
                    #    # img2 = cv2.imread(new_img, cv2.IMREAD_GRAYSCALE)
                    #    img2 = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
                    #    th, _ = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    #    th2, new_img = cv2.threshold(img2, math.floor(th), 255, cv2.THRESH_BINARY)
                    case 'unsharp':
                        img2 = cv2.GaussianBlur(img, (5, 5), 1)
                        img3 = img - img2  # mask
                        new_img = img + img3
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
