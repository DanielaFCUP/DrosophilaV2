import numpy as np
import torch
import torch.nn.functional as F

import PreProcess


def run(model, image_path: str, class_names: list) -> (list, int, str):
    probabilities = classify(model=model, image_path=image_path)

    # Pick index with the highest probability
    prediction = np.argmax(probabilities)

    for i, val in enumerate(probabilities):
        print(f"{class_names[i]}'s likelihood of {val:.4f}.")

    classification = class_names[prediction]
    print(f"It most likely is a {classification} with a probability of {probabilities[prediction]:.4f}.")

    return probabilities, prediction, classification


def classify(model, image_path: str) -> list:
    # Let's put our network in classification mode
    model.eval()

    # Load image from file
    image = PreProcess.load_image(image_path=image_path)

    with torch.no_grad():
        # Convert image to batch of 1
        outputs = model(image.unsqueeze(0))

        # Get probabilities
        probabilities = F.softmax(outputs, dim=1).tolist()[0]

    return probabilities
