
import sys
import os  # noqa
sys.path.insert(0, "")  # noqa

import torch
from utils.styled_plot import plt
from utils.dataset import (
    load_test_image,
    preprocess_image,
    normalize_image,
    unnormalize_image,
    convert_idx_to_label
)
from classifiers.cnn_classifier import ImageNetClassifier

torch.manual_seed(0)

def get_gradient(model, image):
    """
    Propagates the cross entropy loss between the model's output (logits) and the label (here the label
    that is predicted by the model is used) back to the input to get the input gradient.

    Parameters:
        model (ImageNetClassifier, torch.nn.Module):
            The image classification model. This is a torch.nn.Module, so you can call its forward method using
            `model()`. The output are logits (class probabilities). Also has a `.predict` method that returns the
            index of the predicted label.

        image (torch.tensor): The input for which to compute the gradient.

    Returns:
        gradient (torch.tensor): The input gradient. Same shape as the input image.
    """
    # Ensure the model is in evaluation mode
    model.eval()

    # Set requires_grad for the input image to True
    image.requires_grad_(True)

    # Forward pass to obtain logits
    logits = model(image)

    # Get the predicted label
    predicted_label = torch.argmax(logits, dim=1)

    # Create a one-hot tensor for the predicted label
    one_hot = torch.zeros_like(logits)
    one_hot[0, predicted_label] = 1.0

    # Compute the cross-entropy loss
    loss = torch.nn.functional.cross_entropy(logits, predicted_label)

    # Backward pass to compute the gradient
    loss.backward()

    # Extract the gradient from the input image
    gradient = image.grad.data

    return gradient

def perturb_image(image, grad, eps):
    """
    Applies a perturbation to an image based on the Fast-Gradient-Sign-Method.

    Parameters:
        image (torch.tensor): The image to perturb.

        grad (torch.tensor): The input gradient corresponding to the image.

        eps (float): The epsilon value for the perturbation, specifying the magnitude of the perturbation.

    Returns:
        image (torch.tensor): The perturbed image.
    """
    perturbation = eps * torch.sign(grad)
    perturbed_image = image + perturbation


    return perturbed_image


import matplotlib.pyplot as plt
import torchvision.transforms as transforms

def create_adversarials(model, image, eps_values):
    """
    Creates adversarial examples for the given image and model using the Fast-Gradient-Sign-Method.

    Parameters:
        model (ImageNetClassifier, torch.nn.Module):
            The image classification model. This is a torch.nn.Module, so you can call its forward method using `model()`.
            Also has a `.predict` method that returns the index of the predicted label.

        image (torch.tensor): The image to generate adversarial examples from.

        eps_values (List[float]): The list of epsilon values for which to generate adversarial examples.

    Returns:
        adversarials (List[torch.tensor]): A list containing one adversarial example for each eps value in eps_values.
    """
    adversarials = []

    for eps in eps_values:
        gradient=get_gradient(model,image)
        perturbed_image=perturb_image(image,gradient,eps)
        # Append the perturbed image to the list of adversarials
        adversarials.append(perturbed_image.detach().clone())
    return adversarials

def plot_adversarials(model, image, adv_images, eps_values):
    """
    Plots the  original image and the adversarial images in a single row.
    Uses the eps value and the predicted label as axis titles.

    Parameters:
        model (ImageNetClassifier, torch.nn.Module):
            The image classification model. This is a torch.nn.Module, so you can call its forward method using `model()`.
            Also has a `.predict` method that returns the index of the predicted label.

        image (torch.tensor): The original image corresponding to the adversarial examples.

        adv_images (List[torch.tensor]): A list containing the adversarial examples to visualize.

        eps_values (List[float]): The list of epsilon values corresponding to each adversarial example in adv_images.

    Hint: 
        - matplotlib expects a channels last format
        - The model works with normalized images. Before visualizing the images, you have to invert the normalization
        using `unnormalize()`
    """
    fig, axes = plt.subplots(len(adv_images) + 1, 1, figsize=(10, 2 * (len(adv_images) + 1)))

    # Plot original image
    original_label = model.predict(image)
    unnormalized_image = unnormalize_image(image)
    axes[0].imshow(unnormalized_image)
    axes[0].set_title(f'Original\nLabel: {original_label.item()}')

    # Plot adversarial images
    for i, adv_image in enumerate(adv_images):
        adv_label = model.predict(adv_image)
        unnormalized_adv_image = unnormalize_image()
        axes[i + 1].imshow(unnormalized_adv_image)
        axes[i + 1].set_title(f'Eps: {eps_values[i]}\nAdv Label: {adv_label.item()}')

    plt.show()


if __name__ == "__main__":
    image = load_test_image()
    image_preprocessed = preprocess_image(image)
    image_preprocessed_norm = normalize_image(image_preprocessed).unsqueeze(0)

    model = ImageNetClassifier()
    
    y_pred, y_prob = model.predict(image_preprocessed_norm, return_probs=True)
    print(f'Predicted class: "{convert_idx_to_label(y_pred.item())}". Confidence: {y_prob.item() * 100:.2f}%')
    assert y_pred == torch.tensor([13])
    assert torch.allclose(y_prob, torch.tensor([0.9483]), atol=1e-4)

    eps_values = [0.0001, 0.001, 0.01, 0.1, 0.3]
    print('Running `create_adversarial` ...')
    adv_images = create_adversarials(model, image_preprocessed_norm, eps_values)

    print('Running `plot_adversarials` ...')
    plot_adversarials(model, image_preprocessed_norm, adv_images, eps_values)
    plt.savefig("01_adversarial.png")
    plt.show()
