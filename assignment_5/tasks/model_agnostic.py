import sys
import os  # noqa

sys.path.insert(0, "")  # noqa

from utils.dataset import Dataset
from classifiers.mlp import MLP
import numpy as np
import torch


def merge(a, b, j):
    """
    Merges elements a and b s.t. two new arrays are created.
    Elements before position j will be from a, whereas elements after position j will be from b.

    Parameters:
        a (np.ndarray): Numpy array with shape (?,)
        b (np.ndarray): Numpy array with shape (?,)
        j (int): Index of the feature.

    Returns:
        plus_j (np.ndarray): Element from a at the position j. Same shape as a or b.
        minus_j (np.ndarray): Element from b at the position j. Same shape as a or b.
    """
    plus_j = np.concatenate((a[:j],a[j], b[j+1:]), axis=None)
    minus_j = np.concatenate((a[:j], [b[j]], b[j+1:]), axis=None)

    return plus_j, minus_j


def get_shapley_by_estimation(model, X, x, j, M=None, seed=0):
    """
    Calculates the Shapley value by approximation.

    Parameters:
        model: A fit ML model.
        X (np.ndarray): Data from which z is sampled from.
        x: Instance of interest.
        j (int): Feature index. Starts from 0.
        M (int): Number of samples to base the estimate on.
        seed (int): Seed for numpy.

    Returns:
        value (float): Shapley value.
    """

    np.random.seed(seed)

    # If M is not provided, use the number of entries in X
    M = M or len(X)

    # Initialize Shapley value
    value = 0.0

    for _ in range(M):
        # Sample a random instance z from X
        z = X[np.random.choice(len(X))]

        # Randomly permute features for both x and z
        perm_x = np.random.permutation(len(x))
        perm_z = np.random.permutation(len(z))

        # Merge the permuted instances
        plus_x, minus_x = merge(x, z, j)
        plus_z, minus_z = merge(z, x, j)

        # Predict using the permuted instances
        plus_x_pred = model.predict(np.expand_dims(plus_x, axis=0))
        minus_x_pred = model.predict(np.expand_dims(minus_x, axis=0))
        plus_z_pred = model.predict(np.expand_dims(plus_z, axis=0))
        minus_z_pred = model.predict(np.expand_dims(minus_z, axis=0))

        # Update Shapley value based on the estimated contribution
        value += (plus_x_pred - minus_x_pred + minus_z_pred - plus_z_pred)

    # Normalize the Shapley value by dividing by the number of samples
    value /= M

    return value



if __name__ == "__main__":
    dataset = Dataset(
        "wheat_seeds",
        [2, 5, 6],
        [7],
        normalize=True,
        categorical=True)

    (X_train, y_train), (X_test, y_test) = dataset.get_data()

    input_units = X_train.shape[1]
    output_units = len(dataset.get_classes())
    features = dataset.get_input_labels()

    model = MLP(3, input_units * 2, input_units, output_units, lr=0.01)
    filename = "weights.pth"
    if not os.path.isfile(filename):
        model.fit(X_train, y_train, num_epochs=50, batch_size=16)
        model.save(filename)
    else:
        model.load(filename)

    a = np.array([0, 1, 2, 3, 4, 5])
    b = np.array([6, 7, 8, 9, 10, 11])
    j = 2

    with open("model_agnostic.txt", "w") as f:
        f.write("Run `merge` ...\n")
        plus, minus = merge(a, b, j)

        f.write(f"--- a: {a}\n")
        f.write(f"--- b: {b}\n")
        f.write(f"--- j: {j}\n")
        f.write(f"--- plus: {plus}\n")
        f.write(f"--- minus: {minus}\n")

        f.write("\nRun `get_shapley_by_estimation` ...\n")
        x = np.array([0.5, 0.5, 0.5])

        for j in range(3):
            value = get_shapley_by_estimation(model, X_train, x, j, M=10)
            f.write(f"--- {features[j]}: {value}\n")
