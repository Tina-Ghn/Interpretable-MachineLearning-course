import sys
import os
import random  # noqa
sys.path.insert(0, "")  # noqa

from utils.styled_plot import plt
from utils.dataset import Dataset
from classifiers.mlp import MLP
import numpy as np
import itertools
from sklearn import tree

def get_mesh(model, dataset, points_per_feature=50):
    """
    Retrieve mesh data for plotting a two-dimensional graph with `points_per_feature` for each axis.
    The space is created by the features' lower and upper values. Only the first two
    features are used.

    Parameters:
        model: Classifier which can call a predict method.
        dataset (utils.Dataset): Dataset with access to features and their bounds.
        points_per_feature: How many points in each dimension.

    Returns:
        x1 (np.ndarray): Values for feature 1 with shape (`points_per_feature`,).
        x2 (np.ndarray): Values for feature 2 with shape (`points_per_feature`,).
        M (np.ndarray): MLP predictions with shape (`points_per_feature`, `points_per_feature`).
    """

    # Get the lower and upper bounds for the first two features
    lower_bound_1, upper_bound_1 = dataset.get_lower_and_upper_bound_for_feature(0)
    lower_bound_2, upper_bound_2 = dataset.get_lower_and_upper_bound_for_feature(1)

    # Generate equidistant points for x1 and x2
    x1 = np.linspace(lower_bound_1, upper_bound_1, points_per_feature)
    x2 = np.linspace(lower_bound_2, upper_bound_2, points_per_feature)

    # Create a meshgrid
    X1, X2 = np.meshgrid(x1, x2)

    # Flatten the meshgrid to obtain points for prediction
    Z = np.c_[X1.ravel(), X2.ravel()]

    # Predict using the model
    predictions = model.predict(Z)

    # Reshape the predictions to match the shape of the meshgrid
    M = predictions.reshape(X1.shape)

    return x1, x2, M



def plot_mesh(x1, x2, M, labels, title=None, embedded=False):
    if not embedded:
        plt.figure()

    plt.pcolormesh(x1, x2, M, cmap='coolwarm', shading='auto')
    plt.colorbar(label='Prediction')

    plt.xlabel(labels[0])
    plt.ylabel(labels[1])

    if title:
        plt.title(title)

    return plt






def sample_points(model, dataset, num_points, seed=0):
    random.seed(seed)
    X = np.random.uniform(low=dataset.get_lower_and_upper_bound_for_feature(0)[0],
                          high=dataset.get_lower_and_upper_bound_for_feature(0)[1],
                          size=(num_points, 1))
    y = model.predict(X)
    return X, y





def weight_points(selected_x, Z, sigma=0.2):
    """
    For every x in `X` returns a weight depending on the distance to `selected_x`.

    Parameters:
        selected_x (np.ndarray): Single point with shape (2,).
        Z (np.ndarray): Points with shape (?, 2).
        sigma (float): Sigma value to calculate distance.

    Returns:
        weights (np.ndarray): Normalized weights between 0..1 with shape (?,).
    """

    # Calculate Euclidean distances
    distances = np.linalg.norm(Z - selected_x, axis=1)

    # Calculate weights using the given formula
    weights = np.exp(-distances**2 / (2 * sigma**2))

    # Normalize weights to be between 0 and 1
    weights = weights / np.sum(weights)

    return weights


def plot_points_in_mesh(plt, X=[], y=[], weights=None, colors={}, selected_x=None, size=8):
    """
    Given a plot, add scatter points from `X` and `selected_x`.

    Parameters:
        plt (matplotlib.pyplot or utils.styled_plot.plt): Plot with color mesh.
        X (np.ndarray): Points with shape (?, 2).
        y (np.ndarray): Target values with shape (?,).
        weights (np.ndarray): Normalized weights with shape (?,).
        colors (dict): Returns the color for a y value.
        selected_x (np.ndarray): Single point with shape (2,).
        size (int): Default size of the markers.
    """

    # Scatter points from X with colors based on y
    if X.shape[1] == 2:
        scatter = plt.scatter(X[:, 0], X[:, 1], c=[colors[val] for val in y], s=weights * size)
    else:
        # Handle the case where X has more than 2 columns
        # You may need to modify this based on your data structure
        scatter = plt.scatter(X[:, 0], X[:, 1], c=[colors[val] for val in y], s=weights * size)

    # Highlight the selected point
    if selected_x is not None:
        plt.scatter(selected_x[0], selected_x[1], color='red', marker='x', s=100, label='Selected Point')

    # Add legend and labels if needed
    plt.legend()
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    return scatter



from sklearn.tree import DecisionTreeRegressor

def fit_explainer_model(Z, y, weights=None, seed=0):
    model = tree.DecisionTreeRegressor(random_state=seed)
    model.fit(Z, y, sample_weight=weights)

    return model


if __name__ == "__main__":
    dataset = Dataset(
        "wheat_seeds", [1, 5], [7], normalize=True, categorical=True)
    (X_train, y_train), (X_test, y_test) = dataset.get_data()

    input_units = X_train.shape[1]
    output_units = len(dataset.get_classes())

    model = MLP(3, input_units*2, input_units, output_units, lr=0.01)

    filename = "weights.pth"
    if not os.path.isfile(filename):
        model.fit(X_train, y_train, num_epochs=50, batch_size=16)
        model.save(filename)
    else:
        model.load(filename)

    selected_x = np.array([0.31, 0.37])
    points_per_feature = 50
    n_points = 1000
    labels = dataset.get_input_labels()
    colors = {
        0: "purple",
        1: "green",
        2: "orange",
    }

    print("Run `get_mesh` ...")
    x1, x2, M = get_mesh(
        model, dataset, points_per_feature=points_per_feature)

    print("Run `plot_mesh` ...")
    plt = plot_mesh(x1, x2, M, labels=labels, title="MLP")
    plt.savefig("01_custom_lime_mesh.png")
    plt.show()

    print("Run `sample_points` ...")
    Z_sampled, y_sampled = sample_points(model, dataset, n_points)

    print("Run `plot_points_in_mesh` ...")
    plt = plot_mesh(x1, x2, M, labels=labels, title="MLP + Sampled Points")
    plot_points_in_mesh(plt, Z_sampled, y_sampled, colors=colors)
    plt.legend()
    plt.savefig("02_custom_lime_sampled.png")
    plt.show()

    print("Run `weight_points` ...")
    weights = weight_points(selected_x, Z_sampled)

    print("Run `plot_points_in_mesh` ...")
    plt = plot_mesh(x1, x2, M, labels=labels,
                    title="MLP + Weighted Sampled Points")
    plot_points_in_mesh(plt, Z_sampled, y_sampled, weights, colors, selected_x)
    plt.legend()
    plt.savefig("03_custom_lime_weighted.png")
    plt.show()

    print("Run `fit_explainer_model` ...")
    explainer = fit_explainer_model(Z_sampled, y_sampled, weights)

    print("Compare models ...")
    plt.subplots(cols=2, rows=1, sharey=True)

    plt.subplot(1, 2, 1)
    plot_mesh(x1, x2, M, labels=labels, title="MLP", embedded=True)
    plot_points_in_mesh(plt, Z_sampled, y_sampled, weights, colors, selected_x)
    plt.legend()

    plt.subplot(1, 2, 2)
    x12, x22, M2 = get_mesh(
        explainer, dataset, points_per_feature=points_per_feature)
    plot_mesh(x12, x22, M2, labels=labels, title="Decision Tree", embedded=True)
    plot_points_in_mesh(plt, Z_sampled, y_sampled, weights, colors, selected_x)
    plt.legend()

    plt.savefig("04_custom_lime_comparison.png")
    plt.show()
