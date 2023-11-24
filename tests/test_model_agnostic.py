import sys
import os  # noqa
sys.path.insert(0, "")  # noqa

import numpy as np
from classifiers.mlp import MLP
from utils.dataset import Dataset

from tests.config import WORKING_DIR
module = __import__(f"{WORKING_DIR}.model_agnostic", fromlist=[
    'merge', 'get_shapley_by_estimation'])


def test_merge():
    a = np.array([0, 1, 22, 3, 4, 55])
    b = np.array([66, 7, 8, 9, 101, 11])
    j = 2
    plus, minus = module.merge(a, b, j)

    assert plus[j] == 22
    assert minus[j] == 8
    assert plus[0] == minus[0]
    assert plus[4] == minus[4]

    j = 0
    plus, minus = module.merge(a, b, j)

    assert plus[0] == 0
    assert minus[0] == 66
    assert minus[5] == 11


def test_get_shapley_by_estimation():
    dataset = Dataset(
        "wheat_seeds",
        [2, 5, 6],
        [7],
        normalize=True,
        categorical=True)

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

    x = np.array([0.2, 0.55, 0.15])

    value = module.get_shapley_by_estimation(
        model, X_train, x, 0, M=10, seed=100)
    assert value == -0.3

    value = module.get_shapley_by_estimation(
        model, X_train, x, 0, M=0, seed=100)
    assert value == 0.0

    value = module.get_shapley_by_estimation(
        model, X_train, x, 0, M=5, seed=8)
    assert value == 0.0

    value = module.get_shapley_by_estimation(
        model, X_train, x, 2, M=20, seed=74)
    assert value == -0.3


if __name__ == "__main__":
    test_merge()
    test_get_shapley_by_estimation()
