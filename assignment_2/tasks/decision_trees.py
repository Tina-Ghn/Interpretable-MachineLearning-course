import sys
sys.path.insert(0, "")

import numpy as np
from sklearn import tree

from tasks.plotting import plot_bar
from utils.dataset import Dataset

def fit_decision_tree(X_train, y_train, random_state=0):
    """
    3.1
    Fits a decision tree on training data.
    
    Inputs:
        X_train (np.ndarray): Training data.
        y_train (np.ndarray): Target values.
        random_state (int): Seed of the decision tree.
        
    Returns:
        model (DecisionTreeClassifier): Fitted decision tree.
    """
    model = tree.DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model

def plot_feature_importance(model, dataset, title=None):
    """
    3.2
    Uses the feature importances of a decision tree and the dataset's input labels to plot a bar.
    Internally, `plot_bar` is called.
    
    Inputs:
        model (DecisionTreeClassifier): Decision tree.
        dataset (utils.Dataset): Used dataset to train the model. Used to receive the labels.
        
    Returns:
        x (list): Labels, which are displayed on the x-axis.
        y (list): Values, which are displayed on the y-axis.
    """
    x = dataset.get_input_labels()
    y = model.feature_importances_
    
    plot_bar(x, y, title=title, ylabel="Feature Importance")
    
    return x, y

import numpy as np

def compute_feature_importance(model):
    """
    Computes the feature importance of a DecisionTreeClassifier from scratch.

    Inputs:
        model (DecisionTreeClassifier): Fitted decision tree.

    Returns:
        feature_importance (np.ndarray): Feature importances with shape (#features,).
    """
    n_nodes = model.tree_.node_count
    n_features = model.tree_.n_features
    feature_importance = np.zeros(n_features)
    impurity = model.tree_.impurity
    samples = model.tree_.weighted_n_node_samples

    for i in range(n_nodes):
        if model.tree_.children_left[i] != model.tree_.children_right[i]:
            feature = model.tree_.feature[i]
            importance = samples[i] * (impurity[i] - (samples[model.tree_.children_left[i]] * impurity[model.tree_.children_left[i]] + samples[model.tree_.children_right[i]] * impurity[model.tree_.children_right[i]]) / samples[i])
            feature_importance[feature] += importance

    # Normalize the feature importance values
    total_importance = np.sum(feature_importance)
    feature_importance /= total_importance

    return feature_importance

def normalize_feature_importance(feature_importance):
    """
    3.4
    Normalizes the given feature importances.
    
    Inputs:
        feature_importance (np.ndarray): Feature importances with shape (#features,).
        
    Returns:
        feature_importance (np.ndarray): Normalized feature importances with shape (#features,).
    """
    total_importance = np.sum(feature_importance)
    normalized_importance = feature_importance / total_importance
    return normalized_importance
