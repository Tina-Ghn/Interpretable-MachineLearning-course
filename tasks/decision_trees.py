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
    x = [dataset.get_input_labels(i) for i in range(len(dataset.input_ids))]  # Use the method to get feature labels
    y = model.feature_importances_
    plot_bar(x, y, name=title, ylabel="Feature Importance")
    return x, y

def compute_feature_importance(model):
    """
    3.3
    Computes the feature importance of DecisionTreeClassifier from scratch.
    
    Inputs:
        model (DecisionTreeClassifier): Fitted decision tree.
        
    Returns:
        feature_importance (np.ndarray): Feature importances with shape (#features,).
    """
    impurity = model.tree_.impurity  # The impurity at node i
    samples = model.tree_.weighted_n_node_samples  # Weighted number of training samples reaching node i

    # Initialize the feature importance with zeros and shape (num_features,)
    feature_importance = np.zeros((model.tree_.n_features,))

    # Calculate the feature importance
    for node in range(model.tree_.node_count):
        if model.tree_.children_left[node] != model.tree_.children_right[node]:
            left = model.tree_.children_left[node]
            right = model.tree_.children_right[node]
            feature = model.tree_.feature[node]
            decrease = impurity[node] - (samples[left] * impurity[left] + samples[right] * impurity[right]) / samples[node]
            feature_importance[feature] += decrease * samples[node]

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

if __name__ == "__main__":
    dataset = Dataset("wheat_seeds", [0,1,2,3,4,5,6], [7], normalize=True, categorical=True)
    (X_train, y_train), (X_test, y_test) = dataset.get_data()
    
    model = fit_decision_tree(X_train, y_train)
    plot_feature_importance(model, dataset, title="Decision Tree")
    
    fi = compute_feature_importance(model)
    fi = normalize_feature_importance(fi)
