import sys
sys.path.insert(0, "")
from utils.dataset import Dataset
from tasks.plotting import plot_bar
from sklearn import linear_model
import numpy as np
import pandas as pd


def fit_linear_regression(X_train, y_train):
    """
    2.1
    Fits a linear regression model on training data.
    
    Inputs:
        X_train (np.ndarray): Training data.
        y_train (np.ndarray): Target values.
        
    Returns:
        model (LinearRegression): Fitted linear regression model.
    """
    
    model = None
    
    return model


def fit_my_linear_regression(X_train, y_train):
    """
    2.2
    Fits a self-written linear regression model on training data.
    
    Inputs:
        X_train (np.ndarray): Training data.
        y_train (np.ndarray): Target values.
        
    Returns:
        model (MyLinearRegression): Fitted linear regression model.
    """
    
    class MyLinearRegression:
        def __init__(self):
            self.coef_ = None
            self.bias_ = 0
            
        def predict(self, X) -> np.ndarray:
            """
            Uses internal coefficients and bias to predict the outcome.
            
            Returns:
                y (np.ndarray): Predictions of X.
            """
            return None
        
        def fit(self, X, y, learning_rate=1e-1, epochs=1000):
            """
            Adapts the coefficients and bias based on the gradients.
            Coefficients are initialized with zeros.
            
            Parameters:
                X: Training data. 
                y: Target values.
                learning_rate (float): Learning rate decides how much the gradients are updated.
                epochs (int): Iterations of gradient changes.
            """
            
            pass

    model = MyLinearRegression()
    model.fit(X_train, y_train)
    
    return model
    


def plot_linear_regression_weights(model, dataset: Dataset, title=None):
    """
    2.3
    Uses the coefficients of a linear regression model and the dataset's input labels to plot a bar.
    Internally, `plot_bar` is called.
    
    Inputs:
        model (LinearRegression or MyLinearRegression): Linear regression model.
        dataset (utils.Dataset): Used dataset to train the model. Used to receive the labels.
        
    Returns:
        x (list): Labels, which are displayed on the x-axis.
        y (list): Values, which are displayed on the y-axis.
    
    """
    x = None
    y = None
    
    return x, y


def fit_generalized_linear_model(X_train, y_train):
    """
    2.4
    Fits a GLM on training data, solving a multi-classification problem.
    
    Inputs:
        X_train (np.ndarray): Training data.
        y_train (np.ndarray): Target values.
        
    Returns:
        model: Fitted GLM.
    """
    model = None
    
    return model


def correlation_analysis(X):
    """
    2.5
    Performs a correlation analysis using X.
    Two features are correlated if the correlation value is higher than 0.9.
    
    Inputs:
        X (np.ndarray): Data to perform correlation analysis on.
    
    Returns:
        correlations (dict):
            Holds the correlated feature ids of a feature id.
            Key: feature id (e.g. X has 8 columns/features so the dict
                 should have 8 keys with 0..7)
            Values: list of correlated feature ids. Exclude the id from the key.
    """
    
    correlations = {} 
    return correlations


if __name__ == "__main__":
    dataset = Dataset("wheat_seeds", [0,1,2,3,4,5,6], [7], normalize=True, categorical=True)
    (X_train, y_train), (X_test, y_test) = dataset.get_data()

    model1 = fit_linear_regression(X_train, y_train)
    model2 = fit_my_linear_regression(X_train, y_train)
    
    plot_linear_regression_weights(
        model1, dataset, title="Linear Regression")
    plot_linear_regression_weights(
        model2, dataset, title="My Linear Regression")
    
    model3 = fit_generalized_linear_model(X_train, y_train)
    
    correlations = correlation_analysis(dataset.X)
    print(correlations)
