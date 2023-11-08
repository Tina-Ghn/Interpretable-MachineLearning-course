import sys
sys.path.insert(0, "")
from utils.dataset import Dataset
from tasks.plotting import plot_bar
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
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
    model = LinearRegression()
    model.fit(X_train, y_train)
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
            return np.dot(X, self.coef_) + self.bias_
        
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
            num_samples, num_features = X.shape
            self.coef_ = np.zeros(num_features)
            self.bias_ = 0

            for _ in range(epochs):
                y_pred = np.dot(X, self.coef_) + self.bias_
                dw = (1/num_samples) * np.dot(X.T, (y_pred - y))
                db = (1/num_samples) * np.sum(y_pred - y)
                self.coef_ -= learning_rate * dw
                self.bias_ -= learning_rate * db

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
        title (str): Title for the plot.
        
    Returns:
        x (list): Labels, which are displayed on the x-axis.
        y (list): Values, which are displayed on the y-axis.
    """
    x = dataset.get_input_labels()  # Get input labels from the dataset
    y = model.coef_  # Get the coefficients from the model
    plot_bar(x, y, title, ylabel="Weight")
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
  
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.linear_model import LogisticRegression
    
    model = OneVsRestClassifier(LogisticRegression())
    model.fit(X_train, y_train)
    return model


def correlation_analysis(X):
    """
    2.5
    Performs a correlation analysis to check the correlations of a single feature.
    
    Inputs:
        X (np.ndarray): Input features.
        y (np.ndarray): Target values.
        
    Returns:
        correlations (pd.Series): Correlations between input features and target values.
    """
    df = pd.DataFrame(X)
    correlations = {}

    for i in range(X.shape[1]):
        correlations[i] = [j for j in range(X.shape[1]) if i != j and abs(df[i].corr(df[j])) > 0.9]

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
