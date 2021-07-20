import numpy as _np
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt
from ipywidgets import interact

def interact_polyreg(max_degree, x, y, regularized=False, verbose=True):
    """
    The function to plot polynomial linear regression.
    
    Args:
        max_degree: int
            Max polynomial degree.
        x,y: numpy.ndarray
            1D Training data.
        regularized: bool
            Whether to add l2-norm regularization term in loss. Default to False.
        verbose: bool
            Whether to print trained weights. Default to True.
    """
    x_plot = _np.linspace(x.min(), x.max(), 30).reshape(-1,1)

    def polyreg_helper(degree):
        plt.figure(figsize=(10,6))
        plt.scatter(x, y, c='r', label='true')
        linear = make_pipeline(PolynomialFeatures(degree, include_bias=False),
                               MinMaxScaler(),
                               LinearRegression(fit_intercept=True))
        linear.fit(x.reshape(-1,1), y)
        mae_linear = mean_absolute_error(y, linear.predict(x.reshape(-1,1)))

        if regularized:
            ridge = make_pipeline(PolynomialFeatures(degree, include_bias=False),
                                  MinMaxScaler(),
                                  Ridge(alpha=1.0))
            ridge.fit(x.reshape(-1,1), y)
            mae_ridge = mean_absolute_error(y, ridge.predict(x.reshape(-1,1)))
            plt.plot(x_plot, linear.predict(x_plot), label='predicted, w/o regularization')
            plt.plot(x_plot, ridge.predict(x_plot), label='predicted, with regularization')
            plt.title(f"Poly degree = {degree:2}, MAE_no_reg = {mae_linear:.3f}, MAE_reg = {mae_ridge:.3f}", fontsize=16)
            if verbose:
                print('weights without regularization')
                print(linear.named_steps['linearregression'].coef_)
                print('weights with regularization')
                print(ridge.named_steps['ridge'].coef_)
        else:
            plt.plot(x_plot, linear.predict(x_plot), label='predicted')
            plt.title(f"Polynomial degree = {degree:2}, MAE = {mae_linear:.3f}", fontsize=16)
        plt.legend(fontsize=16)
        plt.show()
        
    interact(polyreg_helper, degree=(1, max_degree))
