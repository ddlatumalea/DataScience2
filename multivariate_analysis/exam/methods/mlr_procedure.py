"""
When to use:
1. Prediction of variables from other variables that are more easily obtained.
2. To test if variables are linearly related.
"""
import numpy as np
import matplotlib.pyplot as plt



class MLR:
    """Performs Multilinear Regression.

    Aims at finding the linear relationship that best predicts one variable
    given the values of other variables. Multiple regressors predict the regressand.

    y = (1 X) (a, b1, ..., bk) + e, where a will be defined as b0

    Keyword arguments:
        x -- np.array containing multiple input variables, regressors.
        y -- np.array containing the regressand.
    """

    def __init__(self, x, y):
        n = len(y)
        self.X = np.c_[(np.ones(n), x)]
        self.y = y

        # coefficients
        self.coef = None

    def __str__(self):
        return f"y = {self.coef[0]} + {self.coef[1]} * x0 + {self.coef[2]} * x1 + ... + e"

    def __repr__(self):
        return self.__str__()

    def fit(self):
        X = self.X
        self.coef = np.linalg.inv(X.T @ X) @ X.T @ self.y

    def pred(self, X):
        """Return the predicted y value based on the regressors.

        Keyword argments:
        x -- a matrix containing the regressors.
        """
        return X @ self.coef

    def get_r2(self):
        """Returns R2 and adjusted R2 value.

        Adjusted R2 is corrected for the sample size and number of regressors.
        """
        if self.coef is None:
            raise ValueError('Fit the regression model first.')

        n = len(self.y)
        ypred = self.pred(self.X)

        r2 = 1 - ypred.var() / self.y.var()
        r2_adjusted = 1 - ((1 - r2) * (n - 1) / (n - self.X.shape[1] - 1))

        return r2, r2_adjusted

    def plot(self):
        n = len(self.y)
        plt.scatter(np.arange(n), self.y, label='data', color='red', s=5)

        ypred = self.pred(self.X)
        # regression line
        plt.scatter(np.arange(n), ypred, label='prediction', color='blue', s=5)

        plt.show()


class SLR:
    """Performs Simple Linear Regression.

    Aims at finding the linear relationship that best predicts one variable
    given the value of another variable.

    The regressor is the input.

    When to use:
    1. Prediction of variables from other variables that are more easily obtained.
    2. To test if two variables are linearly related.

    Keyword arguments:
        x -- an np.array containing the regressor.
        y -- an np.array containing the regressand.
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

        # coefficients
        self.a = None
        self.b = None

    def __str__(self):
        return f"y = {self.a} + {self.b}x + e"

    def __repr__(self):
        return self.__str__()

    def fit(self):
        """Solve the function and calculate the intercept and slope."""
        n = len(self.y)
        X = np.stack((np.ones(n), self.x), axis=1)

        self.a, self.b = np.linalg.inv(X.T @ X) @ X.T @ self.y

    def get_r2(self):
        """A measure that represents the proportion of the variance for a
        dependent variable that's explained by an independent variable(s) in a regression model."""
        return 1 - (self.a + self.b * self.x).var() / self.y.var()

    def get_residuals(self):
        segs = np.array(((self.x, self.x), (self.y, self.a + self.b * self.x))).T
        return segs

    def plot(self, title, xlabel , ylabel):
        # data
        plt.scatter(self.x, self.y, label='data', color='red', s=5)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        # residuals
        pred = self.a + self.b * self.x
        plt.vlines(self.x, ymin=pred, ymax=self.y, linestyles='dotted', color='lightgrey')

        # regression line
        xs = np.linspace(self.x.min(), self.x.max())
        plt.plot(xs, self.a + self.b * xs, '-', c='orange', label='regression line')

        # individual mean
        plt.axvline(self.x.mean(), ls='--', lw=0.75)
        plt.axhline(self.y.mean(), ls='--', lw=0.75)

        plt.legend()
        plt.show()