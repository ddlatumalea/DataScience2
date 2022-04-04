import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class PCAProcedure:
    """A procedure to do PCA analysis. This class is able to the analysis step by step.

    Steps:
    1. Transform the data if needed (argue why)
    2. Center the data.
    3. Scale the data if needed (argue why)
    4. Calculate the covariance matrix, or correlation matrix
    5. Eigen decomposition of the matrix
    6. Make a scree plot of eigenvalues (reflect on results)
    7. Project/rotate the data (calculate 'scores')
    8. Plot the scores (if possible add eigenvectors generating a 'biplot'
    9. Determine if factor analysis is to be performed:
        - rotate selected components
        - project data on rotated components
        - plot scores (and rotated components)
    10. Interpret

    Keyword arguments:
    data -- a pd.DataFrame containing the data to be converted.
    """

    def __init__(self, data: pd.DataFrame) -> None:
        """df is the dataframe representation, while data represent the raw values."""
        self.df = data
        self.data = data.values
        self.matrix = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.z_scores = None

    def print_summary(self):
        """Check for variance.

        If the variance differs in orders of magnitude there could be different units.
        This means the file should be trasformed.
        """
        var = self.df.var()
        print(var)
        print("Minimum variance: {} -- Maximum variance: {}".format(var.min(), var.max()))

    def center(self):
        """Center the data."""
        self.data = self.data - self.data.mean(axis=0)

    def gen_corr_matrix(self):
        """Use this if data needs to be transformed"""

        zscores = self.data / self.data.std(axis=0)
        self.z_scores = zscores
        self.matrix = np.dot(zscores.T, zscores) / (zscores.shape[0] - 1)

    def gen_covar_matrix(self):
        self.matrix = np.cov(self.data)

    def gen_principal_components(self):
        eigenvalues, eigenvectors = np.linalg.eig(self.matrix)
        order = eigenvalues.argsort()[::-1]
        self.eigenvalues = eigenvalues[order]
        self.eigenvectors = eigenvectors[:, order]

    def scree_plot(self):
        plt.plot(self.eigenvalues, 'o-')
        plt.show()

        # cumulative percentage of eigenvalues
        plt.plot(100 * self.eigenvalues.cumsum() / sum(self.eigenvalues))
        plt.ylim((0, 110))
        plt.show()

    def show_projection(self):
        # How do you do a projection without zscores?
        zscores = self.data / self.data.std(axis=0)
        projections = np.dot(zscores, self.eigenvectors)

        plt.scatter(projections[:, 0], projections[:, 1], c=projections[:, 2], s=1)
        plt.show()

    def show_biplot(self):
        variables = self.df.columns

        zscores = self.data / self.data.std(axis=0)
        projections = np.dot(zscores, self.eigenvectors)

        plt.scatter(projections[:, 0], projections[:, 1], c=projections[:, 2], s=1)
        for var, (x, y) in zip(variables, 10 * self.eigenvectors[:, :2]):
            plt.arrow(0, 0, x, y, head_width=0.5)
            plt.text(1.1 * x, 1.1 * y, var)
        plt.show()
