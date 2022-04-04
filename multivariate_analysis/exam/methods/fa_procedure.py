import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from methods.pca_procedure import PCAProcedure


class FAProcedure:

    def __init__(self, pca_proc: PCAProcedure):
        self.pca_proc = pca_proc

    def varimax(self, components, gamma=1, maxiter=20, tol=1e-8):
        """Perform VariMax (gamma=1) or OrthoMax (gamma=0) rotation on components"""
        p, k = components.shape
        R = np.eye(k)
        f = float(gamma) / p
        d = 0
        for i in range(maxiter):
            d_old = d
            L = np.dot(components, R)
            A = L ** 3 - f * (L * (L ** 2).sum(axis=0)) ** 2
            U, s, V = np.linalg.svd(np.dot(components.T, A))
            R = np.dot(U, V)
            d = sum(s)
            if (d - d_old) ** 2 < tol:
                break
        return np.dot(components, R)

    def plot(self):
        eigenvectors = self.pca_proc.eigenvectors
        z_scores = self.pca_proc.z_scores
        variables = self.pca_proc.df.columns[1:12]
        V = self.varimax(eigenvectors[:, :3])


        fa_projections = np.dot(z_scores, V)
        plt.scatter(fa_projections[:, 0], fa_projections[:, 1], c=fa_projections[:, 2], s=1)
        for var, (x, y) in zip(variables, 10 * V[:, :2]):
            plt.arrow(0, 0, x, y, head_width=0.5)
            plt.text(1.1 * x, 1.1 * y, var)
        plt.show()