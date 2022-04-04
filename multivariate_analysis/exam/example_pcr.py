import pandas as pd
import numpy as np

from methods.pca_procedure import PCAProcedure

if __name__ == '__main__':
    # Example of a dataset that does not have the same units
    wine = pd.read_csv('../wine.csv')
    data = wine.iloc[:, 1:12]

    pca_proc = PCAProcedure(data)
    pca_proc.print_summary()
    pca_proc.center()
    pca_proc.gen_corr_matrix()
    pca_proc.gen_principal_components()
    pca_proc.scree_plot()
    pca_proc.show_projection()
    pca_proc.show_biplot()
