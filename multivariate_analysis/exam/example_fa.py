import pandas as pd
import numpy as np

from methods.fa_procedure import FAProcedure
from methods.pca_procedure import PCAProcedure

if __name__ == '__main__':
    # Example of a dataset that does not have the same units
    wine = pd.read_csv('../wine.csv')
    data = wine.iloc[:, 1:12]

    # PCA
    pca_proc = PCAProcedure(data)
    pca_proc.print_summary()
    pca_proc.center()
    pca_proc.gen_corr_matrix()
    pca_proc.gen_principal_components()

    fa_proc = FAProcedure(pca_proc=pca_proc)
    fa_proc.plot()