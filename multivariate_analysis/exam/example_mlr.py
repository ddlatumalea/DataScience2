import pandas as pd
import numpy as np

from methods.mlr_procedure import SLR, MLR

if __name__ == '__main__':
    # example for simple linear regression
    iq = pd.read_csv('../IQ.dat', sep=' ', header=0)
    x = iq['M_Height']
    y = iq['IQ']

    slr = SLR(x=x, y=y)
    slr.fit()
    print(slr)
    print(slr.get_r2())

    slr.plot(
        title='IQ',
        xlabel='National average male length (m)',
        ylabel='national average IQ'
    )

    # example for multilinear regression
    X = np.stack((iq.Length, iq.Volume), axis=1)
    y = iq.IQ

    mlr = MLR(x=X, y=y)
    mlr.fit()
    print(mlr)
    print(mlr.get_r2())
    mlr.plot()