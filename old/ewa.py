# %%
import pandas as pd
import numpy as np
import taew
import pandas_datareader.data as web
import datetime

start = datetime.datetime(2019, 1, 1)
end = datetime.datetime(2020, 1, 27)

SP500 = web.DataReader(['sp500'], 'fred', start, end)

haha2 = taew.Alternative_ElliottWave_label_upward(
    np.array(SP500[['sp500']].values, dtype=np.double).flatten(order='C'))
print(haha2)

# %%
