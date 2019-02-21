# Use numba in the future
# from numba import njit
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  

import json


with open('AAPL.json') as json_data:
    d = json.load(json_data)

show_plots = False

data = pd.DataFrame(d['graph']['data'],columns=['Date',1,2,3,4,5,6,7,'Open','High','Low','Close',12])
data.drop([1,2,3,4,5,6,7,12],inplace=True,axis=1)

log_returns = np.log(1 + data['Close'].pct_change())
log_returns.tail()

if show_plots:
    data.plot(figsize=(10,6))
    plt.show()
    log_returns.plot(figsize=(10,6))
    plt.show()

print('Mean: ',log_returns.mean())
print('Var: ',log_returns.var())