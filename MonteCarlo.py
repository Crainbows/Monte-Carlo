# Use numba in the future
from numba import njit
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import json

# Import json data
with open('AAPL.json') as json_data:
    d = json.load(json_data)

data = pd.DataFrame(d['graph']['data'],columns=['Date',1,2,3,4,5,6,7,'Open','High','Low','Close',12])
data.drop([1,2,3,4,5,6,7,12],inplace=True,axis=1)

# gc
del d

# numba function provides acceleration of predictor
@njit(parallel=True)
def _predict_price(time_frame, start_price, mean, stddev):
    '''numba enhanced prediction runner
    
    Arguments:
        time_frame {int} -- Number of days to predict
        start_price {int} -- Starting price
        mean {int} -- Mean of the price changes
        stddev {int} -- Standard deviation of the price changes
    
    Returns:
        array -- Single prediction for the price
    '''
    prediction = np.zeros(time_frame)
    prediction[0] = start_price
    for i  in range(time_frame-1):
        prediction[i+1] = np.random.normal(mean,stddev) * prediction[i]
    return prediction

class MonteCarlo():
    def __init__(self, historical_prices):
        '''Monte-Carlo simulator
        
        Arguments:
            historical_prices {array} -- Array of prices
        '''

        self.historical_prices = historical_prices
        self.weighting = []
    
    def set_sample_period(self, days, weighting):
        '''Add a sampleing period 
        
        Arguments:
            days {int} -- Days to calculate volatility
            weighting {int} -- Weighting of this time period
        '''
        self.weighting.append((days,weighting))

    def run(self, days, iterations=100):
        '''Run the simulation
        
        Arguments:
            days {int} -- Number of days to predict
        '''
        total_weight = np.sum(self.weighting,axis=0)[1]
        prediction = np.zeros(days)

        for (time, weight) in self.weighting:
            returns = 1 + data.iloc[0:-days]['Close'].pct_change()
            prediction += self.run_prediction(days, self.historical_prices[-days], iterations, returns[time:].mean(), returns[time:].std()).median(axis=1) * (weight / total_weight)
        return prediction

    def predict_price(self, time_frame, start_price, mean, stddev):
        '''Wrapper for the numba enhanced predictor
        
        Arguments:
            time_frame {int} -- Number of days to predict
            start_price {int} -- Starting price
            mean {int} -- Mean of the price changes
            stddev {int} -- Standard deviation of the price changes
        
        Returns:
            array -- Single prediction for the price
        '''
        return _predict_price(time_frame, start_price, mean, stddev)

    def run_prediction(self, time_frame, start_price, simulation_number, mean, stddev):
        x = np.zeros((time_frame,simulation_number))
        for i in range(simulation_number):
            x[:,i] = self.predict_price(time_frame, start_price, mean, stddev)
        return pd.DataFrame(x)




mc = MonteCarlo(data['Close'].values)
mc.set_sample_period(252, 1)
mc.set_sample_period(127, 2)
mc.set_sample_period(62, 4)
plt.plot(mc.run(252,iterations=1000), 'r')
plt.plot(data.iloc[-252:]['Close'].values)
plt.show()
