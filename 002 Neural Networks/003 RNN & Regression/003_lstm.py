#==========================================#
# Title:  Stock prices prediction with LSTM
# Author: Bogyeong Suh
# Date:   2023-02-03
#==========================================#
import math
import yfinance as yf
import numpy as np
from pandas_datareader import data as pdr 
from sklearn.preprocessing import MinMaxScaler 
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

yf.pdr_override()
df = pdr.get_data_yahoo('SBUX', start='2018-01-01', end='2022-12-31')

close_prices = df['Close']
values = close_prices.values
training_data_len = math.ceil(len(values)* 0.8)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(values.reshape(-1,1))
train_data = scaled_data[0: training_data_len, :]

x_train = []
y_train = []

days = 60 #length of a data
bs = 30 #batch_size
epoch = 20 #epoch


for i in range(days, len(train_data)):
    x_train.append(train_data[i-days:i, 0])
    y_train.append(train_data[i, 0])
    
x_train, y_train = np.array(x_train), np.array(y_train)
#Reshaping the train data to make it as input for LSTM layer input_shape(batchsize, timesteps, input_dim)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

test_data = scaled_data[training_data_len-days:, :]
x_test = []
y_test = values[training_data_len:]

for i in range(days, len(test_data)):
    x_test.append(test_data[i-days:i, 0])
    
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.summary()
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=bs, epochs=epoch)

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
rmse = np.sqrt(np.mean(predictions - y_test)**2)
print(rmse)

#for plotting
GT = df[training_data_len:]
GT['Predictions'] = predictions

plt.figure()
plt.title('Result')
plt.xlabel('Date')
plt.ylabel('Close prise USD ($)')
plt.plot(GT[['Close', 'Predictions']])
plt.legend(['Ground Truth', 'Predictions'], loc='lower right')
plt.show()