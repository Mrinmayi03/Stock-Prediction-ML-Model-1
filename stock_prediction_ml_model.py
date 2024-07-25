import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt   
import yfinance as yf 
from keras.models import load_model
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from keras.layers import Dense , Dropout , LSTM
from keras.models import Sequential



print("Starting to load model.")
model = load_model('C:\\Users\\91998\\OneDrive\\Documents\\GitHub\\Stock-Prediction-ML-Model\\Stock Predictions ML Model.keras')
print("Model loading done.")


st.header("Stock Market Predictor")

stock = st.text_input("Enter the stock symbol: " , "GOOG")
start_date = '2010-01-01'
end_date = '2024-05-31'

data = yf.download(stock , start_date , end_date)
st.subheader("Stock Data: ")
st.write(data)


close_data = data[['Close']]

st.subheader("General trend of the closing amount of the stock: ")
fig0 = plt.figure(figsize=(8,6))
plt.plot(close_data)
plt.ylabel("Price")
plt.show()
st.pyplot(fig0) 

train_data , test_data = train_test_split(close_data , test_size=0.2, shuffle=False)


scaler = MinMaxScaler(feature_range=(0,1))

past_100_days = train_data.tail(100)

test_data = pd.concat([past_100_days , test_data] , ignore_index = True)

test_data_scaled = scaler.fit_transform(test_data)

st.subheader("Price VS Moving Average for 50 days: ")
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days , 'b')
plt.plot(data.Close , 'r')
plt.legend()
plt.show()
st.pyplot(fig1)

st.subheader("Price VS Moving Average for 50 days VS Moving Average for 100 days: ")
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days , 'b')
plt.plot(ma_100_days , 'g')
plt.plot(data.Close , 'r')
plt.legend()
plt.show()
st.pyplot(fig2)

st.subheader("Price VS Moving Average for 50 days VS Moving Average for 100 days VS Moving Average for 200 days: ")
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days , 'b')
plt.plot(ma_100_days , 'g')
plt.plot(ma_200_days , 'magenta')
plt.plot(data.Close , 'r')
plt.legend()
plt.show()
st.pyplot(fig3)


x = []
y = []
y_unscaled = []

for i in range(100 , test_data_scaled.shape[0]):
    x.append(test_data_scaled[i-100 : i])
    y.append(test_data_scaled[i , 0])
    
for i in range(100, len(y) + 100):
    y_unscaled.append(test_data.iloc[i, 0]) 

x , y , y_unscaled = np.array(x) , np.array(y) , np.array(y_unscaled)

'''
x contains the 100 value sequences and the predict = model.predict(x) will just predict the values for those specific sequences.
'''
predict = model.predict(x)

# #Printing x and y just to check:
# st.subheader("Printing x: ")
# st.write(x)
# st.subheader("Printing y: ")
# st.write(y)
# st.subheader("Printing y unscaled: ")
# st.write(y_unscaled)

predict = scaler.inverse_transform(predict)
# st.subheader("Predict unscaled: ")
# st.write(predict)

# y = scaler.inverse_transform(y)
# st.subheader("Y unscaled: ")
# st.write(y)

# # Inverse transform the scaled data back to the original scale
# scale = 1 / scaler.scale_[0]
# predict = predict * scale
# y = y * scale



st.subheader("Original Price VS Predicted Price: ")
fig4 = plt.figure(figsize=(8,6))
plt.plot(predict , 'r' , label="Predicted Price")
plt.plot(y_unscaled , 'g' , label="Original Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()
st.pyplot(fig4)

'''Adding another column to the original stock dataset: Difference.
This column will have the difference between the original price and predicted price.
'''

# Create a DataFrame with the test data and predictions
predicted_dates = data.index[-len(predict):]  # Dates corresponding to the predictions
results_df = pd.DataFrame({
    'Date': predicted_dates,
    'Original Price': y_unscaled,
    'Predicted Price': predict.flatten()
})

# Add the 'Difference' column
results_df['Difference'] = results_df['Original Price'] - results_df['Predicted Price']
results_df["Difference %"] = ((abs(results_df['Original Price'] - results_df['Predicted Price']))*100)/results_df['Original Price']


# Display the results
st.subheader("Predicted vs Original Prices and Difference: ")
st.write(results_df)

# # Plot original vs predicted prices
# st.subheader("Original Price VS Predicted Price: ")
# fig5 = plt.figure(figsize=(8,6))
# plt.plot(results_df['Date'], results_df['Predicted Price'], 'r', label="Predicted Price")
# plt.plot(results_df['Date'], results_df['Original Price'], 'g', label="Original Price")
# plt.xlabel("Date")
# plt.ylabel("Price")
# plt.legend()
# st.pyplot(fig5)


#HERE ON I WILL BE TRYING TO PREDICT THE STOCK PRICES FOR FUTURE DAYS:

# Prepare the last 100 days of data
last_100_days = data[['Close']].tail(100)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_last_100_days = scaler.fit_transform(last_100_days)

number_of_days_to_predict = 100

# Initialize input for predictions
current_input = scaled_last_100_days.reshape(1, 100, 1)  # shape (1, 100, 1) for LSTM input
future_predictions = []

for _ in range(number_of_days_to_predict):

    predict_future = model.predict(current_input)
    
    # Since we only have one feature, reshape it to (1, 1) to match scaler's expectations
    # predict_future_unscaled = scaler.inverse_transform(predict_future.reshape(-1, 1))[0, 0]
    predict_future_unscaled = scaler.inverse_transform(np.concatenate([np.full((1, last_100_days.shape[1] - 1), np.nan), predict_future], axis=1))[:, -1][0]
    
    future_predictions.append(predict_future_unscaled)
    
    # Update input data with the new prediction
    # Remove the oldest timestep and append the new prediction
    new_input = np.concatenate([current_input[:, 1:, :], predict_future.reshape(1, 1, 1)], axis=1)
    current_input = new_input  # Shape remains (1, 100, 1)


future_dates = [data.index[-1] + timedelta(days=i) for i in range(1, number_of_days_to_predict + 1)]
future_predictions_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted Price': future_predictions
})

st.subheader("Future Predictions: ")
st.write(future_predictions_df)

#######################################################################################################################################

'''
From here on , we will try to build a model that learns from the errors made in predicting future prices
and then predict the future dates more accurately.
'''
# Convert the string to a datetime object
start_date_future = datetime.strptime(end_date , '%Y-%m-%d')

# Define the number of days to add
days_to_add = 100

# Add the days to the start date
end_date_future = start_date_future + timedelta(days=days_to_add)

# Convert the result back to a string
end_date_future = end_date_future.strftime('%Y-%m-%d')

# print("Start Date:", start_date_future)
# print("End Date:", end_date_future)


data_with_actual_future_prices = yf.download(stock , start_date_future , end_date_future)

#Selecting only the Close column in this future dataset:
data_with_actual_future_prices = data_with_actual_future_prices[['Close']]

future_predictions_df = future_predictions_df.merge(data_with_actual_future_prices , on = 'Date' , suffixes =('' , '_Actual'))

st.subheader("Future dates Predicted and Actual Prices: ")
st.write(future_predictions_df)

future_predictions_df['Prediction Error'] = future_predictions_df['Close'] - future_predictions_df['Predicted Price']
st.write(future_predictions_df)

#Creating sequences for the model:
def create_sequences(data, time_steps):
    x, y = [], []
    for i in range(len(data) - time_steps):
        seq = data[i:i+time_steps]
        x.append(seq)
        y.append(data[i + time_steps])
    return np.array(x), np.array(y)


#Normalizing the data:
scaler = MinMaxScaler(feature_range=(0,1))
future_predictions_df['Prediction Error'] = scaler.fit_transform(future_predictions_df[['Prediction Error']])

st.subheader("Future prediction dataset with normalized data:")
st.write(future_predictions_df)

time_steps = 100
x_future_seq , y_future_seq = create_sequences(future_predictions_df['Prediction Error'].values, time_steps)

# scaler = MinMaxScaler()
# x_future_seq = scaler.fit_transform(x_future_seq)
# y_future_seq = scaler.transform(y_future_seq.reshape(-1, 1))


#Splitting the data into training and testing for the model:
x_future_train, x_future_test, y_future_train, y_future_test = train_test_split(x_future_seq , y_future_seq , test_size=0.2, shuffle=False)

#Redefining the LSTM Model:
model_future = Sequential()

model_future.add(LSTM(units = 50 , activation = 'relu' , return_sequences = True, input_shape = (x.shape[1] , 1)))
model_future.add(Dropout(0.2))

model_future.add(LSTM(units = 60 , activation = 'relu' , return_sequences = True))
model_future.add(Dropout(0.3))

model_future.add(LSTM(units = 80 , activation = 'relu' , return_sequences = True))
model_future.add(Dropout(0.3))

model_future.add(LSTM(units = 120 , activation = 'relu'))
model_future.add(Dropout(0.5))

model_future.add(Dense(units = 1))

model_future.compile(optimizer='adam', loss='mean_squared_error')

model_future.summary()


#Now, training the model:
history = model_future.fit(x_future_train, y_future_train, epochs=20, batch_size=1, validation_data=(x_future_test, y_future_test), verbose=2)


#Predicting:
predicted_errors = model_future.predict(x_future_test)

#Inverse transforming:
predicted_errors = scaler.inverse_transform(predicted_errors)
y_future_test = scaler.inverse_transform(y_future_test)
