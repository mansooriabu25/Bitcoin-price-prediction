import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# -----------------------------------------
# 🚀 Load Trained Model (.keras file)
# NOTE: Replace this path if needed
model = load_model('Bitcoin_Price_prediction_Model.keras')
# -----------------------------------------

# 🎯 App Title
st.header('📈 Bitcoin Price Prediction Model')
st.subheader('Bitcoin Price Data (2015 - 2023)')

# 📥 Download BTC-USD data
data = yf.download('BTC-USD', '2015-01-01', '2023-11-30')
data = data.reset_index()
st.write(data)

# 🧹 Clean Data and Keep Only 'Close'
st.subheader('Bitcoin Line Chart (Close Price Only)')
data_close = data[['Close']]
st.line_chart(data_close)

# 📊 Train-Test Split
train_data = data_close[:-100]
test_data = data_close[-200:]

# 🔄 Scale Data
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

# 🧠 Prepare Data for Prediction
base_days = 100
x = []
y = []
for i in range(base_days, test_scaled.shape[0]):
    x.append(test_scaled[i - base_days:i])
    y.append(test_scaled[i, 0])

x = np.array(x)
y = np.array(y)
x = np.reshape(x, (x.shape[0], x.shape[1], 1))

# 📈 Make Predictions
pred = model.predict(x)
pred = scaler.inverse_transform(pred)
actual = scaler.inverse_transform(y.reshape(-1, 1))

# 📊 Display Predictions vs Actual
st.subheader('📊 Predicted vs Original Prices')
pred_df = pd.DataFrame({
    'Predicted Price': pred.flatten(),
    'Original Price': actual.flatten()
})
st.write(pred_df)
st.line_chart(pred_df)

# 🔮 Predict Future Days
m = y.copy()
z = []
future_days = 5
for _ in range(future_days):
    last_sequence = m[-base_days:].reshape(1, base_days, 1)
    pred = model.predict(last_sequence)
    m = np.append(m, pred)
    z.append(pred[0][0])

# 📈 Show Future Predictions
z = np.array(z).reshape(-1, 1)
z = scaler.inverse_transform(z)
st.subheader('🕒 Predicted Future Bitcoin Prices (Next 5 Days)')
st.line_chart(z)
