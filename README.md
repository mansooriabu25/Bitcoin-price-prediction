🪙 Bitcoin Price Prediction Using Deep Learning
This project aims to predict Bitcoin prices using historical data and a deep learning model built with Keras. It features an interactive web app powered by Streamlit that visualizes actual vs. predicted prices and forecasts the next few days of Bitcoin prices.

📁 Project Structure
Bitcoin_ML_Model.ipynb: Jupyter notebook used for model training and experimentation.
Bitcoin_Price_prediction_Model.keras: Trained Keras model saved in HDF5 format.
app.py: Streamlit web app for visualizing predictions and interacting with the model.
README.md: Project overview and usage instructions.
📊 Model Overview
The model is trained on Bitcoin's historical closing prices from 2015 to 2023 using a Long Short-Term Memory (LSTM) neural network. It is capable of:

Visualizing historical BTC-USD price trends
Comparing actual vs. predicted prices
Predicting Bitcoin prices for the next 5 days
🚀 How to Run the App
1. Clone the Repository
git clone https://github.com/your-username/bitcoin-price-prediction.git
cd bitcoin-price-prediction
Install Requirements Make sure you have Python 3.x installed. Then, install the required libraries:
pip install -r requirements.txt
If requirements.txt is not available, you can manually install the dependencies:

pip install numpy pandas yfinance keras scikit-learn streamlit
Run the Streamlit App Before running, ensure the path to the Keras model in app.py is correct.
streamlit run app.py
📈 Demo Features Historical data fetched live from Yahoo Finance

Data preprocessing using MinMaxScaler

Predicted vs Actual closing price comparison

Future price prediction chart (next 5 days)

🧠 Technologies Used Python

Keras & TensorFlow

Scikit-learn

Streamlit

yFinance

Pandas & NumPy

📌 Note Ensure the trained model (Bitcoin_Price_prediction_Model.keras) path is correctly set in app.py:

model = load_model('Bitcoin_Price_prediction_Model.keras')  # Adjust path as needed
