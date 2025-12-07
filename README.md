# 🪙 Bitcoin Price Prediction Using Deep Learning

This project predicts Bitcoin prices using historical market data and a deep learning model built with **Keras**. It includes an interactive **Streamlit web application** that visualizes actual vs. predicted prices and forecasts the next few days of Bitcoin price movements.

---

## 📁 Project Structure

- **Bitcoin_ML_Model.ipynb** – Jupyter notebook for model training and experimentation  
- **Bitcoin_Price_prediction_Model.keras** – Trained LSTM model saved in Keras format  
- **app.py** – Streamlit web app for visualization and prediction  
- **README.md** – Project overview and usage guide  

---

## 📊 Model Overview

The model is trained on **Bitcoin’s historical closing prices (2015–2023)** using an **LSTM (Long Short-Term Memory)** neural network.  
It supports:

- Visualizing historical BTC-USD trends  
- Comparing **actual vs. predicted** closing prices  
- Predicting Bitcoin prices for the **next 5 days**  

---

## 🚀 How to Run the App

### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/bitcoin-price-prediction.git
cd bitcoin-price-prediction

2. Install Requirements

Ensure Python 3.x is installed.

If requirements.txt is available:

pip install -r requirements.txt

3. Run the Streamlit App

Ensure the model path inside app.py is correct.

streamlit run app.py

📈 Demo Features

Live historical BTC-USD data from Yahoo Finance

Preprocessing with MinMaxScaler

Actual vs. Predicted closing price comparison

Future price prediction (next 5 days)

Interactive charts and visualization

🧠 Technologies Used

Python

Keras & TensorFlow

Scikit-learn

Streamlit

yFinance

Pandas & NumPy
