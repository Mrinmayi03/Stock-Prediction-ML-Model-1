# Stock Market Prediction Using LSTM - End-to-End Project

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-brightgreen.svg)

## 📈 Project Overview

This project is a practical implementation of deep learning models (LSTM) to predict stock prices, visualize trends, and build intuition about sequential modeling for financial time series data. It integrates **Streamlit** to make the predictions interactive and deployable as a simple app.

It showcases real-world problem solving, E2E ML pipelines, time series preprocessing, model evaluation, and visualization — ideal for demonstrating core Data Science and SWE skills.

---

## 🛠️ Tech Stack

| Area            | Technology   |
|-----------------|--------------|
| Programming     | Python 3.10+  |
| Data Handling   | Pandas, NumPy |
| Visualization   | Matplotlib, Streamlit |
| Machine Learning| TensorFlow, Keras (LSTM) |
| Data Source     | Yahoo Finance via yfinance API |
| Scaling         | MinMaxScaler (sklearn) |

---

## 🚀 Key Features

✅ Download and process historical stock market data via Yahoo Finance  
✅ Visualize trends: raw prices, 50/100/200 day moving averages  
✅ Build and train LSTM models for sequential time series forecasting  
✅ Predict both historical unseen data and future stock prices  
✅ Compare predicted vs. actual prices, calculate error metrics  
✅ Explore future improvements by modeling prediction errors  
✅ Streamlit interface for user-friendly interactivity  
✅ Clear, modular Python implementation for readability  

---

## 📂 Directory Structure

├── stock_prediction_ml_model.py # Main application (Streamlit)
├── Stock Predictions ML Model.keras # Pre-trained LSTM model
├── stock_prediction_ml_model.ipynb # Development notebook
├── README.md # This file

yaml
Copy
Edit

---

## 💻 How to Run Locally

1️⃣ Clone the repo:

```bash
git clone https://github.com/yourusername/stock-market-prediction-lstm.git
cd stock-market-prediction-lstm
```
2️⃣ Create and activate your virtual environment:

bash
```
python -m venv venv
source venv/bin/activate        # Linux/Mac
.\venv\Scripts\activate         # Windows
```
3️⃣ Install dependencies:

bash
```
pip install -r requirements.txt
```
requirements.txt (create this):
```
nginx
Copy
Edit
pandas
numpy
matplotlib
streamlit
yfinance
scikit-learn
tensorflow
keras
```
4️⃣ Run the Streamlit app:

bash
```streamlit run stock_prediction_ml_model.py```
