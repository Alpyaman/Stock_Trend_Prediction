# Stock Trend Prediction 📈

A Python-based project that predicts stock trends using historical data and multiple technical indicators, leveraging statistical models, machine learning, and backtesting.

---

## 📌 Overview

This project investigates stock trend prediction by:

1. Collecting and preprocessing historical stock data.
2. Calculating common technical indicators.
3. Training statistical and machine learning models.
4. Backtesting strategies on historical records.
5. Comparing indicators and models for trade decision accuracy.

---

## ⭐ Features

- Clean, modularized preprocessing pipelines (e.g. MongoDB + Python).
- Computation of SMA, EMA, MACD, RSI, Bollinger Bands, ROC, Williams %R.
- Multiple models: statistical rules, ensemble ML, RNN/LSTM.
- Backtesting framework with performance metrics and visualizations.
- Risk analysis using daily returns and Monte Carlo simulations.

---

## 🛠️ Technologies

- **Language**: Python 3.8+  
- **Libraries**: Pandas, NumPy, Scikit-learn, TensorFlow/Keras (optional), Matplotlib/Seaborn  
- **Database**: MongoDB or equivalent  
- **Visualization**: Matplotlib, Seaborn, Tableau (optional)  
- **Jupyter Notebooks**: For exploratory data analysis  
- **(Optional)** Streamlit app for live predictions

---

## 📁 Repository Structure

📦Stock_Trend_Prediction
 ┣ 📂data/               # raw and cleaned datasets
 ┣ 📂notebooks/          # EDA & indicator analyses
 ┣ 📂models/             # trained models & checkpoints
 ┣ 📂backtest/           # backtesting scripts and metrics
 ┣ 📂utils/              # helper modules (data loaders, indicators)
 ┣ 📜requirements.txt
 ┣ 📜README.md
 ┗ 📜app.py              # (Optional) live demo or streaming app

---

## ⚙️ Installation

1. **Clone the repo**
```bash
   git clone https://github.com/Alpyaman/Stock_Trend_Prediction.git
   cd Stock_Trend_Prediction
```
3. **Create a virtual environment**
```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate      # Windows
```
5. **Install dependencies**
```bash
   pip install -r requirements.txt
```
7. **(Optional) Set up database**  
   - If using MongoDB:
   ```
     mongod --dbpath /path/to/your/db
   ```
---

## 🚀 Usage

- **Prepare your dataset** by placing CSV/JSON data in `data/`.
- **Run preprocessing**:
  `python utils/preprocess.py`
- **Generate indicators**:
  `python utils/indicators.py`
- **Train models**:
  `python backtest/train_model.py --ticker AAPL`
- **Backtest strategies**:
  `python backtest/run_backtest.py --model MACD --ticker MSFT`
- **(Optional) Launch app**:
  `streamlit run app.py`

---

## 🧾 Data Sources

Uses public data such as:

- IEX Cloud API (last 5 years)
- Tiingo (end-of-day historical data)
- Google Finance, Yahoo Finance, QuantQuote

(Current datasets included in `data/`.)

---

## 📊 Technical Indicators

- **SMA / EMA**: Simple & exponential moving averages  
- **MACD**: Trend reversal indicator  
- **RSI**: Momentum-based buying/selling signals  
- **Bollinger Bands**: Volatility envelopes  
- **ROC**: Rate-of-change to assess trend strength  
- **Williams %R**: Overbought/oversold signal  

---

## 🤖 Modeling & Backtesting

- **Rule-based strategies** on technical thresholds
- **Machine Learning** using ML models (Random Forest, XGBoost)
- **Deep Learning** with RNN/LSTM (optional)
- **Backtesting evaluation**: accuracy, Sharpe ratio, drawdown, etc.
- **Risk analysis**: Monte Carlo simulation, VaR estimates

---

## 📈 Results

- Assessment of single vs. combined indicators (e.g. EMA+MACD)
- Performance comparisons across models and tickers
- Risk-return breakdown charts and summaries

---

## ⚠️ Limitations & Future Work

- **Limited stock universe**: currently focuses on 4–8 tickers  
- **Single-market bias**: primarily US stocks  
- **Real-time deployment**: future work includes live feeds  
- Plan to integrate advanced models e.g. attention-based DL

---

## 🤝 Contributing

Contributions welcome! To contribute:

1. Create an issue or feature request  
2. Fork the repo  
3. Commit your changes with clear messages  
4. Submit a pull request

Please follow standard style guides and include doc updates.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.

---

## 📬 Contact

- **Author**: Alpyaman  
- **Email**: alpyaman3@gmail.com  
- **GitHub**: https://github.com/Alpyaman
