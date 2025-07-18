# Cryptocurrency Algorithmic Trading Bot

#### A Cryptocurrency algorithmic trading model that uses Machine Learning Ensemble (LSTM + XGBoost + Transformer) to predict the cryptocurrency market accurately, Reinforcement Learning to manage risk dynamically in real-time, Advanced NLP sentiment analytics to leverage Twitter & Reddit, ChatGPT to provide strategic trading decisions, Advanced position sizing & risk management to ensure profits are maximized with medium risk. All of the model analytics/P&L results are shown on a real-time Streamlit dashboard. 


## Machine Learning Ensemble:
####  •	LSTM for short-term price trends.
####	•	XGBoost for feature importance and market insights.
####	•	Transformer (Time-series Transformer) for long-term pattern capturing.

 
## Reinforcement Learning for Dynamic Risk Management:
####  • Integrate PPO Reinforcement Learning (with Ray RLlib)
#### _Why?_ _The model can dynamically adjust position sizing & stop-loss levels based on evolving market conditions._


## Sentiment Analysis & NLP Integration:
####  •	NLP Strategy: Weighted sentiment analysis to capture deeper market emotion. (60% Reddit sentiment + 40% Twitter sentiment)


## ChatGPT Integration - (Model: GPT-4.0):
####  •	The algorithmic trading model offers hedge-fund level strategic advice based on comprehensive market data.
####  •	Smart ATR-based stop-loss and dynamic position sizing.


## Streamlit Dashboard with Real-Time Monitoring & Analytics:
####  •	Analyze automated bot’s trading performance and strategy.

## Setup
To download data from Binance, `data_utils.fetch_crypto_data` requires the environment variables `BINANCE_API_KEY` and `BINANCE_SECRET`.
Export them in your shell or provide them via a `.env` file.

```bash
export BINANCE_API_KEY=your_api_key
export BINANCE_SECRET=your_api_secret
```

`.env` example:

```
BINANCE_API_KEY=your_api_key
BINANCE_SECRET=your_api_secret
```
