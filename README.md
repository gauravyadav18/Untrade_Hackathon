# Algorithmic Trading Strategies

## Overview

Welcome to the repository for **Algorithmic Trading Strategies**, developed as part of the Untrade Hackathon in collaboration with the Quant Club at BITS Pilani. This project explores the construction and evaluation of automated trading strategies on BTC/USDT and ETH/USDT pairs using a blend of technical analysis and machine learning techniques.

We implemented logic-based strategies enriched by indicators like RSI, MACD, ADX, ATR, and OBV, and integrated them with Random Forest classifiers for signal prediction and trade execution.

## Team

- Gaurav Yadav  
- Bhavik Ostwal  
- Yadnyit Panchabhai  
- Piyush Roy  

## Contents

- Strategy 1: BTC/USDT
- Strategy 2: ETH/USDT
- Technical Indicators Overview
- Machine Learning Integration (Random Forest)
- Backtesting Results and Quarterly Performance

---

## Strategy 1: BTC/USDT

This strategy combines several indicators and a Random Forest model for trade signal generation:

**Indicators Used:**
- ATR (Average True Range)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- ADX (Average Directional Index)

**Machine Learning:**
- Random Forest Classifier with 150 trees
- Used to predict buy/sell signals based on combined features

**Highlights:**
- Total Trades: 205  
- Sharpe Ratio: 9.96  
- Win Rate: 79.02%  
- Profit: +782.77%  
- 14 out of 16 profitable quarters  

---

## Strategy 2: ETH/USDT

This strategy focuses on EMA-based momentum trading and OBV oscillators:

**Indicators Used:**
- EMA (Exponential Moving Average)
- OBV & OBV Oscillator
- MACD
- RSI
- ADX

**Machine Learning:**
- Random Forest Classifier with decision thresholds based on volatility and trailing stop-loss logic

**Highlights:**
- Total Trades: 241  
- Sharpe Ratio: 10.68  
- Win Rate: 79.25%  
- Profit: +649.75%  
- Profitable in all 16 quarters tested  

---

## Backtesting Summary

| Metric                  | BTC/USDT        | ETH/USDT        |
|------------------------|-----------------|-----------------|
| Sharpe Ratio           | 9.96            | 10.68           |
| Win Rate (%)           | 79.02%          | 79.25%          |
| Total Profit (%)       | 782.77%         | 649.75%         |
| Total Trades           | 205             | 241             |
| Profitable Quarters    | 14 / 16         | 16 / 16         |

---

## Requirements

- Python 3.7+
- pandas, numpy, sklearn, matplotlib
- Any OHLCV dataset with minute/hourly/daily crypto prices

## Getting Started

Clone the repository and install the required packages:

```bash
git clone https://github.com/yourusername/algorithmic-trading-strategies.git
cd algorithmic-trading-strategies
pip install -r requirements.txt
