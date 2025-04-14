# Backtesting has been done on Vector.Untrade.io on 1 day dataframe.
# This script implements a trading strategy using machine learning and technical indicators.
# It uses a Random Forest Classifier to predict trade signals based on historical price data and technical indicators.
# The strategy includes features such as ATR, EMA, RSI, MACD, ADX, volume ratio, and volatility.
# The script also handles exceptions and logs errors during execution.
import pandas as pd  
import numpy as np  
import ta 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split, GridSearchCV 
from sklearn.metrics import accuracy_score 
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TradeType:
    LONG = "LONG"
    SHORT = "SHORT"
    CLOSE = "CLOSE"
    HOLD = "HOLD"

class Strategy:
    def run(self, data: pd.DataFrame, initial_balance=10000) -> pd.DataFrame:
        try:
            
            required_cols = {'datetime', 'open', 'high', 'low', 'close', 'volume'}
            if data.empty or not required_cols.issubset(data.columns):
                logging.error("Input data is empty or missing required columns")
                return data.assign(trade_type=TradeType.HOLD, leverage=1, position=50)

            df = data.copy()
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
            
            # Compute technical indicators
            df['daily_returns'] = df['close'].pct_change()
            df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
            df['ema_21'] = ta.trend.ema_indicator(df['close'], window=21)
            df['rsi'] = ta.momentum.rsi(df['close'], window=14)
            df['macd'] = ta.trend.macd_diff(df['close'])
            df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
            df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
            df['volatility'] = df['daily_returns'].rolling(20).std()
            df['sharpe'] = df['daily_returns'].rolling(60).mean() / df['volatility']
            df.fillna(0, inplace=True)
            
            # Define target variable based on future price movement and RSI threshold
            df['future_return'] = df['close'].pct_change().shift(-1)
            df['target'] = np.where((df['future_return'] > 0.0025) & (df['rsi'] > 50), 1,
                                    np.where((df['future_return'] < -0.0002) & (df['rsi'] < 50), -1, 0))
            
            
            feature_columns = ['ema_21', 'rsi', 'macd', 'volume_ratio', 'adx', 'daily_returns', 'sharpe', 'volatility']
            df.dropna(inplace=True)
            
            features = df[feature_columns]
            target = df['target']
            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

            model = RandomForestClassifier(n_estimators=150, max_depth=5, min_samples_split=4, min_samples_leaf=2, random_state=42)
            model.fit(X_train, y_train)
            df['ml_signal'] = model.predict(features)
            
            df['trade_type'] = TradeType.HOLD
            position_open = False
            position_type = None
            last_trade_close = -1

            for i in range(1, len(df)):
                prev_trade = df['trade_type'].iloc[i-1]
                current_signal = df['ml_signal'].iloc[i]
                
                if prev_trade == TradeType.CLOSE:
                    continue
                
                if current_signal == 1 and not position_open:
                    df.at[df.index[i], 'trade_type'] = TradeType.LONG
                    position_open = True
                    position_type = TradeType.LONG
                elif current_signal == -1 and not position_open:
                    df.at[df.index[i], 'trade_type'] = TradeType.SHORT
                    position_open = True
                    position_type = TradeType.SHORT
                elif position_open:
                    df.at[df.index[i], 'trade_type'] = TradeType.CLOSE
                    position_open = False
                    position_type = None
                    last_trade_close = i

            # Assign leverage based on ATR quantile
            df['leverage'] = np.where(df['atr'] > df['atr'].rolling(20).quantile(0.75), 1, 2)
            
            # Calculate position sizing based on risk management
            df['position'] = (0.003 * initial_balance / df['atr']).clip(50, 100)
            df.reset_index(inplace=True)
            return df[['datetime', 'open', 'high', 'low', 'close', 'volume', 'trade_type', 'leverage', 'position']]

        except Exception as e:
            logging.error(f"Strategy execution failed: {str(e)}")
            df_out = data.copy()
            df_out['trade_type'] = TradeType.HOLD
            df_out['leverage'] = 1
            df_out['position'] = 50
            return df_out[['datetime', 'open', 'high', 'low', 'close', 'volume', 'trade_type', 'leverage', 'position']]

if __name__ == "__main__":
    dates = pd.date_range(start="2020-01-01", periods=200, freq="D")
    np.random.seed(42)
    close_prices = np.random.normal(100, 10, 200).cumsum()
    data = pd.DataFrame({
        "datetime": dates,
        "open": close_prices + np.random.uniform(-2, 2, 200),
        "high": close_prices + np.random.uniform(0, 5, 200),
        "low": close_prices - np.random.uniform(0, 5, 200),
        "close": close_prices,
        "volume": np.random.uniform(1000, 5000, 200)
    })
    
    strategy = Strategy()
    result = strategy.run(data)
    print(result)

# Output of each Strategy is shown in report.