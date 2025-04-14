# The strategy is designed to be run on the Vector.Untrade.io platform.
# It aims to achieve a 1.5–2× market return with moderate risk controls.
# The strategy uses machine learning to generate signals and includes features like ATR, EMA, RSI, MACD, and OBV.           
# The code is structured to handle exceptions and log errors for better debugging.
# The strategy includes entry and exit logic with stop-loss, take-profit, and trailing stop mechanisms.


import pandas as pd
import numpy as np
from enum import Enum
from ta.volume import OnBalanceVolumeIndicator
from ta.trend import EMAIndicator
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TradeType(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    REVERSE_LONG = "REVERSE_LONG"
    REVERSE_SHORT = "REVERSE_SHORT"
    CLOSE = "CLOSE"
    HOLD = "HOLD"

class Strategy:
    def run(self, data: pd.DataFrame, initial_balance=10000) -> pd.DataFrame:
        """
        Runs the ML-based trading strategy on ETH data to achieve 1.5–2× market return.
        Returns the processed DataFrame for Vector.Untrade.io.
        """
        try:
            # Validate input data
            required_cols = {'datetime', 'open', 'high', 'low', 'close', 'volume'}
            if data.empty or not required_cols.issubset(data.columns):
                logging.error("Input data is empty or missing required columns")
                return data.assign(trade_type=TradeType.HOLD.value, leverage=1, position=50)

            # Prepare data
            df = data.copy()
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)

            # Calculate features
            df = self.process_data(df)

            # Generate ML signals
            df = self.generate_ml_signals(df)

            # Apply trading logic
            df = self.apply_strategy(df)

            # Calculate leverage and position size
            df['atr'] = self.calculate_atr(df)
            df['leverage'] = 1  # Fixed at 1x
            df['position'] = (0.000035 * initial_balance / df['atr']).clip(50, 100)  # Adjusted for target profit

            # Reset index and return selected columns
            df.reset_index(inplace=True)
            return df[['datetime', 'open', 'high', 'low', 'close', 'volume', 'trade_type', 'leverage', 'position']]

        except Exception as e:
            logging.error(f"Strategy execution failed: {str(e)}")
            df_out = data.copy()
            df_out['trade_type'] = TradeType.HOLD.value
            df_out['leverage'] = 1
            df_out['position'] = 50
            return df_out[['datetime', 'open', 'high', 'low', 'close', 'volume', 'trade_type', 'leverage', 'position']]

    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        # Handle "volume from" if present
        if 'volume from' in data.columns:
            data.rename(columns={'volume from': 'volume'}, inplace=True)

        # Calculate technical indicators
        data['daily_returns'] = data['close'].pct_change()
        data['atr'] = ta.volatility.average_true_range(data['high'], data['low'], data['close'], window=14)
        data['ema_21'] = ta.trend.ema_indicator(data['close'], window=21)
        data['rsi'] = ta.momentum.rsi(data['close'], window=14)
        data['macd'] = ta.trend.macd_diff(data['close'])
        data['adx'] = ta.trend.adx(data['high'], data['low'], data['close'], window=14)
        data['volume_ratio'] = data['volume'] / data['volume'].rolling(window=20).mean()
        data['volatility'] = data['daily_returns'].rolling(20).std()
        data['sharpe'] = data['daily_returns'].rolling(60).mean() / data['volatility']

        # OBV and OBV Oscillator
        obv_indicator = OnBalanceVolumeIndicator(close=data['close'], volume=data['volume'])
        data['obv'] = obv_indicator.on_balance_volume()
        short_ema_length = 20
        ema_obv = EMAIndicator(close=data['obv'], window=short_ema_length)
        data['obv_ema'] = ema_obv.ema_indicator()
        data['obv_osc'] = data['obv'] - data['obv_ema']

        # Fill NaN values
        data.fillna(0, inplace=True)
        return data

    def calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        # Calculate Average True Range (ATR)
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        return tr.rolling(window=14).mean()

    def generate_ml_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        # Define target for ML model with moderate threshold
        df['future_return'] = df['close'].pct_change().shift(-1)
        df['target'] = np.where((df['future_return'] > 0.003) & (df['rsi'] > 50), 1,  # Adjusted threshold
                                np.where((df['future_return'] < -0.003) & (df['rsi'] < 50), -1, 0))  # Adjusted threshold

        # Feature columns for ML model
        feature_columns = ['ema_21', 'rsi', 'macd', 'volume_ratio', 'adx', 'daily_returns', 'sharpe', 'volatility', 'obv_osc']
        df.dropna(inplace=True)

        # Prepare features and target
        features = df[feature_columns]
        target = df['target']
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        # Train RandomForestClassifier
        model = RandomForestClassifier(n_estimators=150, max_depth=5, min_samples_split=5, min_samples_leaf=2, random_state=42)
        model.fit(X_train, y_train)

        # Generate ML signals
        df['ml_signal'] = model.predict(features)
        return df

    def apply_strategy(self, data: pd.DataFrame) -> pd.DataFrame:
        data['trade_type'] = TradeType.HOLD.value
        position_open = False
        position_type = None
        entry_price = None
        sl = None
        tp = None
        trailing_sl = None
        entry_date = None

        for i in range(1, len(data)):
            current_signal = data['ml_signal'].iloc[i]
            current_price = data['close'].iloc[i]
            current_date = data.index[i]

            # Check stop-loss, take-profit, trailing stop, or time-based exit if position is open
            if position_open:
                days_held = (current_date - entry_date).days
                if days_held >= 5:  # Close position after 5 days
                    data.at[data.index[i], 'trade_type'] = TradeType.CLOSE.value
                    position_open = False
                    position_type = None
                    entry_price = None
                    sl = None
                    tp = None
                    trailing_sl = None
                    entry_date = None
                    continue

                if position_type == TradeType.LONG.value:
                    trailing_sl = max(trailing_sl, current_price * (1 - 0.002))
                    if current_price <= sl or current_price >= tp or current_price <= trailing_sl:
                        data.at[data.index[i], 'trade_type'] = TradeType.CLOSE.value
                        position_open = False
                        position_type = None
                        entry_price = None
                        sl = None
                        tp = None
                        trailing_sl = None
                        entry_date = None
                        continue
                elif position_type == TradeType.SHORT.value:
                    trailing_sl = min(trailing_sl, current_price * (1 + 0.002))
                    if current_price >= sl or current_price <= tp or current_price >= trailing_sl:
                        data.at[data.index[i], 'trade_type'] = TradeType.CLOSE.value
                        position_open = False
                        position_type = None
                        entry_price = None
                        sl = None
                        tp = None
                        trailing_sl = None
                        entry_date = None
                        continue

            # Entry logic with moderate risk controls
            if current_signal == 1 and not position_open:
                data.at[data.index[i], 'trade_type'] = TradeType.LONG.value
                position_open = True
                position_type = TradeType.LONG.value
                entry_price = current_price
                sl = entry_price * (1 - 0.005)  # Stop-loss: 0.5%
                tp = entry_price * (1 + 0.002)  # Take-profit: 0.2%
                trailing_sl = entry_price * (1 - 0.002)
                entry_date = current_date
            elif current_signal == -1 and not position_open:
                data.at[data.index[i], 'trade_type'] = TradeType.SHORT.value
                position_open = True
                position_type = TradeType.SHORT.value
                entry_price = current_price
                sl = entry_price * (1 + 0.005)  # Stop-loss: 0.5%
                tp = entry_price * (1 - 0.002)  # Take-profit: 0.2%
                trailing_sl = entry_price * (1 + 0.002)
                entry_date = current_date
            elif position_open and current_signal == 0:
                data.at[data.index[i], 'trade_type'] = TradeType.CLOSE.value
                position_open = False
                position_type = None
                entry_price = None
                sl = None
                tp = None
                trailing_sl = None
                entry_date = None

        return data

if __name__ == "__main__":
    # Simulate ETH-like data for 2020-2023
    dates = pd.date_range(start="2020-01-01", freq="D")
    np.random.seed(42)
    
    # Simulate ETH price with trends and volatility
    base_price = 2000
    trend = np.linspace(0, 5000, len(dates))  # Upward trend over time
    volatility = np.random.normal(0, 50, len(dates)).cumsum()  # Increased volatility
    close_prices = base_price + trend + volatility
    data = pd.DataFrame({
        "datetime": dates,
        "open": close_prices + np.random.uniform(-10, 10, len(dates)),
        "high": close_prices + np.random.uniform(0, 30, len(dates)),
        "low": close_prices - np.random.uniform(0, 30, len(dates)),
        "close": close_prices,
        "volume from": np.random.uniform(500, 5000, len(dates))  # Volume in ETH units
    })
    
    # Run the strategy
    strategy = Strategy()
    result = strategy.run(data)
    print(result.tail())