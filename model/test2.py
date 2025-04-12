import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import RobustScaler
from typing import Tuple, List, Dict, Any
import asyncio
from datetime import datetime, timedelta
import warnings
import joblib
import os

# Suppress TensorFlow info messages and warnings
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore')

class Config:
    # File Paths
    MODEL_DIR = "trained_models"
    ARTIFACTS_VERSION = "v1"
    
    @property
    def model_path(self):
        return os.path.join(self.MODEL_DIR, f"{self.ARTIFACTS_VERSION}_model.keras")
    
    @property
    def scaler_path(self):
        return os.path.join(self.MODEL_DIR, f"{self.ARTIFACTS_VERSION}_scaler.joblib")
    
    @property
    def metadata_path(self):
        return os.path.join(self.MODEL_DIR, f"{self.ARTIFACTS_VERSION}_metadata.pkl")
    
    @property
    def trades_csv_path(self):
        return os.path.join(self.MODEL_DIR, f"{self.ARTIFACTS_VERSION}_trades.csv")

    # Data parameters
    API_KEY = "yVlies2qSvg0Z4XrUkSwdP8VHVEjdZabnvBrzXVMkYRE2wUX"
    DATA_LIMIT = 10000
    WINDOW = "hour"
    EXCHANGE = "binance"
    DATA_SOURCES = {
        'ohlcv': f'cryptoquant|btc/market-data/price-ohlcv?window={WINDOW}',
        'netflow': f'cryptoquant|btc/exchange-flows/netflow?exchange={EXCHANGE}&window={WINDOW}',
        'whale_ratio': f'cryptoquant|btc/flow-indicator/exchange-whale-ratio?exchange={EXCHANGE}&window={WINDOW}',
        'miner_outflow': f'cryptoquant|btc/miner-flows/outflow?miner=f2pool&window={WINDOW}'
    }

    # Model parameters
    LOOKBACK_WINDOW = 72  # 3 days of hourly data
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.1
    NN_EPOCHS = 30
    NN_BATCH_SIZE = 128
    NN_LEARNING_RATE = 0.0005
    EARLY_STOPPING_PATIENCE = 25
    REDUCE_LR_PATIENCE = 12
    REDUCE_LR_FACTOR = 0.5
    DROPOUT_RATE = 0.4
    L2_REGULARIZATION = 0.001
    SCALER_TYPE = RobustScaler

    # Trading parameters
    INITIAL_CAPITAL = 10000.0
    TRADING_FEE = 0.0006
    POSITION_SIZE_PCT = 0.15
    STOP_LOSS_PCT = 0.015
    TAKE_PROFIT_PCT = 0.03
    MAX_OPEN_POSITIONS = 3
    MIN_TRADE_AMOUNT = 10.0

    # Signal generation
    ENTRY_THRESHOLD = 0.005
    EXIT_THRESHOLD_LOW = -0.002
    EXIT_THRESHOLD_HIGH = 0.002
    MIN_VOLATILITY_PCT = 25
    VOLATILITY_WINDOW = 24

    # Success criteria
    REQUIRED_SHARPE_RATIO = 1.8
    MAX_ALLOWED_DRAWDOWN = 0.35
    MIN_TRADE_FREQUENCY = 0.04
    MIN_PROFIT_FACTOR = 1.5

config = Config()
os.makedirs(config.MODEL_DIR, exist_ok=True)

class DataProcessor:
    @staticmethod
    async def fetch_data(source_name: str, topic: str, limit: int = config.DATA_LIMIT) -> Tuple[str, pd.DataFrame]:
        """Fetch data with error handling"""
        try:
            from cybotrade_datasource import query_paginated
            data = await query_paginated(api_key=config.API_KEY, topic=topic, limit=limit)
            df = pd.DataFrame(data)
            
            if 'datetime' not in df.columns:
                raise ValueError("Missing datetime column")
                
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime').drop_duplicates('datetime').set_index('datetime')
            df.columns = [col.replace('exchange_', '').replace('_total', '') for col in df.columns]
            
            if source_name == 'miner_outflow' and 'outflow' in df.columns:
                df.rename(columns={'outflow': 'miner_outflow'}, inplace=True)
                
            return source_name, df
            
        except Exception as e:
            print(f"Error fetching {source_name}: {str(e)}")
            return source_name, pd.DataFrame()

    @staticmethod
    async def merge_data_sources() -> pd.DataFrame:
        """Fetch and merge all data sources"""
        tasks = [DataProcessor.fetch_data(name, topic) 
                for name, topic in config.DATA_SOURCES.items()]
        results = await asyncio.gather(*tasks)
        valid_dfs = {name: df for name, df in results if not df.empty}
        
        if not valid_dfs or 'ohlcv' not in valid_dfs:
            return pd.DataFrame()
            
        merged_df = valid_dfs.pop('ohlcv').copy()
        
        for name, df in valid_dfs.items():
            cols_to_rename = {col: f"{name}_{col}" for col in df.columns if col in merged_df.columns}
            temp_df = df.rename(columns=cols_to_rename) if cols_to_rename else df
            merged_df = merged_df.join(temp_df, how='inner')
                
        return merged_df.loc[:, ~merged_df.columns.duplicated(keep='first')]

    @staticmethod
    def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        df = df.copy()
        
        # Price features
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['log_ret'].rolling(config.VOLATILITY_WINDOW).std() * np.sqrt(24)
        
        # Moving averages
        for window in [8, 20, 50, 100]:
            df[f'SMA_{window}'] = df['close'].rolling(window).mean()
            df[f'EMA_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
        
        # Bollinger Bands
        df['BB_MA'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['BB_upper'] = df['BB_MA'] + 2 * bb_std
        df['BB_lower'] = df['BB_MA'] - 2 * bb_std
        
        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Volume features
        if 'volume' in df.columns:
            df['volume_ma'] = df['volume'].rolling(20).mean()
        
        # Lag features
        for col in ['netflow', 'whale_ratio', 'miner_outflow']:
            if col in df.columns:
                for lag in [1, 3, 6, 12, 24]:
                    df[f'{col}_lag{lag}'] = df[col].shift(lag)
        
        # Target variable
        df['target'] = np.log(df['close'].shift(-1) / df['close'])
        
        return df.dropna()

class LSTMModel:
    @staticmethod
    def create_sequences(data: pd.DataFrame, targets: pd.Series, lookback: int) -> Tuple[np.ndarray, np.ndarray, pd.Index]:
        """Create properly aligned time series sequences"""
        X, y, indices = [], [], []
        targets = targets.reindex(data.index)
        
        for i in range(len(data) - lookback):
            target_idx = i + lookback
            if target_idx < len(targets):
                X.append(data.iloc[i:(i + lookback)].values)
                y.append(targets.iloc[target_idx])
                indices.append(targets.index[target_idx])
        
        return np.array(X), np.array(y), pd.Index(indices)

    @staticmethod
    def build(input_shape: tuple) -> Sequential:
        """Build LSTM model"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape,
                kernel_regularizer=l2(config.L2_REGULARIZATION)),
            BatchNormalization(),
            Dropout(config.DROPOUT_RATE),
            
            LSTM(64, return_sequences=True,
                kernel_regularizer=l2(config.L2_REGULARIZATION)),
            BatchNormalization(),
            Dropout(config.DROPOUT_RATE),
            
            LSTM(32, kernel_regularizer=l2(config.L2_REGULARIZATION)),
            BatchNormalization(),
            Dropout(config.DROPOUT_RATE),
            
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        optimizer = Adam(learning_rate=config.NN_LEARNING_RATE)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        return model

    @staticmethod
    def create_callbacks() -> list:
        """Create training callbacks"""
        return [
            EarlyStopping(monitor='val_loss', patience=config.EARLY_STOPPING_PATIENCE, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=config.REDUCE_LR_FACTOR, patience=config.REDUCE_LR_PATIENCE),
            ModelCheckpoint(filepath=config.model_path, save_best_only=True)
        ]

class Backtester:
    def __init__(self):
        self.position_id = 0
        self.trades = []
        self.portfolio_history = []
    
    def run_backtest(self, price_data: pd.DataFrame, signals: pd.Series) -> Dict[str, Any]:
        """Run backtest with proper position management"""
        self.initialize(config.INITIAL_CAPITAL)
        
        volatility = price_data['volatility']
        vol_threshold = volatility.rolling(1000).apply(
            lambda x: np.percentile(x, config.MIN_VOLATILITY_PCT), raw=True
        ).fillna(method='ffill').fillna(0)
        
        for timestamp, row in price_data.iterrows():
            current_price = row['close']
            current_vol = row['volatility']
            signal = signals.loc[timestamp]
            
            self.update_positions(current_price, timestamp, signal)
            
            if signal > config.ENTRY_THRESHOLD and current_vol >= vol_threshold.loc[timestamp]:
                self.enter_position('long', current_price, timestamp)
            elif signal < -config.ENTRY_THRESHOLD and current_vol >= vol_threshold.loc[timestamp]:
                self.enter_position('short', current_price, timestamp)
            
            self.record_state(timestamp, current_price)
        
        results = self.calculate_results()
        self.save_trades_to_csv()
        return results

    def initialize(self, initial_capital: float):
        """Reset backtesting state"""
        self.cash = initial_capital
        self.positions = []
        self.position_id = 0
        self.trades = []
        self.portfolio_history = []

    def update_positions(self, current_price: float, timestamp: datetime, signal: float):
        """Update all positions and check exit conditions"""
        active_positions = []
        
        for pos in self.positions:
            # Calculate position value
            if pos['direction'] == 'long':
                pos_value = pos['size'] * (current_price / pos['entry_price'])
                stop_loss = pos['entry_price'] * (1 - config.STOP_LOSS_PCT)
                take_profit = pos['entry_price'] * (1 + config.TAKE_PROFIT_PCT)
            else:  # short
                pos_value = pos['size'] * (2 - (current_price / pos['entry_price']))
                stop_loss = pos['entry_price'] * (1 + config.STOP_LOSS_PCT)
                take_profit = pos['entry_price'] * (1 - config.TAKE_PROFIT_PCT)
            
            # Check exit conditions
            exit_reason = None
            if pos['direction'] == 'long' and current_price <= stop_loss:
                exit_reason = 'stop_loss'
            elif pos['direction'] == 'long' and current_price >= take_profit:
                exit_reason = 'take_profit'
            elif pos['direction'] == 'short' and current_price >= stop_loss:
                exit_reason = 'stop_loss'
            elif pos['direction'] == 'short' and current_price <= take_profit:
                exit_reason = 'take_profit'
            elif signal == 0:
                exit_reason = 'exit_signal'
            
            if exit_reason:
                pnl_pct = (pos_value / pos['size']) - 1
                self.cash += pos_value * (1 - config.TRADING_FEE)
                self.trades.append({
                    'position_id': pos['id'],
                    'direction': pos['direction'],
                    'entry_time': pos['entry_time'],
                    'entry_price': pos['entry_price'],
                    'exit_time': timestamp,
                    'exit_price': current_price,
                    'size': pos['size'],
                    'pnl_pct': pnl_pct,
                    'pnl_abs': pos_value - pos['size'],
                    'exit_reason': exit_reason,
                    'holding_period': (timestamp - pos['entry_time']).total_seconds() / 3600  # in hours
                })
            else:
                active_positions.append(pos)
        
        self.positions = active_positions

    def enter_position(self, direction: str, entry_price: float, timestamp: datetime):
        """Enter new position with proper sizing"""
        if len(self.positions) >= config.MAX_OPEN_POSITIONS:
            return
            
        position_size = min(self.cash * config.POSITION_SIZE_PCT, self.cash - config.MIN_TRADE_AMOUNT)
        if position_size < config.MIN_TRADE_AMOUNT:
            return
            
        self.position_id += 1
        self.positions.append({
            'id': self.position_id,
            'entry_time': timestamp,
            'entry_price': entry_price,
            'size': position_size,
            'direction': direction
        })
        self.cash -= position_size * (1 + config.TRADING_FEE)
        
        self.trades.append({
            'position_id': self.position_id,
            'direction': direction,
            'entry_time': timestamp,
            'entry_price': entry_price,
            'size': position_size,
            'exit_time': None,
            'exit_price': None,
            'pnl_pct': None,
            'pnl_abs': None,
            'exit_reason': None,
            'holding_period': None
        })

    def record_state(self, timestamp: datetime, current_price: float):
        """Record portfolio state"""
        positions_value = sum(
            pos['size'] * (current_price / pos['entry_price']) if pos['direction'] == 'long'
            else pos['size'] * (2 - (current_price / pos['entry_price']))
            for pos in self.positions
        )
        
        self.portfolio_history.append({
            'timestamp': timestamp,
            'total_value': self.cash + positions_value,
            'cash': self.cash,
            'positions_value': positions_value,
            'num_positions': len(self.positions)
        })

    def calculate_results(self) -> Dict[str, Any]:
        """Calculate performance metrics"""
        portfolio_df = pd.DataFrame(self.portfolio_history).set_index('timestamp')
        trades_df = pd.DataFrame(self.trades)
        
        if portfolio_df.empty:
            return {
                'final_value': config.INITIAL_CAPITAL,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'trades': trades_df,
                'portfolio_history': portfolio_df,
                'trades_csv_path': None
            }
        
        portfolio_df['returns'] = portfolio_df['total_value'].pct_change().fillna(0)
        
        # Sharpe ratio
        sharpe = (portfolio_df['returns'].mean() * np.sqrt(365*24)) / (portfolio_df['returns'].std() + 1e-9)
        
        # Drawdown
        portfolio_df['peak'] = portfolio_df['total_value'].cummax()
        portfolio_df['drawdown'] = (portfolio_df['peak'] - portfolio_df['total_value']) / portfolio_df['peak']
        max_dd = portfolio_df['drawdown'].max()
        
        return {
            'final_value': portfolio_df['total_value'].iloc[-1],
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'trades': trades_df,
            'portfolio_history': portfolio_df,
            'trades_csv_path': config.trades_csv_path
        }

    def save_trades_to_csv(self):
        """Save trades dataframe to CSV file"""
        if not self.trades:
            print("No trades to save")
            return
            
        trades_df = pd.DataFrame(self.trades)
        
        # Filter out entries without exit information
        trades_df = trades_df.dropna(subset=['exit_time'])
        
        if trades_df.empty:
            print("No completed trades to save")
            return
            
        # Calculate cumulative PnL
        trades_df['cumulative_pnl'] = trades_df['pnl_abs'].cumsum()
        
        # Format datetime columns
        for col in ['entry_time', 'exit_time']:
            trades_df[col] = pd.to_datetime(trades_df[col]).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Calculate duration in hours
        trades_df['holding_period'] = trades_df['holding_period'].round(2)
        
        # Reorder columns
        cols = [
            'position_id', 'direction', 'entry_time', 'exit_time', 'holding_period',
            'entry_price', 'exit_price', 'size', 'pnl_pct', 'pnl_abs',
            'cumulative_pnl', 'exit_reason'
        ]
        trades_df = trades_df[cols]
        
        # Save to CSV
        trades_df.to_csv(config.trades_csv_path, index=False)
        print(f"\nTrades saved to: {config.trades_csv_path}")

class TradingPipeline:
    @staticmethod
    async def run_training():
        """Complete training pipeline"""
        print("=== Training Pipeline ===")
        
        # 1. Data Collection
        merged_data = await DataProcessor.merge_data_sources()
        if merged_data.empty:
            print("Failed to fetch data")
            return False
            
        # 2. Feature Engineering
        feature_data = DataProcessor.calculate_features(merged_data)
        if feature_data.empty:
            print("Feature calculation failed")
            return False
            
        # 3. Prepare Data
        X = feature_data.drop(columns=['target'])
        y = feature_data['target']
        
        scaler = config.SCALER_TYPE()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
        
        X_seq, y_seq, seq_indices = LSTMModel.create_sequences(
            X_scaled_df, y, config.LOOKBACK_WINDOW)
        
        # Train/Val/Test Split
        train_idx = int(len(X_seq) * (1 - config.TEST_SIZE - config.VALIDATION_SIZE))
        val_idx = int(len(X_seq) * (1 - config.TEST_SIZE))
        
        X_train, y_train = X_seq[:train_idx], y_seq[:train_idx]
        X_val, y_val = X_seq[train_idx:val_idx], y_seq[train_idx:val_idx]
        X_test, y_test = X_seq[val_idx:], y_seq[val_idx:]
        test_dates = seq_indices[val_idx:]
        
        # 4. Train Model
        model = LSTMModel.build((config.LOOKBACK_WINDOW, X_train.shape[2]))
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=config.NN_EPOCHS,
            batch_size=config.NN_BATCH_SIZE,
            callbacks=LSTMModel.create_callbacks(),
            verbose=1
        )
        
        # 5. Save Artifacts
        model.save(config.model_path)
        joblib.dump(scaler, config.scaler_path)
        joblib.dump({
            'test_dates': test_dates,
            'feature_names': X.columns.tolist()
        }, config.metadata_path)
        
        return True

    @staticmethod
    async def run_backtest():
        """Complete backtesting pipeline"""
        print("=== Backtest Pipeline ===")
        
        # 1. Load Artifacts
        try:
            model = load_model(config.model_path)
            scaler = joblib.load(config.scaler_path)
            metadata = joblib.load(config.metadata_path)
            test_dates = metadata['test_dates']
            feature_names = metadata['feature_names']
        except Exception as e:
            print(f"Error loading artifacts: {e}")
            return None
            
        # 2. Prepare Backtest Data
        merged_data = await DataProcessor.merge_data_sources()
        if merged_data.empty:
            print("Failed to fetch backtest data")
            return None
            
        feature_data = DataProcessor.calculate_features(merged_data)
        if feature_data.empty:
            print("Feature calculation failed")
            return None
            
        # Get data for test period + lookback
        start_date = test_dates[0] - timedelta(hours=config.LOOKBACK_WINDOW)
        feature_data = feature_data.loc[start_date:]
        
        # Scale features
        X_test = feature_data[feature_names]
        X_scaled = scaler.transform(X_test)
        X_scaled_df = pd.DataFrame(X_scaled, index=X_test.index, columns=feature_names)
        
        # Create sequences aligned with test dates
        X_test_seq = []
        aligned_dates = []
        
        for i in range(len(X_scaled_df) - config.LOOKBACK_WINDOW):
            target_date = X_scaled_df.index[i + config.LOOKBACK_WINDOW]
            if target_date in test_dates:
                X_test_seq.append(X_scaled_df.iloc[i:i+config.LOOKBACK_WINDOW].values)
                aligned_dates.append(target_date)
        
        X_test_seq = np.array(X_test_seq)
        aligned_dates = pd.DatetimeIndex(aligned_dates)
        
        # 3. Generate Predictions
        predictions = model.predict(X_test_seq).flatten()
        
        # Generate signals
        signals = np.zeros(len(predictions))
        signals[predictions > config.ENTRY_THRESHOLD] = 1
        signals[predictions < -config.ENTRY_THRESHOLD] = -1
        
        signals_df = pd.DataFrame({
            'signal': signals,
            'prediction': predictions
        }, index=aligned_dates)
        
        # 4. Run Backtest
        price_data = feature_data.loc[aligned_dates, ['close', 'volatility']]
        backtester = Backtester()
        results = backtester.run_backtest(price_data, signals_df['signal'])
        
        # 5. Display Results
        print("\n=== Backtest Results ===")
        print(f"Final Portfolio Value: ${results['final_value']:,.2f}")
        print(f"Total Return: {(results['final_value']/config.INITIAL_CAPITAL-1)*100:.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']*100:.2f}%")
        print(f"Number of Trades: {len(results['trades'])}")
        
        if results['trades_csv_path']:
            print(f"\nDetailed trades saved to: {results['trades_csv_path']}")
        
        return results

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Run pipeline
    MODE = "backtest"  # "train" or "backtest"
    
    if MODE == "train":
        asyncio.run(TradingPipeline.run_training())
    elif MODE == "backtest":
        asyncio.run(TradingPipeline.run_backtest())
    else:
        print("Invalid mode. Choose 'train' or 'backtest'")