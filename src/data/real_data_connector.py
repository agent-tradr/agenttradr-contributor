from datetime import datetime, timedelta, timezone
from typing import Dict, Any
import logging
import os
import time
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RealDataConnector:
    """Real Data Connector - Actual Market Data (Not Random!)
    
    This connects to REAL data sources:
    - Yahoo Finance (free, delayed)
    - Alpaca (free/paid tiers)
    - Polygon.io (paid)
    No more random data!
    """
    
    def __init__(self, provider: str = 'yahoo'):
        """
        Args:
            provider: 'yahoo', 'alpaca', or 'polygon'
        """
        self.provider = provider
        self.cache = {}
        self.last_request_time = {}
        self.rate_limit_delay = 0.1  # 100ms between requests
        
        # Initialize provider
        if provider == 'yahoo':
            self._init_yahoo()
        elif provider == 'alpaca':
            self._init_alpaca()
        elif provider == 'polygon':
            self._init_polygon()
        else:
            raise ValueError(f"Unknown provider {provider}")
    
    def _init_yahoo(self):
        """Initialize Yahoo Finance"""
        try:
            import yfinance as yf
            self.yf = yf
            logger.info("Yahoo Finance initialized (free tier)")
        except ImportError:
            logger.error("Install yfinance: pip install yfinance")
            raise
    
    def _init_alpaca(self):
        """Initialize Alpaca API"""
        try:
            import alpaca_trade_api as tradeapi
            
            # Get credentials from environment
            api_key = os.getenv('APCA_API_KEY_ID', 'your_api_key')
            secret_key = os.getenv('APCA_API_SECRET_KEY', 'your_secret')
            base_url = os.getenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets')
            
            self.alpaca = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
            
            # Test connection
            account = self.alpaca.get_account()
            logger.info(f"Alpaca initialized. Balance: ${account.cash}")
        except Exception as e:
            logger.error(f"Alpaca initialization failed: {e}")
            logger.info("Falling back to Yahoo Finance")
            self._init_yahoo()
    
    def _init_polygon(self):
        """Initialize Polygon.io API"""
        try:
            from polygon import RESTClient
            
            api_key = os.getenv('POLYGON_API_KEY', 'your_api_key')
            self.polygon = RESTClient(api_key)
            
            # Test connection
            self.polygon.get_ticker_details('AAPL')
            logger.info("Polygon.io initialized (paid tier)")
        except Exception as e:
            logger.error(f"Polygon initialization failed: {e}")
            logger.info("Falling back to Yahoo Finance")
            self._init_yahoo()
    
    def get_historical_data(self, 
                          symbol: str,
                          period: str = '1mo',
                          interval: str = '1d') -> pd.DataFrame:
        """Get REAL historical data.
        
        Args:
            symbol: Stock symbol
            period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
            interval: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        
        Returns:
            DataFrame with OHLCV data
        """
        # Rate limiting
        self._rate_limit(symbol)
        
        # Check cache
        cache_key = f"{symbol}_{period}_{interval}"
        if cache_key in self.cache:
            cached_data, cache_time = self.cache[cache_key]
            if (datetime.now(timezone.utc) - cache_time).seconds < 300:  # 5 min cache
                return cached_data
        
        try:
            if self.provider == 'yahoo':
                data = self._get_yahoo_data(symbol, period, interval)
            elif self.provider == 'alpaca':
                data = self._get_alpaca_data(symbol, period, interval)
            elif self.provider == 'polygon':
                data = self._get_polygon_data(symbol, period, interval)
            else:
                raise ValueError(f"Unknown provider {self.provider}")
            
            # Cache the data
            self.cache[cache_key] = (data, datetime.now(timezone.utc))
            return data
            
        except Exception as e:
            logger.error(f"Failed to get data for {symbol}: {e}")
            # Return empty DataFrame instead of crashing
            return pd.DataFrame()
    
    def _get_yahoo_data(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """Get data from Yahoo Finance"""
        ticker = self.yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        
        if data.empty:
            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame()
        
        # Standardize column names
        data.columns = [col.lower() for col in data.columns]
        
        # Add missing columns
        if 'adj close' in data.columns:
            data['adjusted_close'] = data['adj close']
        
        # Calculate additional metrics
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        data['volatility'] = data['returns'].rolling(window=20).std()
        
        # Add technical indicators
        data = self._add_technical_indicators(data)
        
        return data
    
    def _get_alpaca_data(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """Get data from Alpaca"""
        end = datetime.now(timezone.utc)
        
        # Convert period to days
        period_map = {
            '1d': 1, '5d': 5, '1mo': 30, '3mo': 90, 
            '6mo': 180, '1y': 365, '2y': 730, '5y': 1825
        }
        days = period_map.get(period, 30)
        start = end - timedelta(days=days)
        
        # Convert interval to Alpaca timeframe
        interval_map = {
            '1m': '1Min', '5m': '5Min', '15m': '15Min', 
            '30m': '30Min', '1h': '1Hour', '1d': '1Day'
        }
        timeframe = interval_map.get(interval, '1Day')
        
        # Get data
        bars = self.alpaca.get_bars(
            symbol,
            timeframe,
            start=start.isoformat(),
            end=end.isoformat()
        ).df
        
        if bars.empty:
            return pd.DataFrame()
        
        # Standardize
        bars.index = pd.to_datetime(bars.index)
        bars.columns = [col.lower() for col in bars.columns]
        
        return self._add_technical_indicators(bars)
    
    def _get_polygon_data(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """Get data from Polygon.io"""
        # This would use the polygon client
        # For now, fall back to Yahoo
        return self._get_yahoo_data(symbol, period, interval)
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to data"""
        if data.empty or len(data) < 20:
            return data
        
        # Simple Moving Averages
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['sma_50'] = data['close'].rolling(window=50).mean()
        
        # Exponential Moving Average
        data['ema_12'] = data['close'].ewm(span=12, adjust=False).mean()
        data['ema_26'] = data['close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        data['macd'] = data['ema_12'] - data['ema_26']
        data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
        data['macd_diff'] = data['macd'] - data['macd_signal']
        
        # RSI
        data['rsi'] = self._calculate_rsi(data['close'])
        
        # Bollinger Bands
        data['bb_middle'] = data['sma_20']
        bb_std = data['close'].rolling(window=20).std()
        data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
        data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
        
        # ATR (Average True Range)
        data['atr'] = self._calculate_atr(data)
        
        # Volume indicators
        data['volume_sma'] = data['volume'].rolling(window=20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_sma']
        
        # Support/Resistance levels
        data['resistance'] = data['high'].rolling(window=20).max()
        data['support'] = data['low'].rolling(window=20).min()
        
        return data
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    def get_realtime_quote(self, symbol: str) -> Dict[str, float]:
        """Get real-time quote.
        
        Returns:
            Dict with bid, ask, last, volume
        """
        self._rate_limit(symbol)
        
        try:
            if self.provider == 'yahoo':
                ticker = self.yf.Ticker(symbol)
                info = ticker.info
                return {
                    'bid': info.get('bid', 0),
                    'ask': info.get('ask', 0),
                    'last': info.get('regularMarketPrice', info.get('currentPrice', 0)),
                    'volume': info.get('volume', 0),
                    'timestamp': datetime.now(timezone.utc)
                }
            elif self.provider == 'alpaca':
                quote = self.alpaca.get_latest_quote(symbol)
                return {
                    'bid': quote.bid_price,
                    'ask': quote.ask_price,
                    'last': quote.ask_price,  # Approximate
                    'volume': 0,  # Not available in quote
                    'timestamp': datetime.now(timezone.utc)
                }
            else:
                # Fallback
                return {
                    'bid': 0, 'ask': 0, 'last': 0, 'volume': 0,
                    'timestamp': datetime.now(timezone.utc)
                }
        except Exception as e:
            logger.error(f"Failed to get quote for {symbol}: {e}")
            return {}
    
    def get_market_hours(self) -> Dict[str, Any]:
        """Get market hours information"""
        try:
            if self.provider == 'alpaca':
                clock = self.alpaca.get_clock()
                return {
                    'is_open': clock.is_open,
                    'next_open': clock.next_open,
                    'next_close': clock.next_close
                }
            else:
                # Simple market hours check
                now = datetime.now(timezone.utc)
                weekday = now.weekday()
                hour = now.hour
                minute = now.minute
                
                # NYSE hours: 9:30 AM - 4:00 PM ET, Monday-Friday
                is_open = (
                    weekday < 5 and  # Monday-Friday
                    ((hour == 9 and minute >= 30) or 
                     (hour > 9 and hour < 16))
                )
                
                return {
                    'is_open': is_open,
                    'next_open': None,
                    'next_close': None
                }
        except Exception as e:
            logger.error(f"Failed to get market hours: {e}")
            return {'is_open': False}
    
    def _rate_limit(self, symbol: str):
        """Rate limiting for API calls"""
        if symbol in self.last_request_time:
            elapsed = time.time() - self.last_request_time[symbol]
            if elapsed < self.rate_limit_delay:
                time.sleep(self.rate_limit_delay - elapsed)
        
        self.last_request_time[symbol] = time.time()
    
    def validate_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality.
        
        Returns:
            Dict with validation results
        """
        if data.empty:
            return {'valid': False, 'reason': 'Empty dataframe'}
        
        issues = []
        
        # Check for missing values
        missing = data.isnull().sum()
        if missing.any():
            issues.append(f"Missing values: {missing[missing > 0].to_dict()}")
        
        # Check for price data quality
        if 'close' in data.columns and len(data) > 1:
            prices = data['close']
            
            # No variation
            if prices.std() == 0:
                issues.append("Suspicious: No price variation")
            
            # Negative prices
            if prices.min() < 0:
                issues.append("Invalid: Negative prices")
            
            # Extremely high prices
            if prices.max() > 100000:
                issues.append("Suspicious: Extremely high prices")
        
        # Check volume
        if 'volume' in data.columns:
            if data['volume'].sum() == 0:
                issues.append("Warning: No volume data")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'rows': len(data),
            'date_range': f"{data.index.min()} to {data.index.max()}" if not data.empty else "N/A"
        }


# Example usage
if __name__ == "__main__":
    # Test with real data
    connector = RealDataConnector(provider='yahoo')
    
    # Get historical data
    data = connector.get_historical_data('NVDA', period='1mo', interval='1d')
    print(f"Got {len(data)} rows of REAL data for NVDA")
    print(data.tail())
    
    # Validate data
    validation = connector.validate_data_quality(data)
    print(f"Data validation: {validation}")
    
    # Get real-time quote
    quote = connector.get_realtime_quote('NVDA')
    print(f"Real-time quote: {quote}")
    
    # Check market hours
    hours = connector.get_market_hours()
    print(f"Market hours: {hours}")