from datetime import datetime, timedelta
import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, AsyncGenerator, Union
from uuid import uuid4

# Optional imports - will be used if available
try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

logger = logging.getLogger(__name__)


class DataSource(Enum):
    ALPHA_VANTAGE = "alpha_vantage"
    IEX_CLOUD = "iex_cloud"
    POLYGON = "polygon"
    QUANDL = "quandl"
    BLOOMBERG = "bloomberg"
    REFINITIV = "refinitiv"
    COINBASE = "coinbase"
    BINANCE = "binance"
    IBKR = "ibkr"
    CUSTOM_FEED = "custom_feed"
    YAHOO_FINANCE = "yahoo_finance"


class DataType(Enum):
    QUOTE = "quote"
    BAR = "bar"
    ORDER_BOOK = "order_book"
    NEWS = "news"
    FUNDAMENTALS = "fundamentals"
    OPTIONS = "options"
    FUTURES = "futures"
    CRYPTO = "crypto"
    ECONOMIC = "economic"
    TRADE = "trade"
    TICK = "tick"


class DataFrequency(Enum):
    TICK = "tick"
    MINUTE = "1m"
    FIVE_MINUTE = "5m"
    FIFTEEN_MINUTE = "15m"
    HOUR = "1h"
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1M"


@dataclass
class MarketDataPoint:
    symbol: str
    timestamp: datetime
    data_type: DataType = None
    source: DataSource = None
    
    # Price/Quote data
    price: Optional[Decimal] = None
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    
    # Trade data
    volume: Optional[int] = None
    trade_id: Optional[str] = None
    
    # OHLCV bar data
    open: Optional[Decimal] = None
    high: Optional[Decimal] = None
    low: Optional[Decimal] = None
    close: Optional[Decimal] = None
    
    # Order book data
    order_book: Optional[Dict[str, List]] = None
    
    # Additional data
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Quality metrics
    quality_score: float = 1.0
    latency_ms: Optional[float] = None


@dataclass
class DataRequest:
    symbols: List[str] = field(default_factory=list)
    data_types: List[DataType] = field(default_factory=list)
    sources: List[DataSource] = field(default_factory=list)
    frequency: DataFrequency = DataFrequency.MINUTE
    
    # Time range
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    # Real-time vs historical
    real_time: bool = False
    backfill: bool = True
    
    # Quality requirements
    min_quality_score: float = 0.8
    max_latency_ms: Optional[float] = None
    
    # Callback for real-time data
    data_callback: Optional[Callable] = None


class DataAdapter(ABC):
    def __init__(self, source: DataSource, config: Dict[str, Any]):
        self.source = source
        self.config = config
        self.connected = False
        self.rate_limits = {
            'requests_per_minute': config.get('rate_limit', 60),
            'requests_made': []
        }
    
    @abstractmethod
    async def connect(self) -> bool:
        pass
    
    @abstractmethod
    async def disconnect(self):
        pass
    
    @abstractmethod
    async def get_historical_data(self, request: DataRequest) -> List[MarketDataPoint]:
        pass
    
    @abstractmethod
    async def subscribe_real_time(self, request: DataRequest) -> AsyncGenerator[MarketDataPoint, None]:
        pass
    
    def check_rate_limit(self) -> bool:
        now = time.time()
        cutoff = now - 60  # 1 minute window
        
        # Remove old requests
        self.rate_limits['requests_made'] = [
            ts for ts in self.rate_limits['requests_made'] if ts > cutoff
        ]
        
        # Check limit
        if len(self.rate_limits['requests_made']) >= self.rate_limits['requests_per_minute']:
            return False
        
        self.rate_limits['requests_made'].append(now)
        return True
    
    def calculate_quality_score(self, data_point: MarketDataPoint) -> float:
        score = 1.0
        
        # Penalize high latency
        if data_point.latency_ms and data_point.latency_ms > 1000:
            score -= 0.2  # > 1s
        elif data_point.latency_ms and data_point.latency_ms > 500:
            score -= 0.1  # > 500ms
        
        # Penalize missing data
        required_fields = ['price', 'volume'] if data_point.data_type == DataType.TRADE else ['bid', 'ask']
        missing_fields = sum(1 for field in required_fields if not getattr(data_point, field))
        score -= missing_fields * 0.2
        
        return max(0.0, score)


class AlphaVantageAdapter(DataAdapter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(DataSource.ALPHA_VANTAGE, config)
        self.base_url = "https://www.alphavantage.co/query"
    
    async def connect(self) -> bool:
        self.connected = True
        return True
    
    async def disconnect(self):
        self.connected = False
    
    async def get_historical_data(self, request: DataRequest) -> List[MarketDataPoint]:
        data_points = []
        
        for symbol in request.symbols:
            try:
                # Map frequency to Alpha Vantage function
                function_map = {
                    DataFrequency.MINUTE: "TIME_SERIES_INTRADAY",
                    DataFrequency.FIVE_MINUTE: "TIME_SERIES_INTRADAY",
                    DataFrequency.DAILY: "TIME_SERIES_DAILY",
                    DataFrequency.WEEKLY: "TIME_SERIES_WEEKLY",
                    DataFrequency.MONTHLY: "TIME_SERIES_MONTHLY"
                }
                
                function = function_map.get(request.frequency, "TIME_SERIES_DAILY")
                
                params = {
                    'function': function,
                    'symbol': symbol,
                    'apikey': self.config.get('api_key', '')
                }
                
                if function == "TIME_SERIES_INTRADAY":
                    interval_map = {
                        DataFrequency.MINUTE: "1min",
                        DataFrequency.FIVE_MINUTE: "5min",
                        DataFrequency.FIFTEEN_MINUTE: "15min"
                    }
                    params['interval'] = interval_map.get(request.frequency, "1min")
                
                if not HAS_AIOHTTP:
                    logger.warning("aiohttp not available - cannot fetch data")
                    continue
                    
                async with aiohttp.ClientSession() as session:
                    async with session.get(self.base_url, params=params) as response:
                        data = await response.json()
                        
                        # Parse response
                        time_series_key = [k for k in data.keys() if "Time Series" in k]
                        if not time_series_key:
                            logger.warning(f"No time series data for {symbol}")
                            continue
                        
                        time_series = data[time_series_key[0]]
                        
                        for timestamp_str, ohlcv in time_series.items():
                            timestamp = datetime.strptime(
                                timestamp_str, 
                                '%Y-%m-%d %H:%M:%S' if ':' in timestamp_str else '%Y-%m-%d'
                            )
                            
                            data_point = MarketDataPoint(
                                symbol=symbol,
                                timestamp=timestamp,
                                data_type=DataType.BAR,
                                source=self.source,
                                open=Decimal(ohlcv['1. open']),
                                high=Decimal(ohlcv['2. high']),
                                low=Decimal(ohlcv['3. low']),
                                close=Decimal(ohlcv['4. close']),
                                volume=int(ohlcv['5. volume']),
                                price=Decimal(ohlcv['4. close'])
                            )
                            
                            data_point.quality_score = self.calculate_quality_score(data_point)
                            data_points.append(data_point)
                            
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {str(e)}")
                
        return data_points
    
    async def subscribe_real_time(self, request: DataRequest) -> AsyncGenerator[MarketDataPoint, None]:
        # Alpha Vantage doesn't support real-time streaming
        # Make this an async generator
        yield  # This makes it a generator
        return


class YahooFinanceAdapter(DataAdapter):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(DataSource.YAHOO_FINANCE, config)
        self.base_url = "https://query1.finance.yahoo.com/v8/finance/chart"
    
    async def connect(self) -> bool:
        self.connected = True
        return True
    
    async def disconnect(self):
        self.connected = False
    
    async def get_historical_data(self, request: DataRequest) -> List[MarketDataPoint]:
        data_points = []
        
        for symbol in request.symbols:
            try:
                # Build URL
                url = f"{self.base_url}/{symbol}"
                params = {
                    'interval': self._map_frequency(request.frequency),
                }
                
                if request.start_date and request.end_date:
                    params['period1'] = int(request.start_date.timestamp())
                    params['period2'] = int(request.end_date.timestamp())
                
                if not HAS_AIOHTTP:
                    logger.warning("aiohttp not available - cannot fetch data")
                    continue
                    
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params) as response:
                        data = await response.json()
                        
                        result = data.get('chart', {}).get('result', [])
                        if not result:
                            continue
                            
                        chart_data = result[0]
                        timestamps = chart_data.get('timestamp', [])
                        quotes = chart_data.get('indicators', {}).get('quote', [{}])[0]
                        
                        for i, ts in enumerate(timestamps):
                            timestamp = datetime.fromtimestamp(ts)
                            
                            data_point = MarketDataPoint(
                                symbol=symbol,
                                timestamp=timestamp,
                                data_type=DataType.BAR,
                                source=self.source,
                                open=Decimal(str(quotes.get('open', [None])[i] or 0)),
                                high=Decimal(str(quotes.get('high', [None])[i] or 0)),
                                low=Decimal(str(quotes.get('low', [None])[i] or 0)),
                                close=Decimal(str(quotes.get('close', [None])[i] or 0)),
                                volume=int(quotes.get('volume', [None])[i] or 0),
                                price=Decimal(str(quotes.get('close', [None])[i] or 0))
                            )
                            
                            data_point.quality_score = self.calculate_quality_score(data_point)
                            data_points.append(data_point)
                            
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {str(e)}")
                
        return data_points
    
    def _map_frequency(self, frequency: DataFrequency) -> str:
        frequency_map = {
            DataFrequency.MINUTE: "1m",
            DataFrequency.FIVE_MINUTE: "5m",
            DataFrequency.FIFTEEN_MINUTE: "15m",
            DataFrequency.HOUR: "1h",
            DataFrequency.DAILY: "1d",
            DataFrequency.WEEKLY: "1wk",
            DataFrequency.MONTHLY: "1mo"
        }
        return frequency_map.get(frequency, "1d")
    
    async def subscribe_real_time(self, request: DataRequest) -> AsyncGenerator[MarketDataPoint, None]:
        # Yahoo Finance doesn't support real-time streaming via this API
        yield  # This makes it a generator
        return


class QualityMonitor:
    def __init__(self):
        self.quality_history = defaultdict(list)
        self.thresholds = {
            'min_quality_score': 0.8,
            'max_latency_ms': 1000,
            'min_data_rate': 0.1  # data points per second
        }
    
    def record_data_quality(self, data_point: MarketDataPoint):
        key = f"{data_point.source.value}_{data_point.symbol}"
        self.quality_history[key].append({
            'timestamp': datetime.utcnow(),
            'quality_score': data_point.quality_score,
            'latency_ms': data_point.latency_ms
        })
        
        # Keep only recent history (last 1000 points)
        if len(self.quality_history[key]) > 1000:
            self.quality_history[key] = self.quality_history[key][-1000:]
    
    def get_quality_report(self) -> Dict[str, Any]:
        report = {}
        
        for source_symbol, history in self.quality_history.items():
            if not history:
                continue
                
            recent_history = [h for h in history 
                             if h['timestamp'] > datetime.utcnow() - timedelta(minutes=30)]
            
            if not recent_history:
                continue
                
            avg_quality = sum(h['quality_score'] for h in recent_history) / len(recent_history)
            avg_latency = sum(h['latency_ms'] or 0 for h in recent_history) / len(recent_history)
            
            report[source_symbol] = {
                'avg_quality_score': avg_quality,
                'avg_latency_ms': avg_latency,
                'data_points': len(recent_history),
                'meets_quality_threshold': avg_quality >= self.thresholds['min_quality_score'],
                'meets_latency_threshold': avg_latency <= self.thresholds['max_latency_ms']
            }
        
        return report


class MarketDataPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.adapters = {}
        self.subscribers = {}
        self.data_buffer = []
        self.quality_monitor = QualityMonitor()
        self.stats = {
            'start_time': datetime.utcnow(),
            'total_data_points': 0,
            'data_points_by_source': defaultdict(int),
            'data_points_by_type': defaultdict(int)
        }
        
        # Initialize adapters based on config
        self._initialize_adapters()
    
    def _initialize_adapters(self):
        adapter_configs = self.config.get('adapters', {})
        
        for source_name, adapter_config in adapter_configs.items():
            if source_name == 'alpha_vantage' and adapter_config.get('enabled', False):
                self.adapters[DataSource.ALPHA_VANTAGE] = AlphaVantageAdapter(adapter_config)
            elif source_name == 'yahoo_finance' and adapter_config.get('enabled', False):
                self.adapters[DataSource.YAHOO_FINANCE] = YahooFinanceAdapter(adapter_config)
    
    async def connect_all_sources(self):
        connection_tasks = []
        for adapter in self.adapters.values():
            connection_tasks.append(adapter.connect())
        
        results = await asyncio.gather(*connection_tasks, return_exceptions=True)
        
        connected_count = sum(1 for result in results if result is True)
        logger.info(f"Connected to {connected_count}/{len(self.adapters)} data sources")
        
        return connected_count > 0
    
    async def get_historical_data(self, request: DataRequest) -> List[MarketDataPoint]:
        all_data = []
        
        # Filter adapters based on request sources
        target_adapters = []
        if request.sources:
            target_adapters = [self.adapters[source] for source in request.sources 
                             if source in self.adapters]
        else:
            target_adapters = list(self.adapters.values())
        
        # Fetch data from all target adapters
        fetch_tasks = []
        for adapter in target_adapters:
            if adapter.connected:
                fetch_tasks.append(adapter.get_historical_data(request))
        
        if not fetch_tasks:
            logger.warning("No connected adapters available for data fetch")
            return []
        
        results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                all_data.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Error fetching historical data: {result}")
        
        # Filter by quality requirements
        filtered_data = []
        for data_point in all_data:
            if data_point.quality_score >= request.min_quality_score:
                if request.max_latency_ms is None or (
                    data_point.latency_ms is None or 
                    data_point.latency_ms <= request.max_latency_ms
                ):
                    filtered_data.append(data_point)
                    
                    # Record quality metrics
                    self.quality_monitor.record_data_quality(data_point)
                    
                    # Update stats
                    self.stats['total_data_points'] += 1
                    self.stats['data_points_by_source'][data_point.source.value] += 1
                    self.stats['data_points_by_type'][data_point.data_type.value] += 1
        
        # Sort by timestamp
        filtered_data.sort(key=lambda x: x.timestamp)
        
        return filtered_data
    
    async def start_real_time_feed(self, request: DataRequest) -> str:
        subscriber_id = str(uuid4())
        self.subscribers[subscriber_id] = {
            'request': request,
            'active': True,
            'created_at': datetime.utcnow()
        }
        
        # Start async task for this subscriber
        asyncio.create_task(self._handle_real_time_subscription(subscriber_id))
        
        logger.info(f"Started real-time feed for subscriber {subscriber_id}")
        return subscriber_id
    
    async def _handle_real_time_subscription(self, subscriber_id: str):
        subscriber = self.subscribers.get(subscriber_id)
        if not subscriber:
            return
        
        request = subscriber['request']
        
        try:
            # Get relevant adapters
            target_adapters = []
            if request.sources:
                target_adapters = [self.adapters[source] for source in request.sources 
                                 if source in self.adapters]
            else:
                target_adapters = list(self.adapters.values())
            
            # Subscribe to real-time feeds
            async def process_adapter_feed(adapter):
                async for data_point in adapter.subscribe_real_time(request):
                    if not self.subscribers.get(subscriber_id, {}).get('active'):
                        break
                        
                    # Apply quality filters
                    if data_point.quality_score >= request.min_quality_score:
                        if request.data_callback:
                            try:
                                await request.data_callback(data_point)
                            except Exception as e:
                                logger.error(f"Error in data callback: {e}")
                        
                        # Record quality and stats
                        self.quality_monitor.record_data_quality(data_point)
                        self.stats['total_data_points'] += 1
                        self.stats['data_points_by_source'][data_point.source.value] += 1
                        self.stats['data_points_by_type'][data_point.data_type.value] += 1
            
            # Start feed processing tasks
            feed_tasks = []
            for adapter in target_adapters:
                if adapter.connected:
                    feed_tasks.append(process_adapter_feed(adapter))
            
            if feed_tasks:
                await asyncio.gather(*feed_tasks, return_exceptions=True)
                
        except Exception as e:
            logger.error(f"Error in real-time subscription {subscriber_id}: {e}")
        finally:
            # Clean up
            if subscriber_id in self.subscribers:
                del self.subscribers[subscriber_id]
    
    def stop_real_time_feed(self, subscriber_id: str):
        if subscriber_id in self.subscribers:
            self.subscribers[subscriber_id]['active'] = False
            logger.info(f"Stopped real-time feed for subscriber {subscriber_id}")
    
    def subscribe_to_data(self, callback: Callable[[MarketDataPoint], None]) -> str:
        subscriber_id = str(uuid4())
        self.subscribers[subscriber_id] = {
            'callback': callback,
            'created_at': datetime.utcnow()
        }
        logger.info(f"Added subscriber {subscriber_id}")
        return subscriber_id
    
    def unsubscribe_from_data(self, subscriber_id: str):
        if subscriber_id in self.subscribers:
            del self.subscribers[subscriber_id]
            logger.info(f"Removed subscriber {subscriber_id}")
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        uptime = (datetime.utcnow() - self.stats['start_time']).total_seconds()
        
        return {
            'uptime_seconds': uptime,
            'total_data_points': self.stats['total_data_points'],
            'data_rate_per_second': self.stats['total_data_points'] / max(uptime, 1),
            'data_points_by_source': dict(self.stats['data_points_by_source']),
            'data_points_by_type': dict(self.stats['data_points_by_type']),
            'buffer_size': len(self.data_buffer),
            'active_subscribers': len(self.subscribers),
            'connected_sources': len([a for a in self.adapters.values() if a.connected]),
            'quality_report': self.quality_monitor.get_quality_report()
        }
    
    async def disconnect_all_sources(self):
        disconnect_tasks = []
        for adapter in self.adapters.values():
            if adapter.connected:
                disconnect_tasks.append(adapter.disconnect())
        
        if disconnect_tasks:
            await asyncio.gather(*disconnect_tasks, return_exceptions=True)
            
        logger.info("Disconnected from all data sources")


# Example usage
async def example_market_data_pipeline():
    config = {
        'adapters': {
            'yahoo_finance': {
                'enabled': True,
                'rate_limit': 100
            },
            'alpha_vantage': {
                'enabled': True,
                'api_key': 'your_api_key_here',
                'rate_limit': 5
            }
        }
    }
    
    # Initialize pipeline
    pipeline = MarketDataPipeline(config)
    
    # Connect to sources
    await pipeline.connect_all_sources()
    
    # Create data request
    request = DataRequest(
        symbols=['AAPL', 'MSFT', 'GOOGL'],
        data_types=[DataType.BAR, DataType.QUOTE],
        sources=[DataSource.YAHOO_FINANCE],
        frequency=DataFrequency.DAILY,
        start_date=datetime.utcnow() - timedelta(days=30),
        end_date=datetime.utcnow(),
        min_quality_score=0.8
    )
    
    # Get historical data
    historical_data = await pipeline.get_historical_data(request)
    print(f"Retrieved {len(historical_data)} historical data points")
    
    # Show some data
    for data_point in historical_data[:5]:
        print(f"{data_point.symbol} @ {data_point.timestamp} ${data_point.close} (Quality: {data_point.quality_score:.2f})")
    
    # Get pipeline statistics
    stats = pipeline.get_pipeline_stats()
    print(f"Pipeline Stats: {stats}")
    
    # Start real-time feed (commented out for demo)
    # rt_request = DataRequest(
    #     symbols=['AAPL'],
    #     data_types=[DataType.QUOTE],
    #     sources=[DataSource.YAHOO_FINANCE],
    #     real_time=True,
    #     callback=lambda dp: print(f"Real-time: {dp.symbol} ${dp.price}")
    # )
    # await pipeline.start_real_time_feed(rt_request)
    
    # Disconnect
    await pipeline.disconnect_all_sources()


if __name__ == "__main__":
    # Run example
    asyncio.run(example_market_data_pipeline())