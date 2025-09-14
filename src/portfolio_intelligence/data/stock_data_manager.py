"""
Stock Data Pipeline Manager for Portfolio Intelligence System

Manages the fetching, caching, and serving of stock market data with:
- Multi-timeframe data support (daily, hourly, 5-minute)
- Intelligent Redis caching with proper TTL
- Rate limiting and error handling
- Batch processing for efficiency
- Integration with existing market data providers

Data Flow:
1. Request comes in for stock data
2. Check Redis cache first
3. If cache miss, fetch from data provider
4. Cache result with appropriate TTL
5. Return data to caller

Caching Strategy:
- Daily bars: 24 hour TTL (refreshed after market close)
- Hourly bars: 1 hour TTL
- 5-minute bars: 5 minute TTL
- Fundamentals: 6 hour TTL (refreshed twice daily)
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum
import json

logger = logging.getLogger(__name__)


class DataTimeframe(Enum):
    """Data timeframe options"""
    DAILY = "daily"
    HOURLY = "hourly" 
    FIVE_MIN = "5min"
    ONE_MIN = "1min"


class DataType(Enum):
    """Data type options"""
    PRICE = "price"
    VOLUME = "volume"
    OHLC = "ohlc"
    FUNDAMENTALS = "fundamentals"
    NEWS = "news"
    TECHNICAL = "technical"


@dataclass
class DataRequest:
    """Data request specification"""
    symbol: str
    timeframe: DataTimeframe
    data_type: DataType = DataType.OHLC
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    limit: Optional[int] = None
    force_refresh: bool = False


@dataclass
class CacheConfig:
    """Cache configuration"""
    ttl_seconds: int
    key_prefix: str
    max_size: Optional[int] = None


@dataclass
class StockDataPoint:
    """Single stock data point"""
    symbol: str
    timestamp: datetime
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "metadata": self.metadata or {}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StockDataPoint':
        """Create from dictionary"""
        return cls(
            symbol=data["symbol"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            open=data.get("open"),
            high=data.get("high"),
            low=data.get("low"),
            close=data.get("close"),
            volume=data.get("volume"),
            metadata=data.get("metadata", {})
        )


class RateLimiter:
    """Rate limiter for API calls"""

    def __init__(self, max_calls: int, time_window: int):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls: List[datetime] = []

    async def wait_if_needed(self) -> None:
        """Wait if rate limit would be exceeded"""
        now = datetime.now(timezone.utc)

        # Remove calls outside the time window
        cutoff = now - timedelta(seconds=self.time_window)
        self.calls = [call_time for call_time in self.calls if call_time > cutoff]

        # Check if we're at the limit
        if len(self.calls) >= self.max_calls:
            sleep_time = self.time_window - (now - self.calls[0]).total_seconds()
            if sleep_time > 0:
                logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
                await asyncio.sleep(sleep_time)

        # Record this call
        self.calls.append(now)


class StockDataManager:
    """
    Central manager for stock market data with caching, rate limiting,
    and error handling
    """

    def __init__(self):
        """Initialize stock data manager"""
        self.cache_configs = {
            DataTimeframe.DAILY: CacheConfig(
                ttl_seconds=24 * 3600,  # 24 hours
                key_prefix="stock_daily",
                max_size=10000
            ),
            DataTimeframe.HOURLY: CacheConfig(
                ttl_seconds=3600,  # 1 hour
                key_prefix="stock_hourly",
                max_size=5000
            ),
            DataTimeframe.FIVE_MIN: CacheConfig(
                ttl_seconds=300,  # 5 minutes
                key_prefix="stock_5min",
                max_size=1000
            ),
            DataTimeframe.ONE_MIN: CacheConfig(
                ttl_seconds=60,  # 1 minute
                key_prefix="stock_1min",
                max_size=500
            )
        }

        # Rate limiters for different providers
        self.rate_limiters = {
            "alpha_vantage": RateLimiter(max_calls=5, time_window=60),
            "ib": RateLimiter(max_calls=50, time_window=60),
            "polygon": RateLimiter(max_calls=5, time_window=60)
        }

        self.redis_client = None  # Will be initialized when needed
        self.data_providers = {}  # Will be populated with actual providers

    async def _get_redis_client(self):
        """Get Redis client (lazy initialization)"""
        if self.redis_client is None:
            try:
                import aioredis
                self.redis_client = await aioredis.from_url("redis://localhost:6379")
            except ImportError:
                logger.warning("Redis not available, using in-memory cache")
                self.redis_client = {}  # Fallback to dict
        return self.redis_client

    def _generate_cache_key(self, request: DataRequest) -> str:
        """Generate cache key for request"""
        config = self.cache_configs[request.timeframe]
        key_parts = [
            config.key_prefix,
            request.symbol,
            request.data_type.value,
            request.timeframe.value
        ]

        if request.start_date:
            key_parts.append(request.start_date.strftime("%Y%m%d"))
        if request.end_date:
            key_parts.append(request.end_date.strftime("%Y%m%d"))
        if request.limit:
            key_parts.append(str(request.limit))

        return ":".join(key_parts)

    async def _get_from_cache(self, cache_key: str) -> Optional[List[StockDataPoint]]:
        """Get data from cache"""
        try:
            redis_client = await self._get_redis_client()

            if isinstance(redis_client, dict):
                # Fallback in-memory cache
                cached_data = redis_client.get(cache_key)
            else:
                # Redis cache
                cached_data = await redis_client.get(cache_key)

            if cached_data:
                if isinstance(cached_data, bytes):
                    cached_data = cached_data.decode('utf-8')

                data_list = json.loads(cached_data)
                return [StockDataPoint.from_dict(item) for item in data_list]

            pass
        except Exception as e:
            logger.warning(f"Cache read error: {e}")

        return None

    async def _set_cache(self, cache_key: str, data: List[StockDataPoint], ttl: int):
        """Set data in cache"""
        try:
            redis_client = await self._get_redis_client()
            data_json = json.dumps([point.to_dict() for point in data])

            if isinstance(redis_client, dict):
                # Fallback in-memory cache
                redis_client[cache_key] = data_json
            else:
                # Redis cache
                await redis_client.setex(cache_key, ttl, data_json)

            pass
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    async def _fetch_from_provider(self, request: DataRequest) -> List[StockDataPoint]:
        """Fetch data from external provider"""
        try:
            # Rate limit the request
            provider = "alpha_vantage"  # Default provider
            await self.rate_limiters[provider].wait_if_needed()

            # This would integrate with actual data providers
            # For now, return sample data
            logger.info(f"Fetching {request.symbol} {request.timeframe.value} data from {provider}")

            # Sample data for demonstration
            now = datetime.now(timezone.utc)
            data_points = []

            for i in range(min(request.limit or 100, 100)):
                timestamp = now - timedelta(minutes=i * 5)  # 5-minute intervals
                data_points.append(StockDataPoint(
                    symbol=request.symbol,
                    timestamp=timestamp,
                    open=100.0 + i * 0.1,
                    high=100.0 + i * 0.15,
                    low=100.0 + i * 0.05,
                    close=100.0 + i * 0.12,
                    volume=1000000 + i * 1000
                ))

            return data_points

            pass
        except Exception as e:
            logger.error(f"Failed to fetch data for {request.symbol}: {e}")
            return []

    async def get_stock_data(self, request: DataRequest) -> List[StockDataPoint]:
        """
        Get stock data with caching and fallback

        Args:
            request: Data request specification

        Returns:
            List of stock data points
        """
        try:
            cache_key = self._generate_cache_key(request)

            # Check cache first (unless force refresh)
            if not request.force_refresh:
                cached_data = await self._get_from_cache(cache_key)
                if cached_data:
                    logger.debug(f"Cache hit for {request.symbol}")
                    return cached_data

            # Fetch from provider
            data = await self._fetch_from_provider(request)

            # Cache the result
            if data:
                config = self.cache_configs[request.timeframe]
                await self._set_cache(cache_key, data, config.ttl_seconds)

            return data

            pass
        except Exception as e:
            logger.error(f"Error getting stock data: {e}")
            return []

    async def get_multiple_stocks(self, symbols: List[str], timeframe: DataTimeframe) -> Dict[str, List[StockDataPoint]]:
        """Get data for multiple stocks efficiently"""
        results = {}

        # Create requests
        requests = [
            DataRequest(symbol=symbol, timeframe=timeframe)
            for symbol in symbols
        ]

        # Process in batches to respect rate limits
        batch_size = 10
        for i in range(0, len(requests), batch_size):
            batch = requests[i:i + batch_size]

            # Process batch concurrently
            batch_results = await asyncio.gather(*[
                self.get_stock_data(request) for request in batch
            ], return_exceptions=True)

            # Collect results
            for request, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Error fetching {request.symbol}: {result}")
                    results[request.symbol] = []
                else:
                    results[request.symbol] = result

        return results

    async def invalidate_cache(self, symbol: str, timeframe: Optional[DataTimeframe] = None):
        """Invalidate cache for symbol"""
        try:
            redis_client = await self._get_redis_client()

            if timeframe:
                # Invalidate specific timeframe
                config = self.cache_configs[timeframe]
                pattern = f"{config.key_prefix}:{symbol}:*"
            else:
                # Invalidate all timeframes
                pattern = f"stock_*:{symbol}:*"

            if isinstance(redis_client, dict):
                # In-memory cache
                keys_to_delete = [key for key in redis_client.keys() if key.startswith(f"{symbol}:")]
                for key in keys_to_delete:
                    del redis_client[key]
            else:
                # Redis cache
                keys = await redis_client.keys(pattern)
                if keys:
                    await redis_client.delete(*keys)

            logger.info(f"Cache invalidated for {symbol}")

            pass
        except Exception as e:
            logger.error(f"Error invalidating cache: {e}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {}

        for timeframe, config in self.cache_configs.items():
            stats[timeframe.value] = {
                "ttl_seconds": config.ttl_seconds,
                "key_prefix": config.key_prefix,
                "max_size": config.max_size
            }

        return stats

    async def cleanup_expired_cache(self):
        """Clean up expired cache entries (maintenance task)"""
        try:
            redis_client = await self._get_redis_client()

            if not isinstance(redis_client, dict):
                # Redis automatically handles TTL expiration
                logger.debug("Redis handles TTL expiration automatically")
            else:
                # For in-memory cache, we'd need to track expiration times
                logger.debug("In-memory cache cleanup not implemented")

            pass
        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")


# Global instance
_stock_data_manager: Optional[StockDataManager] = None


def get_stock_data_manager() -> StockDataManager:
    """Get global stock data manager instance"""
    global _stock_data_manager
    if _stock_data_manager is None:
        _stock_data_manager = StockDataManager()
    return _stock_data_manager


# Helper functions for common use cases
async def get_daily_prices(symbol: str, days: int = 30) -> List[StockDataPoint]:
    """Get daily price data for a symbol"""
    manager = get_stock_data_manager()
    request = DataRequest(
        symbol=symbol,
        timeframe=DataTimeframe.DAILY,
        data_type=DataType.OHLC,
        limit=days
    )
    return await manager.get_stock_data(request)


async def get_intraday_data(symbol: str, hours: int = 6) -> List[StockDataPoint]:
    """Get 5-minute intraday data for a symbol"""
    manager = get_stock_data_manager()
    request = DataRequest(
        symbol=symbol,
        timeframe=DataTimeframe.FIVE_MIN,
        data_type=DataType.OHLC,
        limit=hours * 12  # 12 data points per hour
    )
    return await manager.get_stock_data(request)


async def refresh_stock_cache(symbol: str):
    """Force refresh cache for a symbol"""
    manager = get_stock_data_manager()
    await manager.invalidate_cache(symbol)

    # Fetch fresh data for all timeframes
    timeframes = [DataTimeframe.DAILY, DataTimeframe.HOURLY, DataTimeframe.FIVE_MIN]
    for timeframe in timeframes:
        request = DataRequest(
            symbol=symbol,
            timeframe=timeframe,
            force_refresh=True
        )
        await manager.get_stock_data(request)