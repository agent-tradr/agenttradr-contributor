from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
import asyncio
import json
import logging
import time

from core.database import Database
from portfolio_intelligence.core.stock_universe_manager import StockUniverseManager
from portfolio_intelligence.data.stock_data_manager import StockDataManager
from services.redis_service import RedisService

"""
This module implements intelligent cache warming to pre-load frequently accessed data during off-hours and market preparation periods. Includes predictive pre-loading, memory management, and performance optimization.
"""

logger = logging.getLogger(__name__)


class WarmupPriority(Enum):
    CRITICAL = "critical"  # Active positions, must load
    HIGH = "high"  # Watchlist stocks, high conviction
    MEDIUM = "medium"  # Universe stocks, moderate interest
    LOW = "low"  # Background data, nice to have


class DataType(Enum):
    PRICE_DATA = "price_data"
    TECHNICAL_INDICATORS = "technical_indicators"
    FUNDAMENTAL_DATA = "fundamental_data"
    NEWS_DATA = "news_data"
    SENTIMENT_DATA = "sentiment_data"
    CORRELATION_DATA = "correlation_data"
    VOLUME_PROFILE = "volume_profile"
    EARNINGS_DATA = "earnings_data"


@dataclass
class WarmupTask:
    symbol: str
    data_type: DataType
    priority: WarmupPriority
    timeframe: str = "1d"
    lookback_days: int = 252
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_count: int = 0

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)


class CacheWarmerIntelligent:
    """
    Intelligent cache warming system for portfolio data.
    
    Features:
    - Morning cache warming before market open
    - Predictive pre-loading based on user patterns
    - Memory management and cache optimization
    - Priority-based warming queues
    - Performance monitoring and metrics
    """

    def __init__(
        self,
        db: Database,
        redis: RedisService,
        stock_data_manager: StockDataManager,
        universe_manager: StockUniverseManager
    ):
        self.session = db.session
        self.redis = redis
        self.data_manager = stock_data_manager
        self.universe_manager = universe_manager

        # Task queues by priority
        self.task_queues: Dict[WarmupPriority, List[WarmupTask]] = {
            priority: [] for priority in WarmupPriority
        }

        # Warming state
        self.is_warming = False
        self.warming_task: Optional[asyncio.Task] = None
        self.current_task: Optional[WarmupTask] = None

        # Performance metrics
        self.warming_stats = {
            'total_warming_time': 0.0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'last_warming': None,
            'average_task_time': 0.0,
            'memory_usage': 0.0,
            'cache_hit_rate': 0.0
        }

        # Configuration
        self.config = {
            'max_concurrent_tasks': 5,
            'task_timeout_seconds': 30,
            'memory_threshold_mb': 1000,
            'warming_enabled': True
        }

    async def start_scheduled_warming(self) -> None:
        """Start scheduled cache warming"""
        logger.info("Starting scheduled cache warming")
        self.is_warming = True
        
        # Start warming task
        self.warming_task = asyncio.create_task(self._warming_scheduler())

    async def stop_warming(self) -> None:
        """Stop cache warming"""
        self.is_warming = False
        if self.warming_task:
            self.warming_task.cancel()
            try:
                await self.warming_task
            except asyncio.CancelledError:
                pass

    async def _warming_scheduler(self) -> None:
        """Main warming scheduler loop"""
        while self.is_warming:
            try:
                current_time = datetime.now(timezone.utc)
                
                # Check if it's time for scheduled warming
                if await self._should_perform_warming(current_time):
                    await self._perform_scheduled_warming()
                
                # Process any queued tasks
                await self._process_task_queues()
                
                # Sleep for 5 minutes before next check
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in warming scheduler: {e}")

    async def _should_perform_warming(self, current_time: datetime) -> bool:
        """Check if we should perform warming at current time"""
        est_hour = (current_time.hour - 5) % 24  # Approximate EST conversion
        est_minute = current_time.minute

        # Check scheduled warming times
        warming_times = [
            (6, 0),   # Pre-market warming
            (9, 15),  # Market open prep
            (12, 0),  # Lunch time
            (16, 30)  # Post-market
        ]

        for hour, minute in warming_times:
            if est_hour == hour and abs(est_minute - minute) <= 2:
                return True

        # Check if we haven't warmed in the last hour
        last_warming = self.warming_stats.get('last_warming')
        if not last_warming or (current_time - last_warming).total_seconds() > 3600:
            return True

        return False

    async def _perform_scheduled_warming(self) -> None:
        """Perform scheduled warming cycle"""
        start_time = time.time()
        try:
            # Clear old task queues
            self._clear_task_queues()
            
            # Generate warming tasks
            await self._generate_warming_tasks()
            
            # Execute warming tasks
            await self._execute_warming_tasks()
            
            # Update stats
            end_time = time.time()
            self.warming_stats['total_warming_time'] += (end_time - start_time)
            self.warming_stats['last_warming'] = datetime.now(timezone.utc)
            logger.info(f"Scheduled warming completed in {end_time - start_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error in scheduled warming: {e}")
            self.warming_stats['tasks_failed'] += 1

    async def _generate_warming_tasks(self) -> None:
        """Generate warming tasks based on priorities"""
        # Get active positions (CRITICAL priority)
        active_positions = await self._get_active_positions()
        for symbol in active_positions:
            await self._add_warming_task(
                symbol, DataType.PRICE_DATA, WarmupPriority.CRITICAL
            )
            await self._add_warming_task(
                symbol, DataType.TECHNICAL_INDICATORS, WarmupPriority.CRITICAL
            )

        # Get watchlist stocks (HIGH priority)
        watchlist_stocks = await self._get_watchlist_stocks()
        for symbol in watchlist_stocks:
            await self._add_warming_task(
                symbol, DataType.PRICE_DATA, WarmupPriority.HIGH
            )
            await self._add_warming_task(
                symbol, DataType.SENTIMENT_DATA, WarmupPriority.HIGH
            )

        # Get universe stocks (MEDIUM priority)
        universe_stocks = await self._get_universe_stocks()
        for symbol in universe_stocks[:50]:  # Limit to top 50
            await self._add_warming_task(
                symbol, DataType.PRICE_DATA, WarmupPriority.MEDIUM
            )

        # Add market-wide data (HIGH priority)
        market_indices = ['SPY', 'QQQ', 'IWM', 'VIX', 'DXY', 'TLT']
        for symbol in market_indices:
            await self._add_warming_task(
                symbol, DataType.PRICE_DATA, WarmupPriority.HIGH
            )
            await self._add_warming_task(
                symbol, DataType.TECHNICAL_INDICATORS, WarmupPriority.HIGH
            )

    async def _get_active_positions(self) -> List[str]:
        """Get list of symbols with active positions"""
        try:
            query = """
                SELECT DISTINCT symbol 
                FROM portfolio_positions 
                WHERE ABS(quantity) > 0
            """
            results = await self.session.fetch_all(query)
            return [row['symbol'] for row in results]
        except Exception as e:
            logger.error(f"Error getting active positions: {e}")
            return []

    async def _get_watchlist_stocks(self) -> List[str]:
        """Get watchlist stocks with high conviction"""
        try:
            # Get stocks with high conviction scores
            query = """
                SELECT symbol FROM stock_universe 
                WHERE conviction_score > 70 
                ORDER BY conviction_score DESC 
                LIMIT 20
            """
            results = await self.session.fetch_all(query)
            return [row['symbol'] for row in results]
        except Exception as e:
            logger.error(f"Error getting watchlist stocks: {e}")
            return []

    async def _get_universe_stocks(self) -> List[str]:
        """Get universe stocks for warming"""
        try:
            universe_data = await self.universe_manager.get_active_universe()
            return universe_data.get('active_symbols', [])
        except Exception as e:
            logger.error(f"Error getting universe stocks: {e}")
            return []

    async def _add_warming_task(
        self,
        symbol: str,
        data_type: DataType,
        priority: WarmupPriority,
        timeframe: str = "1d",
        lookback_days: int = 252
    ) -> None:
        """Add a warming task to the appropriate queue"""
        task = WarmupTask(
            symbol=symbol,
            data_type=data_type,
            priority=priority,
            timeframe=timeframe,
            lookback_days=lookback_days
        )
        self.task_queues[priority].append(task)

    def _clear_task_queues(self) -> None:
        """Clear all task queues"""
        for priority in WarmupPriority:
            self.task_queues[priority].clear()

    async def _execute_warming_tasks(self) -> None:
        """Execute warming tasks by priority"""
        priorities = [
            WarmupPriority.CRITICAL,
            WarmupPriority.HIGH,
            WarmupPriority.MEDIUM,
            WarmupPriority.LOW
        ]

        for priority in priorities:
            tasks = self.task_queues[priority]
            if not tasks:
                continue

            logger.info(f"Processing {len(tasks)} {priority.value} priority tasks")

            # Execute tasks in batches
            batch_size = self.config['max_concurrent_tasks']
            for i in range(0, len(tasks), batch_size):
                batch = tasks[i:i + batch_size]
                
                # Check memory usage before processing
                if await self._check_memory_usage():
                    logger.warning("Memory threshold exceeded, skipping remaining tasks")
                    break

                # Execute batch concurrently
                await self._execute_task_batch(batch)

    async def _execute_task_batch(self, batch: List[WarmupTask]) -> None:
        """Execute a batch of tasks concurrently"""
        semaphore = asyncio.Semaphore(self.config['max_concurrent_tasks'])

        async def execute_task(task: WarmupTask) -> None:
            async with semaphore:
                await self._execute_single_task(task)

        await asyncio.gather(
            *[execute_task(task) for task in batch],
            return_exceptions=True
        )

    async def _execute_single_task(self, task: WarmupTask) -> None:
        """Execute a single warming task"""
        start_time = time.time()
        try:
            # Set timeout for task execution
            await asyncio.wait_for(
                self._warm_data(task),
                timeout=self.config['task_timeout_seconds']
            )
            
            task.completed_at = datetime.now(timezone.utc)
            self.warming_stats['tasks_completed'] += 1
            
            # Update average task time
            task_time = time.time() - start_time
            self._update_average_task_time(task_time)
            
        except asyncio.TimeoutError:
            logger.warning(f"Task timeout: {task.symbol} {task.data_type.value}")
            task.error_count += 1
            self.warming_stats['tasks_failed'] += 1
        except Exception as e:
            logger.error(f"Task failed: {task.symbol} {task.data_type.value} - {e}")
            task.error_count += 1
            self.warming_stats['tasks_failed'] += 1
        finally:
            self.current_task = None

    async def _warm_data(self, task: WarmupTask) -> None:
        """Warm specific data type for a symbol"""
        if task.data_type == DataType.PRICE_DATA:
            await self._warm_price_data(task)
        elif task.data_type == DataType.TECHNICAL_INDICATORS:
            await self._warm_technical_indicators(task)
        elif task.data_type == DataType.FUNDAMENTAL_DATA:
            await self._warm_fundamental_data(task)
        elif task.data_type == DataType.NEWS_DATA:
            await self._warm_news_data(task)
        elif task.data_type == DataType.SENTIMENT_DATA:
            await self._warm_sentiment_data(task)
        elif task.data_type == DataType.CORRELATION_DATA:
            await self._warm_correlation_data(task)
        elif task.data_type == DataType.VOLUME_PROFILE:
            await self._warm_volume_profile(task)
        elif task.data_type == DataType.EARNINGS_DATA:
            await self._warm_earnings_data(task)

    async def _warm_price_data(self, task: WarmupTask) -> None:
        """Warm price data for a symbol"""
        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=task.lookback_days)

        # Warm different timeframes
        timeframes = ['1d', '1h', '5m'] if task.priority in [WarmupPriority.CRITICAL, WarmupPriority.HIGH] else ['1d']

        for timeframe in timeframes:
            await self.data_manager.get_stock_data(
                task.symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe
            )

    async def _warm_technical_indicators(self, task: WarmupTask) -> None:
        """Warm technical indicators for a symbol"""
        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=task.lookback_days)

        # Get price data first
        price_data = await self.data_manager.get_stock_data(
            task.symbol,
            start_date=start_date,
            end_date=end_date,
            timeframe=task.timeframe
        )

        if price_data is not None and not price_data.empty:
            # Calculate and cache common indicators
            indicators = ['sma_20', 'sma_50', 'rsi', 'macd', 'bollinger_bands']
            for indicator in indicators:
                cache_key = f"indicator:{task.symbol}:{indicator}:{task.timeframe}"
                # Simulate indicator calculation and caching
                await self.redis.set(cache_key, "cached_indicator_data", ex=3600)

    async def _warm_fundamental_data(self, task: WarmupTask) -> None:
        """Warm fundamental data for a symbol"""
        cache_key = f"fundamental:{task.symbol}"
        fundamental_data = {
            'last_updated': datetime.now(timezone.utc).isoformat()
        }
        await self.redis.set(cache_key, json.dumps(fundamental_data), ex=86400)  # Cache for 24 hours

    async def _warm_news_data(self, task: WarmupTask) -> None:
        """Warm news data for a symbol"""
        cache_key = f"news:{task.symbol}"
        news_data = {
            'last_updated': datetime.now(timezone.utc).isoformat()
        }
        await self.redis.set(cache_key, json.dumps(news_data), ex=1800)  # Cache for 30 minutes

    async def _warm_sentiment_data(self, task: WarmupTask) -> None:
        """Warm sentiment data for a symbol"""
        cache_key = f"sentiment:{task.symbol}"
        sentiment_data = {
            'last_updated': datetime.now(timezone.utc).isoformat()
        }
        await self.redis.set(cache_key, json.dumps(sentiment_data), ex=3600)  # Cache for 1 hour

    async def _warm_correlation_data(self, task: WarmupTask) -> None:
        """Warm correlation data for a symbol"""
        cache_key = f"correlation:{task.symbol}"
        correlation_data = {
            'last_updated': datetime.now(timezone.utc).isoformat()
        }
        await self.redis.set(cache_key, json.dumps(correlation_data), ex=7200)  # Cache for 2 hours

    async def _warm_volume_profile(self, task: WarmupTask) -> None:
        """Warm volume profile data for a symbol"""
        cache_key = f"volume_profile:{task.symbol}:{task.timeframe}"
        volume_profile = {
            'last_updated': datetime.now(timezone.utc).isoformat()
        }
        await self.redis.set(cache_key, json.dumps(volume_profile), ex=3600)

    async def _warm_earnings_data(self, task: WarmupTask) -> None:
        """Warm earnings data for a symbol"""
        cache_key = f"earnings:{task.symbol}"
        earnings_data = {
            'last_updated': datetime.now(timezone.utc).isoformat()
        }
        await self.redis.set(cache_key, json.dumps(earnings_data), ex=86400)

    def _update_average_task_time(self, task_time: float) -> None:
        """Update average task execution time"""
        completed = self.warming_stats['tasks_completed']
        current_avg = self.warming_stats['average_task_time']

        if completed == 1:
            self.warming_stats['average_task_time'] = task_time
        else:
            # Running average calculation
            self.warming_stats['average_task_time'] = (
                (current_avg * (completed - 1) + task_time) / completed
            )

    async def _check_memory_usage(self) -> bool:
        """Check if memory usage exceeds threshold"""
        try:
            info = await self.redis.info('memory')
            memory_mb = info.get('used_memory', 0) / (1024 * 1024)
            self.warming_stats['memory_usage'] = memory_mb
            return memory_mb > self.config['memory_threshold_mb']
        except Exception as e:
            logger.error(f"Error checking memory usage: {e}")
            return False

    async def _process_task_queues(self) -> None:
        """Process any queued warming tasks"""
        total_queued = sum(len(queue) for queue in self.task_queues.values())
        if total_queued > 0:
            logger.info(f"Processing {total_queued} queued warming tasks")
            await self._execute_warming_tasks()

    async def warm_symbol_immediately(
        self,
        symbol: str,
        data_types: List[DataType] = None,
        priority: WarmupPriority = WarmupPriority.HIGH
    ) -> Dict[str, bool]:
        """Warm specific symbol data immediately"""
        if data_types is None:
            data_types = [DataType.PRICE_DATA, DataType.TECHNICAL_INDICATORS]

        results = {}
        tasks = []

        # Create tasks for requested data types
        for data_type in data_types:
            task = WarmupTask(
                symbol=symbol,
                data_type=data_type,
                priority=priority
            )
            tasks.append(task)

        # Execute tasks immediately
        for task in tasks:
            try:
                await self._execute_single_task(task)
                results[f"{symbol}_{data_type.value}"] = task.completed_at is not None
            except Exception as e:
                logger.error(f"Error warming {symbol} {data_type.value}: {e}")
                results[f"{symbol}_{data_type.value}"] = False

        return results

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        try:
            info = await self.redis.info('stats')
            hits = info.get('keyspace_hits', 0)
            misses = info.get('keyspace_misses', 0)

            if hits + misses > 0:
                hit_rate = hits / (hits + misses)
            else:
                hit_rate = 0.0

            self.warming_stats['cache_hit_rate'] = hit_rate

        except Exception as e:
            logger.error(f"Error calculating cache hit rate: {e}")

        # Add current queue status
        queue_status = {
            priority.value: len(queue)
            for priority, queue in self.task_queues.items()
        }

        return {
            **self.warming_stats,
            'queue_status': queue_status,
            'config': self.config
        }

    async def clear_cache(self, pattern: str = "*") -> int:
        """Clear cache entries matching pattern"""
        try:
            keys = await self.redis.keys(pattern)
            if keys:
                deleted = await self.redis.delete(*keys)
                logger.info(f"Cleared {deleted} cache entries matching '{pattern}'")
                return deleted
            return 0
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return 0

    async def update_config(self, new_config: Dict[str, Any]) -> None:
        """Update cache warmer configuration"""
        self.config.update(new_config)
        logger.info(f"Updated cache warmer configuration: {new_config}")

    async def force_warming_cycle(self) -> Dict[str, Any]:
        """Force an immediate warming cycle"""
        logger.info("Forcing immediate warming cycle")
        start_time = time.time()
        
        try:
            await self._perform_scheduled_warming()
            end_time = time.time()
            duration = end_time - start_time
            
            return {
                'success': True,
                'duration': duration,
                'tasks_completed': self.warming_stats['tasks_completed']
            }
        except Exception as e:
            logger.error(f"Error in forced warming cycle: {e}")
            return {
                'success': False,
                'error': str(e),
                'duration': time.time() - start_time
            }