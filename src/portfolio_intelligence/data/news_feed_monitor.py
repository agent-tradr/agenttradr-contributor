from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
import asyncio
import logging
import re

logger = logging.getLogger(__name__)


class NewsSource:
    REUTERS = "reuters"
    BLOOMBERG = "bloomberg"
    MARKETWATCH = "marketwatch"
    YAHOO_FINANCE = "yahoo_finance"
    BENZINGA = "benzinga"
    SEEKING_ALPHA = "seeking_alpha"
    CNBC = "cnbc"
    FINANCIAL_TIMES = "financial_times"
    WALL_STREET_JOURNAL = "wall_street_journal"
    TWITTER_FINANCIAL = "twitter_financial"


class NewsPriority:
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    BREAKING = 5


class NewsCategory:
    EARNINGS = "earnings"
    ANALYST_RATING = "analyst_rating"
    MERGER_ACQUISITION = "merger_acquisition"
    REGULATORY = "regulatory"
    MANAGEMENT_CHANGE = "management_change"
    PRODUCT_LAUNCH = "product_launch"
    PARTNERSHIP = "partnership"
    MARKET_NEWS = "market_news"
    ECONOMIC_DATA = "economic_data"
    GEOPOLITICAL = "geopolitical"
    SECTOR_NEWS = "sector_news"
    TECHNICAL_ANALYSIS = "technical_analysis"


@dataclass
class NewsItem:
    item_id: str
    source: NewsSource
    headline: str
    summary: str
    full_text: Optional[str] = None
    symbols: List[str] = field(default_factory=list)
    category: NewsCategory = None
    priority: NewsPriority = None
    sentiment_score: float = 0.0  # -1.0 to 1.0
    relevance_score: float = 0.0  # 0.0 to 1.0
    published_time: datetime = None
    received_time: datetime = None
    url: Optional[str] = None
    author: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class NewsAlert:
    alert_id: str
    news_item: NewsItem
    affected_symbols: List[str] = field(default_factory=list)
    alert_reason: str = ""
    potential_impact: str = ""
    recommended_action: str = ""
    user_id: Optional[int] = None
    triggered_time: datetime = None


@dataclass
class NewsSourceConfig:
    source: NewsSource
    websocket_url: Optional[str] = None
    api_url: Optional[str] = None
    api_key: Optional[str] = None
    rate_limit: int = 100
    enabled: bool = True


class LiveNewsFeedMonitor:
    """
    Live News Feed Integration System
    
    Features:
    - Real-time WebSocket connections to news sources
    - Position-specific keyword alerts
    - Breaking news detection and instant notifications
    - Sentiment analysis integration
    - Market impact assessment
    - News categorization and prioritization
    - Duplicate detection and filtering
    """
    
    def __init__(self):
        from ...services.notification_service import NotificationService
        from ...cache.redis_cache import RedisCache
        
        self.cache = RedisCache()
        self.notification_service = NotificationService()
        self.is_monitoring = False
        
        # News source configurations
        self.source_configs = {
            NewsSource.REUTERS: NewsSourceConfig(
                source=NewsSource.REUTERS,
                websocket_url="wss://streams.reuters.com/financial-news",
                api_url="https://api.reuters.com/v1/news",
                rate_limit=1000,
                enabled=True
            ),
            NewsSource.BLOOMBERG: NewsSourceConfig(
                source=NewsSource.BLOOMBERG,
                api_url="https://api.bloomberg.com/v1/news",
                rate_limit=500,
                enabled=True
            ),
            NewsSource.MARKETWATCH: NewsSourceConfig(
                source=NewsSource.MARKETWATCH,
                api_url="https://api.marketwatch.com/v1/news",
                rate_limit=200,
                enabled=True
            ),
            NewsSource.YAHOO_FINANCE: NewsSourceConfig(
                source=NewsSource.YAHOO_FINANCE,
                api_url="https://query2.finance.yahoo.com/v1/finance/search",
                rate_limit=2000,
                enabled=True
            ),
            NewsSource.BENZINGA: NewsSourceConfig(
                source=NewsSource.BENZINGA,
                websocket_url="wss://api.benzinga.com/api/v2.1/news/stream",
                api_url="https://api.benzinga.com/api/v2.1/news",
                rate_limit=1000,
                enabled=True
            )
        }
        
        # Monitoring keywords and patterns
        self.critical_keywords = [
            'breaking', 'urgent', 'alert', 'halted', 'suspended',
            'bankruptcy', 'merger', 'acquisition', 'fraud', 'investigation'
        ]
        self.breaking_news_patterns = [
            r'\bBREAKING\b',
            r'\bUPDATE\b',
            r'\bALERT\b',
            r'\bURGENT\b',
        ]
        
        # Position tracking
        self.monitored_positions: Dict[int, Dict[str, Dict]] = {}  # user_id -> symbol -> position_data
        self.position_keywords: Dict[str, Set[str]] = {}  # symbol -> keywords
        
        # News processing
        self.news_history: List[NewsItem] = []
        self.active_alerts: List[NewsAlert] = []
        self.seen_news_ids: Set[str] = set()
        
        # WebSocket connections
        self.websocket_connections = {}
        
        # Alert callbacks
        self.alert_callbacks = []
        
        # Performance metrics
        self.metrics = {
            'news_items_processed': 0,
            'duplicate_news_filtered': 0,
            'breaking_news_detected': 0,
            'alerts_generated': 0,
            'position_alerts_sent': 0,
            'sources_connected': 0
        }

    async def start_monitoring(self) -> None:
        """Start the live news feed monitoring system"""
        logger.info("Starting live news feed monitoring")
        self.is_monitoring = True
        try:
            # Load monitored positions
            await self._load_monitored_positions()
            
            # Start monitoring tasks
            monitoring_tasks = [
                self._monitor_websocket_feeds(),
                self._poll_api_feeds(),
                self._process_news_queue(),
                self._cleanup_old_news(),
                self._update_position_keywords()
            ]
            await asyncio.gather(*monitoring_tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error in news monitoring system: {e}")
            self.is_monitoring = False
            raise
    
    async def stop_monitoring(self) -> None:
        """Stop the live news feed monitoring system"""
        self.is_monitoring = False
        
        # Close WebSocket connections
        for source, ws in self.websocket_connections.items():
            if ws:
                try:
                    await ws.close()
                except Exception:
                    pass
    
    async def add_position_monitor(self, user_id: int, symbol: str, position_data: Dict) -> None:
        """Add a position to be monitored for news alerts"""
        if user_id not in self.monitored_positions:
            self.monitored_positions[user_id] = {}
        self.monitored_positions[user_id][symbol] = position_data
        
        # Generate keywords for this symbol
        keywords = await self._generate_position_keywords(symbol, position_data)
        self.position_keywords[symbol] = keywords
        
        logger.info(f"Added news monitoring for {user_id}/{symbol} with {len(keywords)} keywords")
    
    async def remove_position_monitor(self, user_id: int, symbol: str) -> None:
        """Remove a position from news monitoring"""
        if user_id in self.monitored_positions and symbol in self.monitored_positions[user_id]:
            del self.monitored_positions[user_id][symbol]
            
        # Remove keywords if no other users monitor this symbol
        symbol_monitored = any(
            symbol in positions 
            for positions in self.monitored_positions.values()
        )
        if not symbol_monitored and symbol in self.position_keywords:
            del self.position_keywords[symbol]

    def register_alert_callback(self, callback) -> None:
        """Register a callback function to be called when news alerts are generated"""
        self.alert_callbacks.append(callback)

    async def _load_monitored_positions(self) -> None:
        """Load monitored positions from database"""
        try:
            from ...database.connection import get_db
            from ...models.portfolio import PortfolioPosition
            
            async with get_db() as session:
                positions = session.query(PortfolioPosition).filter(
                    PortfolioPosition.quantity != 0,
                    PortfolioPosition.status == 'active'
                ).all()
                
                for pos in positions:
                    await self.add_position_monitor(
                        pos.user_id,
                        pos.symbol,
                        {
                            'quantity': float(pos.quantity),
                            'avg_cost': float(pos.avg_cost_basis),
                            'sector': getattr(pos, 'sector', 'Unknown')
                        }
                    )
        except Exception as e:
            logger.error(f"Error loading monitored positions: {e}")

    async def _generate_position_keywords(self, symbol: str, position_data: Dict) -> Set[str]:
        """Generate keywords to monitor for a specific position"""
        keywords = {symbol.lower(), symbol.upper()}
        
        # Add company name variations (would fetch from stock universe)
        company_name = await self._get_company_name(symbol)
        if company_name:
            keywords.add(company_name.lower())
            # Add variations like "Apple Inc", "Apple", "AAPL"
            keywords.update(company_name.split())
        
        # Add sector-specific keywords
        sector = position_data.get('sector', '')
        if sector:
            keywords.add(sector.lower())
        
        return keywords

    async def _get_company_name(self, symbol: str) -> Optional[str]:
        """Get company name for a symbol - would integrate with stock universe"""
        # Check cache first
        cached_name = await self.cache.get(f"company_name:{symbol}")
        if cached_name:
            return cached_name
        
        # In production, would fetch from stock universe or external API
        # For now, return None - would be populated from actual data source
        return None

    async def _monitor_websocket_feeds(self) -> None:
        """Monitor WebSocket feeds from news sources"""
        while self.is_monitoring:
            try:
                # Connect to enabled WebSocket sources
                for source, config in self.source_configs.items():
                    if (config.enabled and config.websocket_url and 
                        source not in self.websocket_connections):
                        await self._connect_websocket_source(source, config)
                        
                await asyncio.sleep(1.0)
            except Exception as e:
                logger.error(f"Error monitoring WebSocket feeds: {e}")
                await asyncio.sleep(5.0)

    async def _connect_websocket_source(self, source: NewsSource, config: NewsSourceConfig) -> None:
        """Connect to a WebSocket news source"""
        try:
            # For now, simulate connection
            logger.info(f"Connecting to {source} WebSocket feed")
            # Simulate WebSocket connection
            self.websocket_connections[source] = "simulated_connection"
            self.metrics['sources_connected'] += 1
            
            # In real implementation:
            # ws = await websockets.connect(config.websocket_url)
            # self.websocket_connections[source] = ws
            # asyncio.create_task(self._handle_websocket_messages(source, ws))
        except Exception as e:
            logger.error(f"Error connecting to {source}: {e}")

    async def _poll_api_feeds(self) -> None:
        """Poll API feeds from news sources"""
        while self.is_monitoring:
            try:
                for source, config in self.source_configs.items():
                    if config.enabled and config.api_url:
                        await self._poll_api_source(source, config)
                        
                await asyncio.sleep(60.0)  # Poll every minute
            except Exception as e:
                logger.error(f"Error polling API feeds: {e}")
                await asyncio.sleep(60.0)

    async def _poll_api_source(self, source: NewsSource, config: NewsSourceConfig) -> None:
        """Poll a specific API source for news"""
        try:
            monitored_symbols = set()
            for positions in self.monitored_positions.values():
                monitored_symbols.update(positions.keys())
                
            for symbol in list(monitored_symbols)[:10]:  # Limit to 10 symbols per poll
                await self._fetch_symbol_news(source, config, symbol)
        except Exception as e:
            logger.error(f"Error polling {source}: {e}")

    async def _fetch_symbol_news(self, source: NewsSource, config: NewsSourceConfig, symbol: str) -> None:
        """Fetch news for a specific symbol from an API source"""
        try:
            # Simulate news fetching - in production would make actual API calls
            if source == NewsSource.YAHOO_FINANCE:
                news_items = await self._simulate_yahoo_news(symbol)
            elif source == NewsSource.BENZINGA:
                news_items = await self._simulate_benzinga_news(symbol)
            else:
                news_items = await self._simulate_generic_news(source, symbol)
            
            # Process fetched news
            for news_item in news_items:
                await self._process_news_item(news_item)
        except Exception as e:
            logger.error(f"Error fetching news for {symbol} from {source}: {e}")

    async def _simulate_yahoo_news(self, symbol: str) -> List[NewsItem]:
        """Simulate Yahoo Finance news - placeholder for actual API integration"""
        return []  # Return empty for simulation
    
    async def _simulate_benzinga_news(self, symbol: str) -> List[NewsItem]:
        """Simulate Benzinga news - placeholder for actual API integration"""
        return []  # Return empty for simulation
    
    async def _simulate_generic_news(self, source: NewsSource, symbol: str) -> List[NewsItem]:
        """Simulate generic news - placeholder for actual API integration"""
        return []  # Return empty for simulation

    async def _process_news_queue(self) -> None:
        """Process queued news items from cache"""
        while self.is_monitoring:
            try:
                # Check for queued news items from cache
                queued_items = await self.cache.get("news_processing_queue")
                if queued_items:
                    for item_data in queued_items:
                        news_item = NewsItem(**item_data)
                        await self._process_news_item(news_item)
                    
                    # Clear processed items
                    await self.cache.delete("news_processing_queue")
                
                await asyncio.sleep(2.0)  # Process every 2 seconds
            except Exception as e:
                logger.error(f"Error processing news queue: {e}")
                await asyncio.sleep(5.0)

    async def _process_news_item(self, news_item: NewsItem) -> None:
        """Process a single news item"""
        try:
            # Check for duplicates
            if news_item.item_id in self.seen_news_ids:
                self.metrics['duplicate_news_filtered'] += 1
                return
                
            self.seen_news_ids.add(news_item.item_id)
            self.metrics['news_items_processed'] += 1
            
            # Add to history
            self.news_history.append(news_item)
            
            # Cache the news item
            await self.cache.set(
                f"news_item:{news_item.item_id}",
                news_item.__dict__,
                ttl=86400 * 7  # 7 days
            )
            
            # Check for breaking news
            if await self._is_breaking_news(news_item):
                self.metrics['breaking_news_detected'] += 1
                await self._handle_breaking_news(news_item)
            
            # Check position-specific alerts
            affected_symbols = await self._find_affected_positions(news_item)
            if affected_symbols:
                await self._generate_position_alerts(news_item, affected_symbols)
            
            # Update news sentiment cache
            await self._update_sentiment_cache(news_item)
            
        except Exception as e:
            logger.error(f"Error processing news item {news_item.item_id}: {e}")

    async def _is_breaking_news(self, news_item: NewsItem) -> bool:
        """Check if a news item is breaking news"""
        # Check priority level
        if news_item.priority and news_item.priority >= NewsPriority.CRITICAL:
            return True
        
        # Check headline patterns
        for pattern in self.breaking_news_patterns:
            if re.search(pattern, news_item.headline, re.IGNORECASE):
                return True
        
        # Check for multiple critical keywords
        headline_lower = news_item.headline.lower()
        critical_matches = sum(1 for keyword in self.critical_keywords if keyword in headline_lower)
        return critical_matches >= 2  # 2+ critical keywords = breaking

    async def _handle_breaking_news(self, news_item: NewsItem) -> None:
        """Handle breaking news by sending immediate notifications"""
        try:
            # Send immediate notification to all users with affected positions
            affected_users = set()
            for symbol in news_item.symbols:
                for user_id, positions in self.monitored_positions.items():
                    if symbol in positions:
                        affected_users.add(user_id)
            
            for user_id in affected_users:
                await self.notification_service.send_notification(
                    user_id=user_id,
                    title=f"🚨 BREAKING: {': '.join(news_item.symbols)}",
                    message=f"{news_item.headline}\n\nSource: {news_item.source}",
                    channels=["SMS", "EMAIL"],
                    priority="urgent"
                )
            
            # Cache as breaking news
            await self.cache.set(
                f"news_breaking:{news_item.item_id}",
                news_item.__dict__,
                ttl=3600  # 1 hour
            )
            
        except Exception as e:
            logger.error(f"Error handling breaking news: {e}")

    async def _find_affected_positions(self, news_item: NewsItem) -> List[str]:
        """Find positions affected by a news item"""
        affected_symbols = []
        
        # Direct symbol matches
        for symbol in news_item.symbols:
            if any(symbol in positions for positions in self.monitored_positions.values()):
                affected_symbols.append(symbol)
        
        # Keyword-based matches
        headline_text = f"{news_item.headline} {news_item.summary}".lower()
        for symbol, keywords in self.position_keywords.items():
            for keyword in keywords:
                if keyword.lower() in headline_text:
                    if symbol not in affected_symbols:
                        affected_symbols.append(symbol)
                    break
        
        return affected_symbols

    async def _generate_position_alerts(self, news_item: NewsItem, affected_symbols: List[str]) -> None:
        """Generate alerts for affected positions"""
        try:
            for symbol in affected_symbols:
                # Find users with this position
                affected_users = []
                for user_id, positions in self.monitored_positions.items():
                    if symbol in positions:
                        affected_users.append(user_id)
                
                # Assess potential impact
                impact = await self._assess_news_impact(news_item, symbol)
                
                # Generate alert
                alert = NewsAlert(
                    alert_id=f"alert_{news_item.item_id}_{symbol}",
                    news_item=news_item,
                    affected_symbols=[symbol],
                    alert_reason=f"News affecting {symbol} position",
                    potential_impact=impact['description'],
                    recommended_action=impact['action'],
                    triggered_time=datetime.now(timezone.utc)
                )
                
                # Send alerts to affected users
                for user_id in affected_users:
                    alert.user_id = user_id
                    await self._send_position_alert(alert)
                
                self.active_alerts.append(alert)
                self.metrics['alerts_generated'] += 1
                
        except Exception as e:
            logger.error(f"Error generating position alerts: {e}")

    async def _assess_news_impact(self, news_item: NewsItem, symbol: str) -> Dict[str, str]:
        """Assess the potential impact of news on a position"""
        impact_level = "MEDIUM"
        action = "MONITOR"
        
        # High impact categories
        if news_item.category in [NewsCategory.EARNINGS, NewsCategory.MERGER_ACQUISITION]:
            impact_level = "HIGH"
            action = "REVIEW POSITION"
        
        # Very negative sentiment
        if news_item.sentiment_score < -0.7:
            impact_level = "HIGH NEGATIVE"
            action = "CONSIDER REDUCING POSITION"
        # Very positive sentiment
        elif news_item.sentiment_score > 0.7:
            impact_level = "HIGH POSITIVE"
            action = "CONSIDER INCREASING POSITION"
        
        return {
            'description': impact_level,
            'action': action
        }

    async def _send_position_alert(self, alert: NewsAlert) -> None:
        """Send a position alert to a user"""
        try:
            symbol = alert.affected_symbols[0]
            message = f"""
News: {alert.news_item.headline}
Source: {alert.news_item.source}
Sentiment: {alert.news_item.sentiment_score:.2f}
Impact: {alert.potential_impact}
Recommendation: {alert.recommended_action}
Published: {alert.news_item.published_time.strftime('%H:%M:%S') if alert.news_item.published_time else 'N/A'}
"""
            
            # Determine notification priority
            priority = "high" if alert.news_item.priority and alert.news_item.priority >= 4 else "medium"
            
            await self.notification_service.send_notification(
                user_id=alert.user_id,
                title=f"📰 News Alert: {symbol}",
                message=message,
                channels=["IN_APP"],
                priority=priority
            )
            
            self.metrics['position_alerts_sent'] += 1
            
            # Call registered callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error sending position alert: {e}")

    async def _update_sentiment_cache(self, news_item: NewsItem) -> None:
        """Update sentiment cache for affected symbols"""
        try:
            for symbol in news_item.symbols:
                # Get current sentiment data
                current_data = await self.cache.get(f"news_sentiment:{symbol}") or {
                    'scores': [],
                    'updated': datetime.now(timezone.utc).isoformat()
                }
                
                # Add new sentiment score
                current_data['scores'].append({
                    'score': news_item.sentiment_score,
                    'timestamp': news_item.published_time.isoformat() if news_item.published_time else datetime.now(timezone.utc).isoformat()
                })
                
                # Keep only last 24 hours of sentiment
                cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
                current_data['scores'] = [
                    s for s in current_data['scores']
                    if datetime.fromisoformat(s['timestamp']) > cutoff
                ]
                
                # Calculate average sentiment
                if current_data['scores']:
                    current_data['avg_score'] = sum(s['score'] for s in current_data['scores']) / len(current_data['scores'])
                    current_data['updated'] = datetime.now(timezone.utc).isoformat()
                
                # Cache updated sentiment
                await self.cache.set(f"news_sentiment:{symbol}", current_data, ttl=86400)
                
        except Exception as e:
            logger.error(f"Error updating sentiment cache: {e}")

    async def _cleanup_old_news(self) -> None:
        """Clean up old news items and alerts"""
        while self.is_monitoring:
            try:
                cutoff_time = datetime.now(timezone.utc) - timedelta(days=7)
                
                # Clean up old news items
                self.news_history = [
                    item for item in self.news_history
                    if item.received_time and item.received_time > cutoff_time
                ]
                
                # Clean up active alerts (older than 24 hours)
                alert_cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
                self.active_alerts = [
                    alert for alert in self.active_alerts
                    if alert.triggered_time and alert.triggered_time > alert_cutoff
                ]
                
                # Clean up seen news IDs (keep last 10000)
                if len(self.seen_news_ids) > 10000:
                    # Keep most recent news IDs based on recent news items
                    recent_ids = {item.item_id for item in self.news_history[-5000:]}
                    self.seen_news_ids = recent_ids
                
                await asyncio.sleep(3600)  # Clean up every hour
                
            except Exception as e:
                logger.error(f"Error in cleanup: {e}")
                await asyncio.sleep(3600)

    async def _update_position_keywords(self) -> None:
        """Update position keywords periodically"""
        while self.is_monitoring:
            try:
                # Reload monitored positions
                await self._load_monitored_positions()
                await asyncio.sleep(300)  # Update every 5 minutes
            except Exception as e:
                logger.error(f"Error updating position keywords: {e}")
                await asyncio.sleep(300)

    # Public API methods
    async def get_recent_news(self, symbols: Optional[List[str]] = None, limit: int = 10) -> List[NewsItem]:
        """Get recent news items, optionally filtered by symbols"""
        news_items = self.news_history.copy()
        
        # Filter by symbols if provided
        if symbols:
            symbol_set = set(symbols)
            news_items = [
                item for item in news_items
                if any(symbol in symbol_set for symbol in item.symbols)
            ]
        
        # Sort by recency and return limited results
        news_items.sort(key=lambda x: x.received_time or datetime.min, reverse=True)
        return news_items[:limit]

    async def get_active_alerts(self, user_id: Optional[int] = None) -> List[NewsAlert]:
        """Get active alerts, optionally filtered by user"""
        alerts = self.active_alerts.copy()
        if user_id:
            alerts = [alert for alert in alerts if alert.user_id == user_id]
        return alerts

    async def get_symbol_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get sentiment data for a symbol"""
        return await self.cache.get(f"news_sentiment:{symbol}") or {}

    async def get_breaking_news(self, limit: int = 5) -> List[NewsItem]:
        """Get recent breaking news items"""
        breaking_items = []
        for item in self.news_history:
            if await self._is_breaking_news(item):
                breaking_items.append(item)
        
        breaking_items.sort(key=lambda x: x.received_time or datetime.min, reverse=True)
        return breaking_items[:limit]

    async def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring system statistics"""
        return {
            'is_monitoring': self.is_monitoring,
            'monitored_positions': sum(len(positions) for positions in self.monitored_positions.values()),
            'monitored_symbols': len(self.position_keywords),
            'active_alerts': len(self.active_alerts),
            'news_history_size': len(self.news_history),
            **self.metrics
        }


# Global news monitor instance
_news_monitor_instance: Optional[LiveNewsFeedMonitor] = None


async def get_news_monitor() -> LiveNewsFeedMonitor:
    """Get or create the global news monitor instance"""
    global _news_monitor_instance
    if _news_monitor_instance is None:
        _news_monitor_instance = LiveNewsFeedMonitor()
    return _news_monitor_instance


async def start_global_news_monitoring() -> None:
    """Start the global news monitoring system"""
    monitor = await get_news_monitor()
    await monitor.start_monitoring()


async def stop_global_news_monitoring() -> None:
    """Stop the global news monitoring system"""
    global _news_monitor_instance
    if _news_monitor_instance:
        await _news_monitor_instance.stop_monitoring()
        _news_monitor_instance = None