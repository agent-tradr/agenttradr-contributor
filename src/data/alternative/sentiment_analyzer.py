from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from collections import Counter, deque
import logging
import asyncio
import re
import warnings

# Optional scientific computing imports
try:
    import numpy as np
except ImportError:
    # Provide basic numpy-like functions
    class MockNumpy:
        @staticmethod
        def mean(values):
            return sum(values) / len(values) if values else 0
        
        @staticmethod
        def std(values):
            if not values:
                return 0
            mean_val = sum(values) / len(values)
            variance = sum((x - mean_val) ** 2 for x in values) / len(values)
            return variance ** 0.5
        
        @staticmethod
        class random:
            @staticmethod
            def normal(mean=0, std=1):
                # Simple approximation for normal distribution
                import random
                return random.gauss(mean, std)
    
    np = MockNumpy()

try:
    import pandas as pd
except ImportError:
    pd = None  # pandas not used in current implementation

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class SentimentType(Enum):
    SOCIAL_MEDIA = "social_media"
    NEWS_ARTICLES = "news_articles"
    ANALYST_REPORTS = "analyst_reports"
    EARNINGS_CALLS = "earnings_calls"
    REGULATORY_FILINGS = "regulatory_filings"
    MARKET_MICROSTRUCTURE = "market_microstructure"


class SentimentTrend(Enum):
    STRONGLY_POSITIVE = "strongly_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    STRONGLY_NEGATIVE = "strongly_negative"
    IMPROVING = "improving"
    DETERIORATING = "deteriorating"
    VOLATILE = "volatile"


@dataclass
class SentimentScore:
    symbol: str
    timestamp: datetime
    sentiment_type: SentimentType
    score: float  # -1.0 (very negative) to 1.0 (very positive)
    confidence: float  # 0.0 to 1.0
    volume: int  # Number of mentions/sources
    keywords: List[str]
    source_breakdown: Dict[str, float]
    raw_text_sample: Optional[str] = None


@dataclass
class SentimentSignal:
    symbol: str
    timestamp: datetime
    overall_sentiment: float
    sentiment_trend: SentimentTrend
    confidence_level: float
    volume_weighted_sentiment: float
    sentiment_momentum: float  # Rate of change
    sentiment_divergence: float  # Vs price action
    component_scores: Dict[SentimentType, float]
    significant_events: List[str]
    trading_recommendation: str
    risk_alerts: List[str]


class SentimentAnalyzer:
    """
    Advanced sentiment analysis engine
    Features:
    - Social media sentiment (Twitter, Reddit, StockTwits)
    - News article analysis
    - Analyst report sentiment
    - Market microstructure sentiment
    """

    def __init__(self, db_adapter, cache, lookback_hours=24):
        self.db = db_adapter
        self.cache = cache
        self.lookback_hours = lookback_hours
        
        # Sentiment configuration
        self.SENTIMENT_WEIGHTS = {
            SentimentType.NEWS_ARTICLES: 0.35,
            SentimentType.SOCIAL_MEDIA: 0.25,
            SentimentType.ANALYST_REPORTS: 0.20,
            SentimentType.EARNINGS_CALLS: 0.15,
            SentimentType.REGULATORY_FILINGS: 0.03,
            SentimentType.MARKET_MICROSTRUCTURE: 0.02,
        }
        
        # Thresholds
        self.STRONG_SENTIMENT_THRESHOLD = 0.6
        self.HIGH_VOLUME_THRESHOLD = 100
        self.MOMENTUM_THRESHOLD = 0.3
        self.DIVERGENCE_THRESHOLD = 0.4
        
        # Historical data storage
        self.sentiment_history = {}  # symbol -> deque of scores
        self.price_history = {}      # symbol -> deque of prices
        
        # NLP components (simplified - would use advanced models in production)
        self.positive_keywords = {
            'bullish', 'positive', 'growth', 'profit', 'earnings', 'beat',
            'strong', 'outperform', 'buy', 'upgrade', 'optimistic'
        }
        self.negative_keywords = {
            'bearish', 'negative', 'decline', 'loss', 'miss', 'weak',
            'underperform', 'sell', 'downgrade', 'pessimistic'
        }

    async def analyze_sentiment(self, symbols: List[str], data_points: List[Any]) -> Dict[str, SentimentSignal]:
        """
        Comprehensive sentiment analysis for given symbols
        
        Args:
            symbols: List of symbols to analyze
            data_points: Raw data points from various sources
            
        Returns:
            Dictionary mapping symbol -> sentiment signal
        """
        try:
            logger.info(f"Analyzing sentiment for {len(symbols)} symbols with {len(data_points)} data points")
            
            # Group data points by symbol and type
            grouped_data = self._group_data_by_symbol_and_type(data_points)
            sentiment_signals = {}
            
            for symbol in symbols:
                try:
                    # Get data for this symbol
                    symbol_data = grouped_data.get(symbol, {})
                    if not symbol_data:
                        logger.debug(f"No sentiment data available for {symbol}")
                        continue
                        
                    # Calculate individual sentiment scores
                    sentiment_scores = await self._calculate_sentiment_scores(symbol, symbol_data)
                    
                    # Aggregate into overall signal
                    sentiment_signal = await self._generate_sentiment_signal(symbol, sentiment_scores)
                    sentiment_signals[symbol] = sentiment_signal
                    
                    # Update historical tracking
                    await self._update_sentiment_history(symbol, sentiment_signal)
                    
                except Exception as e:
                    logger.error(f"Sentiment analysis failed for {symbol}: {e}")
                    continue
                    
            logger.info(f"Sentiment analysis completed for {len(sentiment_signals)} symbols")
            return sentiment_signals
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {}

    def _group_data_by_symbol_and_type(self, data_points: List[Any]) -> Dict[str, Dict[SentimentType, List[Any]]]:
        """Group data points by symbol and sentiment type"""
        grouped = {}
        
        for point in data_points:
            if not hasattr(point, 'symbol'):  # Skip non-symbol-specific data
                continue
                
            if point.symbol not in grouped:
                grouped[point.symbol] = {}
                
            # Map data source type to sentiment type
            sentiment_type = self._map_data_type_to_sentiment_type(point.data_type)
            
            if sentiment_type not in grouped[point.symbol]:
                grouped[point.symbol][sentiment_type] = []
                
            grouped[point.symbol][sentiment_type].append(point)
            
        return grouped

    def _map_data_type_to_sentiment_type(self, data_type: Any) -> SentimentType:
        """Map data source type to sentiment type"""
        mapping = {
            # Assuming data_type has these values - adjust based on actual implementation
            'news': SentimentType.NEWS_ARTICLES,
            'social': SentimentType.SOCIAL_MEDIA,
            'analyst': SentimentType.ANALYST_REPORTS,
            'earnings': SentimentType.EARNINGS_CALLS,
            'regulatory': SentimentType.REGULATORY_FILINGS,
            'microstructure': SentimentType.MARKET_MICROSTRUCTURE,
        }
        return mapping.get(data_type, SentimentType.SOCIAL_MEDIA)

    async def _calculate_sentiment_scores(self, symbol: str, symbol_data: Dict[SentimentType, List[Any]]) -> List[SentimentScore]:
        """Calculate sentiment scores for each sentiment type"""
        sentiment_scores = []
        
        for sentiment_type, data_points in symbol_data.items():
            try:
                # Process data points for this sentiment type
                processed_score = await self._process_sentiment_data(sentiment_type, data_points)
                
                if processed_score:
                    sentiment_score = SentimentScore(
                        symbol=symbol,
                        timestamp=datetime.now(timezone.utc),
                        sentiment_type=sentiment_type,
                        score=processed_score['score'],
                        confidence=processed_score['confidence'],
                        volume=processed_score['volume'],
                        keywords=processed_score['keywords'],
                        source_breakdown=processed_score['source_breakdown'],
                        raw_text_sample=processed_score.get('raw_text_sample')
                    )
                    sentiment_scores.append(sentiment_score)
                    
            except Exception as e:
                logger.error(f"Sentiment score calculation failed for {sentiment_type}: {e}")
                pass
                
        return sentiment_scores

    async def _process_sentiment_data(self, sentiment_type: SentimentType, data_points: List[Any]) -> Optional[Dict]:
        """Process sentiment data based on type"""
        try:
            if not data_points:
                return None
                
            if sentiment_type == SentimentType.SOCIAL_MEDIA:
                return await self._process_social_media_sentiment(data_points)
            elif sentiment_type == SentimentType.NEWS_ARTICLES:
                return await self._process_news_sentiment(data_points)
            elif sentiment_type == SentimentType.MARKET_MICROSTRUCTURE:
                return await self._process_microstructure_sentiment(data_points)
            else:
                return await self._process_generic_sentiment(data_points)
                
        except Exception as e:
            logger.error(f"Sentiment data processing failed: {e}")
            return None

    async def _process_social_media_sentiment(self, data_points: List[Any]) -> Dict:
        """Process social media sentiment data"""
        try:
            scores = []
            volumes = []
            keywords_counter = Counter()
            source_scores = {}
            
            for point in data_points:
                if isinstance(point.value, (int, float)):
                    scores.append(float(point.value))
                    volumes.append(getattr(point, 'volume', 1))
                    
                    # Extract keywords from metadata
                    if hasattr(point, 'metadata') and 'keywords' in point.metadata:
                        keywords_counter.update(point.metadata['keywords'])
                        
                    # Track source breakdown
                    source = getattr(point, 'source', 'unknown')
                    if source not in source_scores:
                        source_scores[source] = []
                    source_scores[source].append(float(point.value))
                    
            if not scores:
                return None
                
            # Calculate volume-weighted average
            total_volume = sum(volumes)
            if total_volume > 0:
                weighted_score = sum(s * v for s, v in zip(scores, volumes)) / total_volume
            else:
                weighted_score = np.mean(scores)
                
            # Calculate confidence based on volume and consistency
            score_std = np.std(scores)
            confidence = min(1.0, total_volume / 1000) * (1.0 - min(1.0, score_std))
            
            # Top keywords
            top_keywords = [word for word, count in keywords_counter.most_common(10)]
            
            # Source breakdown averages
            source_breakdown = {
                source: float(np.mean(scores))
                for source, scores in source_scores.items()
            }
            
            return {
                'score': float(weighted_score),
                'confidence': float(confidence),
                'volume': total_volume,
                'keywords': top_keywords,
                'source_breakdown': source_breakdown
            }
            
        except Exception as e:
            logger.error(f"Social media sentiment processing failed: {e}")
            return None

    async def _process_news_sentiment(self, data_points: List[Any]) -> Dict:
        """Process news sentiment data"""
        try:
            scores = []
            impact_weights = []
            keywords_counter = Counter()
            source_scores = {}
            headlines = []
            
            for point in data_points:
                if isinstance(point.value, (int, float)):
                    scores.append(float(point.value))
                    
                    impact_score = getattr(point, 'impact_score', 0.5)
                    impact_weights.append(impact_score)
                    
                    # Extract keywords from headline
                    headline = getattr(point, 'headline', '')
                    headlines.append(headline)
                    if headline:
                        keywords = self._extract_keywords_from_text(headline)
                        keywords_counter.update(keywords)
                        
                    # Track source breakdown
                    source = getattr(point, 'source', 'unknown')
                    if source not in source_scores:
                        source_scores[source] = []
                    source_scores[source].append(float(point.value))
                    
            if not scores:
                return None
                
            # Calculate impact-weighted average
            total_weight = sum(impact_weights)
            if total_weight > 0:
                weighted_score = sum(s * w for s, w in zip(scores, impact_weights)) / total_weight
            else:
                weighted_score = np.mean(scores)
                
            # Confidence based on number of articles and source diversity
            confidence = min(1.0, len(scores) / 20) * min(1.0, len(source_scores) / 3)
            
            # Top keywords
            top_keywords = [word for word, count in keywords_counter.most_common(10)]
            
            # Source breakdown
            source_breakdown = {
                source: float(np.mean(scores))
                for source, scores in source_scores.items()
            }
            
            return {
                'score': float(weighted_score),
                'confidence': float(confidence),
                'volume': len(scores),
                'keywords': top_keywords,
                'source_breakdown': source_breakdown,
                'raw_text_sample': headlines[0] if headlines else None
            }
            
        except Exception as e:
            logger.error(f"News sentiment processing failed: {e}")
            return None

    async def _process_microstructure_sentiment(self, data_points: List[Any]) -> Dict:
        """Process market microstructure sentiment data"""
        try:
            # Microstructure signals: order flow, bid-ask dynamics, etc.
            scores = []
            
            for point in data_points:
                if isinstance(point.value, (int, float)):
                    # Convert microstructure data to sentiment score
                    raw_value = float(point.value)
                    # Normalize to -1 to 1 scale (simplified)
                    normalized_score = max(-1.0, min(1.0, raw_value / 100.0))
                    scores.append(normalized_score)
                    
            if not scores:
                return None
                
            avg_score = float(np.mean(scores))
            confidence = 0.8  # High confidence for market-based signals
            
            return {
                'score': avg_score,
                'confidence': confidence,
                'volume': len(scores),
                'keywords': ['order_flow', 'bid_ask', 'microstructure'],
                'source_breakdown': {'microstructure': avg_score}
            }
            
        except Exception as e:
            logger.error(f"Microstructure sentiment processing failed: {e}")
            return None

    async def _process_generic_sentiment(self, data_points: List[Any]) -> Dict:
        """Process generic sentiment data"""
        try:
            scores = [float(point.value) for point in data_points 
                     if isinstance(point.value, (int, float))]
            
            if not scores:
                return None
                
            return {
                'score': float(np.mean(scores)),
                'confidence': 0.5,
                'volume': len(scores),
                'keywords': [],
                'source_breakdown': {'generic': np.mean(scores)}
            }
            
        except Exception as e:
            logger.error(f"Generic sentiment processing failed: {e}")
            return None

    def _extract_keywords_from_text(self, text: str) -> List[str]:
        """Extract relevant keywords from text"""
        try:
            # Convert to lowercase and remove punctuation
            clean_text = re.sub(r'[^\w\s]', '', text.lower())
            words = clean_text.split()
            
            # Filter for relevant keywords
            keywords = []
            
            # Check for positive/negative keywords
            for word in words:
                if word in self.positive_keywords or word in self.negative_keywords:
                    keywords.append(word)
                    
            # Add financial terms
            financial_terms = ['earnings', 'revenue', 'profit', 'loss', 'growth', 'decline']
            for word in words:
                if word in financial_terms:
                    keywords.append(word)
                    
            return list(set(keywords))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            return []

    async def _generate_sentiment_signal(self, symbol: str, sentiment_scores: List[SentimentScore]) -> SentimentSignal:
        """Generate overall sentiment signal from individual scores"""
        try:
            if not sentiment_scores:
                return self._create_neutral_sentiment_signal(symbol)
                
            # Calculate weighted overall sentiment
            weighted_scores = []
            total_weight = 0.0
            component_scores = {}
            
            for score in sentiment_scores:
                weight = self.SENTIMENT_WEIGHTS.get(score.sentiment_type, 0.1)
                weighted_scores.append(score.score * weight * score.confidence)
                total_weight += weight * score.confidence
                component_scores[score.sentiment_type] = score.score
                
            if total_weight > 0:
                overall_sentiment = sum(weighted_scores) / total_weight
            else:
                overall_sentiment = 0.0
                
            # Calculate volume-weighted sentiment
            volume_weighted_sentiment = self._calculate_volume_weighted_sentiment(sentiment_scores)
            
            # Determine sentiment trend
            sentiment_trend = self._determine_sentiment_trend(overall_sentiment)
            
            # Calculate sentiment momentum
            sentiment_momentum = await self._calculate_sentiment_momentum(symbol, overall_sentiment)
            
            # Calculate price-sentiment divergence
            sentiment_divergence = await self._calculate_sentiment_divergence(symbol, overall_sentiment)
            
            # Calculate confidence level
            confidence_level = self._calculate_overall_confidence(sentiment_scores)
            
            # Detect significant events
            significant_events = self._detect_significant_events(sentiment_scores)
            
            # Generate trading recommendation
            trading_recommendation = self._generate_trading_recommendation(
                overall_sentiment, confidence_level, sentiment_momentum, sentiment_divergence
            )
            
            # Generate risk alerts
            risk_alerts = self._generate_risk_alerts(
                overall_sentiment, sentiment_momentum, sentiment_divergence, confidence_level
            )
            
            return SentimentSignal(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                overall_sentiment=overall_sentiment,
                sentiment_trend=sentiment_trend,
                confidence_level=confidence_level,
                volume_weighted_sentiment=volume_weighted_sentiment,
                sentiment_momentum=sentiment_momentum,
                sentiment_divergence=sentiment_divergence,
                component_scores=component_scores,
                significant_events=significant_events,
                trading_recommendation=trading_recommendation,
                risk_alerts=risk_alerts
            )
            
        except Exception as e:
            logger.error(f"Sentiment signal generation failed for {symbol}: {e}")
            return self._create_neutral_sentiment_signal(symbol)

    def _calculate_volume_weighted_sentiment(self, sentiment_scores: List[SentimentScore]) -> float:
        """Calculate volume-weighted sentiment"""
        try:
            weighted_sum = 0.0
            total_volume = 0
            
            for score in sentiment_scores:
                weighted_sum += score.score * score.volume
                total_volume += score.volume
                
            if total_volume > 0:
                return weighted_sum / total_volume
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Volume-weighted sentiment calculation failed: {e}")
            return 0.0

    def _determine_sentiment_trend(self, overall_sentiment: float) -> SentimentTrend:
        """Determine sentiment trend from overall sentiment"""
        if overall_sentiment >= 0.6:
            return SentimentTrend.STRONGLY_POSITIVE
        elif overall_sentiment >= 0.2:
            return SentimentTrend.POSITIVE
        elif overall_sentiment >= -0.2:
            return SentimentTrend.NEUTRAL
        elif overall_sentiment >= -0.6:
            return SentimentTrend.NEGATIVE
        else:
            return SentimentTrend.STRONGLY_NEGATIVE

    async def _calculate_sentiment_momentum(self, symbol: str, current_sentiment: float) -> float:
        """Calculate sentiment momentum"""
        try:
            if symbol not in self.sentiment_history:
                self.sentiment_history[symbol] = deque(maxlen=50)
                
            history = self.sentiment_history[symbol]
            if len(history) < 2:
                return 0.0
                
            # Calculate momentum as recent change
            recent_sentiments = [s.overall_sentiment for s in list(history)[-5:]]  # Last 5 readings
            if len(recent_sentiments) >= 2:
                momentum = recent_sentiments[-1] - recent_sentiments[0]
                return float(momentum)
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Sentiment momentum calculation failed: {e}")
            return 0.0

    async def _calculate_sentiment_divergence(self, symbol: str, current_sentiment: float) -> float:
        """Calculate sentiment-price divergence"""
        try:
            # Simplified implementation - would need actual price data
            # Simulate price data (in production, get from market data)
            simulated_price_change = np.random.normal(0.01, 0.02)  # Daily price change
            
            # Calculate divergence
            # Positive sentiment + negative price = positive divergence (bullish)
            # Negative sentiment + positive price = negative divergence (bearish)
            divergence = current_sentiment - simulated_price_change
            return float(max(-1.0, min(1.0, divergence)))
            
        except Exception as e:
            logger.error(f"Sentiment divergence calculation failed: {e}")
            return 0.0

    def _calculate_overall_confidence(self, sentiment_scores: List[SentimentScore]) -> float:
        """Calculate overall confidence level"""
        try:
            if not sentiment_scores:
                return 0.0
                
            # Weight confidences by sentiment type importance
            weighted_confidences = []
            total_weight = 0.0
            
            for score in sentiment_scores:
                weight = self.SENTIMENT_WEIGHTS.get(score.sentiment_type, 0.1)
                weighted_confidences.append(score.confidence * weight)
                total_weight += weight
                
            if total_weight > 0:
                overall_confidence = sum(weighted_confidences) / total_weight
            else:
                overall_confidence = np.mean([s.confidence for s in sentiment_scores])
                
            # Boost confidence with source diversity
            unique_types = len(set(s.sentiment_type for s in sentiment_scores))
            diversity_boost = min(0.2, unique_types * 0.05)
            
            return float(min(1.0, overall_confidence + diversity_boost))
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.5

    def _detect_significant_events(self, sentiment_scores: List[SentimentScore]) -> List[str]:
        """Detect significant events from sentiment data"""
        events = []
        
        try:
            for score in sentiment_scores:
                # High volume events
                if score.volume > self.HIGH_VOLUME_THRESHOLD:
                    if abs(score.score) > self.STRONG_SENTIMENT_THRESHOLD:
                        event_type = "positive" if score.score > 0 else "negative"
                        events.append(f"High volume {event_type} sentiment spike ({score.sentiment_type.value})")
                        
                # Keyword-based event detection
                for keyword in score.keywords:
                    if keyword in ['earnings', 'merger', 'acquisition', 'fda', 'approval']:
                        events.append(f"Significant event detected: {keyword}")
                        
            return list(set(events))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Event detection failed: {e}")
            return []

    def _generate_trading_recommendation(self, sentiment: float, confidence: float, 
                                       momentum: float, divergence: float) -> str:
        """Generate trading recommendation"""
        try:
            # Strong signals with high confidence
            if confidence > 0.7:
                if sentiment > 0.6 and momentum > 0.2:
                    return "STRONG BUY - Highly positive sentiment with strong momentum"
                elif sentiment < -0.6 and momentum < -0.2:
                    return "STRONG SELL - Highly negative sentiment with negative momentum"
                    
            # Moderate signals
            if confidence > 0.5:
                if sentiment > 0.3:
                    return "BUY - Positive sentiment indicates upward potential"
                elif sentiment < -0.3:
                    return "SELL - Negative sentiment suggests downward pressure"
                    
            # Divergence-based signals
            if abs(divergence) > self.DIVERGENCE_THRESHOLD:
                if divergence > 0:
                    return "CONTRARIAN BUY - Positive sentiment vs negative price action"
                else:
                    return "CONTRARIAN SELL - Negative sentiment vs positive price action"
                    
            return "HOLD - Mixed or neutral sentiment signals"
            
        except Exception as e:
            logger.error(f"Trading recommendation generation failed: {e}")
            return "HOLD - Unable to generate recommendation"

    def _generate_risk_alerts(self, sentiment: float, momentum: float, 
                            divergence: float, confidence: float) -> List[str]:
        """Generate risk alerts"""
        alerts = []
        
        try:
            # Extreme sentiment alerts
            if abs(sentiment) > 0.8:
                alerts.append("EXTREME SENTIMENT - Potential reversal risk")
                
            # Momentum alerts
            if abs(momentum) > self.MOMENTUM_THRESHOLD:
                if momentum > 0:
                    alerts.append("RAPID SENTIMENT IMPROVEMENT - Monitor for sustainability")
                else:
                    alerts.append("RAPID SENTIMENT DETERIORATION - Risk of further decline")
                    
            # Confidence alerts
            if confidence < 0.4:
                alerts.append("LOW CONFIDENCE - Sentiment signals may be unreliable")
                
            # Divergence alerts
            if abs(divergence) > self.DIVERGENCE_THRESHOLD:
                alerts.append("SENTIMENT-PRICE DIVERGENCE - Monitor for potential reversal")
                
            return alerts
            
        except Exception as e:
            logger.error(f"Risk alert generation failed: {e}")
            return ["Unable to generate risk alerts"]

    def _create_neutral_sentiment_signal(self, symbol: str) -> SentimentSignal:
        """Create neutral sentiment signal for symbols with no data"""
        return SentimentSignal(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            overall_sentiment=0.0,
            sentiment_trend=SentimentTrend.NEUTRAL,
            confidence_level=0.0,
            volume_weighted_sentiment=0.0,
            sentiment_momentum=0.0,
            sentiment_divergence=0.0,
            component_scores={},
            significant_events=[],
            trading_recommendation="HOLD - Insufficient sentiment data",
            risk_alerts=["No sentiment data available"]
        )

    async def _update_sentiment_history(self, symbol: str, sentiment_signal: SentimentSignal):
        """Update sentiment history for a symbol"""
        try:
            if symbol not in self.sentiment_history:
                self.sentiment_history[symbol] = deque(maxlen=50)
                
            self.sentiment_history[symbol].append(sentiment_signal)
            
            # Cache recent sentiment for external access
            cache_key = f"sentiment_history:{symbol}"
            recent_history = list(self.sentiment_history[symbol])[-10:]  # Last 10 readings
            
            serialized_history = []
            for signal in recent_history:
                serialized_history.append({
                    'timestamp': signal.timestamp.isoformat(),
                    'sentiment': signal.overall_sentiment,
                    'confidence': signal.confidence_level,
                    'trend': signal.sentiment_trend.value
                })
                
            await self.cache.set(cache_key, serialized_history, ttl=3600)
            
        except Exception as e:
            logger.error(f"Sentiment history update failed for {symbol}: {e}")

    async def get_sentiment_summary(self, symbols: List[str]) -> Dict:
        """Get sentiment summary for multiple symbols"""
        try:
            summary = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'symbols_analyzed': len(symbols),
                'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0},
                'high_confidence_signals': []
            }
            
            # Collect data from cache
            for symbol in symbols:
                cache_key = f"sentiment_history:{symbol}"
                history = await self.cache.get(cache_key)
                
                if history and len(history) > 0:
                    latest = history[-1]
                    sentiment = latest['sentiment']
                    confidence = latest['confidence']
                    
                    # Categorize sentiment
                    if sentiment > 0.3:
                        category = 'positive'
                    elif sentiment < -0.3:
                        category = 'negative'
                    else:
                        category = 'neutral'
                        
                    summary['sentiment_distribution'][category] = summary['sentiment_distribution'].get(category, 0) + 1
                    
                    # High confidence signals
                    if confidence > 0.7:
                        summary['high_confidence_signals'].append({
                            'symbol': symbol,
                            'sentiment': sentiment,
                            'confidence': confidence
                        })
                        
            return summary
            
        except Exception as e:
            logger.error(f"Sentiment summary generation failed: {e}")
            return {'error': str(e)}