"""Data Quality Validation System

Comprehensive data validation for all incoming market data including:
- Stale price detection and filtering
- Outlier identification using statistical methods
- Bad tick detection (prices, volumes, spreads)

This system is critical for portfolio integrity as bad data can lead
to poor investment decisions, incorrect risk calculations, and
unexpected losses.

Integration with AgentTradr infrastructure.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)


class DataQualityIssue:
    STALE_PRICE = "STALE_PRICE"  # Price hasn't updated
    PRICE_OUTLIER = "PRICE_OUTLIER"  # Extreme price movement
    VOLUME_OUTLIER = "VOLUME_OUTLIER"  # Unusual volume
    SPREAD_ANOMALY = "SPREAD_ANOMALY"  # Abnormal bid-ask spread
    MISSING_DATA = "MISSING_DATA"  # Data gaps
    NEGATIVE_PRICE = "NEGATIVE_PRICE"  # Invalid negative price
    ZERO_VOLUME = "ZERO_VOLUME"  # Suspicious zero volume
    TIMESTAMP_ERROR = "TIMESTAMP_ERROR"  # Incorrect timestamps
    SOURCE_INCONSISTENCY = "SOURCE_INCONSISTENCY"  # Data sources disagree
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"  # Trading halted


class DataQualitySeverity:
    LOW = "LOW"  # Minor issue, data still usable
    MEDIUM = "MEDIUM"  # Significant issue, use with caution
    HIGH = "HIGH"  # Major issue, avoid using data
    CRITICAL = "CRITICAL"  # Severe issue, data unusable


class DataSource:
    INTERACTIVE_BROKERS = "INTERACTIVE_BROKERS"
    ALPHA_VANTAGE = "ALPHA_VANTAGE"
    YAHOO_FINANCE = "YAHOO_FINANCE"
    BLOOMBERG = "BLOOMBERG"
    INTERNAL_CACHE = "INTERNAL_CACHE"


@dataclass
class DataPoint:
    symbol: str
    price: float
    volume: int
    bid: Optional[float] = None
    ask: Optional[float] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = DataSource.INTERNAL_CACHE
    
    @property
    def spread(self) -> Optional[float]:
        if self.bid and self.ask:
            return self.ask - self.bid
        return None
    
    @property
    def spread_pct(self) -> Optional[float]:
        if self.bid and self.ask and self.price > 0:
            return (self.ask - self.bid) / self.price * 100
        return None


@dataclass
class QualityIssue:
    symbol: str
    issue_type: str
    severity: str
    description: str
    affected_fields: List[str] = field(default_factory=list)
    expected_value: Optional[float] = None
    actual_value: Optional[float] = None
    confidence: float = 0.0  # 0.0 to 1.0
    source: str = DataSource.INTERNAL_CACHE
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    suggested_fix: str = ""


@dataclass
class DataQualityReport:
    symbol: str
    overall_score: float  # 0.0 to 100.0
    total_issues: int
    issues_by_severity: Dict[str, int] = field(default_factory=dict)
    issues: List[QualityIssue] = field(default_factory=list)
    data_freshness_score: float = 100.0  # How fresh is the data
    consistency_score: float = 100.0  # Consistency across sources
    completeness_score: float = 100.0  # Data completeness
    accuracy_score: float = 100.0  # Estimated accuracy
    recommended_action: str = ""
    last_validated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class DataQualityValidator:
    """Comprehensive data quality validation system.
    
    Validates all incoming market data to ensure portfolio decisions
    are based on accurate, timely, and consistent information.
    """
    
    def __init__(self):
        # Validation thresholds
        self.thresholds = {
            # Price validation
            'max_daily_change': 0.50,       # 50% max daily change
            'max_minute_change': 0.10,      # 10% max minute change
            'min_price': 0.01,              # Minimum valid price
            'max_price': 10000.0,           # Maximum reasonable price
            # Volume validation
            'min_volume': 0,                # Minimum volume (can be 0)
            # Spread validation
            'max_spread_pct': 5.0,          # 5% maximum spread
            'normal_spread_pct': 0.5,       # 0.5% normal spread threshold
            # Freshness validation
            'stale_threshold_minutes': 15,   # 15 minutes before considering stale
            'critical_stale_hours': 2,      # 2 hours = critical staleness
            # Consistency validation
            'price_difference_threshold': 0.02,  # 2% difference between sources
            'volume_difference_threshold': 0.30  # 30% volume difference
        }
        
        # Quality scoring weights
        self.quality_weights = {
            'freshness': 0.25,
            'consistency': 0.25,
            'completeness': 0.25,
            'accuracy': 0.25
        }
        
        # Cache settings
        self.cache_duration = 300  # 5 minutes
        
        logger.info("🔍 DataQualityValidator initialized")
        logger.info(f"  Validation thresholds: {self.thresholds}")
        logger.info(f"  Quality weights: {self.quality_weights}")
    
    async def validate_data_point(self, data_point: DataPoint) -> List[QualityIssue]:
        """Validate a single data point.
        
        Args:
            data_point: DataPoint to validate
            
        Returns:
            List of detected quality issues
        """
        try:
            issues = []
            
            # Basic validation checks
            issues.extend(await self._validate_price(data_point))
            issues.extend(await self._validate_volume(data_point))
            issues.extend(await self._validate_spread(data_point))
            issues.extend(await self._validate_timestamp(data_point))
            issues.extend(await self._validate_staleness(data_point))
            
            # Advanced validation checks
            issues.extend(await self._detect_price_outliers(data_point))
            issues.extend(await self._detect_volume_outliers(data_point))
            
            logger.debug(f"🔍 Validated {data_point.symbol}: {len(issues)} issues found")
            return issues
            
        except Exception as e:
            logger.error(f"❌ Error validating data point for {data_point.symbol}: {e}")
            return [QualityIssue(
                symbol=data_point.symbol,
                issue_type=DataQualityIssue.SOURCE_INCONSISTENCY,
                severity=DataQualitySeverity.HIGH,
                description=f"Validation error: {str(e)}",
                affected_fields=['all'],
                confidence=0.9,
                source=data_point.source,
                timestamp=datetime.now(timezone.utc),
                suggested_fix="Review data validation logic"
            )]
    
    async def validate_symbol_data(self, symbol: str) -> DataQualityReport:
        """Validate data quality for a symbol.
        
        Args:
            symbol: Symbol to validate
            
        Returns:
            Data quality report
        """
        try:
            logger.debug(f"🔍 Validating data quality for {symbol}")
            
            # Gather data from multiple sources
            data_points = await self._gather_multi_source_data(symbol)
            if not data_points:
                return self._create_no_data_report(symbol)
            
            # Validate each data point
            all_issues = []
            for data_point in data_points:
                issues = await self.validate_data_point(data_point)
                all_issues.extend(issues)
            
            # Cross-source consistency checks
            consistency_issues = await self._validate_cross_source_consistency(data_points)
            all_issues.extend(consistency_issues)
            
            # Calculate quality scores
            freshness_score = self._calculate_freshness_score(data_points)
            consistency_score = self._calculate_consistency_score(data_points, consistency_issues)
            completeness_score = self._calculate_completeness_score(data_points)
            accuracy_score = self._calculate_accuracy_score(all_issues)
            
            # Calculate overall score
            overall_score = (
                freshness_score * self.quality_weights['freshness'] +
                consistency_score * self.quality_weights['consistency'] +
                completeness_score * self.quality_weights['completeness'] +
                accuracy_score * self.quality_weights['accuracy']
            )
            
            # Categorize issues by severity
            issues_by_severity = {}
            for severity in [DataQualitySeverity.LOW, DataQualitySeverity.MEDIUM, 
                           DataQualitySeverity.HIGH, DataQualitySeverity.CRITICAL]:
                issues_by_severity[severity] = len([
                    issue for issue in all_issues if issue.severity == severity
                ])
            
            # Generate recommended action
            recommended_action = self._generate_recommendation(overall_score, all_issues)
            
            # Create report
            report = DataQualityReport(
                symbol=symbol,
                overall_score=overall_score,
                total_issues=len(all_issues),
                issues_by_severity=issues_by_severity,
                issues=all_issues,
                data_freshness_score=freshness_score,
                consistency_score=consistency_score,
                completeness_score=completeness_score,
                accuracy_score=accuracy_score,
                recommended_action=recommended_action,
                last_validated=datetime.now(timezone.utc)
            )
            
            # Send alerts for critical issues
            if overall_score < 60 or any(issue.severity == DataQualitySeverity.CRITICAL for issue in all_issues):
                await self._send_quality_alert(symbol, report)
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Error validating symbol data for {symbol}: {e}")
            return self._create_error_report(symbol, str(e))
    
    async def validate_portfolio_data(self, symbols: List[str]) -> Dict[str, DataQualityReport]:
        """Validate data quality for multiple symbols.
        
        Args:
            symbols: List of symbols to validate
            
        Returns:
            Dictionary mapping symbols to their quality reports
        """
        try:
            logger.info(f"🔍 Validating data quality for {len(symbols)} symbols")
            
            # Validate each symbol
            reports = {}
            critical_issues = []
            
            for symbol in symbols:
                report = await self.validate_symbol_data(symbol)
                reports[symbol] = report
                
                # Track critical issues
                if report.overall_score < 50:
                    critical_issues.append(symbol)
            
            # Send portfolio-level alert if many issues
            if len(critical_issues) > len(symbols) * 0.2:  # More than 20% have issues
                await self._send_portfolio_quality_alert(critical_issues, reports)
            
            logger.info(f"✅ Portfolio data validation complete: {len(critical_issues)} symbols with critical issues")
            return reports
            
        except Exception as e:
            logger.error(f"❌ Error validating portfolio data: {e}")
            return {}
    
    async def get_data_with_quality_filter(
        self, 
        symbol: str,
        min_quality_score: float = 70.0
    ) -> Tuple[Optional[DataPoint], DataQualityReport]:
        """Get data for symbol with quality filtering.
        
        Args:
            symbol: Stock symbol
            min_quality_score: Minimum acceptable quality score
            
        Returns:
            Tuple of (best_data_point, quality_report) or (None, report) if quality too low
        """
        try:
            # Validate data quality
            quality_report = await self.validate_symbol_data(symbol)
            
            if quality_report.overall_score < min_quality_score:
                logger.warning(
                    f"⚠️ Data quality too low for {symbol}: "
                    f"{quality_report.overall_score:.1f} < {min_quality_score}"
                )
                return None, quality_report
            
            # Get the best available data point
            data_points = await self._gather_multi_source_data(symbol)
            if not data_points:
                return None, quality_report
            
            # Find the highest quality data point
            best_data_point = self._select_best_data_point(data_points, quality_report)
            return best_data_point, quality_report
            
        except Exception as e:
            logger.error(f"❌ Error getting filtered data for {symbol}: {e}")
            error_report = self._create_error_report(symbol, str(e))
            return None, error_report
    
    async def _gather_multi_source_data(self, symbol: str) -> List[DataPoint]:
        """Gather data from multiple sources."""
        try:
            data_points = []
            current_time = datetime.now(timezone.utc)
            
            # Mock data from different sources (in production, would fetch from real sources)
            # Interactive Brokers data
            ib_price = 100 + np.random.uniform(-5, 5)
            ib_volume = int(np.random.uniform(100000, 1000000))
            data_points.append(DataPoint(
                symbol=symbol,
                price=ib_price,
                volume=ib_volume,
                bid=ib_price - 0.05,
                ask=ib_price + 0.05,
                timestamp=current_time - timedelta(seconds=np.random.randint(0, 60)),
                source=DataSource.INTERACTIVE_BROKERS
            ))
            
            # Alpha Vantage data (slightly delayed)
            av_price = ib_price + np.random.uniform(-0.5, 0.5)  # Small difference
            av_volume = int(ib_volume * np.random.uniform(0.8, 1.2))
            data_points.append(DataPoint(
                symbol=symbol,
                price=av_price,
                volume=av_volume,
                bid=None,  # Alpha Vantage doesn't provide bid/ask
                ask=None,
                timestamp=current_time - timedelta(minutes=np.random.randint(1, 5)),
                source=DataSource.ALPHA_VANTAGE
            ))
            
            # Occasionally add a third source
            if np.random.random() < 0.3:
                yahoo_price = ib_price + np.random.uniform(-0.2, 0.2)
                yahoo_volume = int(ib_volume * np.random.uniform(0.9, 1.1))
                data_points.append(DataPoint(
                    symbol=symbol,
                    price=yahoo_price,
                    volume=yahoo_volume,
                    bid=yahoo_price - 0.03,
                    ask=yahoo_price + 0.03,
                    timestamp=current_time - timedelta(seconds=np.random.randint(30, 120)),
                    source=DataSource.YAHOO_FINANCE
                ))
            
            return data_points
            
        except Exception as e:
            logger.warning(f"⚠️ Error gathering multi-source data for {symbol}: {e}")
            return []
    
    async def _validate_price(self, data_point: DataPoint) -> List[QualityIssue]:
        """Validate price data."""
        issues = []
        
        # Check for negative prices
        if data_point.price < 0:
            issues.append(QualityIssue(
                symbol=data_point.symbol,
                issue_type=DataQualityIssue.NEGATIVE_PRICE,
                severity=DataQualitySeverity.CRITICAL,
                description="Negative price detected",
                affected_fields=['price'],
                expected_value=None,
                actual_value=data_point.price,
                confidence=1.0,
                source=data_point.source,
                suggested_fix="Use alternative data source or interpolation"
            ))
        # Check for unreasonably low prices
        elif data_point.price < self.thresholds['min_price']:
            issues.append(QualityIssue(
                symbol=data_point.symbol,
                issue_type=DataQualityIssue.PRICE_OUTLIER,
                severity=DataQualitySeverity.HIGH,
                description=f"Price below minimum threshold: {data_point.price}",
                affected_fields=['price'],
                expected_value=self.thresholds['min_price'],
                actual_value=data_point.price,
                confidence=0.9,
                source=data_point.source,
                suggested_fix="Verify price with alternative sources"
            ))
        # Check for unreasonably high prices
        elif data_point.price > self.thresholds['max_price']:
            issues.append(QualityIssue(
                symbol=data_point.symbol,
                issue_type=DataQualityIssue.PRICE_OUTLIER,
                severity=DataQualitySeverity.MEDIUM,
                description=f"Price above maximum threshold: {data_point.price}",
                affected_fields=['price'],
                expected_value=self.thresholds['max_price'],
                actual_value=data_point.price,
                confidence=0.8,
                source=data_point.source,
                suggested_fix="Verify price with alternative sources"
            ))
        
        return issues
    
    async def _validate_volume(self, data_point: DataPoint) -> List[QualityIssue]:
        """Validate volume data."""
        issues = []
        
        # Check for negative volume
        if data_point.volume < 0:
            issues.append(QualityIssue(
                symbol=data_point.symbol,
                issue_type=DataQualityIssue.VOLUME_OUTLIER,
                severity=DataQualitySeverity.CRITICAL,
                description="Negative volume detected",
                affected_fields=['volume'],
                expected_value=0,
                actual_value=data_point.volume,
                confidence=1.0,
                source=data_point.source,
                suggested_fix="Use alternative data source"
            ))
        
        return issues
    
    async def _validate_spread(self, data_point: DataPoint) -> List[QualityIssue]:
        """Validate bid-ask spread."""
        issues = []
        
        if data_point.spread_pct is not None:
            if data_point.spread_pct > self.thresholds['max_spread_pct']:
                issues.append(QualityIssue(
                    symbol=data_point.symbol,
                    issue_type=DataQualityIssue.SPREAD_ANOMALY,
                    severity=DataQualitySeverity.HIGH,
                    description=f"Excessive spread: {data_point.spread_pct:.2f}%",
                    affected_fields=['bid', 'ask'],
                    expected_value=self.thresholds['max_spread_pct'],
                    actual_value=data_point.spread_pct,
                    confidence=0.9,
                    source=data_point.source,
                    suggested_fix="Check market conditions or use alternative source"
                ))
            elif data_point.spread_pct < 0:
                issues.append(QualityIssue(
                    symbol=data_point.symbol,
                    issue_type=DataQualityIssue.SPREAD_ANOMALY,
                    severity=DataQualitySeverity.CRITICAL,
                    description="Negative spread detected",
                    affected_fields=['bid', 'ask'],
                    confidence=1.0,
                    source=data_point.source,
                    suggested_fix="Data error - bid/ask inverted"
                ))
        
        return issues
    
    async def _validate_timestamp(self, data_point: DataPoint) -> List[QualityIssue]:
        """Validate timestamp."""
        issues = []
        current_time = datetime.now(timezone.utc)
        
        # Check for future timestamps
        if data_point.timestamp > current_time:
            issues.append(QualityIssue(
                symbol=data_point.symbol,
                issue_type=DataQualityIssue.TIMESTAMP_ERROR,
                severity=DataQualitySeverity.HIGH,
                description="Future timestamp detected",
                affected_fields=['timestamp'],
                expected_value=None,
                actual_value=data_point.timestamp.timestamp(),
                confidence=1.0,
                source=data_point.source,
                suggested_fix="Synchronize system clock or check data source"
            ))
        
        return issues
    
    async def _validate_staleness(self, data_point: DataPoint) -> List[QualityIssue]:
        """Check for stale data."""
        issues = []
        current_time = datetime.now(timezone.utc)
        age = current_time - data_point.timestamp
        
        if age > timedelta(hours=self.thresholds['critical_stale_hours']):
            issues.append(QualityIssue(
                symbol=data_point.symbol,
                issue_type=DataQualityIssue.STALE_PRICE,
                severity=DataQualitySeverity.CRITICAL,
                description=f"Data critically stale: {age.total_seconds()/3600:.1f} hours old",
                affected_fields=['timestamp'],
                confidence=1.0,
                source=data_point.source,
                suggested_fix="Refresh data from source"
            ))
        elif age > timedelta(minutes=self.thresholds['stale_threshold_minutes']):
            issues.append(QualityIssue(
                symbol=data_point.symbol,
                issue_type=DataQualityIssue.STALE_PRICE,
                severity=DataQualitySeverity.MEDIUM,
                description=f"Data stale: {age.total_seconds()/60:.1f} minutes old",
                affected_fields=['timestamp'],
                confidence=0.8,
                source=data_point.source,
                suggested_fix="Consider refreshing data"
            ))
        
        return issues
    
    async def _detect_price_outliers(self, data_point: DataPoint) -> List[QualityIssue]:
        """Detect price outliers using statistical methods."""
        # Simplified outlier detection (in production, would use historical data)
        return []
    
    async def _detect_volume_outliers(self, data_point: DataPoint) -> List[QualityIssue]:
        """Detect volume outliers."""
        # Simplified outlier detection (in production, would use historical data)
        return []
    
    async def _validate_cross_source_consistency(self, data_points: List[DataPoint]) -> List[QualityIssue]:
        """Validate consistency across data sources."""
        issues = []
        
        if len(data_points) < 2:
            return issues
        
        # Compare prices across sources
        prices = [dp.price for dp in data_points]
        mean_price = np.mean(prices)
        
        for dp in data_points:
            price_diff_pct = abs(dp.price - mean_price) / mean_price
            if price_diff_pct > self.thresholds['price_difference_threshold']:
                issues.append(QualityIssue(
                    symbol=dp.symbol,
                    issue_type=DataQualityIssue.SOURCE_INCONSISTENCY,
                    severity=DataQualitySeverity.MEDIUM,
                    description=f"Price inconsistency: {price_diff_pct*100:.1f}% from mean",
                    affected_fields=['price'],
                    expected_value=mean_price,
                    actual_value=dp.price,
                    confidence=0.7,
                    source=dp.source,
                    suggested_fix="Investigate source discrepancy"
                ))
        
        return issues
    
    def _calculate_freshness_score(self, data_points: List[DataPoint]) -> float:
        """Calculate data freshness score."""
        if not data_points:
            return 0.0
        
        current_time = datetime.now(timezone.utc)
        ages = [(current_time - dp.timestamp).total_seconds() for dp in data_points]
        avg_age_minutes = np.mean(ages) / 60
        
        if avg_age_minutes < 1:
            return 100.0
        elif avg_age_minutes < self.thresholds['stale_threshold_minutes']:
            return 100.0 * (1 - avg_age_minutes / self.thresholds['stale_threshold_minutes'])
        else:
            return max(0.0, 50.0 * (1 - avg_age_minutes / (self.thresholds['critical_stale_hours'] * 60)))
    
    def _calculate_consistency_score(self, data_points: List[DataPoint], issues: List[QualityIssue]) -> float:
        """Calculate consistency score."""
        if not data_points:
            return 0.0
        
        consistency_issues = [i for i in issues if i.issue_type == DataQualityIssue.SOURCE_INCONSISTENCY]
        if not consistency_issues:
            return 100.0
        
        return max(0.0, 100.0 - len(consistency_issues) * 20)
    
    def _calculate_completeness_score(self, data_points: List[DataPoint]) -> float:
        """Calculate data completeness score."""
        if not data_points:
            return 0.0
        
        scores = []
        for dp in data_points:
            field_count = 0
            total_fields = 5  # price, volume, bid, ask, timestamp
            
            if dp.price is not None:
                field_count += 1
            if dp.volume is not None:
                field_count += 1
            if dp.bid is not None:
                field_count += 1
            if dp.ask is not None:
                field_count += 1
            if dp.timestamp is not None:
                field_count += 1
            
            scores.append(field_count / total_fields * 100)
        
        return np.mean(scores)
    
    def _calculate_accuracy_score(self, issues: List[QualityIssue]) -> float:
        """Calculate accuracy score based on issues."""
        if not issues:
            return 100.0
        
        severity_weights = {
            DataQualitySeverity.LOW: 5,
            DataQualitySeverity.MEDIUM: 15,
            DataQualitySeverity.HIGH: 30,
            DataQualitySeverity.CRITICAL: 50
        }
        
        total_penalty = sum(severity_weights.get(issue.severity, 0) for issue in issues)
        return max(0.0, 100.0 - total_penalty)
    
    def _generate_recommendation(self, overall_score: float, issues: List[QualityIssue]) -> str:
        """Generate recommended action based on quality assessment."""
        if overall_score >= 90:
            return "Data quality excellent - safe to use"
        elif overall_score >= 70:
            return "Data quality good - proceed with minor caution"
        elif overall_score >= 50:
            return "Data quality moderate - verify critical values"
        elif overall_score >= 30:
            return "Data quality poor - use alternative sources if available"
        else:
            return "Data quality critical - do not use for trading decisions"
    
    def _select_best_data_point(self, data_points: List[DataPoint], report: DataQualityReport) -> Optional[DataPoint]:
        """Select the best quality data point."""
        if not data_points:
            return None
        
        # Simple selection: prefer most recent with complete data
        best_score = -1
        best_dp = None
        
        for dp in data_points:
            score = 0
            
            # Prefer recent data
            age_minutes = (datetime.now(timezone.utc) - dp.timestamp).total_seconds() / 60
            if age_minutes < 1:
                score += 40
            elif age_minutes < 5:
                score += 30
            elif age_minutes < 15:
                score += 20
            
            # Prefer complete data
            if dp.bid and dp.ask:
                score += 30
            
            # Prefer Interactive Brokers
            if dp.source == DataSource.INTERACTIVE_BROKERS:
                score += 20
            
            if score > best_score:
                best_score = score
                best_dp = dp
        
        return best_dp
    
    def _create_no_data_report(self, symbol: str) -> DataQualityReport:
        """Create report when no data is available."""
        return DataQualityReport(
            symbol=symbol,
            overall_score=0.0,
            total_issues=1,
            issues_by_severity={
                DataQualitySeverity.CRITICAL: 1
            },
            issues=[QualityIssue(
                symbol=symbol,
                issue_type=DataQualityIssue.MISSING_DATA,
                severity=DataQualitySeverity.CRITICAL,
                description="No data available",
                affected_fields=['all'],
                confidence=1.0,
                suggested_fix="Check data source connections"
            )],
            data_freshness_score=0.0,
            consistency_score=0.0,
            completeness_score=0.0,
            accuracy_score=0.0,
            recommended_action="Cannot proceed - no data available"
        )
    
    def _create_error_report(self, symbol: str, error: str) -> DataQualityReport:
        """Create report for validation errors."""
        return DataQualityReport(
            symbol=symbol,
            overall_score=0.0,
            total_issues=1,
            issues_by_severity={
                DataQualitySeverity.CRITICAL: 1
            },
            issues=[QualityIssue(
                symbol=symbol,
                issue_type=DataQualityIssue.SOURCE_INCONSISTENCY,
                severity=DataQualitySeverity.CRITICAL,
                description=f"Validation error: {error}",
                affected_fields=['all'],
                confidence=1.0,
                suggested_fix="Review validation logic"
            )],
            data_freshness_score=0.0,
            consistency_score=0.0,
            completeness_score=0.0,
            accuracy_score=0.0,
            recommended_action="Cannot validate - system error"
        )
    
    async def _send_quality_alert(self, symbol: str, report: DataQualityReport):
        """Send alert for quality issues."""
        try:
            logger.warning(f"⚠️ Data quality alert for {symbol}: score={report.overall_score:.1f}")
        except Exception as e:
            logger.error(f"Error sending quality alert: {e}")
    
    async def _send_portfolio_quality_alert(self, critical_symbols: List[str], reports: Dict[str, DataQualityReport]):
        """Send portfolio-level quality alert."""
        try:
            logger.warning(f"⚠️ Portfolio quality alert: {len(critical_symbols)} symbols with critical issues")
        except Exception as e:
            logger.error(f"Error sending portfolio quality alert: {e}")