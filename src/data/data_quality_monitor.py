from datetime import datetime, timedelta
import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
import numpy as np
import pandas as pd
from scipy import stats
import hashlib

logger = logging.getLogger(__name__)

class QualityIssueType(Enum):
    MISSING_DATA = "missing_data"
    DUPLICATE_DATA = "duplicate_data"
    OUTLIER = "outlier"
    INCONSISTENT_DATA = "inconsistent_data"
    FORMAT_ERROR = "format_error"
    LATENCY_HIGH = "latency_high"
    COVERAGE_GAP = "coverage_gap"
    CORRELATION_BREAK = "correlation_break"
    STALE_DATA = "stale_data"

class SeverityLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class QualityCheck:
    name: str
    description: str
    enabled: bool = True
    issue_type: QualityIssueType = QualityIssueType.MISSING_DATA
    threshold: float = 0.0
    severity: SeverityLevel = SeverityLevel.MEDIUM
    symbols: Optional[List[str]] = None  # If None, applies to all symbols
    data_types: Optional[List[str]] = None  # If None, applies to all data types
    sources: Optional[List[str]] = None  # If None, applies to all sources
    
    # Advanced parameters
    lookback_minutes: int = 60
    min_samples: int = 10
    alert_frequency_minutes: int = 30
    auto_disable_threshold: int = 10  # Disable check after N consecutive failures

@dataclass
class QualityIssue:
    id: Any
    timestamp: datetime
    issue_type: QualityIssueType = None
    severity: SeverityLevel = None
    symbol: str = ""
    data_source: str = ""
    data_type: str = ""
    
    # Issue details
    description: str = ""
    metric_value: float = 0.0
    threshold_value: float = 0.0
    impact_assessment: str = ""
    
    # Resolution tracking
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolution_notes: str = ""
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

class StatisticalAnalyzer:
    def __init__(self, lookback_periods=100):
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=lookback_periods))
        self.volume_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=lookback_periods))
        self.timestamp_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=lookback_periods))
    
    def add_data_point(self, symbol: str, price: float, volume: int, timestamp: datetime):
        key = symbol
        self.price_history[key].append(price)
        self.volume_history[key].append(volume)
        self.timestamp_history[key].append(timestamp)
    
    def detect_price_outliers(self, symbol: str, current_price: float, z_score_threshold: float = 3.0) -> Tuple[bool, float]:
        key = symbol
        history = self.price_history[key]
        
        if len(history) < 10:
            return False, 0.0
        
        prices = list(history)
        mean_price = np.mean(prices)
        std_price = np.std(prices)
        
        if std_price == 0:
            return False, 0.0
        
        z_score = abs((current_price - mean_price) / std_price)
        is_outlier = z_score > z_score_threshold
        
        return is_outlier, z_score
    
    def detect_volume_anomalies(self, symbol: str, current_volume: int, threshold_multiplier: float = 5.0) -> Tuple[bool, float]:
        key = symbol
        history = self.volume_history[key]
        
        if len(history) < 10:
            return False, 0.0
        
        volumes = list(history)
        median_volume = np.median(volumes)
        
        if median_volume == 0:
            return False, 0.0
        
        volume_ratio = current_volume / median_volume
        is_anomaly = volume_ratio > threshold_multiplier or volume_ratio < (1 / threshold_multiplier)
        
        return is_anomaly, volume_ratio
    
    def detect_timing_gaps(self, symbol: str, expected_interval_seconds: int = 60, gap_threshold_multiplier: float = 2.0) -> Tuple[bool, float]:
        key = symbol
        history = self.timestamp_history[key]
        
        if len(history) < 2:
            return False, 0.0
        
        timestamps = list(history)
        timestamps.sort()
        
        # Calculate intervals
        intervals = []
        for i in range(1, len(timestamps)):
            interval = (timestamps[i] - timestamps[i-1]).total_seconds()
            intervals.append(interval)
        
        if not intervals:
            return False, 0.0
        
        # Find largest gap
        max_gap = max(intervals)
        threshold = expected_interval_seconds * gap_threshold_multiplier
        has_gap = max_gap > threshold
        gap_ratio = max_gap / expected_interval_seconds
        
        return has_gap, gap_ratio
    
    def calculate_correlation_matrix(self, symbols: List[str], min_observations: int = 30) -> Optional[pd.DataFrame]:
        # Get price data for all symbols
        price_data = {}
        for symbol in symbols:
            history = self.price_history[symbol]
            if len(history) >= min_observations:
                price_data[symbol] = list(history)
        
        if len(price_data) < 2:
            return None
        
        # Align data lengths
        min_length = min(len(prices) for prices in price_data.values())
        aligned_data = {symbol: prices[-min_length:] for symbol, prices in price_data.items()}
        
        # Calculate returns
        returns_data = {}
        for symbol, prices in aligned_data.items():
            returns = np.diff(np.log(prices))
            returns_data[symbol] = returns
        
        # Create correlation matrix
        df = pd.DataFrame(returns_data)
        return df.corr()
    
    def detect_correlation_breaks(self, symbol1: str, symbol2: str, tolerance: float = 0.3) -> Tuple[bool, float]:
        if len(self.price_history[symbol1]) < 30 or len(self.price_history[symbol2]) < 30:
            return False, 0.0
        
        # Calculate recent correlation
        prices1 = list(self.price_history[symbol1])
        prices2 = list(self.price_history[symbol2])
        
        # Align lengths
        min_len = min(len(prices1), len(prices2))
        prices1 = prices1[-min_len:]
        prices2 = prices2[-min_len:]
        
        # Calculate returns
        returns1 = np.diff(np.log(prices1))
        returns2 = np.diff(np.log(prices2))
        
        if len(returns1) < 10:
            return False, 0.0
        
        current_correlation = np.corrcoef(returns1, returns2)[0, 1]
        
        if np.isnan(current_correlation):
            return False, 0.0
        
        # For simplicity, assume historical correlation was strong
        historical_correlation = 0.8
        correlation_diff = abs(current_correlation - historical_correlation)
        has_break = correlation_diff > tolerance
        
        return has_break, current_correlation

class DataQualityChecker:
    def __init__(self):
        self.checks: List[QualityCheck] = []
        self.issue_history: List[QualityIssue] = []
        self.check_failures: Dict[str, int] = defaultdict(int)
        self.last_alert_times: Dict[str, datetime] = {}
        
        # Initialize default checks
        self._initialize_default_checks()
    
    def _initialize_default_checks(self):
        default_checks = [
            QualityCheck(
                name="missing_price_data",
                description="Detect missing price data points",
                issue_type=QualityIssueType.MISSING_DATA,
                threshold=0.1,  # 10% missing data threshold
                severity=SeverityLevel.HIGH,
                lookback_minutes=30
            ),
            QualityCheck(
                name="stale_data_detection",
                description="Detect stale data (no updates)",
                issue_type=QualityIssueType.STALE_DATA,
                threshold=300,  # 5 minutes without update
                severity=SeverityLevel.MEDIUM,
                lookback_minutes=10
            ),
            QualityCheck(
                name="price_outlier_detection",
                description="Detect price outliers using statistical methods",
                issue_type=QualityIssueType.OUTLIER,
                threshold=3.0,  # 3 standard deviations
                severity=SeverityLevel.HIGH,
                lookback_minutes=60
            ),
            QualityCheck(
                name="volume_anomaly_detection",
                description="Detect volume anomalies",
                issue_type=QualityIssueType.OUTLIER,
                threshold=5.0,  # 5x normal volume
                severity=SeverityLevel.MEDIUM,
                lookback_minutes=60
            ),
            QualityCheck(
                name="duplicate_data_detection",
                description="Detect duplicate data points",
                issue_type=QualityIssueType.DUPLICATE_DATA,
                threshold=1.0,  # Any duplicates
                severity=SeverityLevel.LOW,
                lookback_minutes=15
            ),
            QualityCheck(
                name="high_latency_detection",
                description="Detect high latency data",
                issue_type=QualityIssueType.LATENCY_HIGH,
                threshold=1000,  # 1 second latency
                severity=SeverityLevel.MEDIUM,
                lookback_minutes=30
            )
        ]
        
        self.checks.extend(default_checks)
    
    def add_custom_check(self, check: QualityCheck):
        self.checks.append(check)
        logger.info(f"Added custom quality check: {check.name}")
    
    def remove_check(self, check_name: str):
        self.checks = [c for c in self.checks if c.name != check_name]
        logger.info(f"Removed quality check: {check_name}")
    
    def run_quality_checks(self, data_points: List[Any], statistical_analyzer: StatisticalAnalyzer) -> List[QualityIssue]:
        issues = []
        
        for check in self.checks:
            if not check.enabled:
                continue
            
            # Skip if check has failed too many times
            if self.check_failures[check.name] >= check.auto_disable_threshold:
                logger.warning(f"Check {check.name} disabled due to repeated failures")
                check.enabled = False
                continue
            
            # Check alert frequency
            if not self._should_alert(check):
                continue
            
            try:
                check_issues = self._run_single_check(check, data_points, statistical_analyzer)
                issues.extend(check_issues)
                
                # Reset failure count on success
                self.check_failures[check.name] = 0
                
            except Exception as e:
                logger.error(f"Error running check {check.name}: {str(e)}")
                self.check_failures[check.name] += 1
        
        # Store issues
        self.issue_history.extend(issues)
        
        # Clean old issues
        cutoff = datetime.utcnow() - timedelta(hours=24)
        self.issue_history = [i for i in self.issue_history if i.timestamp > cutoff]
        
        return issues
    
    def _should_alert(self, check: QualityCheck) -> bool:
        if check.name not in self.last_alert_times:
            return True
        
        time_since_last = datetime.utcnow() - self.last_alert_times[check.name]
        return time_since_last.total_seconds() >= (check.alert_frequency_minutes * 60)
    
    def _run_single_check(self, check: QualityCheck, data_points: List[Any], statistical_analyzer: StatisticalAnalyzer) -> List[QualityIssue]:
        issues = []
        
        # Filter data points for this check
        filtered_points = self._filter_data_points(data_points, check)
        
        if check.issue_type == QualityIssueType.MISSING_DATA:
            issues.extend(self._check_missing_data(check, filtered_points))
        elif check.issue_type == QualityIssueType.STALE_DATA:
            issues.extend(self._check_stale_data(check, filtered_points))
        elif check.issue_type == QualityIssueType.OUTLIER:
            issues.extend(self._check_outliers(check, filtered_points, statistical_analyzer))
        elif check.issue_type == QualityIssueType.DUPLICATE_DATA:
            issues.extend(self._check_duplicates(check, filtered_points))
        elif check.issue_type == QualityIssueType.LATENCY_HIGH:
            issues.extend(self._check_high_latency(check, filtered_points))
        elif check.issue_type == QualityIssueType.COVERAGE_GAP:
            issues.extend(self._check_coverage_gaps(check, filtered_points, statistical_analyzer))
        
        if issues:
            self.last_alert_times[check.name] = datetime.utcnow()
        
        return issues
    
    def _filter_data_points(self, data_points: List[Any], check: QualityCheck) -> List[Any]:
        filtered = data_points
        
        # Filter by symbols
        if check.symbols:
            filtered = [dp for dp in filtered if getattr(dp, 'symbol', None) in check.symbols]
        
        # Filter by data types
        if check.data_types:
            filtered = [dp for dp in filtered if getattr(dp, 'data_type', None) in check.data_types]
        
        # Filter by sources
        if check.sources:
            filtered = [dp for dp in filtered if getattr(dp, 'source', None) in check.sources]
        
        return filtered
    
    def _check_missing_data(self, check: QualityCheck, data_points: List[Any]) -> List[QualityIssue]:
        issues = []
        
        # Group by symbol and source
        grouped = defaultdict(list)
        for dp in data_points:
            key = f"{getattr(dp, 'symbol', 'unknown')}:{getattr(dp, 'source', 'unknown')}"
            grouped[key].append(dp)
        
        # Check each group
        for key, points in grouped.items():
            symbol, source = key.split(':', 1)
            
            # Expected number of points based on frequency
            cutoff = datetime.utcnow() - timedelta(minutes=check.lookback_minutes)
            recent_points = [p for p in points if getattr(p, 'timestamp', datetime.utcnow()) > cutoff]
            
            expected_points = check.lookback_minutes  # Assume 1 point per minute
            actual_points = len(recent_points)
            missing_ratio = (expected_points - actual_points) / expected_points if expected_points > 0 else 0
            
            if missing_ratio > check.threshold:
                issue = QualityIssue(
                    id=f"missing_data_{int(time.time())}_{hash(key)}",
                    timestamp=datetime.utcnow(),
                    issue_type=check.issue_type,
                    severity=check.severity,
                    symbol=symbol,
                    data_source=source,
                    data_type="unknown",
                    description=f"Missing {missing_ratio:.1%} of expected data points",
                    metric_value=missing_ratio,
                    threshold_value=check.threshold,
                    impact_assessment="Medium impact on data completeness",
                    metadata={}
                )
                issues.append(issue)
        
        return issues
    
    def _check_stale_data(self, check: QualityCheck, data_points: List[Any]) -> List[QualityIssue]:
        issues = []
        
        # Group by symbol and source
        grouped = defaultdict(list)
        for dp in data_points:
            key = f"{getattr(dp, 'symbol', 'unknown')}:{getattr(dp, 'source', 'unknown')}"
            grouped[key].append(dp)
        
        # Check each group
        for key, points in grouped.items():
            if not points:
                continue
            
            symbol, source = key.split(':', 1)
            
            # Find most recent data point
            latest_point = max(points, key=lambda p: getattr(p, 'timestamp', datetime.min))
            latest_time = getattr(latest_point, 'timestamp', datetime.utcnow())
            
            # Check staleness
            time_since_update = (datetime.utcnow() - latest_time).total_seconds()
            
            if time_since_update > check.threshold:
                issue = QualityIssue(
                    id=f"stale_data_{int(time.time())}_{hash(key)}",
                    timestamp=datetime.utcnow(),
                    issue_type=check.issue_type,
                    severity=check.severity,
                    symbol=symbol,
                    data_source=source,
                    data_type=getattr(latest_point, 'data_type', 'unknown'),
                    description=f"Data stale for {time_since_update:.0f} seconds",
                    metric_value=time_since_update,
                    threshold_value=check.threshold,
                    impact_assessment="Data freshness compromised",
                    metadata={'last_update': latest_time.isoformat()}
                )
                issues.append(issue)
        
        return issues
    
    def _check_outliers(self, check: QualityCheck, data_points: List[Any], statistical_analyzer: StatisticalAnalyzer) -> List[QualityIssue]:
        issues = []
        
        for dp in data_points:
            symbol = getattr(dp, 'symbol', 'unknown')
            price = getattr(dp, 'price', None)
            volume = getattr(dp, 'volume', None)
            
            if price is not None:
                # Check price outliers
                is_outlier, z_score = statistical_analyzer.detect_price_outliers(
                    symbol, float(price), check.threshold
                )
                
                if is_outlier:
                    issue = QualityIssue(
                        id=f"price_outlier_{int(time.time())}_{hash(symbol)}",
                        timestamp=datetime.utcnow(),
                        issue_type=check.issue_type,
                        severity=check.severity,
                        symbol=symbol,
                        data_source=getattr(dp, 'source', 'unknown'),
                        data_type=getattr(dp, 'data_type', 'unknown'),
                        description=f"Price outlier detected: z-score = {z_score:.2f}",
                        metric_value=z_score,
                        threshold_value=check.threshold,
                        impact_assessment="Potential data quality issue or market event",
                        metadata={'price': float(price)}
                    )
                    issues.append(issue)
            
            if volume is not None:
                # Check volume anomalies
                is_anomaly, volume_ratio = statistical_analyzer.detect_volume_anomalies(
                    symbol, int(volume), check.threshold
                )
                
                if is_anomaly:
                    issue = QualityIssue(
                        id=f"volume_anomaly_{int(time.time())}_{hash(symbol)}",
                        timestamp=datetime.utcnow(),
                        issue_type=check.issue_type,
                        severity=check.severity,
                        symbol=symbol,
                        data_source=getattr(dp, 'source', 'unknown'),
                        data_type=getattr(dp, 'data_type', 'unknown'),
                        description=f"Volume anomaly detected: {volume_ratio:.1f}x normal volume",
                        metric_value=volume_ratio,
                        threshold_value=check.threshold,
                        impact_assessment="Unusual trading activity detected",
                        metadata={'volume': int(volume)}
                    )
                    issues.append(issue)
        
        return issues
    
    def _check_duplicates(self, check: QualityCheck, data_points: List[Any]) -> List[QualityIssue]:
        issues = []
        
        # Group by symbol
        symbol_points = defaultdict(list)
        for dp in data_points:
            symbol = getattr(dp, 'symbol', 'unknown')
            symbol_points[symbol].append(dp)
        
        # Check for duplicates within each symbol
        for symbol, points in symbol_points.items():
            fingerprints = set()
            duplicate_count = 0
            
            for dp in points:
                # Create fingerprint from key attributes
                timestamp = getattr(dp, 'timestamp', datetime.utcnow())
                price = getattr(dp, 'price', 0)
                volume = getattr(dp, 'volume', 0)
                
                fingerprint = f"{timestamp.isoformat()}:{price}:{volume}"
                
                if fingerprint in fingerprints:
                    duplicate_count += 1
                else:
                    fingerprints.add(fingerprint)
            
            if duplicate_count > 0:
                issue = QualityIssue(
                    id=f"duplicate_data_{int(time.time())}_{hash(symbol)}",
                    timestamp=datetime.utcnow(),
                    issue_type=check.issue_type,
                    severity=check.severity,
                    symbol=symbol,
                    data_source=getattr(points[0], 'source', 'unknown'),
                    data_type=getattr(points[0], 'data_type', 'unknown'),
                    description=f"Found {duplicate_count} duplicate data points",
                    metric_value=duplicate_count,
                    threshold_value=check.threshold,
                    impact_assessment="Data integrity compromised by duplicates",
                    metadata={
                        'total_points': len(points),
                        'duplicate_ratio': duplicate_count / len(points) if points else 0
                    }
                )
                issues.append(issue)
        
        return issues
    
    def _check_high_latency(self, check: QualityCheck, data_points: List[Any]) -> List[QualityIssue]:
        issues = []
        
        high_latency_points = []
        for dp in data_points:
            latency = getattr(dp, 'latency_ms', None)
            if latency and latency > check.threshold:
                high_latency_points.append(dp)
        
        if high_latency_points:
            # Group by symbol
            symbol_latencies = defaultdict(list)
            for dp in high_latency_points:
                symbol = getattr(dp, 'symbol', 'unknown')
                symbol_latencies[symbol].append(getattr(dp, 'latency_ms', 0))
            
            for symbol, latencies in symbol_latencies.items():
                avg_latency = np.mean(latencies)
                max_latency = max(latencies)
                
                issue = QualityIssue(
                    id=f"high_latency_{int(time.time())}_{hash(symbol)}",
                    timestamp=datetime.utcnow(),
                    issue_type=check.issue_type,
                    severity=check.severity,
                    symbol=symbol,
                    data_source=getattr(high_latency_points[0], 'source', 'unknown'),
                    data_type=getattr(high_latency_points[0], 'data_type', 'unknown'),
                    description=f"High latency detected: avg {avg_latency:.0f}ms, max {max_latency:.0f}ms",
                    metric_value=avg_latency,
                    threshold_value=check.threshold,
                    impact_assessment="Data timeliness compromised",
                    metadata={'high_latency_count': len(latencies)}
                )
                issues.append(issue)
        
        return issues
    
    def _check_coverage_gaps(self, check: QualityCheck, data_points: List[Any], statistical_analyzer: StatisticalAnalyzer) -> List[QualityIssue]:
        issues = []
        
        # Group by symbol
        symbol_points = defaultdict(list)
        for dp in data_points:
            symbol = getattr(dp, 'symbol', 'unknown')
            symbol_points[symbol].append(dp)
        
        # Check each symbol for gaps
        for symbol, points in symbol_points.items():
            has_gap, gap_ratio = statistical_analyzer.detect_timing_gaps(symbol)
            
            if has_gap:
                issue = QualityIssue(
                    id=f"coverage_gap_{int(time.time())}_{hash(symbol)}",
                    timestamp=datetime.utcnow(),
                    issue_type=QualityIssueType.COVERAGE_GAP,
                    severity=check.severity,
                    symbol=symbol,
                    data_source=getattr(points[0], 'source', 'unknown'),
                    data_type=getattr(points[0], 'data_type', 'unknown'),
                    description=f"Data coverage gap detected: {gap_ratio:.1f}x expected interval",
                    metric_value=gap_ratio,
                    threshold_value=check.threshold,
                    impact_assessment="Data continuity compromised",
                    metadata={}
                )
                issues.append(issue)
        
        return issues
    
    def get_quality_summary(self) -> Dict[str, Any]:
        # Recent issues (last 24 hours)
        recent_cutoff = datetime.utcnow() - timedelta(hours=24)
        recent_issues = [i for i in self.issue_history if i.timestamp > recent_cutoff]
        
        # Group issues by type and severity
        issues_by_type = defaultdict(int)
        issues_by_severity = defaultdict(int)
        issues_by_symbol = defaultdict(int)
        
        for issue in recent_issues:
            if not issue.resolved:
                issues_by_type[issue.issue_type.value] += 1
                issues_by_severity[issue.severity.name] += 1
                issues_by_symbol[issue.symbol] += 1
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'total_issues_24h': len(recent_issues),
            'unresolved_issues': len([i for i in recent_issues if not i.resolved]),
            'issues_by_type': dict(issues_by_type),
            'issues_by_severity': dict(issues_by_severity),
            'top_problematic_symbols': dict(sorted(issues_by_symbol.items(), key=lambda x: x[1], reverse=True)[:10]),
            'enabled_checks': len([c for c in self.checks if c.enabled]),
            'disabled_checks': len([c for c in self.checks if not c.enabled])
        }

class DataQualityMonitor:
    def __init__(self, config: Dict[str, Any] = None):
        if config is None:
            config = {}
        
        self.statistical_analyzer = StatisticalAnalyzer(
            lookback_periods=config.get('lookback_periods', 100)
        )
        self.quality_checker = DataQualityChecker()
        
        # Issue tracking
        self.active_issues: Dict[str, QualityIssue] = {}
        self.resolved_issues: List[QualityIssue] = []
        
        # Callbacks for issue notifications
        self.issue_callbacks: List[Callable] = []
        
        # Performance tracking
        self.checks_performed = 0
        self.last_check_time = datetime.utcnow()
        
        logger.info("Data quality monitor initialized")
    
    def add_data_points(self, data_points: List[Any]):
        # Add data to statistical analyzer
        for dp in data_points:
            symbol = getattr(dp, 'symbol', None)
            price = getattr(dp, 'price', None)
            volume = getattr(dp, 'volume', None)
            timestamp = getattr(dp, 'timestamp', datetime.utcnow())
            
            if symbol and price is not None:
                volume = volume or 0
                self.statistical_analyzer.add_data_point(symbol, float(price), int(volume), timestamp)
        
        # Run quality checks
        issues = self.quality_checker.run_quality_checks(data_points, self.statistical_analyzer)
        
        # Process new issues
        for issue in issues:
            self._handle_new_issue(issue)
        
        self.checks_performed += 1
        self.last_check_time = datetime.utcnow()
    
    def _handle_new_issue(self, issue: QualityIssue):
        # Check if similar issue exists
        similar_issue = self._find_similar_issue(issue)
        
        if similar_issue:
            # Update existing issue
            similar_issue.timestamp = issue.timestamp
            similar_issue.metric_value = issue.metric_value
            similar_issue.metadata.update(issue.metadata)
            logger.debug(f"Updated existing quality issue: {similar_issue.id}")
        else:
            # New issue
            self.active_issues[issue.id] = issue
            logger.warning(f"New quality issue detected: {issue.description}")
            
            # Notify callbacks
            for callback in self.issue_callbacks:
                try:
                    callback(issue)
                except Exception as e:
                    logger.error(f"Error in quality issue callback: {str(e)}")
    
    def _find_similar_issue(self, issue: QualityIssue) -> Optional[QualityIssue]:
        for active_issue in self.active_issues.values():
            if (active_issue.issue_type == issue.issue_type and
                active_issue.symbol == issue.symbol and
                active_issue.data_source == issue.data_source):
                return active_issue
        return None
    
    def resolve_issue(self, issue_id: str, resolution_notes: str = ""):
        if issue_id in self.active_issues:
            issue = self.active_issues[issue_id]
            issue.resolved = True
            issue.resolved_at = datetime.utcnow()
            issue.resolution_notes = resolution_notes
            
            self.resolved_issues.append(issue)
            del self.active_issues[issue_id]
            
            logger.info(f"Resolved quality issue: {issue_id}")
    
    def get_active_issues(self, severity_filter: Optional[SeverityLevel] = None, symbol_filter: Optional[str] = None) -> List[QualityIssue]:
        issues = list(self.active_issues.values())
        
        if severity_filter:
            issues = [i for i in issues if i.severity == severity_filter]
        
        if symbol_filter:
            issues = [i for i in issues if i.symbol == symbol_filter]
        
        # Sort by severity and timestamp
        issues.sort(key=lambda i: (i.severity.value, i.timestamp), reverse=True)
        
        return issues
    
    def add_issue_callback(self, callback: Callable):
        self.issue_callbacks.append(callback)
    
    def get_data_quality_report(self) -> Dict[str, Any]:
        base_summary = self.quality_checker.get_quality_summary()
        
        # Add monitor-specific metrics
        base_summary.update({
            'last_check_time': self.last_check_time.isoformat(),
            'checks_performed': self.checks_performed,
            'active_issues_count': len(self.active_issues),
            'resolved_issues_count': len(self.resolved_issues),
            'issue_callbacks_registered': len(self.issue_callbacks),
            'active_critical_issues': len([i for i in self.active_issues.values() if i.severity == SeverityLevel.CRITICAL]),
            'symbols_monitored': len(self.statistical_analyzer.price_history),
            'correlation_matrix_available': self.statistical_analyzer.calculate_correlation_matrix(
                list(self.statistical_analyzer.price_history.keys())[:10]
            ) is not None
        })
        
        return base_summary

# Example usage
async def example_data_quality_monitoring(config: Dict[str, Any] = None):
    if config is None:
        config = {}
    
    # Initialize monitor
    monitor = DataQualityMonitor(config)
    
    # Add custom quality check
    custom_check = QualityCheck(
        name="custom_price_range_check",
        description="Check if prices are within reasonable range",
        issue_type=QualityIssueType.OUTLIER,
        threshold=1000,  # Price > $1000 is suspicious for most stocks
        severity=SeverityLevel.HIGH,
        symbols=['AAPL', 'MSFT']  # Only check these symbols
    )
    
    monitor.quality_checker.add_custom_check(custom_check)
    
    # Add callback for quality issues
    def issue_callback(issue: QualityIssue):
        print(f"Quality Issue Alert: {issue.description} (Severity: {issue.severity.name})")
    
    monitor.add_issue_callback(issue_callback)
    
    # Simulate some data points with quality issues
    from dataclasses import dataclass
    
    @dataclass
    class MockDataPoint:
        symbol: str
        timestamp: datetime
        price: float
        volume: int
        source: str = "test"
        data_type: str = "quote"
        latency_ms: Optional[float] = None
    
    # Good data
    good_data = [
        MockDataPoint('AAPL', datetime.utcnow(), 150.0, 1000),
        MockDataPoint('MSFT', datetime.utcnow(), 300.0, 1500),
        MockDataPoint('GOOGL', datetime.utcnow(), 2500.0, 500),
    ]
    
    # Add good data first to establish baseline
    for _ in range(50):
        for dp in good_data:
            dp.timestamp = datetime.utcnow()
            dp.price += np.random.randint(-5, 5)
            dp.volume += np.random.randint(-100, 100)
        
        monitor.add_data_points(good_data)
        await asyncio.sleep(0.1)
    
    # Add problematic data
    problematic_data = [
        MockDataPoint('AAPL', datetime.utcnow(), 1500.0, 1000),  # Price outlier
        MockDataPoint('MSFT', datetime.utcnow(), 300.0, 50000),  # Volume anomaly
        MockDataPoint('GOOGL', datetime.utcnow(), 2500.0, 500, latency_ms=5000),  # High latency
    ]
    
    monitor.add_data_points(problematic_data)
    
    # Get quality report
    report = monitor.get_data_quality_report()
    print(f"Data Quality Report:")
    print(f"- Total issues (24h): {report['total_issues_24h']}")
    print(f"- Unresolved issues: {report['unresolved_issues']}")
    print(f"- Issues by type: {report['issues_by_type']}")
    print(f"- Active critical issues: {report['active_critical_issues']}")
    
    # Get active issues
    active_issues = monitor.get_active_issues()
    for issue in active_issues:
        print(f"Issue: {issue.description} (Severity: {issue.severity.name})")
    
    # Resolve first issue if any exist
    if active_issues:
        monitor.resolve_issue(active_issues[0].id, "Fixed data source configuration")
        print(f"Resolved issue: {active_issues[0].id}")

if __name__ == "__main__":
    # Run example
    asyncio.run(example_data_quality_monitoring())