"""
Fundamental Data Updater

This module handles updating fundamental data for stocks including:
- Earnings data and estimates
- Analyst estimates tracking
- Financial ratio calculations
- Data quality validation
- Automated data refresh
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf
from sqlalchemy import select

from ...cache.cache_manager import CacheManager
from ...database import get_db_session
from ...notifications import NotificationService
from ...portfolio_intelligence.models import (
    AnalystEstimate,
    EarningsData,
    FinancialRatio,
    StockData,
)

logger = logging.getLogger(__name__)


class FundamentalMetric(Enum):
    """Available fundamental metrics"""
    EARNINGS_PER_SHARE = "eps"
    REVENUE = "revenue"
    FREE_CASH_FLOW = "free_cash_flow"
    BOOK_VALUE = "book_value"
    DEBT_TO_EQUITY = "debt_to_equity"
    ROE = "roe"
    ROA = "roa"
    GROSS_MARGIN = "gross_margin"
    OPERATING_MARGIN = "operating_margin"
    NET_MARGIN = "net_margin"
    CURRENT_RATIO = "current_ratio"
    QUICK_RATIO = "quick_ratio"
    PEG_RATIO = "peg_ratio"


class DataQuality(Enum):
    """Data quality levels"""
    EXCELLENT = "excellent"  # Fresh, complete data
    GOOD = "good"  # Recent, mostly complete
    FAIR = "fair"  # Somewhat stale or incomplete
    POOR = "poor"  # Stale or very incomplete
    MISSING = "missing"  # No data available


@dataclass
class FundamentalDataPoint:
    """Single fundamental data point"""
    symbol: str
    metric: FundamentalMetric
    value: float
    period: str  # "Q1-2024", "FY-2023", etc.
    period_end_date: datetime
    reported_date: datetime
    source: str
    quality: DataQuality
    currency: str = "USD"
    unit: str = "dollars"  # "dollars", "shares", "ratio", "percentage"
    normalized_value: Optional[float] = None  # Per-share or normalized value


@dataclass
class EarningsUpdate:
    """Earnings update information"""
    symbol: str
    quarter: str
    fiscal_year: int
    reported_eps: Optional[float] = None
    estimated_eps: Optional[float] = None
    surprise_pct: Optional[float] = None
    revenue: Optional[float] = None
    revenue_estimate: Optional[float] = None
    revenue_surprise_pct: Optional[float] = None
    report_date: Optional[datetime] = None
    guidance_updated: bool = False
    guidance_direction: Optional[str] = None  # "RAISED", "LOWERED", "MAINTAINED"


@dataclass
class AnalystConsensus:
    """Analyst consensus data"""
    symbol: str
    target_price: Optional[float] = None
    rating: Optional[str] = None  # "BUY", "HOLD", "SELL"
    num_analysts: int = 0
    eps_current_year: Optional[float] = None
    eps_next_year: Optional[float] = None
    revenue_current_year: Optional[float] = None
    revenue_next_year: Optional[float] = None
    price_target_high: Optional[float] = None
    price_target_low: Optional[float] = None
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class FinancialRatios:
    """Financial ratios data"""
    symbol: str
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    peg_ratio: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None
    asset_turnover: Optional[float] = None
    inventory_turnover: Optional[float] = None
    calculated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class FundamentalUpdater:
    """Updates fundamental data for stocks"""

    def __init__(
        self,
        cache_manager: CacheManager,
        notification_service: NotificationService
    ):
        self.cache = cache_manager
        self.notification_service = notification_service

        # Data sources configuration
        self.data_sources = {
            "yfinance": {"enabled": True, "priority": 1},
            "alpha_vantage": {"enabled": False, "priority": 2},
            "financial_modeling_prep": {"enabled": False, "priority": 3}
        }

        # Request tracking for rate limiting
        self.request_counts: Dict[str, List[datetime]] = {
            source: [] for source in self.data_sources.keys()
        }

        # Update frequencies (in hours)
        self.update_frequencies = {
            FundamentalMetric.EARNINGS_PER_SHARE: 24,
            FundamentalMetric.REVENUE: 24,
            FundamentalMetric.FREE_CASH_FLOW: 48,
            FundamentalMetric.BOOK_VALUE: 48,
            FundamentalMetric.DEBT_TO_EQUITY: 48,
            FundamentalMetric.ROE: 24,
            FundamentalMetric.ROA: 24,
            FundamentalMetric.GROSS_MARGIN: 48,
            FundamentalMetric.OPERATING_MARGIN: 48,
            FundamentalMetric.NET_MARGIN: 48,
            FundamentalMetric.CURRENT_RATIO: 48,
            FundamentalMetric.QUICK_RATIO: 48,
            FundamentalMetric.PEG_RATIO: 24
        }

    async def update_fundamental_data(self, symbol: str) -> bool:
        """Update all fundamental data for a symbol"""
        try:
            logger.info(f"Updating fundamental data for {symbol}")

            # Check cache first
            cache_key = f"fundamental_update_{symbol}"
            last_update = await self.cache.get(cache_key)

            if last_update and self._should_skip_update(last_update):
                logger.debug(f"Skipping update for {symbol} - recently updated")
                return True

            # Update different data types
            tasks = [
                self._update_earnings_data(symbol),
                self._update_analyst_estimates(symbol),
                self._update_financial_ratios(symbol)
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check for failures
            failed_updates = [r for r in results if isinstance(r, Exception)]
            if failed_updates:
                logger.error(f"Some updates failed for {symbol}: {failed_updates}")

            # Update cache
            await self.cache.set(
                cache_key,
                datetime.now(timezone.utc).isoformat(),
                ttl=3600  # 1 hour
            )

            return len(failed_updates) == 0

            pass
        except Exception as e:
            logger.error(f"Error updating fundamental data for {symbol}: {e}")
            return False

    def _should_skip_update(self, last_update_str: str) -> bool:
        """Check if update should be skipped based on last update time"""
        try:
            last_update = datetime.fromisoformat(last_update_str.replace('Z', '+00:00'))
            time_diff = datetime.now(timezone.utc) - last_update
            return time_diff < timedelta(hours=1)  # Skip if updated within 1 hour
        except Exception:
            return False

    async def _update_earnings_data(self, symbol: str) -> bool:
        """Update earnings data for a symbol"""
        try:
            # Get data from Yahoo Finance
            ticker = yf.Ticker(symbol)

            # Get quarterly earnings
            quarterly_earnings = ticker.quarterly_earnings
            if quarterly_earnings is not None and not quarterly_earnings.empty:
                await self._store_earnings_data(symbol, quarterly_earnings, "quarterly")

            # Get annual earnings
            annual_earnings = ticker.earnings
            if annual_earnings is not None and not annual_earnings.empty:
                await self._store_earnings_data(symbol, annual_earnings, "annual")

            return True

            pass
        except Exception as e:
            logger.error(f"Error updating earnings data for {symbol}: {e}")
            return False

    async def _store_earnings_data(
        self, symbol: str, earnings_df: pd.DataFrame, period_type: str
    ):
        """Store earnings data in database"""
        async with get_db_session() as session:
            for date, row in earnings_df.iterrows():
                period = f"{'Q' if period_type == 'quarterly' else 'FY'}-{date.year}"

                # Check if data already exists
                result = await session.execute(
                    select(EarningsData).where(
                        EarningsData.symbol == symbol,
                        EarningsData.period == period
                    )
                )
                existing = result.scalar_one_or_none()

                if existing:
                    # Update existing record
                    for column in ["reported_eps", "revenue", "surprise_pct"]:
                        if column in row and not pd.isna(row[column]):
                            setattr(existing, column, float(row[column]))
                    existing.updated_at = datetime.now(timezone.utc)
                else:
                    # Create new record
                    db_earnings = EarningsData(
                        symbol=symbol,
                        period=period,
                        period_end_date=date,
                        reported_eps=float(row.get("Earnings", 0)) if not pd.isna(row.get("Earnings")) else None,
                        revenue=float(row.get("Revenue", 0)) if not pd.isna(row.get("Revenue")) else None,
                        created_at=datetime.now(timezone.utc),
                        updated_at=datetime.now(timezone.utc)
                    )
                    session.add(db_earnings)

            await session.commit()

    async def _update_analyst_estimates(self, symbol: str) -> bool:
        """Update analyst estimates for a symbol"""
        try:
            ticker = yf.Ticker(symbol)

            # Get analyst recommendations
            recommendations = ticker.recommendations
            if recommendations is not None and not recommendations.empty:
                latest_rec = recommendations.iloc[-1]

                consensus = AnalystConsensus(
                    symbol=symbol,
                    rating=latest_rec.get("To Grade", None),
                    target_price=latest_rec.get("Price Target", None),
                    num_analysts=len(recommendations)
                )

                await self._store_analyst_estimates(symbol, consensus)

            return True

            pass
        except Exception as e:
            logger.error(f"Error updating analyst estimates for {symbol}: {e}")
            return False

    async def _store_analyst_estimates(self, symbol: str, consensus: AnalystConsensus):
        """Store analyst estimates in database"""
        async with get_db_session() as session:
            # Check if estimate already exists
            result = await session.execute(
                select(AnalystEstimate).where(AnalystEstimate.symbol == symbol)
            )
            existing = result.scalar_one_or_none()

            if existing:
                # Update existing record
                for key, value in consensus.__dict__.items():
                    if hasattr(existing, key) and value is not None:
                        setattr(existing, key, value)
                existing.updated_at = datetime.now(timezone.utc)
            else:
                # Create new record
                db_estimate = AnalystEstimate(
                    symbol=symbol,
                    target_price=consensus.target_price,
                    rating=consensus.rating,
                    num_analysts=consensus.num_analysts,
                    updated_at=datetime.now(timezone.utc)
                )
                session.add(db_estimate)

            await session.commit()

    async def _update_financial_ratios(self, symbol: str) -> bool:
        """Update financial ratios for a symbol"""
        try:
            ticker = yf.Ticker(symbol)

            # Get key statistics
            info = ticker.info
            if not info:
                return False

            ratios = FinancialRatios(
                symbol=symbol,
                pe_ratio=info.get("trailingPE"),
                pb_ratio=info.get("priceToBook"),
                ps_ratio=info.get("priceToSalesTrailing12Months"),
                peg_ratio=info.get("pegRatio"),
                debt_to_equity=info.get("debtToEquity"),
                current_ratio=info.get("currentRatio"),
                quick_ratio=info.get("quickRatio"),
                roe=info.get("returnOnEquity"),
                roa=info.get("returnOnAssets"),
                gross_margin=info.get("grossMargins"),
                operating_margin=info.get("operatingMargins"),
                net_margin=info.get("profitMargins")
            )

            await self._store_financial_ratios(symbol, ratios)
            return True

            pass
        except Exception as e:
            logger.error(f"Error updating financial ratios for {symbol}: {e}")
            return False

    async def _store_financial_ratios(self, symbol: str, ratios: FinancialRatios):
        """Store financial ratios in database"""
        async with get_db_session() as session:
            # Check if ratios already exist
            result = await session.execute(
                select(FinancialRatio).where(FinancialRatio.symbol == symbol)
            )
            existing = result.scalar_one_or_none()

            if existing:
                # Update existing record
                for key, value in ratios.__dict__.items():
                    if hasattr(existing, key) and value is not None:
                        setattr(existing, key, value)
                existing.updated_at = datetime.now(timezone.utc)
            else:
                # Create new record
                db_ratios = FinancialRatio(
                    symbol=symbol,
                    pe_ratio=ratios.pe_ratio,
                    pb_ratio=ratios.pb_ratio,
                    ps_ratio=ratios.ps_ratio,
                    peg_ratio=ratios.peg_ratio,
                    debt_to_equity=ratios.debt_to_equity,
                    current_ratio=ratios.current_ratio,
                    quick_ratio=ratios.quick_ratio,
                    roe=ratios.roe,
                    roa=ratios.roa,
                    gross_margin=ratios.gross_margin,
                    operating_margin=ratios.operating_margin,
                    net_margin=ratios.net_margin,
                    updated_at=datetime.now(timezone.utc)
                )
                session.add(db_ratios)

            await session.commit()

    async def bulk_update_fundamentals(self, symbols: List[str]) -> Dict[str, bool]:
        """Update fundamental data for multiple symbols"""
        results = {}

        # Process in batches to avoid overwhelming data sources
        batch_size = 10
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]

            # Process batch concurrently
            tasks = [self.update_fundamental_data(symbol) for symbol in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Store results
            for symbol, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Error updating {symbol}: {result}")
                    results[symbol] = False
                else:
                    results[symbol] = result

            # Rate limiting - wait between batches
            await asyncio.sleep(1)

        return results

    async def get_data_quality_report(self, symbol: str) -> Dict[str, Any]:
        """Get data quality report for a symbol"""
        async with get_db_session() as session:
            report = {
                "symbol": symbol,
                "last_updated": None,
                "data_coverage": {},
                "quality_score": 0,
                "missing_metrics": []
            }

            # Check earnings data
            result = await session.execute(
                select(EarningsData).where(EarningsData.symbol == symbol)
                .order_by(EarningsData.updated_at.desc())
                .limit(1)
            )
            earnings = result.scalar_one_or_none()

            if earnings:
                report["data_coverage"]["earnings"] = True
                report["last_updated"] = earnings.updated_at
            else:
                report["data_coverage"]["earnings"] = False
                report["missing_metrics"].append("earnings")

            # Check analyst estimates
            result = await session.execute(
                select(AnalystEstimate).where(AnalystEstimate.symbol == symbol)
            )
            estimates = result.scalar_one_or_none()

            if estimates:
                report["data_coverage"]["analyst_estimates"] = True
            else:
                report["data_coverage"]["analyst_estimates"] = False
                report["missing_metrics"].append("analyst_estimates")

            # Check financial ratios
            result = await session.execute(
                select(FinancialRatio).where(FinancialRatio.symbol == symbol)
            )
            ratios = result.scalar_one_or_none()

            if ratios:
                report["data_coverage"]["financial_ratios"] = True
            else:
                report["data_coverage"]["financial_ratios"] = False
                report["missing_metrics"].append("financial_ratios")

            # Calculate quality score
            coverage_count = sum(report["data_coverage"].values())
            total_metrics = len(report["data_coverage"])
            report["quality_score"] = (coverage_count / total_metrics) * 100 if total_metrics > 0 else 0

            return report

    async def schedule_updates(self):
        """Schedule regular fundamental data updates"""
        logger.info("Starting scheduled fundamental data updates")

        while True:
            try:
                # Get active symbols from database
                async with get_db_session() as session:
                    result = await session.execute(
                        select(StockData.symbol).where(StockData.is_active)
                    )
                    symbols = [row[0] for row in result.fetchall()]

                if symbols:
                    logger.info(f"Updating fundamental data for {len(symbols)} symbols")
                    results = await self.bulk_update_fundamentals(symbols)

                    # Log summary
                    successful = sum(1 for success in results.values() if success)
                    failed = len(results) - successful
                    logger.info(f"Fundamental data update complete: {successful} successful, {failed} failed")

                # Wait before next update cycle (6 hours)
                await asyncio.sleep(6 * 3600)

                pass
            except Exception as e:
                logger.error(f"Error in scheduled fundamental updates: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour before retrying


# Example usage and testing
async def main():
    """Example usage of FundamentalUpdater"""

    # Initialize components
    cache_manager = CacheManager()
    notification_service = NotificationService()

    # Create updater
    updater = FundamentalUpdater(cache_manager, notification_service)

    # Test updating data for a single symbol
    test_symbols = ["AAPL", "MSFT", "GOOGL"]

    for symbol in test_symbols:
        success = await updater.update_fundamental_data(symbol)
        print(f"Update for {symbol}: {'SUCCESS' if success else 'FAILED'}")

        # Get quality report
        report = await updater.get_data_quality_report(symbol)
        print(f"Quality report for {symbol}: {report['quality_score']:.1f}% coverage")

        if report["missing_metrics"]:
            print(f"Missing metrics: {', '.join(report['missing_metrics'])}")


if __name__ == "__main__":
    asyncio.run(main())