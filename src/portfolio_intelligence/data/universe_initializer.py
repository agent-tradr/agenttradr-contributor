"""
This module initializes the universe with 150-200 quality stocks across sectors and themes, with quality screening and classification.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
import asyncio
import json
import yfinance as yf
import pandas as pd
from enum import Enum


class QualityRating(Enum):
    EXCELLENT = "EXCELLENT"  # A+ rating
    GOOD = "GOOD"  # A rating
    FAIR = "FAIR"  # B rating
    POOR = "POOR"  # C rating
    EXCLUDED = "EXCLUDED"  # Not tradeable


class MarketCap(Enum):
    MEGA = "MEGA"  # $200B+
    LARGE = "LARGE"  # $10B-$200B
    MID = "MID"  # $2B-$10B
    SMALL = "SMALL"  # $300M-$2B
    MICRO = "MICRO"  # Under $300M


@dataclass
class StockInfo:
    symbol: str
    name: str = None
    sector: str = None
    industry: str = None
    market_cap: float = None
    market_cap_category: MarketCap = None
    price: float = None
    volume: float = None
    avg_volume_30d: float = None
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    peg_ratio: Optional[float] = None
    price_to_book: Optional[float] = None
    debt_to_equity: Optional[float] = None
    roe: Optional[float] = None
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    beta: Optional[float] = None
    dividend_yield: Optional[float] = None
    quality_rating: QualityRating = None
    quality_score: float = None
    themes: List[str] = None
    exchange: str = None
    currency: str = None
    country: str = None
    employees: Optional[int] = None
    website: Optional[str] = None
    description: Optional[str] = None
    created_at: datetime = None
    updated_at: datetime = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'name': self.name,
            'sector': self.sector,
            'industry': self.industry,
            'market_cap': self.market_cap,
            'market_cap_category': self.market_cap_category.value if self.market_cap_category else None,
            'price': self.price,
            'volume': self.volume,
            'avg_volume_30d': self.avg_volume_30d,
            'pe_ratio': self.pe_ratio,
            'forward_pe': self.forward_pe,
            'peg_ratio': self.peg_ratio,
            'price_to_book': self.price_to_book,
            'debt_to_equity': self.debt_to_equity,
            'roe': self.roe,
            'revenue_growth': self.revenue_growth,
            'earnings_growth': self.earnings_growth,
            'beta': self.beta,
            'dividend_yield': self.dividend_yield,
            'quality_rating': self.quality_rating.value if self.quality_rating else None,
            'quality_score': self.quality_score,
            'themes': self.themes,
            'exchange': self.exchange,
            'currency': self.currency,
            'country': self.country,
            'employees': self.employees,
            'website': self.website,
            'description': self.description,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class UniverseInitializer:
    """
    Initializes and maintains the stock universe with quality screening.
    
    Creates a curated universe of 150-200 quality stocks across sectors and themes, 
    with comprehensive fundamental analysis and classification.
    """
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        
        # Universe parameters
        self.target_universe_size = 175  # Target number of stocks
        self.min_market_cap = 1_000_000_000  # $1B minimum
        self.min_avg_volume = 100_000  # 100k shares daily minimum
        self.min_price = 5.0  # $5 minimum price
        self.max_price = 1000.0  # $1000 maximum price
        
        # Quality thresholds
        self.quality_thresholds = {
            'min_revenue_growth': -0.10,
            'max_pe_ratio': 50.0,
            'min_roe': 0.05,
            'max_debt_to_equity': 2.0
        }
        
        # Base universe - Core S&P 500 stocks by sector
        self.base_universe = {
            'technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
            'healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO'],
            'financials': ['JPM', 'BAC', 'WFC', 'GS', 'MS'],
            'consumer_discretionary': ['TSLA', 'HD', 'MCD', 'NKE', 'SBUX'],
            'communication': ['DIS', 'NFLX', 'T', 'VZ', 'CMCSA'],
            'industrials': ['BA', 'CAT', 'GE', 'UPS', 'HON'],
            'energy': ['XOM', 'CVX', 'SLB', 'EOG', 'COP'],
            'materials': ['LIN', 'APD', 'SHW', 'FCX', 'NEM'],
            'utilities': ['NEE', 'SO', 'DUK', 'AEP', 'SRE'],
            'real_estate': ['AMT', 'PLD', 'CCI', 'EQIX', 'SPG']
        }
        
        # Theme classifications
        self.theme_classifications = {
            'ai_ml': ['NVDA', 'AMD', 'GOOGL', 'MSFT', 'AMZN'],
            'cloud': ['MSFT', 'AMZN', 'GOOGL', 'CRM', 'ORCL'],
            'ev': ['TSLA', 'GM', 'F', 'NIO', 'RIVN'],
            'renewable_energy': ['ENPH', 'SEDG', 'NEE', 'FSLR', 'BEP'],
            'biotechnology': ['GILD', 'AMGN', 'BIIB', 'REGN', 'VRTX']
        }
    
    async def initialize_universe(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Initialize the stock universe with quality screening.
        
        Args:
            force_refresh: Force refresh of all data
            
        Returns:
            Universe initialization results
        """
        try:
            self.logger.info("Starting universe initialization...")
            
            # Get all candidate symbols
            all_symbols = self._get_all_candidate_symbols()
            self.logger.info(f"Analyzing {len(all_symbols)} candidate symbols")
            
            # Fetch and analyze stock data
            analyzed_stocks = await self._analyze_stocks_batch(all_symbols, force_refresh=force_refresh)
            
            # Apply quality screening
            quality_stocks = self._apply_quality_screening(analyzed_stocks)
            
            # Select final universe
            final_universe = self._select_final_universe(quality_stocks)
            
            # Store in database
            await self._store_universe(final_universe)
            
            # Generate summary statistics
            summary = self._generate_universe_summary(final_universe)
            
            self.logger.info(f"Universe initialization complete: {len(final_universe)} stocks selected")
            return {
                'universe_size': len(final_universe),
                'summary': summary,
                'created_at': datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            self.logger.error(f"Failed to initialize universe: {e}")
            return {
                'error': str(e)
            }
    
    def _get_all_candidate_symbols(self) -> List[str]:
        """
        Get all candidate symbols from base universe and theme classifications.
        """
        all_symbols = set()
        
        # Add all base universe symbols
        for sector_symbols in self.base_universe.values():
            all_symbols.update(sector_symbols)
            
        # Add all theme symbols
        for theme_symbols in self.theme_classifications.values():
            all_symbols.update(theme_symbols)
            
        return sorted(list(all_symbols))
    
    async def _analyze_stocks_batch(self, symbols: List[str], force_refresh: bool = False) -> List[StockInfo]:
        """
        Analyze a batch of stock symbols.
        """
        # Process in batches to avoid rate limits
        batch_size = 20
        analyzed_stocks = []
        semaphore = asyncio.Semaphore(5)  # Limit concurrent requests
        
        async def analyze_single_stock(symbol: str) -> Optional[StockInfo]:
            async with semaphore:
                return await self._analyze_single_stock(symbol, force_refresh)
        
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            tasks = [analyze_single_stock(symbol) for symbol in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter valid results
            valid_results = [result for result in batch_results if isinstance(result, StockInfo)]
            analyzed_stocks.extend(valid_results)
            
            self.logger.info(
                f"Analyzed batch {i//batch_size + 1}/{(len(symbols)-1)//batch_size + 1}: "
                f"{len(valid_results)}/{len(batch)} valid"
            )
            
            # Rate limiting delay
            await asyncio.sleep(1)
            
        return analyzed_stocks
    
    async def _analyze_single_stock(self, symbol: str, force_refresh: bool = False) -> Optional[StockInfo]:
        """
        Analyze a single stock symbol.
        """
        try:
            # Get stock data from yfinance
            ticker = yf.Ticker(symbol)
            
            # Get basic info
            info = ticker.info
            if not info or 'symbol' not in info:
                return None
            
            # Get historical data for technical analysis
            hist = ticker.history(period="1y")
            if hist.empty:
                return None
                
            # Calculate current metrics
            current_price = hist['Close'].iloc[-1]
            current_volume = hist['Volume'].iloc[-1]
            avg_volume_30d = hist['Volume'].iloc[-30:].mean()
            
            # Extract fundamental data
            market_cap = info.get('marketCap', 0)
            if market_cap < self.min_market_cap:
                return None
                
            # Get sector classification
            sector = self._classify_sector(symbol, info.get('sector', 'Unknown'))
            industry = info.get('industry', 'Unknown')
            
            # Get theme classifications
            themes = self._get_symbol_themes(symbol)
            
            # Calculate quality metrics
            quality_score, quality_rating = self._calculate_quality_score(info, hist)
            
            # Create StockInfo object
            stock_info = StockInfo(
                symbol=symbol,
                name=info.get('longName', symbol),
                sector=sector,
                industry=industry,
                market_cap=market_cap,
                market_cap_category=self._classify_market_cap(market_cap),
                price=current_price,
                volume=current_volume,
                avg_volume_30d=avg_volume_30d,
                pe_ratio=info.get('trailingPE'),
                forward_pe=info.get('forwardPE'),
                peg_ratio=info.get('pegRatio'),
                price_to_book=info.get('priceToBook'),
                debt_to_equity=info.get('debtToEquity'),
                roe=info.get('returnOnEquity'),
                revenue_growth=info.get('revenueGrowth'),
                earnings_growth=info.get('earningsGrowth'),
                beta=info.get('beta'),
                dividend_yield=info.get('dividendYield'),
                quality_rating=quality_rating,
                quality_score=quality_score,
                themes=themes,
                exchange=info.get('exchange', ''),
                currency=info.get('currency', 'USD'),
                country=info.get('country', 'US'),
                employees=info.get('fullTimeEmployees'),
                website=info.get('website'),
                description=info.get('longBusinessSummary'),
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )
            
            return stock_info
        except Exception as e:
            self.logger.error(f"Failed to analyze {symbol}: {e}")
            return None

    def _classify_sector(self, symbol: str, yf_sector: str) -> str:
        """
        Classify stock sector based on base universe mapping.
        """
        for sector, symbols in self.base_universe.items():
            if symbol in symbols:
                return sector
        
        # Fallback mapping from yfinance sectors
        sector_mapping = {
            'Technology': 'technology',
            'Healthcare': 'healthcare',
            'Financial Services': 'financials',
            'Consumer Cyclical': 'consumer_discretionary',
            'Communication Services': 'communication',
            'Industrials': 'industrials',
            'Energy': 'energy',
            'Basic Materials': 'materials',
            'Utilities': 'utilities',
            'Real Estate': 'real_estate'
        }
        return sector_mapping.get(yf_sector, 'other')

    def _get_symbol_themes(self, symbol: str) -> List[str]:
        """
        Get theme classifications for a symbol.
        """
        themes = []
        for theme, symbols in self.theme_classifications.items():
            if symbol in symbols:
                themes.append(theme)
        return themes

    def _classify_market_cap(self, market_cap: float) -> MarketCap:
        """
        Classify market cap category.
        """
        if market_cap >= 200_000_000_000:
            return MarketCap.MEGA
        elif market_cap >= 10_000_000_000:
            return MarketCap.LARGE
        elif market_cap >= 2_000_000_000:
            return MarketCap.MID
        elif market_cap >= 300_000_000:
            return MarketCap.SMALL
        else:
            return MarketCap.MICRO

    def _calculate_quality_score(self, info: Dict[str, Any], hist: pd.DataFrame) -> Tuple[float, QualityRating]:
        """
        Calculate quality score and rating for a stock.
        """
        scores = []
        
        # Financial health score (0-1)
        financial_score = 0.0
        financial_factors = 0
        
        # ROE check
        roe = info.get('returnOnEquity')
        if roe is not None:
            financial_score += min(1.0, max(0.0, roe / 0.25))  # 25% ROE = 1.0
            financial_factors += 1
        
        # Debt-to-equity check
        debt_equity = info.get('debtToEquity')
        if debt_equity is not None:
            financial_score += max(0.0, 1.0 - debt_equity / 2.0)  # Lower debt = higher score
            financial_factors += 1
        
        # Revenue growth
        revenue_growth = info.get('revenueGrowth')
        if revenue_growth is not None:
            financial_score += max(0.0, min(1.0, (revenue_growth + 0.10) / 0.20))
            financial_factors += 1
        
        if financial_factors > 0:
            scores.append(financial_score / financial_factors)

        # Valuation score (0-1)
        valuation_score = 0.0
        valuation_factors = 0
        
        # P/E ratio check
        pe_ratio = info.get('trailingPE')
        if pe_ratio is not None and pe_ratio > 0:
            valuation_score += max(0.0, 1.0 - pe_ratio / 50.0)  # Lower P/E = higher score
            valuation_factors += 1
        
        # PEG ratio check
        peg_ratio = info.get('pegRatio')
        if peg_ratio is not None and peg_ratio > 0:
            valuation_score += max(0.0, min(1.0, 2.0 - peg_ratio))  # PEG < 1 is good
            valuation_factors += 1
        
        if valuation_factors > 0:
            scores.append(valuation_score / valuation_factors)

        # Liquidity score (0-1)
        liquidity_score = 0.0
        avg_volume = hist['Volume'].iloc[-30:].mean() if len(hist) >= 30 else hist['Volume'].mean()
        current_price = hist['Close'].iloc[-1]
        daily_dollar_volume = avg_volume * current_price
        
        if daily_dollar_volume >= 10_000_000:
            # $10M+ = excellent liquidity
            liquidity_score = 1.0
        elif daily_dollar_volume >= 1_000_000:
            # $1M+ = good liquidity
            liquidity_score = 0.7
        elif daily_dollar_volume >= 100_000:
            # $100k+ = fair liquidity
            liquidity_score = 0.4
        else:
            # Poor liquidity
            liquidity_score = 0.1
        
        scores.append(liquidity_score)

        # Price stability score (0-1)
        if len(hist) >= 252:  # At least 1 year of data
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * (252 ** 0.5)  # Annualized volatility
            stability_score = max(0.0, 1.0 - volatility / 0.5)  # 50% vol = 0 score
            scores.append(stability_score)

        # Calculate overall quality score
        if not scores:
            overall_score = 0.0
        else:
            overall_score = sum(scores) / len(scores)

        # Convert to rating
        if overall_score >= 0.8:
            rating = QualityRating.EXCELLENT
        elif overall_score >= 0.6:
            rating = QualityRating.GOOD
        elif overall_score >= 0.4:
            rating = QualityRating.FAIR
        elif overall_score >= 0.2:
            rating = QualityRating.POOR
        else:
            rating = QualityRating.EXCLUDED
        
        return overall_score, rating

    def _apply_quality_screening(self, stocks: List[StockInfo]) -> List[StockInfo]:
        """
        Apply quality screening filters to stocks.
        """
        quality_stocks = []
        
        for stock in stocks:
            # Basic filters
            if (stock.price < self.min_price or stock.price > self.max_price or
                stock.avg_volume_30d < self.min_avg_volume):
                continue
            
            # Quality rating filter
            if stock.quality_rating == QualityRating.EXCLUDED:
                continue
            
            # Specific metric filters
            if (stock.revenue_growth is not None and 
                stock.revenue_growth < self.quality_thresholds['min_revenue_growth']):
                continue
            
            if (stock.pe_ratio is not None and 
                stock.pe_ratio > self.quality_thresholds['max_pe_ratio']):
                continue
            
            if (stock.roe is not None and 
                stock.roe < self.quality_thresholds['min_roe']):
                continue
            
            if (stock.debt_to_equity is not None and 
                stock.debt_to_equity > self.quality_thresholds['max_debt_to_equity']):
                continue
            
            quality_stocks.append(stock)
        
        self.logger.info(f"Quality screening: {len(quality_stocks)}/{len(stocks)} passed")
        return quality_stocks

    def _select_final_universe(self, quality_stocks: List[StockInfo]) -> List[StockInfo]:
        """
        Select final universe from quality stocks with diversification.
        """
        if len(quality_stocks) <= self.target_universe_size:
            return quality_stocks

        # Sort by quality score descending
        quality_stocks.sort(key=lambda s: s.quality_score, reverse=True)
        
        # Ensure sector diversification
        selected_stocks = []
        sector_counts = {}
        max_per_sector = max(15, self.target_universe_size // len(self.base_universe))
        
        # First pass: select best from each sector
        for stock in quality_stocks:
            sector_count = sector_counts.get(stock.sector, 0)
            if sector_count < max_per_sector:
                selected_stocks.append(stock)
                sector_counts[stock.sector] = sector_count + 1
                
                if len(selected_stocks) >= self.target_universe_size:
                    break

        # Second pass: fill remaining slots with highest quality
        if len(selected_stocks) < self.target_universe_size:
            remaining_stocks = [stock for stock in quality_stocks if stock not in selected_stocks]
            remaining_slots = self.target_universe_size - len(selected_stocks)
            selected_stocks.extend(remaining_stocks[:remaining_slots])

        self.logger.info(f"Final selection: {len(selected_stocks)} stocks from "
                        f"{len(quality_stocks)} quality candidates")
        return selected_stocks

    async def _store_universe(self, universe: List[StockInfo]) -> None:
        """
        Store universe in database.
        """
        try:
            # Clear existing universe
            await self.db_manager.execute_query("DELETE FROM stock_universe")
            
            # Insert new universe
            query = """
            INSERT INTO stock_universe (
                symbol, name, sector, industry, market_cap, market_cap_category,
                price, volume, avg_volume_30d, pe_ratio, forward_pe, peg_ratio,
                price_to_book, debt_to_equity, roe, revenue_growth, earnings_growth,
                beta, dividend_yield, quality_rating, quality_score, themes,
                exchange, currency, country, employees, website, description,
                created_at, updated_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            for stock in universe:
                await self.db_manager.execute_query(query, [
                    stock.symbol, stock.name, stock.sector, stock.industry,
                    stock.market_cap, stock.market_cap_category.value,
                    stock.price, stock.volume, stock.avg_volume_30d,
                    stock.pe_ratio, stock.forward_pe, stock.peg_ratio,
                    stock.price_to_book, stock.debt_to_equity, stock.roe,
                    stock.revenue_growth, stock.earnings_growth, stock.beta,
                    stock.dividend_yield, stock.quality_rating.value,
                    stock.quality_score, json.dumps(stock.themes),
                    stock.exchange, stock.currency, stock.country,
                    stock.employees, stock.website, stock.description,
                    stock.created_at, stock.updated_at
                ])
                
            self.logger.info(f"Stored {len(universe)} stocks in database")
        except Exception as e:
            self.logger.error(f"Failed to store universe: {e}")

    def _generate_universe_summary(self, universe: List[StockInfo]) -> Dict[str, Any]:
        """
        Generate summary statistics for the universe.
        """
        quality_breakdown = {}
        sector_breakdown = {}
        theme_breakdown = {}
        market_cap_breakdown = {}
        
        for stock in universe:
            # Quality breakdown
            quality = stock.quality_rating.value
            quality_breakdown[quality] = quality_breakdown.get(quality, 0) + 1
            
            # Sector breakdown
            sector_breakdown[stock.sector] = sector_breakdown.get(stock.sector, 0) + 1
            
            # Market cap breakdown
            cap_category = stock.market_cap_category.value
            market_cap_breakdown[cap_category] = market_cap_breakdown.get(cap_category, 0) + 1
            
            # Theme breakdown
            for theme in stock.themes:
                theme_breakdown[theme] = theme_breakdown.get(theme, 0) + 1

        return {
            'total_stocks': len(universe),
            'quality_breakdown': quality_breakdown,
            'sector_breakdown': sector_breakdown,
            'market_cap_breakdown': market_cap_breakdown,
            'theme_breakdown': theme_breakdown,
            'avg_quality_score': sum(s.quality_score for s in universe) / len(universe),
            'avg_market_cap': sum(s.market_cap for s in universe) / len(universe),
            'total_market_cap': sum(s.market_cap for s in universe)
        }

    async def get_universe_stats(self) -> Dict[str, Any]:
        """
        Get current universe statistics from database.
        """
        try:
            query = """
            SELECT COUNT(*) as total_stocks,
                   AVG(quality_score) as avg_quality_score,
                   AVG(market_cap) as avg_market_cap,
                   SUM(market_cap) as total_market_cap
            FROM stock_universe
            """
            result = await self.db_manager.fetch_one(query)
            
            if not result:
                return {'total_stocks': 0}
            
            return {
                'total_stocks': int(result[0]),
                'avg_quality_score': float(result[1]) if result[1] else 0.0,
                'avg_market_cap': float(result[2]) if result[2] else 0.0,
                'total_market_cap': float(result[3]) if result[3] else 0.0
            }
        except Exception as e:
            self.logger.error(f"Failed to get universe stats: {e}")
            return {'error': str(e)}

    async def refresh_universe_data(self, symbols: List[str] = None) -> Dict[str, Any]:
        """
        Refresh universe data for specified symbols or all symbols.
        """
        try:
            # Get all universe symbols if none specified
            if not symbols:
                query = "SELECT symbol FROM stock_universe"
                rows = await self.db_manager.fetch_all(query)
                symbols = [row[0] for row in rows]
            
            # Re-analyze specified symbols
            updated_stocks = await self._analyze_stocks_batch(symbols, force_refresh=True)
            
            # Update database
            query = """
            UPDATE stock_universe SET
                price = %s, volume = %s, avg_volume_30d = %s,
                quality_score = %s, quality_rating = %s,
                updated_at = %s
            WHERE symbol = %s
            """
            
            for stock in updated_stocks:
                await self.db_manager.execute_query(query, [
                    stock.price, stock.volume, stock.avg_volume_30d,
                    stock.quality_score, stock.quality_rating.value,
                    datetime.now(timezone.utc), stock.symbol
                ])
            
            return {
                'updated_symbols': len(updated_stocks),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            self.logger.error(f"Failed to refresh universe data: {e}")
            return {'error': str(e)}