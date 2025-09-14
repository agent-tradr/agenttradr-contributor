"""
Comprehensive unit tests for src/models/strategy.py
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from typing import Dict, Any

from src.models.strategy import StrategyType, PerformanceMetrics, Strategy


class TestStrategyType:
    """Test StrategyType class"""

    def test_strategy_type_constants(self):
        """Test that StrategyType has all expected constants"""
        assert StrategyType.TREND_FOLLOWING == "trend_following"
        assert StrategyType.MEAN_REVERSION == "mean_reversion"
        assert StrategyType.MOMENTUM == "momentum"
        assert StrategyType.ML_ENHANCED == "ml_enhanced"
        assert StrategyType.EXPERIMENTAL == "experimental"
        assert StrategyType.HYBRID == "hybrid"

    def test_strategy_type_values_are_strings(self):
        """Test that all strategy type values are strings"""
        assert isinstance(StrategyType.TREND_FOLLOWING, str)
        assert isinstance(StrategyType.MEAN_REVERSION, str)
        assert isinstance(StrategyType.MOMENTUM, str)
        assert isinstance(StrategyType.ML_ENHANCED, str)
        assert isinstance(StrategyType.EXPERIMENTAL, str)
        assert isinstance(StrategyType.HYBRID, str)

    def test_strategy_type_uniqueness(self):
        """Test that all strategy types are unique"""
        types = [
            StrategyType.TREND_FOLLOWING,
            StrategyType.MEAN_REVERSION,
            StrategyType.MOMENTUM,
            StrategyType.ML_ENHANCED,
            StrategyType.EXPERIMENTAL,
            StrategyType.HYBRID
        ]
        assert len(types) == len(set(types))


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass"""

    def test_default_initialization(self):
        """Test PerformanceMetrics with default values"""
        metrics = PerformanceMetrics()
        assert metrics.sharpe_ratio == 0.0
        assert metrics.total_return == 0.0
        assert metrics.max_drawdown == 0.0
        assert metrics.win_rate == 0.0
        assert metrics.total_trades == 0
        assert metrics.avg_trade_return == 0.0
        assert metrics.volatility == 0.0
        assert metrics.sortino_ratio == 0.0
        assert metrics.calmar_ratio == 0.0

    def test_custom_initialization(self):
        """Test PerformanceMetrics with custom values"""
        metrics = PerformanceMetrics(
            sharpe_ratio=1.5,
            total_return=0.25,
            max_drawdown=-0.15,
            win_rate=0.65,
            total_trades=100,
            avg_trade_return=0.002,
            volatility=0.18,
            sortino_ratio=2.1,
            calmar_ratio=1.8
        )
        assert metrics.sharpe_ratio == 1.5
        assert metrics.total_return == 0.25
        assert metrics.max_drawdown == -0.15
        assert metrics.win_rate == 0.65
        assert metrics.total_trades == 100
        assert metrics.avg_trade_return == 0.002
        assert metrics.volatility == 0.18
        assert metrics.sortino_ratio == 2.1
        assert metrics.calmar_ratio == 1.8

    def test_to_dict(self):
        """Test conversion to dictionary"""
        metrics = PerformanceMetrics(
            sharpe_ratio=1.5,
            total_return=0.25,
            max_drawdown=-0.15,
            win_rate=0.65,
            total_trades=100,
            avg_trade_return=0.002,
            volatility=0.18,
            sortino_ratio=2.1,
            calmar_ratio=1.8
        )
        
        result = metrics.to_dict()
        
        assert isinstance(result, dict)
        assert result["sharpe_ratio"] == 1.5
        assert result["total_return"] == 0.25
        assert result["max_drawdown"] == -0.15
        assert result["win_rate"] == 0.65
        assert result["total_trades"] == 100
        assert result["avg_trade_return"] == 0.002
        assert result["volatility"] == 0.18
        assert result["sortino_ratio"] == 2.1
        assert result["calmar_ratio"] == 1.8

    def test_from_dict(self):
        """Test creation from dictionary"""
        data = {
            "sharpe_ratio": 1.5,
            "total_return": 0.25,
            "max_drawdown": -0.15,
            "win_rate": 0.65,
            "total_trades": 100,
            "avg_trade_return": 0.002,
            "volatility": 0.18,
            "sortino_ratio": 2.1,
            "calmar_ratio": 1.8
        }
        
        metrics = PerformanceMetrics.from_dict(data)
        
        assert metrics.sharpe_ratio == 1.5
        assert metrics.total_return == 0.25
        assert metrics.max_drawdown == -0.15
        assert metrics.win_rate == 0.65
        assert metrics.total_trades == 100
        assert metrics.avg_trade_return == 0.002
        assert metrics.volatility == 0.18
        assert metrics.sortino_ratio == 2.1
        assert metrics.calmar_ratio == 1.8

    def test_from_dict_with_missing_fields(self):
        """Test from_dict with partial data uses defaults for missing fields"""
        data = {
            "sharpe_ratio": 1.5,
            "total_return": 0.25
        }
        
        # Since dataclass uses defaults, this should work with defaults for missing fields
        metrics = PerformanceMetrics.from_dict(data)
        assert metrics.sharpe_ratio == 1.5
        assert metrics.total_return == 0.25
        # Other fields should have their default values
        assert metrics.max_drawdown == 0.0
        assert metrics.win_rate == 0.0
        assert metrics.total_trades == 0
        assert metrics.avg_trade_return == 0.0
        assert metrics.volatility == 0.0
        assert metrics.sortino_ratio == 0.0
        assert metrics.calmar_ratio == 0.0

    def test_to_dict_from_dict_roundtrip(self):
        """Test that to_dict and from_dict are inverses"""
        original = PerformanceMetrics(
            sharpe_ratio=1.5,
            total_return=0.25,
            max_drawdown=-0.15,
            win_rate=0.65,
            total_trades=100,
            avg_trade_return=0.002,
            volatility=0.18,
            sortino_ratio=2.1,
            calmar_ratio=1.8
        )
        
        dict_repr = original.to_dict()
        recreated = PerformanceMetrics.from_dict(dict_repr)
        
        assert recreated.sharpe_ratio == original.sharpe_ratio
        assert recreated.total_return == original.total_return
        assert recreated.max_drawdown == original.max_drawdown
        assert recreated.win_rate == original.win_rate
        assert recreated.total_trades == original.total_trades
        assert recreated.avg_trade_return == original.avg_trade_return
        assert recreated.volatility == original.volatility
        assert recreated.sortino_ratio == original.sortino_ratio
        assert recreated.calmar_ratio == original.calmar_ratio

    def test_negative_values(self):
        """Test that negative values are handled correctly"""
        metrics = PerformanceMetrics(
            sharpe_ratio=-1.0,
            total_return=-0.5,
            max_drawdown=-0.5,
            win_rate=0.0,
            total_trades=0,
            avg_trade_return=-0.01,
            volatility=0.0,
            sortino_ratio=-2.0,
            calmar_ratio=-1.5
        )
        
        assert metrics.sharpe_ratio == -1.0
        assert metrics.total_return == -0.5
        assert metrics.max_drawdown == -0.5


class TestStrategy:
    """Test Strategy dataclass"""

    def test_basic_initialization(self):
        """Test basic Strategy initialization"""
        strategy = Strategy(
            id="test_id",
            name="Test Strategy",
            type=StrategyType.MOMENTUM,
            genes={"param1": 0.5, "param2": 100},
            code="def execute(): pass"
        )
        
        assert strategy.id == "test_id"
        assert strategy.name == "Test Strategy"
        assert strategy.type == StrategyType.MOMENTUM
        assert strategy.genes == {"param1": 0.5, "param2": 100}
        assert strategy.code == "def execute(): pass"
        assert strategy.darwin_score == 0.0
        assert strategy.generation == 0
        assert strategy.parents == []
        assert isinstance(strategy.birth_time, datetime)
        assert strategy.performance is None
        assert strategy.mutations == []
        assert strategy.species is None
        assert strategy.is_alive is True
        assert strategy.death_time is None
        assert strategy.death_reason is None

    def test_full_initialization(self):
        """Test Strategy initialization with all parameters"""
        birth_time = datetime.now()
        death_time = birth_time + timedelta(days=1)
        performance = PerformanceMetrics(sharpe_ratio=1.5, total_return=0.25)
        
        strategy = Strategy(
            id="test_id",
            name="Test Strategy",
            type=StrategyType.ML_ENHANCED,
            genes={"param1": 0.5},
            code="code",
            darwin_score=85.5,
            generation=5,
            parents=["parent1", "parent2"],
            birth_time=birth_time,
            performance=performance,
            mutations=["mutation1", "mutation2"],
            species="species_a",
            is_alive=False,
            death_time=death_time,
            death_reason="Poor performance"
        )
        
        assert strategy.darwin_score == 85.5
        assert strategy.generation == 5
        assert strategy.parents == ["parent1", "parent2"]
        assert strategy.birth_time == birth_time
        assert strategy.performance == performance
        assert strategy.mutations == ["mutation1", "mutation2"]
        assert strategy.species == "species_a"
        assert strategy.is_alive is False
        assert strategy.death_time == death_time
        assert strategy.death_reason == "Poor performance"

    def test_hash(self):
        """Test Strategy hash function"""
        strategy1 = Strategy(
            id="test_id",
            name="Test Strategy",
            type=StrategyType.MOMENTUM,
            genes={},
            code=""
        )
        strategy2 = Strategy(
            id="test_id",
            name="Different Name",
            type=StrategyType.TREND_FOLLOWING,
            genes={},
            code=""
        )
        strategy3 = Strategy(
            id="different_id",
            name="Test Strategy",
            type=StrategyType.MOMENTUM,
            genes={},
            code=""
        )
        
        # Same ID should have same hash
        assert hash(strategy1) == hash(strategy2)
        # Different ID should have different hash
        assert hash(strategy1) != hash(strategy3)

    def test_equality(self):
        """Test Strategy equality comparison"""
        strategy1 = Strategy(
            id="test_id",
            name="Test Strategy",
            type=StrategyType.MOMENTUM,
            genes={},
            code=""
        )
        strategy2 = Strategy(
            id="test_id",
            name="Different Name",
            type=StrategyType.TREND_FOLLOWING,
            genes={},
            code=""
        )
        strategy3 = Strategy(
            id="different_id",
            name="Test Strategy",
            type=StrategyType.MOMENTUM,
            genes={},
            code=""
        )
        
        # Same ID means equal
        assert strategy1 == strategy2
        # Different ID means not equal
        assert strategy1 != strategy3
        # Not equal to non-Strategy objects
        assert strategy1 != "not a strategy"
        assert strategy1 != 123
        assert strategy1 != None

    def test_to_dict_basic(self):
        """Test basic Strategy to_dict conversion"""
        birth_time = datetime(2024, 1, 1, 12, 0, 0)
        strategy = Strategy(
            id="test_id",
            name="Test Strategy",
            type=StrategyType.MOMENTUM,
            genes={"param1": 0.5},
            code="def execute(): pass",  # This won't be in the output
            birth_time=birth_time
        )
        
        result = strategy.to_dict()
        
        assert result["id"] == "test_id"
        assert result["name"] == "Test Strategy"
        assert result["type"] == StrategyType.MOMENTUM
        assert result["genes"] == {"param1": 0.5}
        assert result["darwin_score"] == 0.0
        assert result["generation"] == 0
        assert result["parents"] == []
        assert result["birth_time"] == birth_time.isoformat()
        assert result["performance"] is None
        assert result["mutations"] == []
        assert result["species"] is None
        assert result["is_alive"] is True
        assert result["death_time"] is None
        assert result["death_reason"] is None
        # Verify 'code' is not included in to_dict output
        assert "code" not in result

    def test_to_dict_with_performance(self):
        """Test Strategy to_dict with performance metrics"""
        performance = PerformanceMetrics(sharpe_ratio=1.5, total_return=0.25)
        strategy = Strategy(
            id="test_id",
            name="Test Strategy",
            type=StrategyType.MOMENTUM,
            genes={},
            code="",
            performance=performance
        )
        
        result = strategy.to_dict()
        
        assert result["performance"] is not None
        assert result["performance"]["sharpe_ratio"] == 1.5
        assert result["performance"]["total_return"] == 0.25

    def test_to_dict_with_enum_type(self):
        """Test Strategy to_dict handles enum-like type objects"""
        # Create a mock enum-like object
        mock_type = MagicMock()
        mock_type.value = "custom_type"
        
        strategy = Strategy(
            id="test_id",
            name="Test Strategy",
            type=mock_type,
            genes={},
            code=""
        )
        
        result = strategy.to_dict()
        assert result["type"] == "custom_type"

    def test_to_dict_complete(self):
        """Test Strategy to_dict with all fields populated"""
        birth_time = datetime(2024, 1, 1, 12, 0, 0)
        death_time = datetime(2024, 1, 2, 12, 0, 0)
        performance = PerformanceMetrics(sharpe_ratio=1.5)
        
        strategy = Strategy(
            id="test_id",
            name="Test Strategy",
            type=StrategyType.HYBRID,
            genes={"param1": 0.5},
            code="code",
            darwin_score=75.0,
            generation=3,
            parents=["p1", "p2"],
            birth_time=birth_time,
            performance=performance,
            mutations=["m1", "m2"],
            species="species_a",
            is_alive=False,
            death_time=death_time,
            death_reason="Obsolete"
        )
        
        result = strategy.to_dict()
        
        assert result["darwin_score"] == 75.0
        assert result["generation"] == 3
        assert result["parents"] == ["p1", "p2"]
        assert result["mutations"] == ["m1", "m2"]
        assert result["species"] == "species_a"
        assert result["is_alive"] is False
        assert result["death_time"] == death_time.isoformat()
        assert result["death_reason"] == "Obsolete"

    def test_from_dict_basic(self):
        """Test basic Strategy from_dict conversion"""
        data = {
            "id": "test_id",
            "name": "Test Strategy",
            "type": StrategyType.MOMENTUM,
            "genes": {"param1": 0.5},
            "code": "def execute(): pass"
        }
        
        strategy = Strategy.from_dict(data)
        
        assert strategy.id == "test_id"
        assert strategy.name == "Test Strategy"
        assert strategy.type == StrategyType.MOMENTUM
        assert strategy.genes == {"param1": 0.5}
        assert strategy.code == "def execute(): pass"
        assert strategy.darwin_score == 0.0
        assert strategy.generation == 0
        assert strategy.parents == []

    def test_from_dict_with_defaults(self):
        """Test from_dict handles missing optional fields"""
        data = {
            "id": "test_id",
            "name": "Test Strategy",
            "type": "custom_type",
            "genes": {"param1": 0.5}
        }
        
        strategy = Strategy.from_dict(data)
        
        assert strategy.code == ""
        assert strategy.darwin_score == 0.0
        assert strategy.generation == 0
        assert strategy.parents == []
        assert strategy.mutations == []
        assert strategy.species is None
        assert strategy.is_alive is True

    def test_from_dict_with_performance_dict(self):
        """Test from_dict with performance as dictionary"""
        data = {
            "id": "test_id",
            "name": "Test Strategy",
            "type": StrategyType.MOMENTUM,
            "genes": {},
            "performance": {
                "sharpe_ratio": 1.5,
                "total_return": 0.25,
                "max_drawdown": -0.15,
                "win_rate": 0.65,
                "total_trades": 100,
                "avg_trade_return": 0.002,
                "volatility": 0.18,
                "sortino_ratio": 2.1,
                "calmar_ratio": 1.8
            }
        }
        
        strategy = Strategy.from_dict(data)
        
        assert strategy.performance is not None
        assert isinstance(strategy.performance, PerformanceMetrics)
        assert strategy.performance.sharpe_ratio == 1.5
        assert strategy.performance.total_return == 0.25

    def test_from_dict_with_performance_object(self):
        """Test from_dict with performance as object"""
        performance = PerformanceMetrics(sharpe_ratio=2.0)
        data = {
            "id": "test_id",
            "name": "Test Strategy",
            "type": StrategyType.MOMENTUM,
            "genes": {},
            "performance": performance
        }
        
        strategy = Strategy.from_dict(data)
        
        assert strategy.performance == performance
        assert strategy.performance.sharpe_ratio == 2.0

    def test_from_dict_with_dates(self):
        """Test from_dict with date strings"""
        birth_time = datetime(2024, 1, 1, 12, 0, 0)
        death_time = datetime(2024, 1, 2, 12, 0, 0)
        
        data = {
            "id": "test_id",
            "name": "Test Strategy",
            "type": StrategyType.MOMENTUM,
            "genes": {},
            "birth_time": birth_time.isoformat(),
            "death_time": death_time.isoformat(),
            "death_reason": "Obsolete"
        }
        
        strategy = Strategy.from_dict(data)
        
        assert strategy.birth_time == birth_time
        assert strategy.death_time == death_time
        assert strategy.death_reason == "Obsolete"

    def test_from_dict_complete(self):
        """Test from_dict with all fields"""
        data = {
            "id": "test_id",
            "name": "Test Strategy",
            "type": StrategyType.EXPERIMENTAL,
            "genes": {"param1": 0.5, "param2": 100},
            "code": "code",
            "darwin_score": 80.0,
            "generation": 5,
            "parents": ["p1", "p2", "p3"],
            "birth_time": "2024-01-01T12:00:00",
            "performance": {
                "sharpe_ratio": 1.5,
                "total_return": 0.25,
                "max_drawdown": -0.15,
                "win_rate": 0.65,
                "total_trades": 100,
                "avg_trade_return": 0.002,
                "volatility": 0.18,
                "sortino_ratio": 2.1,
                "calmar_ratio": 1.8
            },
            "mutations": ["mutation1", "mutation2"],
            "species": "species_b",
            "is_alive": False,
            "death_time": "2024-01-02T12:00:00",
            "death_reason": "Replaced"
        }
        
        strategy = Strategy.from_dict(data)
        
        assert strategy.id == "test_id"
        assert strategy.name == "Test Strategy"
        assert strategy.type == StrategyType.EXPERIMENTAL
        assert strategy.genes == {"param1": 0.5, "param2": 100}
        assert strategy.code == "code"
        assert strategy.darwin_score == 80.0
        assert strategy.generation == 5
        assert strategy.parents == ["p1", "p2", "p3"]
        assert strategy.mutations == ["mutation1", "mutation2"]
        assert strategy.species == "species_b"
        assert strategy.is_alive is False
        assert strategy.death_reason == "Replaced"

    def test_to_dict_from_dict_roundtrip(self):
        """Test that to_dict and from_dict are inverses for Strategy
        Note: 'code' field is not included in to_dict() so it won't be preserved
        """
        birth_time = datetime(2024, 1, 1, 12, 0, 0)
        death_time = datetime(2024, 1, 2, 12, 0, 0)
        performance = PerformanceMetrics(sharpe_ratio=1.5, total_return=0.25)
        
        original = Strategy(
            id="test_id",
            name="Test Strategy",
            type=StrategyType.HYBRID,
            genes={"param1": 0.5, "param2": 100},
            code="original code",  # Note: code is not serialized in to_dict
            darwin_score=75.0,
            generation=3,
            parents=["p1", "p2"],
            birth_time=birth_time,
            performance=performance,
            mutations=["m1", "m2"],
            species="species_a",
            is_alive=False,
            death_time=death_time,
            death_reason="Obsolete"
        )
        
        dict_repr = original.to_dict()
        recreated = Strategy.from_dict(dict_repr)
        
        assert recreated.id == original.id
        assert recreated.name == original.name
        assert recreated.type == original.type
        assert recreated.genes == original.genes
        # Code is not included in to_dict, so it defaults to empty string on from_dict
        assert recreated.code == ""  # Default value when not in dict
        assert recreated.darwin_score == original.darwin_score
        assert recreated.generation == original.generation
        assert recreated.parents == original.parents
        assert recreated.birth_time == original.birth_time
        assert recreated.mutations == original.mutations
        assert recreated.species == original.species
        assert recreated.is_alive == original.is_alive
        assert recreated.death_time == original.death_time
        assert recreated.death_reason == original.death_reason
        
        # Check performance separately due to object comparison
        assert recreated.performance.sharpe_ratio == original.performance.sharpe_ratio
        assert recreated.performance.total_return == original.performance.total_return

    def test_is_valid_true(self):
        """Test is_valid returns True for valid strategy"""
        strategy = Strategy(
            id="test_id",
            name="Test Strategy",
            type=StrategyType.MOMENTUM,
            genes={"param1": 0.5},
            code="",
            darwin_score=50.0
        )
        
        assert strategy.is_valid() is True

    def test_is_valid_false_no_id(self):
        """Test is_valid returns False when id is empty"""
        strategy = Strategy(
            id="",
            name="Test Strategy",
            type=StrategyType.MOMENTUM,
            genes={"param1": 0.5},
            code=""
        )
        
        assert strategy.is_valid() is False

    def test_is_valid_false_no_name(self):
        """Test is_valid returns False when name is empty"""
        strategy = Strategy(
            id="test_id",
            name="",
            type=StrategyType.MOMENTUM,
            genes={"param1": 0.5},
            code=""
        )
        
        assert strategy.is_valid() is False

    def test_is_valid_false_no_genes(self):
        """Test is_valid returns False when genes is empty"""
        strategy = Strategy(
            id="test_id",
            name="Test Strategy",
            type=StrategyType.MOMENTUM,
            genes={},
            code=""
        )
        
        assert strategy.is_valid() is False

    def test_is_valid_false_negative_darwin_score(self):
        """Test is_valid returns False when darwin_score is negative"""
        strategy = Strategy(
            id="test_id",
            name="Test Strategy",
            type=StrategyType.MOMENTUM,
            genes={"param1": 0.5},
            code="",
            darwin_score=-1.0
        )
        
        assert strategy.is_valid() is False

    def test_is_valid_false_darwin_score_over_100(self):
        """Test is_valid returns False when darwin_score > 100"""
        strategy = Strategy(
            id="test_id",
            name="Test Strategy",
            type=StrategyType.MOMENTUM,
            genes={"param1": 0.5},
            code="",
            darwin_score=101.0
        )
        
        assert strategy.is_valid() is False

    def test_is_valid_edge_cases(self):
        """Test is_valid with edge case darwin scores"""
        # Darwin score = 0 should be valid
        strategy1 = Strategy(
            id="test_id",
            name="Test Strategy",
            type=StrategyType.MOMENTUM,
            genes={"param1": 0.5},
            code="",
            darwin_score=0.0
        )
        assert strategy1.is_valid() is True
        
        # Darwin score = 100 should be valid
        strategy2 = Strategy(
            id="test_id",
            name="Test Strategy",
            type=StrategyType.MOMENTUM,
            genes={"param1": 0.5},
            code="",
            darwin_score=100.0
        )
        assert strategy2.is_valid() is True

    def test_get_age_in_generations(self):
        """Test get_age_in_generations calculation"""
        strategy = Strategy(
            id="test_id",
            name="Test Strategy",
            type=StrategyType.MOMENTUM,
            genes={"param1": 0.5},
            code="",
            generation=5
        )
        
        assert strategy.get_age_in_generations(10) == 5
        assert strategy.get_age_in_generations(5) == 0
        assert strategy.get_age_in_generations(3) == -2  # Can be negative if current < birth

    def test_get_age_in_generations_edge_cases(self):
        """Test get_age_in_generations with edge cases"""
        strategy = Strategy(
            id="test_id",
            name="Test Strategy",
            type=StrategyType.MOMENTUM,
            genes={"param1": 0.5},
            code="",
            generation=0
        )
        
        assert strategy.get_age_in_generations(0) == 0
        assert strategy.get_age_in_generations(100) == 100
        assert strategy.get_age_in_generations(-5) == -5

    def test_strategy_with_none_values(self):
        """Test Strategy handles None values appropriately"""
        strategy = Strategy(
            id="test_id",
            name="Test Strategy",
            type=StrategyType.MOMENTUM,
            genes={"param1": None, "param2": 0.5},
            code="",
            species=None,
            death_time=None,
            death_reason=None
        )
        
        assert strategy.genes["param1"] is None
        assert strategy.species is None
        assert strategy.death_time is None
        assert strategy.death_reason is None

    def test_strategy_with_empty_collections(self):
        """Test Strategy with empty lists"""
        strategy = Strategy(
            id="test_id",
            name="Test Strategy",
            type=StrategyType.MOMENTUM,
            genes={"param1": 0.5},
            code="",
            parents=[],
            mutations=[]
        )
        
        assert strategy.parents == []
        assert strategy.mutations == []
        assert len(strategy.parents) == 0
        assert len(strategy.mutations) == 0

    def test_strategy_with_large_collections(self):
        """Test Strategy with large collections"""
        parents = [f"parent_{i}" for i in range(100)]
        mutations = [f"mutation_{i}" for i in range(1000)]
        
        strategy = Strategy(
            id="test_id",
            name="Test Strategy",
            type=StrategyType.MOMENTUM,
            genes={"param1": 0.5},
            code="",
            parents=parents,
            mutations=mutations
        )
        
        assert len(strategy.parents) == 100
        assert len(strategy.mutations) == 1000
        assert strategy.parents[0] == "parent_0"
        assert strategy.parents[-1] == "parent_99"
        assert strategy.mutations[0] == "mutation_0"
        assert strategy.mutations[-1] == "mutation_999"

    def test_strategy_immutability_of_dataclass_fields(self):
        """Test that dataclass fields maintain expected behavior"""
        strategy = Strategy(
            id="test_id",
            name="Test Strategy",
            type=StrategyType.MOMENTUM,
            genes={"param1": 0.5},
            code=""
        )
        
        # Modify fields
        strategy.darwin_score = 75.0
        strategy.generation = 10
        strategy.is_alive = False
        
        assert strategy.darwin_score == 75.0
        assert strategy.generation == 10
        assert strategy.is_alive is False

    def test_strategy_genes_mutation(self):
        """Test that genes dictionary can be modified"""
        genes = {"param1": 0.5}
        strategy = Strategy(
            id="test_id",
            name="Test Strategy",
            type=StrategyType.MOMENTUM,
            genes=genes,
            code=""
        )
        
        # Modify genes through strategy
        strategy.genes["param2"] = 1.0
        assert strategy.genes["param2"] == 1.0
        assert len(strategy.genes) == 2
        
        # Original dict is also modified (shared reference)
        assert genes["param2"] == 1.0

    def test_strategy_type_can_be_any_string(self):
        """Test that strategy type can be any string, not just StrategyType constants"""
        strategy = Strategy(
            id="test_id",
            name="Test Strategy",
            type="custom_strategy_type",
            genes={"param1": 0.5},
            code=""
        )
        
        assert strategy.type == "custom_strategy_type"
        
        # Should still work with to_dict/from_dict
        dict_repr = strategy.to_dict()
        assert dict_repr["type"] == "custom_strategy_type"
        
        recreated = Strategy.from_dict(dict_repr)
        assert recreated.type == "custom_strategy_type"

    def test_strategy_hashable_in_set(self):
        """Test that Strategy objects can be used in sets"""
        strat1 = Strategy(
            id="strat_001",
            name="Strategy 1",
            type=StrategyType.MOMENTUM,
            genes={"p": 1},
            code=""
        )
        strat2 = Strategy(
            id="strat_002",
            name="Strategy 2",
            type=StrategyType.HYBRID,
            genes={"p": 2},
            code=""
        )
        strat3 = Strategy(
            id="strat_001",  # Same ID as strat1
            name="Different Name",
            type=StrategyType.EXPERIMENTAL,
            genes={"p": 3},
            code=""
        )
        
        strategy_set = {strat1, strat2, strat3}
        # strat3 should not be added because it has same ID as strat1
        assert len(strategy_set) == 2
        assert strat1 in strategy_set
        assert strat2 in strategy_set
        assert strat3 in strategy_set  # True because strat3 == strat1
    
    def test_performance_metrics_extreme_values(self):
        """Test PerformanceMetrics with extreme and special float values"""
        metrics = PerformanceMetrics(
            sharpe_ratio=float('inf'),
            total_return=float('-inf'),
            max_drawdown=float('nan'),
            win_rate=1.0,
            total_trades=2147483647,  # Max int32
            avg_trade_return=1e-10,  # Very small value
            volatility=1e10,  # Very large value
            sortino_ratio=-0.0,  # Negative zero
            calmar_ratio=0.0
        )
        
        assert metrics.sharpe_ratio == float('inf')
        assert metrics.total_return == float('-inf')
        assert metrics.win_rate == 1.0
        assert metrics.total_trades == 2147483647
        assert metrics.avg_trade_return == 1e-10
        assert metrics.volatility == 1e10
        
        # NaN is special - it's not equal to itself
        import math
        assert math.isnan(metrics.max_drawdown)
    
    def test_strategy_complex_nested_genes(self):
        """Test Strategy with deeply nested and complex genes structure"""
        complex_genes = {
            "indicators": {
                "moving_averages": {
                    "sma": {"periods": [10, 20, 50, 200], "weights": [0.1, 0.2, 0.3, 0.4]},
                    "ema": {"periods": [12, 26], "smoothing": 2.0}
                },
                "oscillators": {
                    "rsi": {"period": 14, "levels": {"overbought": 70, "oversold": 30}},
                    "macd": {"fast": 12, "slow": 26, "signal": 9}
                }
            },
            "risk_management": {
                "position_sizing": {
                    "method": "kelly_criterion",
                    "max_position": 0.25,
                    "scaling": {"enabled": True, "factor": 1.5}
                },
                "stop_loss": {
                    "type": "trailing",
                    "percentage": 0.02,
                    "atr_multiplier": 2.0
                }
            },
            "filters": [
                {"type": "volume", "min_volume": 1000000},
                {"type": "volatility", "max_volatility": 0.5},
                {"type": "trend", "min_strength": 0.7}
            ],
            "metadata": {
                "version": "2.0.1",
                "created_at": "2024-01-01",
                "tags": ["high_frequency", "mean_reversion", "tested"]
            }
        }
        
        strategy = Strategy(
            id="complex_001",
            name="Complex Nested Strategy",
            type=StrategyType.ML_ENHANCED,
            genes=complex_genes,
            code="# Complex strategy implementation"
        )
        
        # Verify deep access works
        assert strategy.genes["indicators"]["moving_averages"]["sma"]["periods"][2] == 50
        assert strategy.genes["risk_management"]["position_sizing"]["scaling"]["enabled"] is True
        assert strategy.genes["filters"][1]["type"] == "volatility"
        
        # Test serialization/deserialization preserves complex structure
        dict_repr = strategy.to_dict()
        restored = Strategy.from_dict(dict_repr)
        assert restored.genes == complex_genes
    
    def test_strategy_unicode_and_special_characters(self):
        """Test Strategy with unicode and special characters in fields"""
        strategy = Strategy(
            id="test_📈_id",
            name="Strategy 名前 🚀 €£¥",
            type="типа_стратегии",
            genes={
                "参数1": 0.5,
                "παράμετρος": 1.0,
                "emoji_param_🎯": True,
                "special!@#$%^&*()": "value"
            },
            code="# コメント\ndef 実行(): pass",
            species="物种_🧬"
        )
        
        assert strategy.id == "test_📈_id"
        assert strategy.name == "Strategy 名前 🚀 €£¥"
        assert strategy.genes["参数1"] == 0.5
        assert strategy.genes["emoji_param_🎯"] is True
        
        # Test serialization handles unicode properly
        dict_repr = strategy.to_dict()
        restored = Strategy.from_dict(dict_repr)
        assert restored.name == strategy.name
        assert restored.genes == strategy.genes
    
    def test_strategy_from_dict_invalid_date_format(self):
        """Test from_dict with various invalid date formats"""
        # Invalid ISO format
        data1 = {
            "id": "test_id",
            "name": "Test",
            "type": "test",
            "genes": {"p": 1},
            "birth_time": "2024-13-01T12:00:00"  # Invalid month
        }
        
        with pytest.raises(ValueError):
            Strategy.from_dict(data1)
        
        # Non-ISO format
        data2 = {
            "id": "test_id",
            "name": "Test",
            "type": "test",
            "genes": {"p": 1},
            "death_time": "01/01/2024 12:00:00"  # Wrong format
        }
        
        with pytest.raises(ValueError):
            Strategy.from_dict(data2)
    
    def test_strategy_very_long_strings(self):
        """Test Strategy with very long string fields"""
        long_id = "id_" + "x" * 10000
        long_name = "Strategy " + "A" * 50000
        long_code = "# " + "code " * 20000
        long_species = "species_" + "B" * 5000
        long_death_reason = "Reason: " + "failed " * 1000
        
        strategy = Strategy(
            id=long_id,
            name=long_name,
            type=StrategyType.EXPERIMENTAL,
            genes={"param": 1},
            code=long_code,
            species=long_species,
            death_reason=long_death_reason
        )
        
        assert len(strategy.id) > 10000
        assert len(strategy.name) > 50000
        assert len(strategy.code) > 100000
        
        # Test serialization works with long strings
        dict_repr = strategy.to_dict()
        assert dict_repr["id"] == long_id
        assert dict_repr["name"] == long_name
        
        restored = Strategy.from_dict(dict_repr)
        assert restored.id == long_id
        # Note: code is not serialized in to_dict
        assert restored.code == ""  # Default when not in dict
    
    def test_mutations_field_operations(self):
        """Test various operations on the mutations field"""
        strategy = Strategy(
            id="mut_test",
            name="Mutation Test",
            type=StrategyType.EXPERIMENTAL,
            genes={"base": 1.0},
            code=""
        )
        
        # Test append
        strategy.mutations.append("mutation1")
        assert "mutation1" in strategy.mutations
        
        # Test extend
        strategy.mutations.extend(["mutation2", "mutation3"])
        assert len(strategy.mutations) == 3
        
        # Test insert
        strategy.mutations.insert(1, "inserted_mutation")
        assert strategy.mutations[1] == "inserted_mutation"
        assert len(strategy.mutations) == 4
        
        # Test remove
        strategy.mutations.remove("inserted_mutation")
        assert "inserted_mutation" not in strategy.mutations
        assert len(strategy.mutations) == 3
        
        # Test clear
        strategy.mutations.clear()
        assert len(strategy.mutations) == 0
    
    def test_performance_metrics_boundary_values(self):
        """Test PerformanceMetrics with boundary values for financial metrics"""
        # Test reasonable boundaries for financial metrics
        metrics = PerformanceMetrics(
            sharpe_ratio=10.0,  # Very high but possible
            total_return=10.0,  # 1000% return
            max_drawdown=-1.0,  # 100% drawdown
            win_rate=0.0,  # All losses
            total_trades=0,  # No trades
            avg_trade_return=0.0,
            volatility=0.0,  # No volatility (impossible in practice)
            sortino_ratio=-10.0,  # Very negative
            calmar_ratio=0.0  # Zero ratio
        )
        
        assert metrics.sharpe_ratio == 10.0
        assert metrics.total_return == 10.0
        assert metrics.max_drawdown == -1.0
        assert metrics.win_rate == 0.0
        
        # Test with all maximum win rate
        metrics2 = PerformanceMetrics(
            win_rate=1.0,  # 100% win rate
            total_trades=1000000  # Million trades
        )
        
        assert metrics2.win_rate == 1.0
        assert metrics2.total_trades == 1000000
    
    def test_strategy_birth_time_default(self):
        """Test that birth_time has a default value close to now"""
        before = datetime.now()
        strategy = Strategy(
            id="mock_time",
            name="Mock Time Test",
            type=StrategyType.MOMENTUM,
            genes={"p": 1},
            code=""
        )
        after = datetime.now()
        
        # Birth time should be between before and after
        assert before <= strategy.birth_time <= after
    
    def test_strategy_with_concurrent_modifications(self):
        """Test Strategy behavior with concurrent field modifications"""
        strategy = Strategy(
            id="concurrent",
            name="Concurrent Test",
            type=StrategyType.HYBRID,
            genes={"param1": 0.5, "param2": 1.0},
            code="initial_code"
        )
        
        # Simulate concurrent modifications
        original_genes = strategy.genes.copy()
        
        # Modify genes
        strategy.genes["param3"] = 2.0
        strategy.genes["param1"] = 0.7
        
        # Modify other fields
        strategy.darwin_score = 85.0
        strategy.generation = 10
        strategy.code = "modified_code"
        
        # Verify modifications
        assert strategy.genes["param1"] == 0.7
        assert strategy.genes["param3"] == 2.0
        assert strategy.darwin_score == 85.0
        assert strategy.code == "modified_code"
        
        # Original genes dict should be different
        assert original_genes != strategy.genes
    
    def test_strategy_comparison_with_different_types(self):
        """Test Strategy equality comparison with various types"""
        strategy = Strategy(
            id="test_id",
            name="Test",
            type=StrategyType.MOMENTUM,
            genes={"p": 1},
            code=""
        )
        
        # Test comparison with different types
        assert strategy != None
        assert strategy != []
        assert strategy != {}
        assert strategy != 123
        assert strategy != 123.45
        assert strategy != "test_id"
        assert strategy != True
        assert strategy != object()
        
        # Test comparison with dict representation
        assert strategy != strategy.to_dict()
    
    def test_parents_field_edge_cases(self):
        """Test edge cases for the parents field"""
        # Empty parents
        strategy1 = Strategy(
            id="no_parents",
            name="Orphan Strategy",
            type=StrategyType.EXPERIMENTAL,
            genes={"p": 1},
            code="",
            parents=[]
        )
        assert len(strategy1.parents) == 0
        
        # Single parent (unusual but valid)
        strategy2 = Strategy(
            id="single_parent",
            name="Single Parent Strategy",
            type=StrategyType.EXPERIMENTAL,
            genes={"p": 1},
            code="",
            parents=["parent_001"]
        )
        assert len(strategy2.parents) == 1
        
        # Many parents (multi-parent crossover)
        many_parents = [f"parent_{i:05d}" for i in range(10000)]
        strategy3 = Strategy(
            id="many_parents",
            name="Many Parents Strategy",
            type=StrategyType.HYBRID,
            genes={"p": 1},
            code="",
            parents=many_parents
        )
        assert len(strategy3.parents) == 10000
        
        # Duplicate parents (shouldn't happen but test handling)
        strategy4 = Strategy(
            id="dup_parents",
            name="Duplicate Parents Strategy",
            type=StrategyType.HYBRID,
            genes={"p": 1},
            code="",
            parents=["parent_001", "parent_001", "parent_002"]
        )
        assert len(strategy4.parents) == 3
        assert strategy4.parents.count("parent_001") == 2
    
    def test_strategy_dataclass_frozen_behavior(self):
        """Test that Strategy dataclass is not frozen (fields can be modified)"""
        strategy = Strategy(
            id="frozen_test",
            name="Frozen Test",
            type=StrategyType.MOMENTUM,
            genes={"p": 1},
            code="initial"
        )
        
        # Should be able to modify fields since dataclass is not frozen
        strategy.id = "modified_id"
        strategy.name = "Modified Name"
        strategy.darwin_score = 95.0
        
        assert strategy.id == "modified_id"
        assert strategy.name == "Modified Name"
        assert strategy.darwin_score == 95.0
    
    def test_strategy_with_none_type(self):
        """Test Strategy behavior when type is None (edge case)"""
        strategy = Strategy(
            id="none_type",
            name="None Type Strategy",
            type=None,  # Edge case: None type
            genes={"p": 1},
            code=""
        )
        
        assert strategy.type is None
        
        # Test serialization with None type
        dict_repr = strategy.to_dict()
        assert dict_repr["type"] is None
        
        # Test deserialization
        restored = Strategy.from_dict(dict_repr)
        assert restored.type is None
    
    def test_performance_metrics_partial_from_dict(self):
        """Test PerformanceMetrics.from_dict with only some fields provided"""
        partial_data = {
            "sharpe_ratio": 2.5,
            "total_return": 0.5,
            "max_drawdown": -0.2
            # Missing other fields
        }
        
        # This should work as dataclass has defaults for all fields
        metrics = PerformanceMetrics.from_dict(partial_data)
        assert metrics.sharpe_ratio == 2.5
        assert metrics.total_return == 0.5
        assert metrics.max_drawdown == -0.2
        # Other fields should have defaults
        assert metrics.win_rate == 0.0
        assert metrics.total_trades == 0
        assert metrics.avg_trade_return == 0.0
    
    def test_strategy_genes_deep_copy_behavior(self):
        """Test that genes dictionary modifications affect the strategy"""
        original_genes = {"param1": 0.5, "nested": {"inner": 1.0}}
        strategy = Strategy(
            id="deep_copy_test",
            name="Deep Copy Test",
            type=StrategyType.MOMENTUM,
            genes=original_genes,
            code=""
        )
        
        # Modify nested structure through strategy
        strategy.genes["nested"]["inner"] = 2.0
        strategy.genes["nested"]["new_inner"] = 3.0
        
        # Check that original is also modified (shared reference)
        assert original_genes["nested"]["inner"] == 2.0
        assert original_genes["nested"]["new_inner"] == 3.0
    
    def test_strategy_is_valid_with_none_fields(self):
        """Test is_valid when fields are None"""
        # Test with None id
        strategy = Strategy(
            id=None,
            name="Test",
            type=StrategyType.MOMENTUM,
            genes={"p": 1},
            code=""
        )
        assert strategy.is_valid() is False
        
        # Test with None name
        strategy.id = "valid_id"
        strategy.name = None
        assert strategy.is_valid() is False
        
        # Test with None genes
        strategy.name = "Valid Name"
        strategy.genes = None
        assert strategy.is_valid() is False
    
    def test_strategy_darwin_score_precision(self):
        """Test Darwin score with various precision levels"""
        strategy = Strategy(
            id="precision_test",
            name="Precision Test",
            type=StrategyType.MOMENTUM,
            genes={"p": 1},
            code=""
        )
        
        # Test various precision levels
        test_scores = [
            0.0000001,  # Very small positive
            99.9999999,  # Just under 100
            50.123456789,  # Many decimal places
            1e-8,  # Scientific notation small
            100.0 - 1e-10  # Very close to 100
        ]
        
        for score in test_scores:
            strategy.darwin_score = score
            assert strategy.is_valid() is True
        
        # Test boundary violations with high precision
        strategy.darwin_score = -0.0000001
        assert strategy.is_valid() is False
        
        strategy.darwin_score = 100.0000001
        assert strategy.is_valid() is False
    
    def test_strategy_code_with_special_content(self):
        """Test Strategy code field with various special content"""
        special_codes = [
            "",  # Empty code
            " ",  # Whitespace only
            "\n\n\n",  # Only newlines
            "a" * 1000000,  # Very long code
            "print('Hello')\n" * 10000,  # Repetitive code
            "# -*- coding: utf-8 -*-\n# 中文注释\nprint('测试')",  # With encoding
            "def f():\n    pass\n\x00\x01\x02",  # With control characters
        ]
        
        for code in special_codes:
            strategy = Strategy(
                id="code_test",
                name="Code Test",
                type=StrategyType.MOMENTUM,
                genes={"p": 1},
                code=code
            )
            
            assert strategy.code == code
            # Note: code is not included in to_dict output
            dict_repr = strategy.to_dict()
            assert "code" not in dict_repr
    
    def test_strategy_species_variations(self):
        """Test species field with various values"""
        species_values = [
            None,
            "",
            "simple_species",
            "species-with-dash",
            "species_with_underscore",
            "Species With Spaces",
            "species.with.dots",
            "species/with/slashes",
            "🧬 species with emoji",
            "species" * 100  # Long species name
        ]
        
        for species in species_values:
            strategy = Strategy(
                id="species_test",
                name="Species Test",
                type=StrategyType.EXPERIMENTAL,
                genes={"p": 1},
                code="",
                species=species
            )
            
            assert strategy.species == species
            
            # Test serialization
            dict_repr = strategy.to_dict()
            assert dict_repr["species"] == species
            
            # Test deserialization
            restored = Strategy.from_dict(dict_repr)
            assert restored.species == species
    
    def test_strategy_death_scenarios(self):
        """Test various death scenarios for strategies"""
        base_time = datetime(2024, 1, 1)
        
        # Test immediate death
        strategy1 = Strategy(
            id="immediate_death",
            name="Immediate Death",
            type=StrategyType.EXPERIMENTAL,
            genes={"p": 1},
            code="",
            birth_time=base_time,
            is_alive=False,
            death_time=base_time,  # Died at birth
            death_reason="Initialization failure"
        )
        
        assert strategy1.get_age_in_generations(0) == 0
        
        # Test death with None death_time (inconsistent state)
        strategy2 = Strategy(
            id="inconsistent_death",
            name="Inconsistent Death",
            type=StrategyType.EXPERIMENTAL,
            genes={"p": 1},
            code="",
            is_alive=False,
            death_time=None,  # Dead but no death time
            death_reason="Unknown"
        )
        
        assert strategy2.is_alive is False
        assert strategy2.death_time is None
        
        # Test alive with death_time set (inconsistent state)
        strategy3 = Strategy(
            id="zombie",
            name="Zombie Strategy",
            type=StrategyType.EXPERIMENTAL,
            genes={"p": 1},
            code="",
            is_alive=True,  # Alive but has death time
            death_time=base_time + timedelta(days=1),
            death_reason="Should be dead"
        )
        
        assert strategy3.is_alive is True
        assert strategy3.death_time is not None
    
    def test_strategy_generation_edge_cases(self):
        """Test generation field edge cases"""
        # Negative generation (shouldn't happen but test handling)
        strategy1 = Strategy(
            id="negative_gen",
            name="Negative Generation",
            type=StrategyType.MOMENTUM,
            genes={"p": 1},
            code="",
            generation=-5
        )
        
        assert strategy1.generation == -5
        assert strategy1.get_age_in_generations(0) == 5
        assert strategy1.get_age_in_generations(-10) == -5
        
        # Very large generation
        strategy2 = Strategy(
            id="large_gen",
            name="Large Generation",
            type=StrategyType.MOMENTUM,
            genes={"p": 1},
            code="",
            generation=2**31 - 1  # Max int32
        )
        
        assert strategy2.generation == 2**31 - 1
        
        # Test overflow behavior with get_age_in_generations
        assert strategy2.get_age_in_generations(2**31) == 1
    
    def test_mutations_with_duplicates(self):
        """Test mutations list with duplicate entries"""
        mutations = [
            "mutation_a",
            "mutation_b",
            "mutation_a",  # Duplicate
            "mutation_c",
            "mutation_b",  # Another duplicate
        ]
        
        strategy = Strategy(
            id="dup_mutations",
            name="Duplicate Mutations",
            type=StrategyType.EXPERIMENTAL,
            genes={"p": 1},
            code="",
            mutations=mutations
        )
        
        # List preserves duplicates
        assert len(strategy.mutations) == 5
        assert strategy.mutations.count("mutation_a") == 2
        assert strategy.mutations.count("mutation_b") == 2
        
        # Test serialization preserves duplicates
        dict_repr = strategy.to_dict()
        assert dict_repr["mutations"] == mutations
        
        restored = Strategy.from_dict(dict_repr)
        assert restored.mutations == mutations
    
    def test_strategy_collection_operations(self):
        """Test using Strategy in various collection operations"""
        strategies = []
        for i in range(5):
            strategies.append(Strategy(
                id=f"strat_{i}",
                name=f"Strategy {i}",
                type=StrategyType.MOMENTUM,
                genes={"param": i},
                code=""
            ))
        
        # Test in list
        assert len(strategies) == 5
        assert strategies[0].id == "strat_0"
        
        # Test in dictionary as values
        strategy_dict = {s.id: s for s in strategies}
        assert len(strategy_dict) == 5
        assert strategy_dict["strat_3"].genes["param"] == 3
        
        # Test in dictionary as keys (using strategy as key)
        strategy_as_key_dict = {s: s.generation for s in strategies}
        assert len(strategy_as_key_dict) == 5
        
        # Test sorting by darwin_score
        strategies[0].darwin_score = 90
        strategies[1].darwin_score = 70
        strategies[2].darwin_score = 95
        strategies[3].darwin_score = 60
        strategies[4].darwin_score = 80
        
        sorted_strategies = sorted(strategies, key=lambda s: s.darwin_score)
        assert sorted_strategies[0].darwin_score == 60
        assert sorted_strategies[-1].darwin_score == 95
        
        # Test filtering
        high_performers = [s for s in strategies if s.darwin_score >= 80]
        assert len(high_performers) == 3
    
    def test_strategy_type_attribute_error_handling(self):
        """Test to_dict handling when type has no 'value' attribute"""
        class MockTypeWithoutValue:
            def __str__(self):
                return "mock_type_str"
        
        strategy = Strategy(
            id="mock_type_test",
            name="Mock Type Test",
            type=MockTypeWithoutValue(),
            genes={"p": 1},
            code=""
        )
        
        # Should use the object itself when no 'value' attribute
        dict_repr = strategy.to_dict()
        assert isinstance(dict_repr["type"], MockTypeWithoutValue)
    
    def test_performance_metrics_json_serialization(self):
        """Test that PerformanceMetrics dict can be JSON serialized"""
        import json
        
        metrics = PerformanceMetrics(
            sharpe_ratio=1.5,
            total_return=0.25,
            max_drawdown=-0.15,
            win_rate=0.65,
            total_trades=100,
            avg_trade_return=0.002,
            volatility=0.18,
            sortino_ratio=2.1,
            calmar_ratio=1.8
        )
        
        dict_repr = metrics.to_dict()
        
        # Should be JSON serializable
        json_str = json.dumps(dict_repr)
        assert isinstance(json_str, str)
        
        # Should be able to reconstruct from JSON
        loaded = json.loads(json_str)
        reconstructed = PerformanceMetrics.from_dict(loaded)
        
        assert reconstructed.sharpe_ratio == metrics.sharpe_ratio
        assert reconstructed.total_return == metrics.total_return
    
    def test_strategy_memory_efficiency(self):
        """Test Strategy memory usage with large numbers of instances"""
        import sys
        
        # Create a strategy and check its size
        strategy = Strategy(
            id="memory_test",
            name="Memory Test",
            type=StrategyType.MOMENTUM,
            genes={"p": 1},
            code="def execute(): pass"
        )
        
        # Get size in bytes (this is implementation-dependent)
        size = sys.getsizeof(strategy)
        assert size > 0  # Should have some size
        
        # Test that dataclass creates efficient instances
        strategies = []
        for i in range(1000):
            strategies.append(Strategy(
                id=f"mem_{i}",
                name=f"Memory Strategy {i}",
                type=StrategyType.MOMENTUM,
                genes={"param": i},
                code="pass"
            ))
        
        # All strategies should be created successfully
        assert len(strategies) == 1000
        assert all(s.id == f"mem_{i}" for i, s in enumerate(strategies))