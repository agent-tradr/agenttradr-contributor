"""
Models package - Re-exports all models from database.models.

This module provides backward compatibility for imports that expect
models to be available at src.models.
"""

from ..database.models import (
    User,
    UserProfile,
    Strategy,
    StrategySignal,
    StrategySubscription,
    PaymentMethod,
    Session,
    AuditLog,
    Transaction
)

# Create aliases for backward compatibility
Signal = StrategySignal
Subscription = StrategySubscription
Payment = Transaction

__all__ = [
    "User",
    "UserProfile",
    "Strategy",
    "StrategySignal",
    "StrategySubscription",
    "PaymentMethod",
    "Session",
    "AuditLog",
    "Transaction",
    "Signal",
    "Subscription",
    "Payment"
]