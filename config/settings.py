"""
Configuration settings for InvestmentIQ MVAS

Centralized configuration management for the system.
"""

from pathlib import Path
from typing import Dict, Any
import os


class Settings:
    """Application settings and configuration"""

    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data" / "mock"
    LOG_DIR = PROJECT_ROOT / "logs"
    OUTPUT_DIR = PROJECT_ROOT / "output"

    # Agent configuration
    AGENT_CONFIG = {
        "financial_analyst": {
            "agent_id": "financial_analyst_001",
            "timeout_seconds": 30,
            "retry_attempts": 3
        },
        "qualitative_signal": {
            "agent_id": "qualitative_signal_001",
            "timeout_seconds": 30,
            "retry_attempts": 3
        },
        "context_engine": {
            "agent_id": "context_engine_001",
            "timeout_seconds": 15,
            "retry_attempts": 2
        },
        "strategic_orchestrator": {
            "agent_id": "strategic_orchestrator_001",
            "timeout_seconds": 60,
            "retry_attempts": 1
        }
    }

    # Logging configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    # Workflow configuration
    WORKFLOW_CONFIG = {
        "enable_parallel_execution": True,
        "conflict_detection_enabled": True,
        "detailed_logging": True,
        "save_workflow_history": True
    }

    # Analysis thresholds
    THRESHOLDS = {
        "high_margin": 40.0,
        "moderate_margin": 30.0,
        "high_debt_ratio": 1.0,
        "moderate_debt_ratio": 0.5,
        "strong_sentiment_threshold": 0.6,
        "weak_sentiment_threshold": -0.6,
        "high_confidence": 0.7,
        "moderate_confidence": 0.5
    }

    @classmethod
    def ensure_directories(cls) -> None:
        """Ensure all required directories exist"""
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_agent_config(cls, agent_type: str) -> Dict[str, Any]:
        """Get configuration for a specific agent type"""
        return cls.AGENT_CONFIG.get(agent_type, {})

    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return {
            "project_root": str(cls.PROJECT_ROOT),
            "data_dir": str(cls.DATA_DIR),
            "log_dir": str(cls.LOG_DIR),
            "output_dir": str(cls.OUTPUT_DIR),
            "agent_config": cls.AGENT_CONFIG,
            "workflow_config": cls.WORKFLOW_CONFIG,
            "thresholds": cls.THRESHOLDS
        }
