"""
Logging utilities for InvestmentIQ MVAS

Provides structured logging capabilities for the entire system.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import json


class WorkflowLogger:
    """
    Structured logger for workflow execution and agent communication.

    Provides detailed logging with context preservation for debugging
    and audit trail purposes.
    """

    def __init__(
        self,
        name: str,
        log_dir: Optional[Path] = None,
        log_level: str = "INFO"
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Clear existing handlers
        self.logger.handlers = []

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File handler (if log_dir provided)
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"workflow_{timestamp}.log"

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

            self.log_file_path = log_file
        else:
            self.log_file_path = None

    def info(self, message: str, **kwargs) -> None:
        """Log info message with optional context"""
        if kwargs:
            message = f"{message} | Context: {json.dumps(kwargs)}"
        self.logger.info(message)

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message with optional context"""
        if kwargs:
            message = f"{message} | Context: {json.dumps(kwargs)}"
        self.logger.debug(message)

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message with optional context"""
        if kwargs:
            message = f"{message} | Context: {json.dumps(kwargs)}"
        self.logger.warning(message)

    def error(self, message: str, **kwargs) -> None:
        """Log error message with optional context"""
        if kwargs:
            message = f"{message} | Context: {json.dumps(kwargs)}"
        self.logger.error(message)

    def workflow_step(self, step_name: str, data: dict) -> None:
        """Log a workflow step with structured data"""
        self.logger.info(
            f"WORKFLOW_STEP: {step_name}",
            extra={"workflow_data": data}
        )

    def agent_communication(
        self,
        sender: str,
        receiver: str,
        message_type: str,
        payload: dict
    ) -> None:
        """Log agent-to-agent communication"""
        self.logger.info(
            f"A2A_COMMUNICATION: {sender} -> {receiver} [{message_type}]",
            extra={"a2a_payload": payload}
        )

    def get_log_path(self) -> Optional[Path]:
        """Get the path to the log file"""
        return self.log_file_path


def setup_logging(
    log_dir: Optional[Path] = None,
    log_level: str = "INFO"
) -> WorkflowLogger:
    """
    Setup logging for the application.

    Args:
        log_dir: Directory to store log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Configured WorkflowLogger instance
    """
    return WorkflowLogger(
        name="InvestmentIQ",
        log_dir=log_dir,
        log_level=log_level
    )
