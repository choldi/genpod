"""Custom logger configuration for LightTTS."""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional


class LightTTSLogger:
    """Custom logger wrapper for LightTTS application."""
    
    _instance: Optional['LightTTSLogger'] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._loggers = {}
        self._setup_root_logger()
    
    def _setup_root_logger(self):
        """Configure root logger based on environment variables."""
        # Get settings from environment
        log_level = os.getenv("LOGLEVEL", "INFO").upper()
        log_dest = os.getenv("LOGDEST", "console").lower()
        log_path = os.getenv("LOGPATH", "./logs/lighttts.log")
        
        # Parse log level
        numeric_level = getattr(logging, log_level, logging.INFO)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(numeric_level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Setup handlers based on LOGDEST
        handlers = []
        
        if log_dest in ("console", "both"):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.setLevel(numeric_level)
            handlers.append(console_handler)
        
        if log_dest in ("file", "both"):
            # Ensure log directory exists
            log_file = Path(log_path)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Use rotating file handler (10MB max, 5 backups)
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                encoding="utf-8"
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(numeric_level)
            handlers.append(file_handler)
        
        # Add handlers to root logger
        for handler in handlers:
            root_logger.addHandler(handler)
        
        # Prevent propagation to avoid duplicate logs
        root_logger.propagate = False
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger instance for the given module name."""
        if name not in self._loggers:
            self._loggers[name] = logging.getLogger(name)
        return self._loggers[name]
    
    def set_level(self, level: str):
        """Change log level dynamically."""
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        root_logger = logging.getLogger()
        root_logger.setLevel(numeric_level)
        for handler in root_logger.handlers:
            handler.setLevel(numeric_level)


# Global instance
_logger_instance = LightTTSLogger()


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given module name.
    
    This is the main function to use throughout the application.
    
    Args:
        name: Usually __name__ of the calling module
        
    Returns:
        Configured logger instance
    """
    return _logger_instance.get_logger(name)


def set_log_level(level: str):
    """Set log level dynamically."""
    _logger_instance.set_level(level)
