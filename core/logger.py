"""Enhanced logger configuration with structured logging and correlation IDs."""

import logging
import logging.handlers
import os
import sys
import json
import uuid
import time
from pathlib import Path
from typing import Optional, Dict, Any
from contextvars import ContextVar
from pythonjsonlogger import jsonlogger


# Context variable for correlation ID
correlation_id_var: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)
request_context_var: ContextVar[Optional[Dict[str, Any]]] = ContextVar('request_context', default=None)


class StructuredFormatter(jsonlogger.JsonFormatter):
    """JSON formatter with correlation ID and request context."""
    
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
        super().add_fields(log_record, record, message_dict)
        
        # Add correlation ID
        correlation_id = correlation_id_var.get()
        if correlation_id:
            log_record['correlation_id'] = correlation_id
        
        # Add request context
        request_context = request_context_var.get()
        if request_context:
            log_record['request_context'] = request_context
        
        # Add standard fields
        log_record['timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        log_record['level'] = record.levelname
        log_record['logger'] = record.name
        log_record['module'] = record.module
        log_record['function'] = record.funcName
        log_record['line'] = record.lineno


class LightTTSLogger:
    """Enhanced logger wrapper for LightTTS application with structured logging."""
    
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
        log_format = os.getenv("LOGFORMAT", "json").lower()  # json or text
        
        # Parse log level
        numeric_level = getattr(logging, log_level, logging.INFO)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(numeric_level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Create formatter based on format preference
        if log_format == "json":
            formatter = StructuredFormatter(
                fmt="%(timestamp)s %(level)s %(logger)s %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%SZ"
            )
        else:
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
        
        # Log startup info
        root_logger.info("Logger initialized", extra={
            "log_level": log_level,
            "log_dest": log_dest,
            "log_format": log_format,
            "log_path": log_path
        })
    
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
    
    @staticmethod
    def set_correlation_id(correlation_id: Optional[str] = None) -> str:
        """Set correlation ID for current context."""
        if correlation_id is None:
            correlation_id = str(uuid.uuid4())[:8]
        correlation_id_var.set(correlation_id)
        return correlation_id
    
    @staticmethod
    def get_correlation_id() -> Optional[str]:
        """Get current correlation ID."""
        return correlation_id_var.get()
    
    @staticmethod
    def set_request_context(context: Dict[str, Any]):
        """Set request context for current context."""
        request_context_var.set(context)
    
    @staticmethod
    def clear_context():
        """Clear correlation ID and request context."""
        correlation_id_var.set(None)
        request_context_var.set(None)


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


def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """Set correlation ID for request tracing."""
    return _logger_instance.set_correlation_id(correlation_id)


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID."""
    return _logger_instance.get_correlation_id()


def set_request_context(context: Dict[str, Any]):
    """Set request context for structured logging."""
    _logger_instance.set_request_context(context)


def clear_log_context():
    """Clear logging context."""
    _logger_instance.clear_context()
