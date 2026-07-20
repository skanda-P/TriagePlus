"""
Production-grade error handling and logging for TriagePlus
Includes request validation, error codes, and structured logging
"""

import logging
import json
from typing import Dict, Optional, Callable
from functools import wraps
from datetime import datetime
import uuid
from enum import Enum

# Configure structured logging
class LogFormatter(logging.Formatter):
    """JSON-formatted logging for structured log aggregation"""
    
    def format(self, record):
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        if hasattr(record, 'request_id'):
            log_data['request_id'] = record.request_id
        
        return json.dumps(log_data)

# Setup logger
logger = logging.getLogger('triageplus')
handler = logging.StreamHandler()
handler.setFormatter(LogFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class ErrorCode(Enum):
    """Standard error codes for API responses"""
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    CONFLICT = "CONFLICT"
    RATE_LIMITED = "RATE_LIMITED"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    TIMEOUT = "TIMEOUT"
    DATABASE_ERROR = "DATABASE_ERROR"
    LLM_ERROR = "LLM_ERROR"
    EXTERNAL_API_ERROR = "EXTERNAL_API_ERROR"

class TriagePlusException(Exception):
    """Base exception for TriagePlus"""
    
    def __init__(self, 
                 code: ErrorCode, 
                 message: str, 
                 status_code: int = 500,
                 details: Optional[Dict] = None,
                 request_id: str = None):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        self.request_id = request_id or str(uuid.uuid4())
        super().__init__(self.message)
    
    def to_dict(self) -> Dict:
        """Convert exception to API response dict"""
        return {
            'error': {
                'code': self.code.value,
                'message': self.message,
                'request_id': self.request_id,
                'details': self.details
            }
        }

class ValidationError(TriagePlusException):
    def __init__(self, message: str, details: Optional[Dict] = None, request_id: str = None):
        super().__init__(
            code=ErrorCode.VALIDATION_ERROR,
            message=message,
            status_code=400,
            details=details,
            request_id=request_id
        )

class NotFoundError(TriagePlusException):
    def __init__(self, message: str, details: Optional[Dict] = None, request_id: str = None):
        super().__init__(
            code=ErrorCode.NOT_FOUND,
            message=message,
            status_code=404,
            details=details,
            request_id=request_id
        )

class UnauthorizedError(TriagePlusException):
    def __init__(self, message: str = "Unauthorized", details: Optional[Dict] = None, request_id: str = None):
        super().__init__(
            code=ErrorCode.UNAUTHORIZED,
            message=message,
            status_code=401,
            details=details,
            request_id=request_id
        )

class DatabaseError(TriagePlusException):
    def __init__(self, message: str, details: Optional[Dict] = None, request_id: str = None):
        super().__init__(
            code=ErrorCode.DATABASE_ERROR,
            message=message,
            status_code=500,
            details=details,
            request_id=request_id
        )

class LLMError(TriagePlusException):
    def __init__(self, message: str, details: Optional[Dict] = None, request_id: str = None):
        super().__init__(
            code=ErrorCode.LLM_ERROR,
            message=message,
            status_code=503,
            details=details,
            request_id=request_id
        )

class RateLimitError(TriagePlusException):
    def __init__(self, message: str = "Rate limit exceeded", reset_time: int = None, request_id: str = None):
        details = {'reset_time': reset_time} if reset_time else {}
        super().__init__(
            code=ErrorCode.RATE_LIMITED,
            message=message,
            status_code=429,
            details=details,
            request_id=request_id
        )

def with_error_handling(func: Callable) -> Callable:
    """Decorator for automatic error handling and logging.

    For FastAPI route handlers, the wrapper RAISES HTTPException for any
    TriagePlusException or unexpected exception, so FastAPI returns the
    correct HTTP status code (instead of a 200-with-error-body dict).
    A global exception handler registered in main.py converts
    TriagePlusException into the structured {error: {...}} envelope.
    """
    from fastapi import HTTPException

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        request_id = str(uuid.uuid4())

        try:
            logger.info(f"Starting {func.__name__}", extra={'request_id': request_id})
            result = await func(*args, **kwargs)
            logger.info(f"Completed {func.__name__}", extra={'request_id': request_id})
            return result

        except HTTPException:
            # Already a well-formed FastAPI error - let it propagate untouched.
            raise

        except TriagePlusException as e:
            e.request_id = request_id
            logger.warning(
                f"TriagePlus exception in {func.__name__}: {e.message}",
                extra={'request_id': request_id},
            )
            raise HTTPException(status_code=e.status_code, detail=e.to_dict()['error'])

        except Exception as e:
            logger.error(
                f"Unhandled exception in {func.__name__}: {str(e)}",
                extra={'request_id': request_id},
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail={
                    'code': ErrorCode.INTERNAL_ERROR.value,
                    'message': 'Internal server error',
                    'request_id': request_id,
                },
            )

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        request_id = str(uuid.uuid4())

        try:
            logger.info(f"Starting {func.__name__}", extra={'request_id': request_id})
            result = func(*args, **kwargs)
            logger.info(f"Completed {func.__name__}", extra={'request_id': request_id})
            return result

        except HTTPException:
            raise

        except TriagePlusException as e:
            e.request_id = request_id
            logger.warning(
                f"TriagePlus exception in {func.__name__}: {e.message}",
                extra={'request_id': request_id},
            )
            raise HTTPException(status_code=e.status_code, detail=e.to_dict()['error'])

        except Exception as e:
            logger.error(
                f"Unhandled exception in {func.__name__}: {str(e)}",
                extra={'request_id': request_id},
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail={
                    'code': ErrorCode.INTERNAL_ERROR.value,
                    'message': 'Internal server error',
                    'request_id': request_id,
                },
            )

    # Return appropriate wrapper
    import inspect
    if inspect.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper

def get_logger(name: str) -> logging.Logger:
    """Get configured logger instance"""
    return logging.getLogger(f'triageplus.{name}')
