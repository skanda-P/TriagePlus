"""
Production metrics and monitoring for TriagePlus
Tracks request latency, error rates, cache performance, LLM performance
"""

import time
import logging
import asyncio
from typing import Dict, Optional, Callable
from functools import wraps
from datetime import datetime
from collections import defaultdict, deque
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class Metric:
    """Single metric data point"""
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}

class MetricsCollector:
    """Collect and aggregate metrics"""
    
    def __init__(self, retention_seconds: int = 3600):
        self.retention_seconds = retention_seconds
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.start_time = datetime.utcnow()
    
    def record(self, name: str, value: float, tags: Dict[str, str] = None):
        """Record a metric"""
        metric = Metric(
            name=name,
            value=value,
            timestamp=time.time(),
            tags=tags or {}
        )
        self.metrics[name].append(metric)
        
        # Log for external monitoring (Datadog, New Relic, etc.)
        self._emit_metric(metric)
    
    def _emit_metric(self, metric: Metric):
        """Send metric to external monitoring service"""
        # This is where you'd send to Datadog, New Relic, CloudWatch, etc.
        log_data = {
            'metric': metric.name,
            'value': metric.value,
            'tags': metric.tags,
            'timestamp': datetime.fromtimestamp(metric.timestamp).isoformat()
        }
        logger.info(f"METRIC: {json.dumps(log_data)}")
    
    def get_stats(self, name: str) -> Optional[Dict]:
        """Get statistics for a metric"""
        if name not in self.metrics:
            return None
        
        values = [m.value for m in self.metrics[name]]
        if not values:
            return None
        
        return {
            'name': name,
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': sum(values) / len(values),
            'p95': sorted(values)[int(len(values) * 0.95)] if len(values) > 0 else 0,
            'p99': sorted(values)[int(len(values) * 0.99)] if len(values) > 0 else 0,
        }
    
    def get_all_stats(self) -> Dict:
        """Get statistics for all metrics"""
        stats = {}
        for name in self.metrics.keys():
            metric_stats = self.get_stats(name)
            if metric_stats:
                stats[name] = metric_stats
        return stats

# Global metrics instance
_metrics_collector = MetricsCollector()

def get_metrics() -> MetricsCollector:
    """Get global metrics collector"""
    return _metrics_collector

def track_latency(operation_name: str):
    """Decorator to track operation latency"""
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                latency = (time.time() - start) * 1000  # Convert to ms
                get_metrics().record(
                    f"{operation_name}_latency_ms",
                    latency,
                    tags={'operation': operation_name}
                )
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                latency = (time.time() - start) * 1000
                get_metrics().record(
                    f"{operation_name}_latency_ms",
                    latency,
                    tags={'operation': operation_name}
                )
        
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

class PerformanceMonitor:
    """Monitor specific system components"""
    
    @staticmethod
    def record_llm_call(model_name: str, latency_ms: float, success: bool, error: str = None):
        """Record LLM API call metrics"""
        metrics = get_metrics()
        metrics.record(
            "llm_call_latency_ms",
            latency_ms,
            tags={'model': model_name, 'success': str(success)}
        )
        
        if success:
            metrics.record("llm_calls_success", 1, tags={'model': model_name})
        else:
            metrics.record("llm_calls_error", 1, tags={'model': model_name, 'error_type': error})
    
    @staticmethod
    def record_db_query(query_type: str, latency_ms: float, success: bool):
        """Record database query metrics"""
        metrics = get_metrics()
        metrics.record(
            "db_query_latency_ms",
            latency_ms,
            tags={'query_type': query_type, 'success': str(success)}
        )
    
    @staticmethod
    def record_cache_operation(operation: str, hit: bool):
        """Record cache hit/miss"""
        metrics = get_metrics()
        if hit:
            metrics.record("cache_hit", 1, tags={'operation': operation})
        else:
            metrics.record("cache_miss", 1, tags={'operation': operation})
    
    @staticmethod
    def record_retrieval_latency(retriever_type: str, latency_ms: float, results_count: int):
        """Record retrieval system performance"""
        metrics = get_metrics()
        metrics.record(
            "retrieval_latency_ms",
            latency_ms,
            tags={'retriever': retriever_type}
        )
        metrics.record(
            "retrieval_results_count",
            results_count,
            tags={'retriever': retriever_type}
        )

class HealthCheck:
    """System health monitoring"""
    
    def __init__(self):
        self.checks: Dict[str, Callable] = {}
        self.last_check: Dict[str, Dict] = {}
    
    def register_check(self, name: str, check_func: Callable):
        """Register a health check"""
        self.checks[name] = check_func
    
    async def run_all_checks(self) -> Dict[str, bool]:
        """Run all registered health checks"""
        results = {}
        
        for name, check_func in self.checks.items():
            try:
                is_healthy = await check_func() if asyncio.iscoroutinefunction(check_func) else check_func()
                results[name] = is_healthy
                self.last_check[name] = {
                    'healthy': is_healthy,
                    'timestamp': datetime.utcnow().isoformat()
                }
            except Exception as e:
                logger.error(f"Health check '{name}' failed: {e}")
                results[name] = False
                self.last_check[name] = {
                    'healthy': False,
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                }
        
        return results
    
    def get_overall_health(self) -> bool:
        """Check if system is overall healthy"""
        if not self.last_check:
            return True
        
        return all(check.get('healthy', False) for check in self.last_check.values())

# Global health check instance
_health_check = HealthCheck()

def get_health_check() -> HealthCheck:
    """Get global health check instance"""
    return _health_check

# Standard health checks
async def check_database() -> bool:
    """Check database connectivity"""
    try:
        # This would be implemented based on your DB client
        return True
    except Exception:
        return False

async def check_llm_service() -> bool:
    """Check LLM service availability"""
    try:
        # Check if Ollama or LLM service is reachable
        # This would be implemented with actual health check
        return True
    except Exception:
        return False

def setup_default_checks():
    """Register default health checks"""
    health = get_health_check()
    health.register_check("database", check_database)
    health.register_check("llm_service", check_llm_service)
