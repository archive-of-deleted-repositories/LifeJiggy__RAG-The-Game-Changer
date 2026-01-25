"""
Performance Monitoring - RAG-The-Game-Changer

Production performance monitoring and alerting.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
import threading
from collections import deque, defaultdict
import json
import statistics

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Single metric data point."""

    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class Alert:
    """Alert definition."""

    alert_id: str
    metric_name: str
    condition: str  # ">", "<", "==", etc.
    threshold: float
    current_value: float
    severity: str  # "info", "warning", "critical"
    message: str
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    resolution_time: Optional[float] = None


class MetricsCollector:
    """High-performance metrics collection system."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Metrics storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)

        # Performance optimization
        self.batch_size = self.config.get("batch_size", 100)
        self.flush_interval = self.config.get("flush_interval", 30)
        self.compression_enabled = self.config.get("compression_enabled", True)

        # Background thread for periodic flushing
        self._flush_thread = None
        self._running = False
        self._callbacks: List[Callable] = []

        # Buffer for batch processing
        self._buffer = []

        self._start_flush_thread()

    def _start_flush_thread(self):
        """Start background thread for periodic flushing."""
        self._running = True
        self._flush_thread = threading.Thread(target=self._flush_worker, daemon=True)
        self._flush_thread.start()

    def _flush_worker(self):
        """Background worker for flushing metrics."""
        while self._running:
            try:
                await asyncio.sleep(self.flush_interval)
                await self.flush_metrics()
            except Exception as e:
                logger.error(f"Error in flush worker: {e}")

    def record_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a single metric value."""
        metric = MetricPoint(name=name, value=value, labels=labels or {}, timestamp=time.time())

        self._buffer.append(metric)

        if len(self._buffer) >= self.batch_size:
            # Trigger flush immediately for buffer overflow
            asyncio.create_task(self.flush_metrics())

    def increment_counter(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        self.counters[name] += value

        # Also record as metric for time series
        self.record_metric(f"{name}_total", self.counters[name], labels)

    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric value."""
        self.gauges[name] = value
        self.record_metric(name, value, labels)

    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a histogram metric value."""
        self.histograms[name].append(value)

        # Also record basic stats
        self.record_metric(f"{name}_sum", sum(self.histograms[name]), labels)
        self.record_metric(f"{name}_count", len(self.histograms[name]), labels)
        self.record_metric(f"{name}_avg", statistics.mean(self.histograms[name]), labels)

    def record_request_latency(self, endpoint: str, latency_ms: float):
        """Record HTTP request latency."""
        self.record_histogram(f"request_latency_{endpoint}", latency_ms, {"endpoint": endpoint})
        self.record_metric(f"request_latency_{endpoint}", latency_ms, {"endpoint": endpoint})

    def record_query_performance(
        self, query_type: str, latency_ms: float, num_retrieved: int, confidence: float
    ):
        """Record RAG query performance metrics."""
        labels = {"query_type": query_type}

        self.record_histogram("rag_query_latency", latency_ms, labels)
        self.record_metric("rag_query_retrieval_count", num_retrieved, labels)
        self.record_metric("rag_query_confidence", confidence, labels)

        # Performance categorization
        if latency_ms < 1000:
            self.increment_counter("rag_queries_fast", 1, labels)
        elif latency_ms < 5000:
            self.increment_counter("rag_queries_medium", 1, labels)
        else:
            self.increment_counter("rag_queries_slow", 1, labels)

    def record_error(self, error_type: str, component: str, count: int = 1):
        """Record an error metric."""
        labels = {"component": component, "error_type": error_type}
        self.increment_counter(f"error_{error_type}", count, labels)
        self.increment_counter("total_errors", count)

    def record_memory_usage(self, component: str, memory_mb: float):
        """Record memory usage."""
        self.record_metric("memory_usage_mb", memory_mb, {"component": component})
        self.record_histogram("memory_usage", memory_mb, {"component": component})

    def record_cpu_usage(self, component: str, cpu_percent: float):
        """Record CPU usage."""
        self.record_metric("cpu_usage_percent", cpu_percent, {"component": component})
        self.record_histogram("cpu_usage", cpu_percent, {"component": component})

    async def flush_metrics(self):
        """Flush buffered metrics to callbacks/storage."""
        if not self._buffer:
            return

        try:
            metrics_to_flush = self._buffer.copy()
            self._buffer.clear()

            # Store in time series
            for metric in metrics_to_flush:
                self.metrics[metric.name].append(metric)

            # Call registered callbacks
            for callback in self._callbacks:
                try:
                    await callback(metrics_to_flush)
                except Exception as e:
                    logger.error(f"Error in metric callback: {e}")

            logger.debug(f"Flushed {len(metrics_to_flush)} metrics")

        except Exception as e:
            logger.error(f"Error flushing metrics: {e}")

    def add_callback(self, callback: Callable[[List[MetricPoint]], None]):
        """Add a callback for metric processing."""
        self._callbacks.append(callback)

    def get_metrics_summary(
        self, metric_name: str, time_window: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get summary statistics for a metric."""
        if metric_name not in self.metrics:
            return {}

        metrics_data = list(self.metrics[metric_name])

        if time_window:
            cutoff_time = time.time() - time_window
            metrics_data = [m for m in metrics_data if m.timestamp >= cutoff_time]

        if not metrics_data:
            return {}

        values = [m.value for m in metrics_data]

        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "sum": sum(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "p95": self._percentile(values, 95),
            "p99": self._percentile(values, 99),
        }

    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = (percentile / 100) * (len(sorted_values) - 1)
        return sorted_values[int(index)]

    def get_all_metrics(self, time_window: Optional[int] = None) -> Dict[str, Any]:
        """Get all metrics with optional time window."""
        result = {}

        for metric_name in self.metrics:
            result[metric_name] = self.get_metrics_summary(metric_name, time_window)

        return {
            "metrics": result,
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "histograms": {
                name: self._get_histogram_summary(values)
                for name, values in self.histograms.items()
            },
            "timestamp": time.time(),
        }

    def _get_histogram_summary(self, values: List[float]) -> Dict[str, float]:
        """Get histogram summary statistics."""
        if not values:
            return {}

        sorted_values = sorted(values)
        length = len(sorted_values)

        return {
            "count": length,
            "sum": sum(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": sorted_values[0],
            "max": sorted_values[-1],
            "p50": sorted_values[int(length * 0.5)],
            "p95": sorted_values[int(length * 0.95)],
            "p99": sorted_values[int(length * 0.99)],
        }

    async def export_metrics(self, format: str = "json", time_window: Optional[int] = None) -> str:
        """Export metrics in specified format."""
        metrics_data = self.get_all_metrics(time_window)

        if format.lower() == "json":
            return json.dumps(metrics_data, indent=2)
        elif format.lower() == "prometheus":
            return self._format_prometheus(metrics_data)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _format_prometheus(self, metrics_data: Dict[str, Any]) -> str:
        """Format metrics for Prometheus."""
        lines = []

        # Counters
        for name, value in metrics_data.get("counters", {}).items():
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value}")

        # Gauges
        for name, value in metrics_data.get("gauges", {}).items():
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")

        # Time series metrics
        for name, summary in metrics_data.get("metrics", {}).items():
            for metric_type, value in summary.items():
                lines.append(f"# TYPE {name}_{metric_type} gauge")
                lines.append(f"{name}_{metric_type} {value}")

        return "\n".join(lines)

    def stop(self):
        """Stop the metrics collector."""
        self._running = False

        if self._flush_thread and self._flush_thread.is_alive():
            self._flush_thread.join(timeout=5.0)

        # Flush any remaining metrics
        asyncio.create_task(self.flush_metrics())


class AlertManager:
    """Alert management system with configurable rules."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Alert storage
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=10000)
        self.alert_rules: Dict[str, Dict[str, Any]] = {}

        # Configuration
        self.alert_cooldown = self.config.get("alert_cooldown", 300)  # 5 minutes
        self.max_active_alerts = self.config.get("max_active_alerts", 100)
        self.severity_levels = ["info", "warning", "critical"]

        # Notification callbacks
        self.notification_callbacks: List[Callable] = []

        # Load default alert rules
        self._load_default_rules()

    def _load_default_rules(self):
        """Load default alerting rules."""
        default_rules = {
            "high_latency": {
                "metric": "rag_query_latency",
                "condition": ">",
                "threshold": 5000,
                "severity": "warning",
            },
            "critical_latency": {
                "metric": "rag_query_latency",
                "condition": ">",
                "threshold": 10000,
                "severity": "critical",
            },
            "low_confidence": {
                "metric": "rag_query_confidence",
                "condition": "<",
                "threshold": 0.3,
                "severity": "warning",
            },
            "high_error_rate": {
                "metric": "total_errors",
                "condition": ">",
                "threshold": 10,
                "severity": "warning",
                "time_window": 300,  # 5 minutes
            },
            "memory_usage_high": {
                "metric": "memory_usage_mb",
                "condition": ">",
                "threshold": 8096,  # 8GB
                "severity": "warning",
            },
            "cpu_usage_high": {
                "metric": "cpu_usage_percent",
                "condition": ">",
                "threshold": 80,
                "severity": "warning",
            },
            "no_retrieval": {
                "metric": "rag_query_retrieval_count",
                "condition": "==",
                "threshold": 0,
                "severity": "warning",
            },
        }

        for rule_name, rule in default_rules.items():
            self.add_alert_rule(rule_name, rule)

    def add_alert_rule(self, rule_name: str, rule_config: Dict[str, Any]):
        """Add a new alert rule."""
        self.alert_rules[rule_name] = rule_config
        logger.info(f"Added alert rule: {rule_name}")

    def remove_alert_rule(self, rule_name: str):
        """Remove an alert rule."""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            logger.info(f"Removed alert rule: {rule_name}")

    async def check_alerts(self, metrics_data: Dict[str, Any]):
        """Check all alert rules against current metrics."""
        for rule_name, rule in self.alert_rules.items():
            try:
                await self._check_single_rule(rule_name, rule, metrics_data)
            except Exception as e:
                logger.error(f"Error checking rule {rule_name}: {e}")

    async def _check_single_rule(
        self, rule_name: str, rule: Dict[str, Any], metrics_data: Dict[str, Any]
    ):
        """Check a single alert rule."""
        metric_name = rule["metric"]
        condition = rule["condition"]
        threshold = rule["threshold"]

        # Get current metric value
        current_value = self._get_metric_value(metric_name, metrics_data, rule.get("time_window"))

        if current_value is None:
            return

        # Evaluate condition
        triggered = self._evaluate_condition(current_value, condition, threshold)

        alert_id = f"{rule_name}_{hash(str(rule))}"

        if triggered:
            await self._trigger_alert(
                alert_id,
                rule_name,
                metric_name,
                condition,
                threshold,
                current_value,
                rule.get("severity", "warning"),
            )
        else:
            # Resolve existing alert if it exists
            if alert_id in self.active_alerts:
                await self._resolve_alert(alert_id)

    def _get_metric_value(
        self, metric_name: str, metrics_data: Dict[str, Any], time_window: Optional[int] = None
    ) -> Optional[float]:
        """Get current value for a metric."""
        # Check different sources of metrics
        if metric_name in metrics_data.get("counters", {}):
            return float(metrics_data["counters"][metric_name])
        elif metric_name in metrics_data.get("gauges", {}):
            return float(metrics_data["gauges"][metric_name])
        elif metric_name in metrics_data.get("metrics", {}):
            metric_summary = metrics_data["metrics"][metric_name]
            if "mean" in metric_summary:
                return float(metric_summary["mean"])

        return None

    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition."""
        if condition == ">":
            return value > threshold
        elif condition == "<":
            return value < threshold
        elif condition == ">=":
            return value >= threshold
        elif condition == "<=":
            return value <= threshold
        elif condition == "==":
            return abs(value - threshold) < 0.001
        elif condition == "!=":
            return abs(value - threshold) >= 0.001
        else:
            logger.warning(f"Unknown alert condition: {condition}")
            return False

    async def _trigger_alert(
        self,
        alert_id: str,
        rule_name: str,
        metric_name: str,
        condition: str,
        threshold: float,
        current_value: float,
        severity: str,
    ):
        """Trigger a new alert."""
        # Check cooldown
        if alert_id in self.active_alerts:
            last_alert = self.active_alerts[alert_id]
            if time.time() - last_alert.timestamp < self.alert_cooldown:
                return  # Still in cooldown

        # Check max active alerts
        if len(self.active_alerts) >= self.max_active_alerts:
            logger.warning(f"Maximum active alerts reached, dropping alert: {alert_id}")
            return

        # Create alert
        alert = Alert(
            alert_id=alert_id,
            metric_name=metric_name,
            condition=condition,
            threshold=threshold,
            current_value=current_value,
            severity=severity,
            message=f"Alert: {metric_name} {condition} {threshold} (current: {current_value:.2f})",
        )

        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)

        logger.warning(f"Alert triggered: {alert.message}")

        # Send notifications
        await self._send_notifications(alert)

    async def _resolve_alert(self, alert_id: str):
        """Resolve an active alert."""
        if alert_id not in self.active_alerts:
            return

        alert = self.active_alerts[alert_id]
        alert.resolved = True
        alert.resolution_time = time.time()

        # Move to history
        self.alert_history.append(alert)
        del self.active_alerts[alert_id]

        logger.info(f"Alert resolved: {alert_id}")

        # Send resolution notification
        await self._send_resolution_notification(alert)

    def add_notification_callback(self, callback: Callable[[Alert], None]):
        """Add a notification callback."""
        self.notification_callbacks.append(callback)

    async def _send_notifications(self, alert: Alert):
        """Send alert notifications."""
        for callback in self.notification_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logger.error(f"Error in notification callback: {e}")

    async def _send_resolution_notification(self, alert: Alert):
        """Send alert resolution notifications."""
        resolution_msg = f"Alert resolved: {alert.message}"

        for callback in self.notification_callbacks:
            try:
                await callback(
                    type(
                        "Alert", (alert.alert_id, resolution_msg, alert.severity, True, time.time())
                    )
                )
            except Exception as e:
                logger.error(f"Error in resolution notification callback: {e}")

    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())

    def get_alert_history(
        self, limit: Optional[int] = None, severity: Optional[str] = None
    ) -> List[Alert]:
        """Get alert history with optional filtering."""
        history = list(self.alert_history)

        # Filter by severity
        if severity:
            history = [alert for alert in history if alert.severity == severity]

        # Limit results
        if limit:
            history = history[-limit:]

        return history

    async def clear_resolved_alerts(self):
        """Clear all resolved alerts from history."""
        self.alert_history = deque(
            [alert for alert in self.alert_history if not alert.resolved], maxlen=10000
        )
        logger.info("Cleared resolved alerts from history")


class PerformanceMonitor:
    """High-level performance monitoring interface."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.metrics_collector = MetricsCollector(config)
        self.alert_manager = AlertManager(config)

        # Connect metrics to alerts
        self.metrics_collector.add_callback(self.alert_manager.check_alerts)

        # Auto-export settings
        self.auto_export_enabled = self.config.get("auto_export_enabled", False)
        self.auto_export_interval = self.config.get("auto_export_interval", 300)  # 5 minutes
        self.auto_export_path = self.config.get("auto_export_path", "./metrics")

        self._export_task = None
        if self.auto_export_enabled:
            self._start_auto_export()

    def _start_auto_export(self):
        """Start automatic metric export."""
        self._export_task = asyncio.create_task(self._auto_export_worker())

    async def _auto_export_worker(self):
        """Background worker for auto-export."""
        while self.auto_export_enabled:
            try:
                await asyncio.sleep(self.auto_export_interval)
                await self.export_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in auto-export: {e}")

    async def export_metrics(self):
        """Export current metrics."""
        try:
            metrics_json = await self.metrics_collector.export_metrics("json")

            import os

            os.makedirs(self.auto_export_path, exist_ok=True)

            timestamp = int(time.time())
            filename = f"{self.auto_export_path}/metrics_{timestamp}.json"

            with open(filename, "w") as f:
                f.write(metrics_json)

            logger.info(f"Exported metrics to {filename}")

        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")

    def record_query(
        self,
        query: str,
        response_time_ms: float,
        retrieval_count: int,
        confidence: float,
        error: Optional[str] = None,
    ):
        """Record a complete query with all its metrics."""
        self.metrics_collector.record_query_performance(
            "standard", response_time_ms, retrieval_count, confidence
        )

        if error:
            self.metrics_collector.record_error("query_error", "rag_system")

    def record_batch_queries(self, queries_data: List[Dict[str, Any]]):
        """Record multiple queries."""
        for query_data in queries_data:
            self.record_query(
                query_data.get("query", ""),
                query_data.get("response_time_ms", 0),
                query_data.get("retrieval_count", 0),
                query_data.get("confidence", 0),
                query_data.get("error"),
            )

    async def get_performance_summary(
        self,
        time_window: int = 3600,  # 1 hour
    ) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        metrics_summary = self.metrics_collector.get_all_metrics(time_window)
        active_alerts = self.alert_manager.get_active_alerts()

        return {
            "time_window": time_window,
            "timestamp": time.time(),
            "metrics": metrics_summary,
            "active_alerts": [
                {
                    "id": alert.alert_id,
                    "message": alert.message,
                    "severity": alert.severity,
                    "timestamp": alert.timestamp,
                }
                for alert in active_alerts
            ],
            "performance_grade": self._calculate_performance_grade(metrics_summary, active_alerts),
        }

    def _calculate_performance_grade(self, metrics: Dict[str, Any], alerts: List[Alert]) -> str:
        """Calculate overall performance grade."""
        critical_alerts = [a for a in alerts if a.severity == "critical"]
        warning_alerts = [a for a in alerts if a.severity == "warning"]

        if critical_alerts:
            return "F"
        elif warning_alerts > 5:
            return "D"
        elif warning_alerts > 2:
            return "C"
        elif len(alerts) > 0:
            return "B"
        else:
            return "A"

    async def stop(self):
        """Stop the performance monitor."""
        if self._export_task:
            self._export_task.cancel()

        self.metrics_collector.stop()

    def add_notification_callback(self, callback: Callable):
        """Add notification callback for alerts."""
        self.alert_manager.add_notification_callback(callback)
