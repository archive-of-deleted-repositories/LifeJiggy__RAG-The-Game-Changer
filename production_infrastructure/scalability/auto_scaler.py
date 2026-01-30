"""
Auto Scaler - RAG-The-Game-Changer

Production-grade auto-scaler for dynamically scaling RAG instances.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ScaleDirection(Enum):
    UP = "up"
    DOWN = "down"


class ScaleTrigger(Enum):
    CPU_HIGH = "cpu_high"
    MEMORY_HIGH = "memory_high"
    QUEUE_DEPTH = "queue_depth"
    RESPONSE_TIME = "response_time"
    ERROR_RATE = "error_rate"


@dataclass
class ScalePolicy:
    """Configuration for auto-scaling policy."""

    min_instances: int = 1
    max_instances: int = 10
    scale_up_threshold: float = 0.7
    scale_down_threshold: float = 0.3
    cooldown_period: int = 300
    scale_step: int = 1
    trigger_metrics: List[str] = None


class AutoScaler:
    """Auto-scaler for RAG pipeline instances."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.policy = ScalePolicy(**self.config.get("policy", {}))
        self._current_instances = self.policy.min_instances
        self._last_scale_time = 0
        self._monitoring = False
        self._monitor_task = None
        self._lock = asyncio.Lock()
        self._metrics_buffer: List[Dict[str, float]] = []
        self._max_buffer_size = 10
        self._scale_callback: Optional[Callable] = None
        self._instance_info: Dict[str, Any] = {}

    def register_scale_callback(self, callback: Callable):
        """Register callback for scaling actions."""
        self._scale_callback = callback

    async def start_monitoring(self, interval: int = 30):
        """Start background monitoring."""
        if self._monitoring:
            logger.warning("Monitoring already started")
            return

        self._monitoring = True

        async def monitor():
            while self._monitoring:
                await asyncio.sleep(interval)
                await self._collect_metrics()
                await self._evaluate_scaling()

        self._monitor_task = asyncio.create_task(monitor())
        logger.info(f"Started monitoring with {interval}s interval")

    async def stop_monitoring(self):
        """Stop background monitoring."""
        self._monitoring = False

        if self._monitor_task:
            self._monitor_task.cancel()
            self._monitor_task = None

        logger.info("Stopped monitoring")

    async def _collect_metrics(self):
        """Collect current system metrics."""
        try:
            metrics = await self._get_system_metrics()

            if metrics:
                self._metrics_buffer.append(metrics)
                if len(self._metrics_buffer) > self._max_buffer_size:
                    self._metrics_buffer.pop(0)

            logger.debug(f"Collected metrics: {metrics}")

        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")

    async def _get_system_metrics(self) -> Optional[Dict[str, float]]:
        """Get current system metrics."""
        import psutil

        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        return {
            "cpu": cpu_percent / 100.0,
            "memory": memory_percent / 100.0,
            "timestamp": time.time(),
        }

    async def _evaluate_scaling(self):
        """Evaluate if scaling is needed."""
        if not self._metrics_buffer:
            return

        # Check cooldown period
        if time.time() - self._last_scale_time < self.policy.cooldown_period:
            return

        # Calculate average metrics
        avg_metrics = self._calculate_average_metrics()

        # Evaluate scale up
        if await self._should_scale_up(avg_metrics):
            await self._scale_up(avg_metrics)

        # Evaluate scale down
        elif await self._should_scale_down(avg_metrics):
            await self._scale_down(avg_metrics)

    def _calculate_average_metrics(self) -> Dict[str, float]:
        """Calculate average of buffered metrics."""
        if not self._metrics_buffer:
            return {}

        avg = {}
        for key in ["cpu", "memory"]:
            values = [m.get(key, 0) for m in self._metrics_buffer]
            avg[key] = sum(values) / len(values) if values else 0

        return avg

    async def _should_scale_up(self, metrics: Dict[str, float]) -> bool:
        """Determine if scaling up is needed."""
        if self._current_instances >= self.policy.max_instances:
            return False

        triggers = []

        if (
            "cpu_high" in self.policy.trigger_metrics
            and metrics.get("cpu", 0) > self.policy.scale_up_threshold
        ):
            triggers.append("cpu")

        if (
            "memory_high" in self.policy.trigger_metrics
            and metrics.get("memory", 0) > self.policy.scale_up_threshold
        ):
            triggers.append("memory")

        if triggers:
            logger.info(f"Scale up triggers: {triggers}")
            return True

        return False

    async def _should_scale_down(self, metrics: Dict[str, float]) -> bool:
        """Determine if scaling down is needed."""
        if self._current_instances <= self.policy.min_instances:
            return False

        cpu_low = metrics.get("cpu", 1.0) < self.policy.scale_down_threshold
        memory_low = metrics.get("memory", 1.0) < self.policy.scale_down_threshold

        if cpu_low and memory_low:
            logger.info("Scale down triggers: low CPU and memory")
            return True

        return False

    async def _scale_up(self, metrics: Dict[str, float]):
        """Scale up by adding instances."""
        async with self._lock:
            new_count = min(
                self._current_instances + self.policy.scale_step, self.policy.max_instances
            )

            scale_amount = new_count - self._current_instances
            self._current_instances = new_count
            self._last_scale_time = time.time()

            logger.info(f"Scaling up from {new_count - scale_amount} to {new_count} instances")

            if self._scale_callback:
                await self._scale_callback(ScaleDirection.UP, scale_amount)

    async def _scale_down(self, metrics: Dict[str, float]):
        """Scale down by removing instances."""
        async with self._lock:
            new_count = max(
                self._current_instances - self.policy.scale_step, self.policy.min_instances
            )

            scale_amount = self._current_instances - new_count
            self._current_instances = new_count
            self._last_scale_time = time.time()

            logger.info(f"Scaling down from {new_count + scale_amount} to {new_count} instances")

            if self._scale_callback:
                await self._scale_callback(ScaleDirection.DOWN, scale_amount)

    def get_metrics(self) -> Dict[str, Any]:
        """Get current scaler metrics."""
        avg_metrics = self._calculate_average_metrics()

        return {
            "current_instances": self._current_instances,
            "min_instances": self.policy.min_instances,
            "max_instances": self.policy.max_instances,
            "avg_cpu": avg_metrics.get("cpu", 0),
            "avg_memory": avg_metrics.get("memory", 0),
            "last_scale_time": self._last_scale_time,
            "monitoring": self._monitoring,
        }

    def set_instance_count(self, count: int):
        """Manually set instance count."""
        self._current_instances = max(
            min(count, self.policy.max_instances), self.policy.min_instances
        )
        logger.info(f"Instance count manually set to {self._current_instances}")
