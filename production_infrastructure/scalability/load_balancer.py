"""
Load Balancer - RAG-The-Game-Changer

Production-grade load balancer for distributing requests across RAG instances.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class Algorithm(Enum):
    ROUND_ROBIN = "round_robin"
    WEIGHTED = "weighted"
    LEAST_CONNECTIONS = "least_connections"


@dataclass
class BackendInstance:
    """Represents a backend RAG instance."""

    instance_id: str
    endpoint: str
    weight: int = 1
    healthy: bool = True
    current_connections: int = 0
    total_requests: int = 0
    error_count: int = 0
    avg_response_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class LoadBalancer:
    """Load balancer for RAG pipeline instances."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.algorithm = Algorithm(self.config.get("algorithm", "round_robin"))
        self.backends: Dict[str, BackendInstance] = {}
        self._current_index = 0
        self._lock = asyncio.Lock()
        self.health_check_interval = self.config.get("health_check_interval", 30)
        self._health_check_task = None

    async def register_backend(
        self,
        instance_id: str,
        endpoint: str,
        weight: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Register a new backend instance."""
        try:
            backend = BackendInstance(
                instance_id=instance_id, endpoint=endpoint, weight=weight, metadata=metadata or {}
            )

            async with self._lock:
                self.backends[instance_id] = backend

            logger.info(f"Registered backend: {instance_id} at {endpoint}")
            return True
        except Exception as e:
            logger.error(f"Error registering backend {instance_id}: {e}")
            return False

    async def unregister_backend(self, instance_id: str) -> bool:
        """Unregister a backend instance."""
        try:
            async with self._lock:
                if instance_id in self.backends:
                    del self.backends[instance_id]

            logger.info(f"Unregistered backend: {instance_id}")
            return True
        except Exception as e:
            logger.error(f"Error unregistering backend {instance_id}: {e}")
            return False

    async def get_backend(self) -> Optional[BackendInstance]:
        """Get next available backend based on algorithm."""
        async with self._lock:
            healthy_backends = [b for b in self.backends.values() if b.healthy]

            if not healthy_backends:
                logger.warning("No healthy backends available")
                return None

            if self.algorithm == Algorithm.ROUND_ROBIN:
                backend = healthy_backends[self._current_index % len(healthy_backends)]
                self._current_index += 1

            elif self.algorithm == Algorithm.WEIGHTED:
                backend = self._weighted_selection(healthy_backends)

            elif self.algorithm == Algorithm.LEAST_CONNECTIONS:
                backend = min(healthy_backends, key=lambda b: b.current_connections)

            else:
                backend = healthy_backends[0]

            backend.current_connections += 1
            backend.total_requests += 1

            return backend

    def _weighted_selection(self, backends: List[BackendInstance]) -> BackendInstance:
        """Select backend using weighted algorithm."""
        total_weight = sum(b.weight for b in backends)
        point = time.time() % total_weight

        current = 0
        for backend in backends:
            current += backend.weight
            if point <= current:
                return backend

        return backends[-1]

    async def release_backend(self, instance_id: str, response_time: float, success: bool = True):
        """Release backend after request completion."""
        try:
            async with self._lock:
                if instance_id in self.backends:
                    backend = self.backends[instance_id]
                    backend.current_connections -= 1

                    if not success:
                        backend.error_count += 1

                    if backend.avg_response_time > 0:
                        backend.avg_response_time = (backend.avg_response_time + response_time) / 2
                    else:
                        backend.avg_response_time = response_time
        except Exception as e:
            logger.error(f"Error releasing backend {instance_id}: {e}")

    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all backends."""
        results = {}
        for instance_id, backend in self.backends.items():
            healthy = await self._health_check(backend)
            results[instance_id] = healthy

            async with self._lock:
                if healthy != backend.healthy:
                    backend.healthy = healthy
                    logger.info(f"Backend {instance_id} health changed to {healthy}")

        return results

    async def _health_check(self, backend: BackendInstance) -> bool:
        """Check health of a single backend."""
        try:
            import aiohttp

            timeout = aiohttp.ClientTimeout(total=5)

            async with aiohttp.ClientSession(timeout=timeout) as session:
                health_url = f"{backend.endpoint}/health"
                async with session.get(health_url) as response:
                    return response.status == 200
        except Exception as e:
            logger.warning(f"Health check failed for {backend.instance_id}: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        total_requests = sum(b.total_requests for b in self.backends.values())
        total_errors = sum(b.error_count for b in self.backends.values())
        total_connections = sum(b.current_connections for b in self.backends.values())
        healthy_count = sum(1 for b in self.backends.values() if b.healthy)

        return {
            "total_backends": len(self.backends),
            "healthy_backends": healthy_count,
            "total_requests": total_requests,
            "total_errors": total_errors,
            "active_connections": total_connections,
            "algorithm": self.algorithm.value,
        }

    async def start_health_monitoring(self):
        """Start background health monitoring."""
        if self._health_check_task:
            return

        async def monitor():
            while True:
                await asyncio.sleep(self.health_check_interval)
                await self.health_check_all()

        self._health_check_task = asyncio.create_task(monitor())
        logger.info("Started health monitoring")

    async def stop_health_monitoring(self):
        """Stop background health monitoring."""
        if self._health_check_task:
            self._health_check_task.cancel()
            self._health_check_task = None
            logger.info("Stopped health monitoring")
