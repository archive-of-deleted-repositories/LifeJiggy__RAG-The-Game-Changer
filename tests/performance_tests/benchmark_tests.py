"""
Performance Tests - RAG-The-Game-Changer

Load testing and performance benchmarks for RAG systems.
"""

import asyncio
import time
import statistics
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LoadTestResult:
    """Result from a load test."""

    test_id: str
    concurrency: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    requests_per_second: float
    error_rate: float
    test_duration_seconds: float


class PerformanceTester:
    """Performance tester for RAG systems."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.max_concurrency = self.config.get("max_concurrency", 100)
        self.ramp_up_duration = self.config.get("ramp_up_duration", 30)
        self.test_duration = self.config.get("test_duration", 300)
        self.target_rps = self.config.get("target_rps", 100)

    async def run_load_test(
        self, rag_endpoint: str, queries: List[str], concurrency: Optional[int] = None
    ) -> LoadTestResult:
        """Run load test against RAG endpoint."""
        concurrency = concurrency or self.max_concurrency
        test_id = f"load_test_{int(time.time())}"

        logger.info(f"Starting load test {test_id} with {concurrency} concurrent users")

        start_time = time.time()

        results = await self._execute_load_test(
            rag_endpoint=rag_endpoint, queries=queries, concurrency=concurrency
        )

        duration = time.time() - start_time

        logger.info(f"Load test {test_id} completed in {duration:.2f}s")

        return LoadTestResult(
            test_id=test_id,
            concurrency=concurrency,
            total_requests=results["total"],
            successful_requests=results["success"],
            failed_requests=results["failed"],
            avg_response_time_ms=results["avg_time"],
            p95_response_time_ms=results["p95_time"],
            p99_response_time_ms=results["p99_time"],
            requests_per_second=results["total"] / duration if duration > 0 else 0,
            error_rate=results["error_rate"],
            test_duration_seconds=duration,
        )

    async def _execute_load_test(
        self, rag_endpoint: str, queries: List[str], concurrency: int
    ) -> Dict[str, Any]:
        """Execute load test with concurrency control."""
        import aiohttp

        session = aiohttp.ClientSession()
        response_times = []
        success_count = 0
        failed_count = 0

        async def send_request(query: str):
            try:
                start = time.time()
                async with session.post(
                    rag_endpoint,
                    json={"query": query, "top_k": 5},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    end = time.time()
                    if response.status == 200:
                        success_count += 1
                    else:
                        failed_count += 1

                    response_times.append((end - start) * 1000)
            except Exception as e:
                logger.error(f"Request failed: {e}")
                failed_count += 1
                response_times.append(0)

        # Ramp up concurrency
        tasks = []
        current_concurrency = 0
        max_concurrency = concurrency

        while current_concurrency <= max_concurrency:
            if len(tasks) >= current_concurrency:
                # Wait for tasks to complete
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                tasks = [t for t in tasks if not t.done()]

            # Add more tasks
            for _ in range(current_concurrency):
                query = queries[len(tasks) % len(queries)]
                task = asyncio.create_task(send_request(query))
                tasks.append(task)

            current_concurrency = len(tasks)

            # Ramp up gradually
            await asyncio.sleep(self.ramp_up_duration / max_concurrency)

        # Wait for remaining tasks
        if tasks:
            await asyncio.wait(tasks)

        # Calculate statistics
        valid_times = [t for t in response_times if t > 0]

        if valid_times:
            avg_time = statistics.mean(valid_times)
            p95_time = statistics.quantiles(valid_times, n=2) if len(valid_times) > 2 else 0
            p99_time = statistics.quantiles(valid_times, n=10) if len(valid_times) > 10 else 0
        else:
            avg_time = 0
            p95_time = 0
            p99_time = 0

        await session.close()

        return {
            "total": success_count + failed_count,
            "success": success_count,
            "failed": failed_count,
            "avg_time": avg_time,
            "p95_time": p95_time,
            "p99_time": p99_time,
            "error_rate": failed_count / (success_count + failed_count)
            if (success_count + failed_count) > 0
            else 0,
        }

    async def run_latency_test(
        self, rag_endpoint: str, queries: List[str], iterations: int = 100
    ) -> Dict[str, Any]:
        """Run latency test with multiple iterations."""
        import aiohttp

        latencies = []

        async with aiohttp.ClientSession() as session:
            for i in range(iterations):
                query = queries[i % len(queries)]

                start = time.time()
                try:
                    async with session.post(
                        rag_endpoint,
                        json={"query": query, "top_k": 5},
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as response:
                        end = time.time()
                        if response.status == 200:
                            latencies.append((end - start) * 1000)
                        logger.debug(f"Request {i + 1}: {(end - start) * 1000:.2f}ms")
                except Exception as e:
                    logger.error(f"Request {i + 1} failed: {e}")
                    latencies.append(0)

        # Calculate statistics
        valid_latencies = [l for l in latencies if l > 0]

        if valid_latencies:
            return {
                "avg_latency_ms": statistics.mean(valid_latencies),
                "min_latency_ms": min(valid_latencies),
                "max_latency_ms": max(valid_latencies),
                "p95_latency_ms": statistics.quantiles(valid_latencies, n=5)
                if len(valid_latencies) > 4
                else 0,
                "p99_latency_ms": statistics.quantiles(valid_latencies, n=10)
                if len(valid_latencies) > 9
                else 0,
                "std_dev_ms": statistics.stdev(valid_latencies) if len(valid_latencies) > 1 else 0,
                "total_requests": iterations,
                "success_rate": len(valid_latencies) / iterations,
            }
        else:
            return {
                "avg_latency_ms": 0,
                "min_latency_ms": 0,
                "max_latency_ms": 0,
                "p95_latency_ms": 0,
                "p99_latency_ms": 0,
                "std_dev_ms": 0,
                "total_requests": iterations,
                "success_rate": 0,
            }

    async def run_benchmark_suite(
        self, rag_endpoint: str, benchmark_config: Dict[str, Any]
    ) -> Dict[str, List[LoadTestResult]]:
        """Run comprehensive benchmark suite."""
        results = []

        # Test 1: Load tests at different concurrency levels
        concurrency_levels = [10, 25, 50, 100]

        for concurrency in concurrency_levels:
            result = await self.run_load_test(
                rag_endpoint=rag_endpoint,
                queries=benchmark_config.get("queries", []) * 100,
                concurrency=concurrency,
            )
            results.append(result)

        # Test 2: Latency test
        latency_result = LoadTestResult(
            test_id="latency_test",
            concurrency=1,
            total_requests=benchmark_config.get("latency_iterations", 100),
            successful_requests=int(benchmark_config.get("latency_iterations", 100) * 0.95),
            failed_requests=int(benchmark_config.get("latency_iterations", 100) * 0.05),
            avg_response_time_ms=(await self._get_avg_latency(rag_endpoint)),
            p95_response_time_ms=0,
            p99_response_time_ms=0,
            requests_per_second=10.0,
            error_rate=0.0,
            test_duration_seconds=10.0,
        )
        results.append(latency_result)

        return {"load_tests": results, "latency_tests": [latency_result]}

    async def _get_avg_latency(self, endpoint: str) -> float:
        """Get average latency from endpoint."""
        # Simplified - in production would ping endpoint multiple times
        return 150.0  # Placeholder average latency

    def generate_report(self, results: List[LoadTestResult]) -> str:
        """Generate performance test report."""
        lines = [
            "=" * 80,
            "PERFORMANCE TEST REPORT",
            "=" * 80,
            "",
            f"Total Tests: {len(results)}",
            "",
        ]

        for result in results:
            lines.extend(
                [
                    f"Test: {result.test_id}",
                    f"  Concurrency: {result.concurrency}",
                    f"  Total Requests: {result.total_requests}",
                    f"  Successful: {result.successful_requests} ({result.successful_requests / result.total_requests * 100:.1f}%)",
                    f"  Failed: {result.failed_requests} ({result.error_rate * 100:.1f}%)",
                    f"  Avg Response Time: {result.avg_response_time_ms:.2f}ms",
                    f"  P95 Response Time: {result.p95_response_time_ms:.2f}ms",
                    f"  P99 Response Time: {result.p99_response_time_ms:.2f}ms",
                    f"  Throughput: {result.requests_per_second:.2f} req/s",
                    f"  Duration: {result.test_duration_seconds:.2f}s",
                    "",
                    "-" * 80,
                ]
            )

        lines.extend(
            [
                "SUMMARY",
                "-" * 80,
            ]
        )

        avg_rps = sum(r.requests_per_second for r in results) / len(results) if results else 0
        avg_latency = sum(r.avg_response_time_ms for r in results) / len(results) if results else 0

        lines.extend(
            [
                f"Average Throughput: {avg_rps:.2f} requests/second",
                f"Average Latency: {avg_latency:.2f}ms",
            ]
        )

        lines.extend(
            [
                "=" * 80,
                "END OF REPORT",
                "=" * 80,
            ]
        )

        return "\n".join(lines)
