from dataclasses import dataclass
from typing import Any, Optional, Dict, List

from prometheus_client import Gauge, Info, Histogram


LABELS = ["request_type", "service"]

@dataclass
class Metrics:
    """
    A dataclass to store metrics for a request.
    """
    # System metrics
    system_info: Dict[str, Any]
    state_size: int
    cpu_utilization: float
    cpu_ram_pid: float
    gpu_utilization: float
    gpu_ram_usage: float
    gpu_ram_free: float
    gpu_ram_pid: float

    # Metrics from the C++ backend
    load_time: float
    sample_time: float
    sample_throughput: float
    time_to_first_token: float
    time_per_output_token: List[float]
    prompt_eval_time: float
    prompt_eval_throughput: float
    completion_eval_time: float
    completion_eval_throughput: float
    end_to_end_latency: float
    prefill_tokens: int
    generation_tokens: int  
    kv_cache_usage_ratio: float


class MetricsExporter:
    """
    A custom Prometheus Metrics Explorer for the LLAMA C++ backend.
    Collects metrics per request sent to the backend.
    """
    def __init__(self):
        self.labels = LABELS
        # One-time metrics
        self._histogram_load_time = Histogram(
            name="llama_cpp_python:load_t_seconds",
            documentation="Histogram of load time in seconds",
            labelnames=self.labels,
            buckets=[
                0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                8.0, 9.0, 10.0, 12.5, 15.0, 20.0, 25.0, 30.0
            ]
        )
        # Request-level latencies
        self._histogram_sample_time = Histogram(
            name="llama_cpp_python:sample_t_seconds",
            documentation="Histogram of token sampling time in seconds",
            labelnames=self.labels,
            buckets=[
                0.00001, 0.00005, 0.0001, 0.00025, 0.0005, 0.001, 0.0025,
                0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5,
            ]
        )
        self._histogram_time_to_first_token = Histogram(
            name="llama_cpp_python:ttft_seconds",
            documentation="Histogram of time to first token in seconds",
            labelnames=self.labels,
            buckets=[
                0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5,
                0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 25.0, 30.0
            ]
        )
        self._histogram_time_per_output_token = Histogram(
            name="llama_cpp_python:tpot_seconds",
            documentation="Histogram of time per output token in seconds",
            labelnames=self.labels,
            buckets=[
                0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5,
                0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 25.0, 30.0
            ]
        )
        self._histogram_prompt_eval_time = Histogram(
            name="llama_cpp_python:p_eval_t_seconds",
            documentation="Histogram of prompt evaluation time in seconds",
            labelnames=self.labels,
            buckets=[
                0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0,
                20.0, 25.0, 30.0, 40.0, 50.0, 60.0
            ]
        )
        self._histogram_completion_eval_time = Histogram(
            name="llama_cpp_python:c_eval_t_seconds",
            documentation="Histogram of completion evaluation time in seconds",
            labelnames=self.labels,
            buckets=[
                0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0,
                20.0, 25.0, 30.0, 40.0, 50.0, 60.0
            ]
        )
        self._histogram_e2e_request_latency = Histogram(
            name="llama_cpp_python:e2e_seconds",
            documentation="Histogram of end-to-end request latency in seconds",
            labelnames=self.labels,
            buckets=[
                0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0,
                20.0, 25.0, 30.0, 40.0, 50.0, 60.0
            ]
        )
        # Prefill and generation tokens
        self._histogram_prefill_tokens = Histogram(
            name="llama_cpp_python:prefill_tokens_total",
            documentation="Histogram of number of prefill tokens processed",
            labelnames=self.labels,
            buckets=[
                1, 10, 25, 50, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000,
                3500, 4000, 4500, 5000
            ]
        )
        self._histogram_generation_tokens = Histogram(
            name="llama_cpp_python:completion_tokens_total",
            documentation="Histogram of number of generation tokens processed",
            labelnames=self.labels,
            buckets=[
                1, 10, 25, 50, 100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000,
                3500, 4000, 4500, 5000
            ]
        )
        # Current throughput
        self._gauge_prompt_eval_throughput = Gauge(
            name="llama_cpp_python:prompt_eval_throughput",
            documentation="Current throughput of the prompt evaluation process (in tokens/second)",
            labelnames=self.labels
        )
        self._gauge_completion_eval_throughput = Gauge(
            name="llama_cpp_python:completion_eval_throughput",
            documentation="Current throughput of the completion evaluation process (in tokens/second)",
            labelnames=self.labels
        )
        self._gauge_sample_throughput = Gauge(
            name="llama_cpp_python:sample_throughput",
            documentation="Current throughput of the token sampling process (in tokens/second)",
            labelnames=self.labels
        )
        # System info
        self._gauge_state_size = Gauge(
            name="llama_cpp_python:state_size",
            documentation="Current state size in bytes of various components such as rng (random number generator), logits, embedding, and kv_cache (key-value cache)",
            labelnames=self.labels
        )
        self._gauge_cpu_utilization = Gauge(
            name="llama_cpp_python:cpu_utilization",
            documentation="Current CPU utilization",
            labelnames=self.labels
        )
        self._gauge_cpu_ram_usage_by_pid = Gauge(
            name="llama_cpp_python:cpu_memory_usage_by_pid",
            documentation="Current CPU memory usage during the request",
            labelnames=self.labels
        )
        self._gauge_gpu_utilization = Gauge(
            name="llama_cpp_python:gpu_utilization",
            documentation="Current GPU utilization",
            labelnames=self.labels
        )
        self._gauge_gpu_memory_usage = Gauge(
            name="llama_cpp_python:gpu_memory_usage",
            documentation="Current GPU memory usage",
            labelnames=self.labels
        )
        self._gauge_gpu_memory_free = Gauge(
            name="llama_cpp_python:gpu_memory_free",
            documentation="Current free GPU memory",
            labelnames=self.labels
        )
        self._gauge_gpu_memory_usage_by_pid = Gauge(
            name="llama_cpp_python:gpu_memory_usage_by_pid",
            documentation="Current GPU memory usage during the request",
            labelnames=self.labels
        )
        self._gauge_kv_cache_usage_ratio = Gauge(
            name="llama_cpp_python:kv_cache_usage_ratio",
            documentation="KV-cache usage. 1 means 100 percent usage",
            labelnames=self.labels
        )
        self._info = Info(
            name="llama_cpp_python:info",
            documentation="Server metadata"
        )

    def log_metrics(self, metrics: Metrics, labels: Dict[str, str]):
        """
        Log the metrics using the Prometheus client.
        """
        self._histogram_load_time.labels(**labels).observe(metrics.load_time)
        self._histogram_sample_time.labels(**labels).observe(metrics.sample_time)
        if metrics.time_to_first_token:
            self._histogram_time_to_first_token.labels(**labels).observe(metrics.time_to_first_token)
        for _tpot in metrics.time_per_output_token:
            self._histogram_time_per_output_token.labels(**labels).observe(_tpot)
        self._histogram_prompt_eval_time.labels(**labels).observe(metrics.prompt_eval_time)
        self._histogram_completion_eval_time.labels(**labels).observe(metrics.completion_eval_time)
        self._histogram_e2e_request_latency.labels(**labels).observe(metrics.end_to_end_latency)
        self._histogram_prefill_tokens.labels(**labels).observe(metrics.prefill_tokens)
        self._histogram_generation_tokens.labels(**labels).observe(metrics.generation_tokens)
        self._gauge_prompt_eval_throughput.labels(**labels).set(metrics.prompt_eval_throughput)
        self._gauge_completion_eval_throughput.labels(**labels).set(metrics.completion_eval_throughput)
        self._gauge_sample_throughput.labels(**labels).set(metrics.sample_throughput)
        self._gauge_cpu_utilization.labels(**labels).set(metrics.cpu_utilization)
        self._gauge_cpu_ram_usage_by_pid.labels(**labels).set(metrics.cpu_ram_pid)
        self._gauge_gpu_utilization.labels(**labels).set(metrics.gpu_utilization)
        self._gauge_gpu_memory_usage.labels(**labels).set(metrics.gpu_ram_usage)
        self._gauge_gpu_memory_free.labels(**labels).set(metrics.gpu_ram_free)
        self._gauge_gpu_memory_usage_by_pid.labels(**labels).set(metrics.gpu_ram_pid)
        self._gauge_state_size.labels(**labels).set(metrics.state_size)
        self._gauge_kv_cache_usage_ratio.labels(**labels).set(metrics.kv_cache_usage_ratio)
        self._info.info(metrics.system_info)