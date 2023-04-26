from prometheus_client import CollectorRegistry, Gauge, push_to_gateway


class PrometheusClient():
    def __init__(self, pushgateway_server) -> None:
        self.push_gateway_server = pushgateway_server

    def push_to_prometheus(self, metric_name, metric_value, itr) -> None:
        registry = CollectorRegistry()
        gauge: Gauge = Gauge(metric_name, f"iteration_{itr}", registry=registry)
        gauge.set(metric_value)
        push_to_gateway(self.push_gateway_server, job="ablator_logs", registry=registry)
