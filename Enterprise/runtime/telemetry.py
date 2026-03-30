from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class TelemetryCollector:
    counters: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    latencies_ms: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def increment(self, key: str, amount: int = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + amount

    def record_latency(self, key: str, value_ms: float) -> None:
        self.latencies_ms.setdefault(key, []).append(float(value_ms))

    def to_dict(self) -> dict[str, object]:
        summary: dict[str, object] = {"counters": dict(self.counters), "latencies_ms": {}}
        for key, values in self.latencies_ms.items():
            if not values:
                continue
            summary["latencies_ms"][key] = {
                "count": len(values),
                "min": round(min(values), 3),
                "max": round(max(values), 3),
                "mean": round(statistics.mean(values), 3),
            }
        return summary
