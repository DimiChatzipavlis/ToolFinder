from __future__ import annotations

import json
import os
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from time import time

from .policy import SecurityPolicyViolation


@dataclass
class TelemetryCollector:
    counters: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    latencies_ms: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    sink_path: str | None = None
    allowed_root: str | None = None
    max_latency_samples_per_metric: int = 500
    _external_latency_stats: dict[str, dict[str, float]] = field(default_factory=dict)

    def increment(self, key: str, amount: int = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + amount

    def record_latency(self, key: str, value_ms: float) -> None:
        samples = self.latencies_ms.setdefault(key, [])
        samples.append(float(value_ms))
        overflow = len(samples) - self.max_latency_samples_per_metric
        if overflow > 0:
            del samples[:overflow]

    def merge_snapshot(self, snapshot: dict[str, object]) -> None:
        counters = snapshot.get("counters", {}) if isinstance(snapshot, dict) else {}
        if isinstance(counters, dict):
            for key, value in counters.items():
                if isinstance(value, int):
                    self.increment(str(key), value)

        latencies = snapshot.get("latencies_ms", {}) if isinstance(snapshot, dict) else {}
        if not isinstance(latencies, dict):
            return

        for key, stats in latencies.items():
            if not isinstance(stats, dict):
                continue

            count = stats.get("count")
            min_value = stats.get("min")
            max_value = stats.get("max")
            mean_value = stats.get("mean")
            if not isinstance(count, (int, float)):
                continue
            if not isinstance(min_value, (int, float)):
                continue
            if not isinstance(max_value, (int, float)):
                continue
            if not isinstance(mean_value, (int, float)):
                continue

            incoming = {
                "count": float(count),
                "min": float(min_value),
                "max": float(max_value),
                "mean": float(mean_value),
            }
            existing = self._external_latency_stats.get(str(key))
            if existing is None:
                self._external_latency_stats[str(key)] = incoming
                continue

            total_count = existing["count"] + incoming["count"]
            if total_count <= 0:
                continue
            weighted_mean = (
                (existing["mean"] * existing["count"]) + (incoming["mean"] * incoming["count"])
            ) / total_count
            self._external_latency_stats[str(key)] = {
                "count": total_count,
                "min": min(existing["min"], incoming["min"]),
                "max": max(existing["max"], incoming["max"]),
                "mean": weighted_mean,
            }

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

        for key, external in self._external_latency_stats.items():
            count = int(external["count"])
            if count <= 0:
                continue
            existing = summary["latencies_ms"].get(key)
            if not isinstance(existing, dict):
                summary["latencies_ms"][key] = {
                    "count": count,
                    "min": round(external["min"], 3),
                    "max": round(external["max"], 3),
                    "mean": round(external["mean"], 3),
                }
                continue

            existing_count = int(existing.get("count", 0))
            existing_mean = float(existing.get("mean", 0.0))
            combined_count = existing_count + count
            if combined_count <= 0:
                continue
            combined_mean = (
                (existing_mean * existing_count) + (external["mean"] * count)
            ) / combined_count
            summary["latencies_ms"][key] = {
                "count": combined_count,
                "min": round(min(float(existing.get("min", external["min"])), external["min"]), 3),
                "max": round(max(float(existing.get("max", external["max"])), external["max"]), 3),
                "mean": round(combined_mean, 3),
            }
        return summary

    def persist_snapshot(self, context: dict[str, object] | None = None) -> None:
        if not self.sink_path:
            return

        payload: dict[str, object] = {
            "timestamp": time(),
            "telemetry": self.to_dict(),
        }
        if context:
            payload["context"] = context

        sink = Path(os.path.abspath(self.sink_path))
        allowed_root = self.allowed_root or os.getenv("ENTERPRISE_TELEMETRY_ALLOWED_ROOT", "")
        if allowed_root:
            resolved_allowed_root = os.path.abspath(allowed_root)
            try:
                is_within_root = os.path.commonpath([str(sink), resolved_allowed_root]) == resolved_allowed_root
            except ValueError as exc:
                raise SecurityPolicyViolation(
                    f"Telemetry sink path {sink} is not under allowed telemetry root {resolved_allowed_root}"
                ) from exc
            if not is_within_root:
                raise SecurityPolicyViolation(
                    f"Telemetry sink path {sink} is not under allowed telemetry root {resolved_allowed_root}"
                )
        sink.parent.mkdir(parents=True, exist_ok=True)
        with sink.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True))
            handle.write("\n")
