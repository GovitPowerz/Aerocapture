"""Per-update JSONL logger for RL training. Mirrors the GA TrainingLogger contract."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class RLLogger:
    def __init__(self, output_dir: Path, config_hash: str) -> None:
        self._config_hash = config_hash
        self._buffer: list[dict[str, Any]] = []
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S")
        self._filepath = output_dir / f"rl_training_{timestamp}.jsonl"
        self._file = open(self._filepath, "a")  # noqa: SIM115

    def log_update(self, record: dict[str, Any]) -> None:
        record = {**record, "timestamp": datetime.now(tz=UTC).isoformat(), "config_hash": self._config_hash}
        self._buffer.append(record)
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    @property
    def buffer(self) -> list[dict[str, Any]]:
        return self._buffer

    @property
    def filepath(self) -> Path:
        return self._filepath

    def close(self) -> None:
        self._file.close()
