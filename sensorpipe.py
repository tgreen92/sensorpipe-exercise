"""sensorpipe: summarize sensor readings and flag anomalies.

The pipeline reads `timestamp_ms,sensor_id,value` rows, reports each sensor's
mean over a recent time window, and flags readings that look anomalous. Rows
can arrive out of order, and some are malformed.

Setup (Python 3.10 or later):

    pip install pytest
    python -m pytest sensorpipe.py     # run the tests
    python sensorpipe.py               # run on built-in sample data
    python sensorpipe.py readings.csv  # run on a CSV file with a header row

Some tests fail and the command-line run crashes. That is the starting point.

The session has three tasks, in this order. Talk us through what you are
thinking as you work. We care more about how you approach each problem than
about finishing all three.

1. Debug. Some of the existing code is broken. Find and fix the defects so
   that the parse, window and summarize tests pass.
2. Instrument. Add logging and metrics so that someone running this pipeline
   in production can tell what it did and how long it took. Decide what is
   worth recording.
3. Implement. Write `detect_anomalies`. Its docstring is the specification,
   and the `test_anomaly_*` tests check it.

Use the editor and tools you normally use. We will agree on the use of AI
assistants at the start of the session.
"""

import argparse
import random
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev


@dataclass(frozen=True)
class Reading:
    timestamp: float
    """Seconds since the Unix epoch."""
    sensor_id: str
    value: float


@dataclass(frozen=True)
class Anomaly:
    reading: Reading
    history_mean: float
    history_stdev: float


def parse_line(line: str) -> Reading | None:
    """Parse one `timestamp_ms,sensor_id,value` line. Return None if the line is malformed."""
    parts = line.strip().split(",")
    if len(parts) != 3:
        return None
    raw_timestamp, sensor_id, raw_value = parts
    try:
        timestamp = int(raw_timestamp) / 100
        value = float(raw_value)
    except ValueError:
        return None
    return Reading(timestamp=timestamp, sensor_id=sensor_id.strip(), value=value)


def parse_lines(lines: Iterable[str]) -> list[Reading]:
    """Parse CSV lines that start with a header row, skipping malformed lines."""
    rows = iter(lines)
    next(rows)
    readings: list[Reading] = []
    for line in rows:
        reading = parse_line(line)
        if reading is not None:
            readings.append(reading)
    return readings


class SlidingWindow:
    """The readings from the last `seconds` seconds, oldest first.

    Readings must be added in timestamp order. A reading exactly `seconds`
    older than the newest one stays in the window.
    """

    def __init__(self, seconds: float, readings: list[Reading] = []) -> None:
        self.seconds = seconds
        self.readings = readings

    def add(self, reading: Reading) -> None:
        self.readings.append(reading)
        cutoff = reading.timestamp - self.seconds
        while self.readings[0].timestamp < cutoff:
            self.readings.pop(0)

    def values(self) -> list[float]:
        return [reading.value for reading in self.readings]

    def mean(self) -> float:
        return fmean(self.values())

    def stdev(self) -> float:
        return pstdev(self.values())


def summarize(readings: Iterable[Reading], window_seconds: float) -> dict[str, float]:
    """Return each sensor's mean value over the window ending at that sensor's newest reading."""
    windows: dict[str, SlidingWindow] = {}
    for reading in sorted(readings, key=lambda r: r.timestamp):
        window = windows.setdefault(reading.sensor_id, SlidingWindow(window_seconds))
        window.add(reading)
    return {sensor_id: window.mean() for sensor_id, window in windows.items()}


def detect_anomalies(
    readings: Iterable[Reading],
    window_seconds: float,
    threshold: float = 3.0,
    min_history: int = 5,
) -> list[Anomaly]:
    """Return the readings that sit far from their sensor's recent history.

    Process the readings in timestamp order. The history of a reading at time
    `t` is the earlier readings from the same sensor with a timestamp in
    `[t - window_seconds, t)`. The reading is an anomaly when its history holds
    at least `min_history` readings and

        abs(value - mean(history)) > threshold * pstdev(history)

    Every reading, anomalous or not, joins the history of later readings.
    Return the anomalies in timestamp order.
    """
    raise NotImplementedError


def sample_lines() -> list[str]:
    """Shuffled sample readings with two injected spikes and a few malformed rows."""
    rng = random.Random(7)
    base_ms = 1_760_000_000_000
    sensors = {"temp-1": (21.0, 0.4), "temp-2": (35.0, 0.8), "pressure-1": (101.3, 0.2)}
    rows: list[str] = []
    for i in range(120):
        for sensor_id, (mean, stdev) in sensors.items():
            timestamp_ms = base_ms + i * 500 + rng.randint(0, 40)
            value = rng.gauss(mean, stdev)
            if (sensor_id, i) in {("temp-2", 80), ("pressure-1", 101)}:
                value = mean * 1.5 if sensor_id == "temp-2" else mean - 3.4
            rows.append(f"{timestamp_ms},{sensor_id},{value:.3f}")
    rng.shuffle(rows)
    rows[40:40] = ["1760000020250,temp-1,"]
    rows[95:95] = ["1760000031000;temp-2;35.1"]
    rows[200:200] = [""]
    rows[260:260] = ["NaT,pressure-1,101.2"]
    return ["timestamp_ms,sensor_id,value", *rows]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize sensor readings.")
    parser.add_argument("path", type=Path, nargs="?", help="CSV file; omit to use built-in sample data")
    parser.add_argument("--window", type=float, default=10.0, help="window length in seconds")
    parser.add_argument("--threshold", type=float, default=3.0, help="anomaly threshold in standard deviations")
    args = parser.parse_args(argv)

    lines = args.path.read_text().splitlines() if args.path else sample_lines()
    readings = parse_lines(lines)

    print("Sensor means:")
    for sensor_id, mean in sorted(summarize(readings, args.window).items()):
        print(f"  {sensor_id}: {mean:.2f}")

    print("Anomalies:")
    for anomaly in detect_anomalies(readings, args.window, args.threshold):
        reading = anomaly.reading
        print(
            f"  t={reading.timestamp:.3f} {reading.sensor_id}={reading.value:.2f} "
            f"(history mean {anomaly.history_mean:.2f}, stdev {anomaly.history_stdev:.2f})"
        )
    return 0


def series(sensor_id: str, start: float, values: list[float]) -> list[Reading]:
    """One reading per second starting at `start`."""
    return [Reading(timestamp=start + i, sensor_id=sensor_id, value=v) for i, v in enumerate(values)]


def test_parse_line_converts_milliseconds_to_seconds() -> None:
    assert parse_line("1500,temp-1,2.5\n") == Reading(timestamp=1.5, sensor_id="temp-1", value=2.5)


def test_parse_line_rejects_malformed_lines() -> None:
    assert parse_line("") is None
    assert parse_line("1500,temp-1") is None
    assert parse_line("noon,temp-1,2.5") is None
    assert parse_line("1500,temp-1,warm") is None


def test_parse_lines_skips_header_and_malformed_lines() -> None:
    lines = ["timestamp_ms,sensor_id,value", "1000,a,1.0", "garbage", "2000,b,2.0"]

    assert parse_lines(lines) == [
        Reading(timestamp=1.0, sensor_id="a", value=1.0),
        Reading(timestamp=2.0, sensor_id="b", value=2.0),
    ]


def test_window_evicts_readings_older_than_window() -> None:
    window = SlidingWindow(seconds=10)
    for reading in series("a", 0, [1.0]) + series("a", 5, [2.0]) + series("a", 20, [3.0]):
        window.add(reading)

    assert window.values() == [3.0]


def test_window_keeps_reading_on_boundary() -> None:
    window = SlidingWindow(seconds=10)
    window.add(Reading(timestamp=0, sensor_id="a", value=1.0))
    window.add(Reading(timestamp=10, sensor_id="a", value=3.0))

    assert window.values() == [1.0, 3.0]
    assert window.mean() == 2.0
    assert window.stdev() == 1.0


def test_summarize_reports_mean_per_sensor() -> None:
    readings = series("a", 0, [1.0, 3.0]) + series("b", 0, [100.0, 200.0])

    assert summarize(readings, window_seconds=10) == {"a": 2.0, "b": 150.0}


def test_summarize_uses_only_recent_readings() -> None:
    readings = series("a", 0, [1000.0]) + series("a", 50, [4.0, 6.0])

    assert summarize(readings, window_seconds=10) == {"a": 5.0}


def test_anomaly_detects_spike_after_steady_history() -> None:
    readings = series("a", 0, [10.0, 10.0, 10.0, 10.0, 10.0, 50.0])

    assert detect_anomalies(readings, window_seconds=10) == [
        Anomaly(reading=readings[-1], history_mean=10.0, history_stdev=0.0)
    ]


def test_anomaly_ignores_normal_noise() -> None:
    readings = series("a", 0, [10.0, 11.0, 9.0, 10.0, 11.0, 9.0, 10.5])

    assert detect_anomalies(readings, window_seconds=10) == []


def test_anomaly_needs_min_history_before_flagging() -> None:
    readings = series("a", 0, [10.0, 10.0, 10.0, 10.0, 50.0])

    assert detect_anomalies(readings, window_seconds=10, min_history=5) == []


def test_anomaly_history_excludes_readings_older_than_window() -> None:
    readings = series("a", 0, [10.0, 10.0, 10.0, 10.0, 10.0]) + series("a", 20, [50.0])

    assert detect_anomalies(readings, window_seconds=10) == []


def test_anomaly_reading_joins_later_history() -> None:
    readings = series("a", 0, [10.0, 10.0, 10.0, 10.0, 10.0, 50.0, 11.0])

    anomalies = detect_anomalies(readings, window_seconds=10)

    assert [a.reading for a in anomalies] == [readings[5]]


def test_anomaly_sensors_have_separate_histories_and_results_are_time_ordered() -> None:
    readings = series("b", 0.5, [100.0, 100.0, 100.0, 100.0, 100.0, 0.0]) + series(
        "a", 0, [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 50.0]
    )

    anomalies = detect_anomalies(readings, window_seconds=10)

    assert [(a.reading.sensor_id, a.reading.timestamp) for a in anomalies] == [("b", 5.5), ("a", 6.0)]


if __name__ == "__main__":
    sys.exit(main())
