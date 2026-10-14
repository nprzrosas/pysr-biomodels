"""Low-overhead process-tree telemetry for model-level batch workers."""

import csv
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any, Self, TextIO

import psutil

DEFAULT_SAMPLING_INTERVAL_SECONDS = 5.0
BYTES_PER_GIBIBYTE = 1024**3

TELEMETRY_COLUMNS = (
    "timestamp_utc",
    "elapsed_seconds",
    "run_id",
    "worker_id",
    "model_id",
    "stage",
    "worker_state",
    "parent_pid",
    "child_pids",
    "process_tree_cpu_percent_raw",
    "process_tree_cpu_percent_capacity_normalized",
    "process_tree_rss_bytes",
    "system_cpu_percent",
    "system_memory_used_bytes",
    "system_memory_available_bytes",
    "effective_memory_available_bytes",
    "system_memory_percent",
    "system_swap_total_bytes",
    "system_swap_used_bytes",
)


def _read_text(path: Path) -> str | None:
    """Read one optional system-control file.

    Args:
        path: File to read.

    Returns:
        Stripped content, or ``None`` when unavailable.
    """
    try:
        return path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None


def read_cgroup_v2_cpu_capacity(
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> float | None:
    """Read the cgroup-v2 CPU quota as a fractional CPU capacity.

    Args:
        cgroup_root: Mounted cgroup-v2 control directory.

    Returns:
        CPU capacity implied by quota/period, or ``None`` for no limit.
    """
    value = _read_text(Path(cgroup_root) / "cpu.max")
    if value is None:
        return None

    fields = value.split()
    if len(fields) != 2 or fields[0] == "max":
        return None

    try:
        quota = float(fields[0])
        period = float(fields[1])
    except ValueError:
        return None

    if quota <= 0.0 or period <= 0.0:
        return None
    return quota / period


def read_cgroup_v2_memory_limit(
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> int | None:
    """Read the cgroup-v2 memory limit.

    Args:
        cgroup_root: Mounted cgroup-v2 control directory.

    Returns:
        Limit in bytes, or ``None`` when unlimited or unavailable.
    """
    value = _read_text(Path(cgroup_root) / "memory.max")
    if value is None or value == "max":
        return None

    try:
        limit = int(value)
    except ValueError:
        return None
    return limit if limit > 0 else None


def read_cgroup_v2_memory_current(
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> int | None:
    """Read current cgroup-v2 memory consumption.

    Args:
        cgroup_root: Mounted cgroup-v2 control directory.

    Returns:
        Current consumption in bytes, or ``None`` when unavailable.
    """
    value = _read_text(Path(cgroup_root) / "memory.current")
    if value is None:
        return None

    try:
        current = int(value)
    except ValueError:
        return None
    return current if current >= 0 else None


def memory_snapshot(
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> dict[str, int | float | None]:
    """Capture host and cgroup-aware available-memory information.

    Args:
        cgroup_root: Mounted cgroup-v2 control directory.

    Returns:
        JSON-safe memory and swap measurements. Effective available memory is
        the smaller of host-available memory and remaining cgroup capacity.
    """
    virtual_memory = psutil.virtual_memory()
    swap_memory = psutil.swap_memory()
    cgroup_limit = read_cgroup_v2_memory_limit(cgroup_root)
    cgroup_current = read_cgroup_v2_memory_current(cgroup_root)
    effective_available = int(virtual_memory.available)
    if cgroup_limit is not None and cgroup_current is not None:
        effective_available = min(
            effective_available,
            max(0, cgroup_limit - cgroup_current),
        )

    return {
        "system_memory_total_bytes": int(virtual_memory.total),
        "system_memory_used_bytes": int(virtual_memory.used),
        "system_memory_available_bytes": int(virtual_memory.available),
        "system_memory_percent": float(virtual_memory.percent),
        "system_swap_total_bytes": int(swap_memory.total),
        "system_swap_used_bytes": int(swap_memory.used),
        "cgroup_v2_memory_limit_bytes": cgroup_limit,
        "cgroup_v2_memory_current_bytes": cgroup_current,
        "effective_memory_available_bytes": effective_available,
    }


@dataclass(frozen=True)
class MemorySafetyPolicy:
    """Configurable coordinator-only memory thresholds.

    A zero threshold disables the corresponding gate. The policy influences
    orchestration only; it never changes scientific settings or results.
    """

    minimum_launch_available_bytes: int = 0
    emergency_available_bytes: int = 0
    emergency_consecutive_samples: int = 2

    def __post_init__(self) -> None:
        """Validate non-negative thresholds and a positive sample count."""
        if self.minimum_launch_available_bytes < 0:
            raise ValueError("Minimum launch memory cannot be negative.")
        if self.emergency_available_bytes < 0:
            raise ValueError("Emergency memory cannot be negative.")
        if self.emergency_consecutive_samples < 1:
            raise ValueError("Emergency sample count must be at least one.")


class MemoryPressureGuard:
    """Detect persistent low-memory conditions without reacting to spikes."""

    def __init__(self, policy: MemorySafetyPolicy) -> None:
        """Initialize the guard.

        Args:
            policy: Thresholds controlling launch and runtime decisions.
        """
        self.policy = policy
        self.consecutive_low_samples = 0

    def observe(self, available_bytes: int) -> bool:
        """Record one sample and report whether emergency action is required.

        Args:
            available_bytes: Effective host/cgroup memory currently available.

        Returns:
            ``True`` after the configured number of consecutive low samples.
        """
        if self.policy.emergency_available_bytes <= 0:
            self.consecutive_low_samples = 0
            return False
        if available_bytes < self.policy.emergency_available_bytes:
            self.consecutive_low_samples += 1
        else:
            self.consecutive_low_samples = 0
        return self.consecutive_low_samples >= self.policy.emergency_consecutive_samples


def validate_prelaunch_memory(
    policy: MemorySafetyPolicy,
    snapshot: dict[str, int | float | None],
) -> None:
    """Reject a batch before filesystem or worker side effects when unsafe.

    Args:
        policy: Configured memory thresholds.
        snapshot: Current result from :func:`memory_snapshot`.

    Raises:
        MemoryError: If effective available memory is below the launch floor.
    """
    threshold = policy.minimum_launch_available_bytes
    available = int(snapshot["effective_memory_available_bytes"] or 0)
    if threshold > 0 and available < threshold:
        raise MemoryError(
            "Pre-launch memory safety gate blocked the batch: "
            f"available={available} bytes, required={threshold} bytes."
        )


def discover_resources(cgroup_root: Path = Path("/sys/fs/cgroup")) -> dict[str, Any]:
    """Discover host, affinity, and cgroup-visible compute resources.

    Args:
        cgroup_root: Mounted cgroup-v2 control directory.

    Returns:
        JSON-safe CPU and memory capacity metadata.
    """
    physical = psutil.cpu_count(logical=False)
    logical = psutil.cpu_count(logical=True)

    try:
        affinity_count = len(psutil.Process().cpu_affinity())
    except (AttributeError, psutil.AccessDenied, psutil.Error):
        affinity_count = None

    cgroup_cpu_capacity = read_cgroup_v2_cpu_capacity(cgroup_root)
    capacity_candidates = [
        float(value)
        for value in (logical, affinity_count, cgroup_cpu_capacity)
        if value is not None and float(value) > 0.0
    ]
    effective_cpu_capacity = min(capacity_candidates) if capacity_candidates else 1.0

    memory = memory_snapshot(cgroup_root)
    cgroup_memory_limit = memory["cgroup_v2_memory_limit_bytes"]
    effective_memory_limit = int(memory["system_memory_total_bytes"] or 0)
    if cgroup_memory_limit is not None:
        effective_memory_limit = min(effective_memory_limit, cgroup_memory_limit)

    return {
        "physical_cpu_count": physical,
        "logical_cpu_count": logical,
        "affinity_visible_cpu_count": affinity_count,
        "cgroup_v2_cpu_capacity": cgroup_cpu_capacity,
        "effective_cpu_capacity": effective_cpu_capacity,
        **memory,
        "cgroup_v2_memory_limit_bytes": cgroup_memory_limit,
        "effective_memory_limit_bytes": effective_memory_limit,
    }


@dataclass(frozen=True)
class WorkerTelemetryContext:
    """Stable labels attached to one worker process-tree sample."""

    run_id: str
    worker_id: str
    model_id: str
    stage: str
    worker_state: str


class ProcessTreeSampler:
    """Sample CPU and RSS for a parent process and recursive children.

    Summed RSS may double-count physical pages shared by related processes.
    This monitor is observational only and never changes worker allocation,
    seeds, scientific settings, equations, or selection behavior.
    """

    def __init__(self, *, parent_pid: int, effective_cpu_capacity: float) -> None:
        """Initialize a sampler and capture the parent's creation time.

        Args:
            parent_pid: Root PID for recursive process accounting.
            effective_cpu_capacity: Capacity used for normalized CPU percent.
        """
        if effective_cpu_capacity <= 0.0:
            raise ValueError("effective_cpu_capacity must be positive.")

        self.parent_pid = int(parent_pid)
        self.effective_cpu_capacity = float(effective_cpu_capacity)
        parent = psutil.Process(self.parent_pid)
        self._parent_create_time = parent.create_time()
        self._known_processes: dict[tuple[int, float], psutil.Process] = {}
        self._primed = False

    def _process_tree(self) -> list[psutil.Process]:
        """Return live processes while guarding against PID reuse."""
        try:
            parent = psutil.Process(self.parent_pid)
            if parent.create_time() != self._parent_create_time:
                return []
            processes = [parent, *parent.children(recursive=True)]
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return []

        live: list[psutil.Process] = []
        for process in processes:
            try:
                process.create_time()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
            live.append(process)
        return live

    def prime(self) -> None:
        """Prime psutil CPU counters before the first real sample."""
        for process in self._process_tree():
            try:
                key = (process.pid, process.create_time())
                process.cpu_percent(interval=None)
                self._known_processes[key] = process
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        psutil.cpu_percent(interval=None)
        self._primed = True

    def sample(
        self,
        *,
        context: WorkerTelemetryContext,
        elapsed_seconds: float,
    ) -> dict[str, Any]:
        """Collect one process-tree and system telemetry row.

        Args:
            context: Run, model, worker, and stage labels.
            elapsed_seconds: Elapsed run time at sampling.

        Returns:
            A row matching ``TELEMETRY_COLUMNS``.
        """
        if not self._primed:
            self.prime()

        processes = self._process_tree()
        raw_cpu_percent = 0.0
        rss_bytes = 0
        child_pids: list[int] = []

        for process in processes:
            try:
                key = (process.pid, process.create_time())
                if key in self._known_processes:
                    known_process = self._known_processes[key]
                    raw_cpu_percent += known_process.cpu_percent(interval=None)
                else:
                    process.cpu_percent(interval=None)
                    self._known_processes[key] = process
                rss_bytes += process.memory_info().rss
                if process.pid != self.parent_pid:
                    child_pids.append(process.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

        memory = memory_snapshot()
        return {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "elapsed_seconds": round(float(elapsed_seconds), 6),
            "run_id": context.run_id,
            "worker_id": context.worker_id,
            "model_id": context.model_id,
            "stage": context.stage,
            "worker_state": context.worker_state,
            "parent_pid": self.parent_pid,
            "child_pids": ";".join(str(pid) for pid in sorted(child_pids)),
            "process_tree_cpu_percent_raw": raw_cpu_percent,
            "process_tree_cpu_percent_capacity_normalized": (
                raw_cpu_percent / self.effective_cpu_capacity
            ),
            "process_tree_rss_bytes": rss_bytes,
            "system_cpu_percent": psutil.cpu_percent(interval=None),
            "system_memory_used_bytes": memory["system_memory_used_bytes"],
            "system_memory_available_bytes": memory["system_memory_available_bytes"],
            "effective_memory_available_bytes": memory[
                "effective_memory_available_bytes"
            ],
            "system_memory_percent": memory["system_memory_percent"],
            "system_swap_total_bytes": memory["system_swap_total_bytes"],
            "system_swap_used_bytes": memory["system_swap_used_bytes"],
        }


class TelemetryWriter:
    """Coordinator-owned append-only CSV telemetry writer."""

    def __init__(self, path: Path) -> None:
        """Create the CSV and write its header when needed.

        Args:
            path: Persistent telemetry CSV path.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._owner_thread = threading.get_ident()
        existed = self.path.exists() and self.path.stat().st_size > 0
        self._stream = self.path.open("a", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._stream, fieldnames=TELEMETRY_COLUMNS)
        if not existed:
            self._writer.writeheader()
            self._stream.flush()

    def write(self, row: dict[str, Any]) -> None:
        """Append and flush one telemetry row.

        Args:
            row: Complete telemetry record.
        """
        if threading.get_ident() != self._owner_thread:
            raise RuntimeError("Only the coordinator may write telemetry.")
        self._writer.writerow({column: row[column] for column in TELEMETRY_COLUMNS})
        self._stream.flush()

    def close(self) -> None:
        """Close the telemetry stream."""
        self._stream.close()

    def __enter__(self) -> Self:
        """Return this writer for context-manager use."""
        return self

    def __exit__(self, *_args: object) -> None:
        """Close the writer on context-manager exit."""
        self.close()


class PlainTextProgressReporter:
    """Non-interactive progress output independent of terminal features."""

    def __init__(self, stream: TextIO) -> None:
        """Configure output.

        Args:
            stream: Text stream receiving progress lines.
        """
        self.stream = stream

    def report(
        self,
        *,
        completed: int,
        total: int,
        failed: int,
        active_rows: list[dict[str, Any]],
    ) -> None:
        """Print one summary and one line per active worker.

        Args:
            completed: Number of finished models.
            total: Total model count.
            failed: Number of failed models.
            active_rows: Current telemetry rows for active workers.
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        summary = (
            f"[{timestamp}] completed={completed}/{total} "
            f"active={len(active_rows)} failed={failed}"
        )
        if active_rows:
            elapsed_seconds = max(float(row["elapsed_seconds"]) for row in active_rows)
            system_cpu = float(active_rows[0]["system_cpu_percent"])
            system_memory = float(active_rows[0]["system_memory_percent"])
            available_gib = (
                float(active_rows[0]["effective_memory_available_bytes"])
                / BYTES_PER_GIBIBYTE
            )
            summary += (
                f" elapsed={elapsed_seconds:.1f}s"
                f" system_cpu={system_cpu:.1f}%"
                f" system_ram={system_memory:.1f}%"
                f" available={available_gib:.2f}GiB"
            )
        self.stream.write(f"{summary}\n")
        for row in active_rows:
            gib = float(row["process_tree_rss_bytes"]) / (1024**3)
            self.stream.write(
                f"worker={row['worker_id']} model={row['model_id']} "
                f"stage={row['stage']} "
                f"pid={row['parent_pid']} "
                f"cpu={float(row['process_tree_cpu_percent_raw']):.1f}% "
                "capacity="
                f"{float(row['process_tree_cpu_percent_capacity_normalized']):.1f}% "
                f"rss={gib:.2f}GiB "
                f"elapsed={float(row['elapsed_seconds']):.1f}s\n"
            )
        self.stream.flush()


def monotonic_elapsed(started_at: float) -> float:
    """Return elapsed monotonic seconds for telemetry.

    Args:
        started_at: Value previously returned by ``time.monotonic``.

    Returns:
        Non-negative elapsed seconds.
    """
    return max(0.0, monotonic() - started_at)
