"""The run gate: one object every worker passes through before each model call.

It does these things for the whole run at once:

- Rate limit: at most `rpm` calls per minute across the workers, and on any HTTP 429
  a shared pause: one 429 stops new calls for every worker until the wait is over.
  The wait is the host's Retry-After value, held between 2 seconds (floor) and 120
  seconds (ceiling); the backoff step (15 seconds on the first try) applies only when
  the host sends no Retry-After.
- Escalation: the gate counts 429s per minute. Above 10 in the last minute the pause
  is twice the Retry-After, above 20 four times (still capped at 120 seconds).
- Saturation: when 429s outnumber successful calls in each of three consecutive whole
  minutes the host is "saturated": the gate reports it (run.py writes state
  host_saturated with the per-minute counts into the manifest, and the plan runner's
  status line shows that word) and the run keeps
  going; the operator decides whether to move to the backup host. It clears when three
  consecutive minutes have 429s at or below the successes again. Nothing stops by
  itself.
- Pause accounting: total seconds the run spent paused on 429s (overlapping pauses
  counted once) and the rows-per-minute pace outside those pauses, for the manifest.
- Outage detection: when every worker that has a call in flight fails with a
  connection-level error (DNS, connect, read timeout) inside one 60-second window,
  and no call succeeded in that window, that is a network outage, not a set of bad
  rows. The gate raises NetworkOutage to
  the workers so their rows go back to the queue with nothing marked failed, then
  blocks new calls, probes the host every 30 seconds with a HEAD request, and lets
  the workers continue when it answers. After `give_up_s` (30 minutes) down it gives
  up and the run exits with code 75 so the plan runner can relaunch it later.
- Records: every 429 (Retry-After, metadata) and every outage (start, end, rows in
  flight) for the manifest.
"""
import threading
import time
from typing import Callable, Dict, List, Optional

from adapters.base import is_connection_error, is_rate_limit, rate_limit_info

EXIT_PAUSED_ON_NETWORK = 75      # sysexits' "temporary failure": relaunch later with --resume
EXIT_PAUSED_BY_OWNER = 76        # stopped cleanly by a signal or a STOP file; relaunch with --resume when wanted
OUTAGE_WINDOW_S = 60.0
PROBE_EVERY_S = 30.0
RETRY_AFTER_FLOOR_S = 2.0
RETRY_AFTER_CEILING_S = 120.0
ESCALATION_TIERS = ((20, 4.0), (10, 2.0))     # (429s in the last minute above which, multiplier)
SATURATION_MINUTES = 3                         # consecutive whole minutes with more 429s than successes


class NetworkOutage(Exception):
    """Raised to a worker whose call failed (or was about to start) during an outage.
    Carries the attempt records made so far so the caller can log them; the row itself
    goes back to the queue."""

    def __init__(self, message: str, attempts: Optional[List[Dict]] = None):
        super().__init__(message)
        self.attempts = attempts or []


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class RunGate:
    def __init__(self, rpm: float, backoff: List[float], probe: Optional[Callable[[], bool]] = None,
                 give_up_s: float = 1800.0, sleep=time.sleep, clock=time.time,
                 on_saturation: Optional[Callable[[Optional[Dict]], None]] = None,
                 saturation_minutes: int = SATURATION_MINUTES):
        self.interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
        self.backoff = list(backoff)
        self.probe = probe
        self.give_up_s = give_up_s
        self.sleep, self.clock = sleep, clock
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.next_at = 0.0                 # rate limiter
        self.pause_until = 0.0             # shared 429 pause
        self.in_flight: Dict[str, int] = {}          # worker -> row_id
        self.recent_conn_errors: Dict[str, float] = {}   # worker -> time of last connection error
        self.last_success_at = 0.0
        self.outage: Optional[Dict] = None           # the current outage record, while down
        self.gave_up = False
        self.probing = False
        # records for the manifest
        self.rate_limit_events: List[Dict] = []
        self.outages: List[Dict] = []
        # 429 escalation and saturation
        self.started_at = self.clock()
        self.hits_429: List[float] = []                  # clock times of every 429
        self.hits_per_minute: Dict[int, int] = {}        # minute index -> 429s in it
        self.successes_per_minute: Dict[int, int] = {}   # minute index -> successful calls in it
        self.escalation_max_multiplier = 1.0
        self.minutes_over: Dict[int, int] = {10: 0, 20: 0}     # whole minutes over each escalation line
        self.minutes_429s_outnumber_ok = 0                       # whole minutes where 429s beat successes
        self.on_saturation = on_saturation
        self.saturation_minutes = saturation_minutes
        self.saturated: Optional[Dict] = None            # the open saturation record while saturated
        self.saturation_events: List[Dict] = []
        self.last_saturation_minute_checked: Optional[int] = None
        # pause accounting: merged pause intervals and success times
        self.pauses: List[List[float]] = []
        self.success_times: List[float] = []

    # ---- before / after a call -------------------------------------------------
    def before_call(self, worker: str, row_id) -> None:
        """Wait for the rate limiter and any shared 429 pause; refuse (NetworkOutage) while
        the network is down or after the gate gave up."""
        with self.cond:
            if self.gave_up or self.outage is not None:
                raise NetworkOutage("network outage in progress; row goes back to the queue")
            now = self.clock()
            start = max(now, self.next_at, self.pause_until)
            if self.interval:
                self.next_at = start + self.interval
            self.in_flight[worker] = row_id
        if start > now:
            self.sleep(start - now)

    def after_call(self, worker: str) -> None:
        with self.lock:
            self.in_flight.pop(worker, None)
            self.recent_conn_errors.pop(worker, None)
            self.last_success_at = self.clock()
            self.success_times.append(self.last_success_at)
            minute = int(self.last_success_at // 60)
            self.successes_per_minute[minute] = self.successes_per_minute.get(minute, 0) + 1
            self._check_saturation(self.last_success_at)

    # ---- 429 escalation and saturation ------------------------------------------
    def _note_429(self, now: float) -> float:
        """Record a 429 at `now`; return the pause multiplier for it (1, 2 or 4)."""
        self.hits_429.append(now)
        minute = int(now // 60)
        self.hits_per_minute[minute] = self.hits_per_minute.get(minute, 0) + 1
        last_minute = sum(1 for t in self.hits_429 if now - t <= 60.0)
        mult = 1.0
        for threshold, m in ESCALATION_TIERS:
            if last_minute > threshold:
                mult = m
                break
        self.escalation_max_multiplier = max(self.escalation_max_multiplier, mult)
        self._check_saturation(now)
        return mult

    def _check_saturation(self, now: float) -> None:
        """Saturated: 429s outnumber successful calls in each of the last
        `saturation_minutes` whole minutes. Cleared: that many consecutive whole minutes
        with 429s at or below the successes. Checked once per new minute; the callback
        gets the record on entry and None on exit (run.py writes the manifest)."""
        minute = int(now // 60)
        if self.last_saturation_minute_checked == minute:
            return
        self.last_saturation_minute_checked = minute
        prev = minute - 1
        if prev >= int(self.started_at // 60):                    # the minute just completed
            for threshold in self.minutes_over:
                if self.hits_per_minute.get(prev, 0) > threshold:
                    self.minutes_over[threshold] += 1
            if self.hits_per_minute.get(prev, 0) > self.successes_per_minute.get(prev, 0):
                self.minutes_429s_outnumber_ok += 1
        minutes = [minute - self.saturation_minutes + k for k in range(self.saturation_minutes)]   # oldest first, whole minutes only
        window = [(self.hits_per_minute.get(mi, 0), self.successes_per_minute.get(mi, 0)) for mi in minutes]
        first_minute = int(self.started_at // 60)
        if minute - self.saturation_minutes < first_minute:
            return                                                # not enough whole minutes yet
        if self.saturated is None and all(h > ok for h, ok in window):
            self.saturated = {"started_at": now_iso(), "ended_at": None,
                              "counts_per_minute": [{"429s": h, "ok": ok} for h, ok in window],
                              "rule": "429s outnumber successful calls in each of %d consecutive minutes" % self.saturation_minutes,
                              "minutes": self.saturation_minutes}
            self.saturation_events.append(self.saturated)
            if self.on_saturation:
                self.on_saturation(self.saturated)
        elif self.saturated is not None and all(h <= ok for h, ok in window):
            self.saturated["ended_at"] = now_iso()
            self.saturated["counts_per_minute_at_clear"] = [{"429s": h, "ok": ok} for h, ok in window]
            self.saturated = None
            if self.on_saturation:
                self.on_saturation(None)

    def _add_pause(self, start: float, end: float) -> None:
        if self.pauses and start <= self.pauses[-1][1]:
            self.pauses[-1][1] = max(self.pauses[-1][1], end)
        else:
            self.pauses.append([start, end])

    def seconds_paused(self) -> float:
        with self.lock:
            now = self.clock()
            return round(sum(min(b, now) - a for a, b in self.pauses if a < now), 1)

    def pace_outside_pauses(self) -> Optional[float]:
        """Rows (successful calls) per minute over the time not spent paused on 429s."""
        with self.lock:
            now = self.clock()
            paused = sum(min(b, now) - a for a, b in self.pauses if a < now)
            active_min = (now - self.started_at - paused) / 60.0
            if active_min <= 0:
                return None
            def in_pause(t):
                return any(a <= t <= b for a, b in self.pauses)
            n = sum(1 for t in self.success_times if not in_pause(t))
            return round(n / active_min, 2)

    def on_error(self, worker: str, exc: Exception, transport_try: int) -> Optional[Dict]:
        """Called by the adapter wrapper when a call raised. Returns the 429 record when the
        error was a rate limit (and sets the shared pause). Raises NetworkOutage when this
        error completes an outage (every in-flight worker failed on the connection)."""
        with self.cond:
            self.in_flight.pop(worker, None)
            if is_rate_limit(exc):
                info = rate_limit_info(exc)
                now = self.clock()
                step = self.backoff[min(max(transport_try - 1, 0), len(self.backoff) - 1)] if self.backoff else 0.0
                ra = info.get("retry_after")
                mult = self._note_429(now)
                if ra is None:
                    base, source = step, "no Retry-After: backoff step"
                else:
                    base, source = min(max(float(ra), RETRY_AFTER_FLOOR_S), RETRY_AFTER_CEILING_S), "Retry-After"
                wait = min(base * mult, RETRY_AFTER_CEILING_S)
                info.update(wait_s=wait, wait_source=source, multiplier=mult,
                            hits_last_minute=sum(1 for t in self.hits_429 if now - t <= 60.0),
                            at=now_iso(), worker=worker)
                self.pause_until = max(self.pause_until, now + wait)
                self._add_pause(now, self.pause_until)
                self.rate_limit_events.append(info)
                return info
            if is_connection_error(exc):
                now = self.clock()
                self.recent_conn_errors[worker] = now
                recent = {w for w, t in self.recent_conn_errors.items() if now - t <= OUTAGE_WINDOW_S}
                others_in_flight = set(self.in_flight)      # workers still waiting on a call
                # outage = nobody with a call in flight has succeeded lately: every worker that
                # is or was in flight inside the window failed on the connection
                no_recent_success = now - self.last_success_at > OUTAGE_WINDOW_S
                if recent and no_recent_success and not (others_in_flight - recent) and self.outage is None:
                    self.outage = {"started_at": now_iso(), "ended_at": None, "rows_in_flight": sorted(
                        set(self.in_flight.values()) | {self._row_of(w) for w in recent if self._row_of(w) is not None}),
                        "probes": 0, "workers_failed": sorted(recent)}
                    self.rows_in_flight_at_start = set(self.outage["rows_in_flight"])
                    self.cond.notify_all()
                if self.outage is not None:
                    raise NetworkOutage("connection errors on every in-flight worker inside %.0f s" % OUTAGE_WINDOW_S)
            return None

    def _row_of(self, worker):
        return self.in_flight.get(worker)

    def note_row_in_flight(self, row_id) -> None:
        """Rows requeued because of an outage are added to that outage's record."""
        with self.lock:
            if self.outage is not None and row_id not in self.outage["rows_in_flight"]:
                self.outage["rows_in_flight"].append(row_id)
                self.outage["rows_in_flight"].sort()

    # ---- outage wait -----------------------------------------------------------
    def wait_for_network(self) -> bool:
        """Block until the network is back (True) or the gate gave up (False). The first
        worker to arrive runs the probe loop; the others wait on the condition."""
        with self.cond:
            if self.outage is None:
                return not self.gave_up
            if self.probing:
                while self.outage is not None and not self.gave_up:
                    self.cond.wait(1.0)
                return not self.gave_up
            self.probing = True
            outage = self.outage
            started = self.clock()
        try:
            while True:
                with self.cond:
                    outage["probes"] += 1
                up = self.probe() if self.probe else True
                if up:
                    with self.cond:
                        outage["ended_at"] = now_iso()
                        outage["down_s"] = round(self.clock() - started, 1)
                        self.outages.append(outage)
                        self.outage = None
                        self.recent_conn_errors.clear()
                        self.cond.notify_all()
                    return True
                if self.clock() - started >= self.give_up_s:
                    with self.cond:
                        outage["ended_at"] = None
                        outage["down_s"] = round(self.clock() - started, 1)
                        outage["gave_up"] = True
                        self.outages.append(outage)
                        self.gave_up = True
                        self.cond.notify_all()
                    return False
                self.sleep(PROBE_EVERY_S)
        finally:
            with self.cond:
                self.probing = False

    def summary(self) -> Dict:
        peak = max(self.hits_per_minute.values()) if self.hits_per_minute else 0
        return {"rate_limit_429s": len(self.rate_limit_events),
                "retry_after_values_seen": [e.get("retry_after") for e in self.rate_limit_events],
                "rate_limit_events": self.rate_limit_events,
                "seconds_paused_total": self.seconds_paused(),
                "rows_per_min_outside_pauses": self.pace_outside_pauses(),
                "pause_rule": {"floor_s": RETRY_AFTER_FLOOR_S, "ceiling_s": RETRY_AFTER_CEILING_S,
                               "no_retry_after_step_s": self.backoff[0] if self.backoff else 0.0,
                               "escalation": {"over_10_per_min": "x2", "over_20_per_min": "x4"},
                               "saturation": "429s outnumber successful calls in each of %d consecutive minutes" % self.saturation_minutes},
                "rate_limit_peak_per_minute": peak,
                "rate_limit_minutes_over": {str(k): v for k, v in self.minutes_over.items()},
                "minutes_429s_outnumber_ok": self.minutes_429s_outnumber_ok,
                "rate_limit_escalation_max_multiplier": self.escalation_max_multiplier,
                "saturation_events": self.saturation_events,
                "host_saturated_now": self.saturated is not None,
                "network_outages": self.outages, "paused_on_network": self.gave_up}


class GatedAdapter:
    """Wraps an adapter so every call passes the gate. Attribute reads fall through to the
    adapter (name, price table, expected served model, and so on)."""

    def __init__(self, adapter, gate: RunGate):
        self._adapter, self._gate = adapter, gate

    def complete(self, prompt, meta=None):
        meta = meta or {}
        worker = threading.current_thread().name
        t_wait = time.time()
        self._gate.before_call(worker, meta.get("row_id"))
        t_call = time.time()
        try:
            reply = self._adapter.complete(prompt, meta)
            reply.raw = dict(reply.raw or {})
            reply.raw["latency_s_call"] = round(time.time() - t_call, 3)   # the call alone, gate wait excluded
            reply.raw["gate_wait_s"] = round(t_call - t_wait, 3)
        except NetworkOutage:
            raise
        except Exception as exc:
            info = self._gate.on_error(worker, exc, meta.get("transport_try", 1))   # may raise NetworkOutage
            if info is not None:
                try:
                    exc.rate_limit = info      # forecast.py logs it on the attempt record
                except Exception:
                    pass
            raise
        self._gate.after_call(worker)
        return reply

    def __getattr__(self, name):
        return getattr(self._adapter, name)
