"""Out-of-process responder worker for the continual driver.

The adaptation method (an ``event_responders`` responder -- bo / safegp / grid /
oracle / noadapt) runs in a SEPARATE process so the physics loop never blocks
while the optimizer fits a GP or runs acquisition. The sim (main) process posts
propose-requests and outcome-updates over a queue and polls a result queue WITHOUT
blocking; the longer ``propose()`` takes in wall-clock, the more sim steps elapse
before the new gait is applied -- exactly the real-robot latency the continual
experiments want to measure.

Nothing that touches PyBullet ever crosses the process boundary: the responder is
rebuilt inside the worker from a picklable :class:`ResponderSpec`, and only small
parameter arrays / scalars travel on the queues.

Protocol (single FIFO request queue, so an ``update`` posted before the next
event's ``propose`` is always folded into memory first):

    sim -> worker :  ("update", cand, V, fell)          # fire-and-forget
                     ("propose", event_id)              # request a gait
                     ("stop",)                          # shut down
    worker -> sim :  ("cand", event_id, cand, mode)     # a proposed gait

The ``event_id`` tags each request; the sim applies a returned gait only if the
event is still current, discarding stale proposals for events that already ended.

Nesting note: a daemonic ``multiprocessing.Pool`` worker cannot spawn children, so
if this is constructed inside one we transparently fall back to an in-thread
responder (GIL-bound, but correct). Prefer a non-daemonic
``concurrent.futures.ProcessPoolExecutor`` for outer parallelism so the true
separate-process path is used.
"""

import multiprocessing as mp
import queue as _queue
import sys
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ResponderSpec:
    """Everything needed to (re)build a responder and its initial memory seed,
    fully picklable so it can cross a process boundary to the worker."""
    name: str
    incumbent: np.ndarray
    box: tuple
    free_dims: list
    oracle_target: Optional[np.ndarray]
    seed: int
    safegp_kwargs: Optional[dict] = None
    seed_fall: bool = False        # pre-seed the incumbent as a known fall (bo/safegp)
    v_fall: float = -2.0


def _build(spec):
    """Construct the responder and apply the incumbent-fall seed (bo/safegp)."""
    from methods import event_responders as er
    r = er.make_responder(spec.name, spec.incumbent, spec.box, spec.free_dims,
                          spec.oracle_target, spec.seed,
                          safegp_kwargs=spec.safegp_kwargs)
    if spec.seed_fall and spec.name in ("bo", "safegp"):
        r.update(np.asarray(spec.incumbent, float), float(spec.v_fall), True)
    return r


def _serve(responder, incumbent, req_q, res_q):
    """Consume request messages until ``("stop",)``. On any propose failure fall
    back to the incumbent so the sim never waits forever for a gait."""
    inc = np.asarray(incumbent, float)
    while True:
        msg = req_q.get()
        cmd = msg[0]
        if cmd == "stop":
            break
        if cmd == "update":
            _, cand, V, fell = msg
            try:
                responder.update(np.asarray(cand, float), float(V), bool(fell))
            except Exception as e:                        # never kill the worker
                print(f"[responder_worker] update failed: {e}", flush=True)
        elif cmd == "propose":
            _, event_id = msg
            try:
                cand, mode = responder.propose()
                res_q.put(("cand", event_id, np.asarray(cand, float), mode))
            except Exception as e:
                print(f"[responder_worker] propose failed ({e}); "
                      f"returning incumbent", flush=True)
                res_q.put(("cand", event_id, inc.copy(), "fallback"))


def _worker_main(spec, req_q, res_q):
    """Child-process entry point: build the responder, then serve requests."""
    try:                              # keep each GP fit single-core so many bouts
        import torch                  # can run concurrently without fighting over
        torch.set_num_threads(1)      # cores -- and so compute_latency is consistent
    except Exception:                 # regardless of how many run in parallel
        pass
    responder = _build(spec)
    _serve(responder, spec.incumbent, req_q, res_q)
    res_q.cancel_join_thread()        # don't block exit on an unconsumed cand


class ResponderWorker:
    """Owns the responder's execution context (a spawned process, or a thread when
    nested in a daemonic pool) and exposes a non-blocking interface to the sim."""

    def __init__(self, spec):
        self.spec = spec
        self._proc = None
        self._thread = None
        self._responder = None
        self.thread_mode = mp.current_process().daemon   # can't spawn children here

        if self.thread_mode:
            print("[responder_worker] constructed inside a daemon process; "
                  "falling back to an in-thread responder (GIL-bound). Use a "
                  "ProcessPoolExecutor for outer parallelism to get true "
                  "separate-process async.", file=sys.stderr, flush=True)
            self._req = _queue.Queue()
            self._res = _queue.Queue()
            self._responder = _build(spec)
            self._thread = threading.Thread(
                target=_serve,
                args=(self._responder, spec.incumbent, self._req, self._res),
                daemon=True)
            self._thread.start()
        else:
            ctx = mp.get_context("spawn")
            self._req = ctx.Queue()
            self._res = ctx.Queue()
            self._proc = ctx.Process(target=_worker_main,
                                     args=(spec, self._req, self._res),
                                     daemon=True)
            self._proc.start()

    # ── sim-side interface (all non-blocking) ────────────────────────────────
    def request_propose(self, event_id):
        self._req.put(("propose", int(event_id)))

    def push_update(self, cand, V, fell):
        self._req.put(("update", np.asarray(cand, float), float(V), bool(fell)))

    def poll(self):
        """Return the oldest ready proposal as ``(event_id, cand, mode)``, or
        ``None`` if nothing is ready yet. Non-blocking."""
        try:
            msg = self._res.get_nowait()
        except _queue.Empty:
            return None
        return int(msg[1]), np.asarray(msg[2], float), msg[3]

    def close(self, timeout=5.0):
        """Post ``stop`` and reap the worker; terminate it if it doesn't exit."""
        try:
            self._req.put(("stop",))
        except Exception:
            pass
        if self._proc is not None:
            self._proc.join(timeout)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(1.0)
        elif self._thread is not None:
            self._thread.join(timeout)
