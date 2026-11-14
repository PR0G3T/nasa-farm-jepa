"""Hardware layer — topology, discovery, memory, pools, streams, graphs,
probe cache, scheduler ladder, telemetry. Everything runs on CPU-only
hosts; accelerator paths skip cleanly."""

import threading
import time

import pytest
import torch

from nasa_farm.constants import DTYPE
from nasa_farm.hardware import capabilities as capabilities
from nasa_farm.hardware import graphs, pool, probe
from nasa_farm.hardware import memory as memory
from nasa_farm.hardware import scheduler as schedule
from nasa_farm.hardware import streams as streams
from nasa_farm.hardware import topology as topology
from nasa_farm.hardware.telemetry import Monitor


class TestTopo:
    def test_physical_cores_positive(self):
        assert topology.physical_cores() >= 1

    def test_topo_structure(self):
        t = topology.topology()
        assert t.cores == len(t.ids) >= 1
        assert sorted(c for g in t.numa for c in g) == sorted(t.ids)

    def test_worker_cpus_round_robin(self):
        t = topology.topology()
        got = {topology.worker_processors(i, t.cores) for i in range(t.cores)}
        assert len(got) == min(t.cores, len(set(got)))
        assert all(c in t.ids for (c,) in got)

    def test_pin_returns_bool(self):
        assert isinstance(topology.pin(topology.topology().ids[:1]), bool)
        assert topology.pin(()) is False


class TestCaps:
    def test_lanes_cpu_tail(self):
        lanes = capabilities.lanes(DTYPE)
        assert lanes[-1].type == "cpu"
        assert all(d.type != "cpu" for d in lanes[:-1])

    def test_supports_mps_f64(self):
        assert not capabilities.supports(torch.float64, torch.device("mps"))
        assert capabilities.supports(torch.float64, torch.device("cpu"))

    def test_device_mem_cpu(self):
        free, _ = capabilities.device_memory(torch.device("cpu"))
        assert free > 0

    def test_name(self):
        assert capabilities.name(torch.device("cpu")) == "cpu"

    def test_cpu_workers_bounds(self):
        assert capabilities.processor_workers() >= 1
        assert capabilities.processor_workers(rows=10) == 1   # shard floor


class TestMem:
    def test_avail_budget(self):
        d = torch.device("cpu")
        assert 0 < memory.available(d, 0.5) < memory.available(d, 1.0) * 2

    def test_reserve_pool_cpu_none(self):
        assert memory.reserve_pool(torch.device("cpu")) is None

    def test_empty_noop_cpu(self):
        memory.empty(torch.device("cpu"))


class TestPool:
    def test_backend_known(self):
        assert pool.backend() in ("threads", "interpreters")

    def test_make_requires_two(self):
        with pytest.raises(ValueError):
            pool.make(1)

    def test_map_order(self):
        if pool.backend() == "interpreters":
            pytest.skip("interpreter pool pickles args; covered via evaluate")
        ex = pool.make(2)
        try:
            assert list(ex.map(lambda x: x * 2, range(6))) == \
                [0, 2, 4, 6, 8, 10]
        finally:
            ex.shutdown(wait=True)


class TestPipe:
    def test_cpu_streams_none(self):
        assert streams.streams(torch.device("cpu")) == (None, None)

    def test_stage_cpu_passthrough(self):
        st = streams.Stage(torch.device("cpu"), None, None)
        x = torch.ones(4, 2)
        assert torch.equal(st.in_(x), x)


class TestGraphs:
    def test_failed_capture_marks_none(self, monkeypatch):
        calls = {"n": 0}

        def boom(*a, **keywords):
            calls["n"] += 1
            raise RuntimeError("no graph device")

        monkeypatch.setattr(graphs, "capture", boom)
        c = graphs.Cache.__new__(graphs.Cache)
        c.device, c.pool, c.max, c._d = torch.device("cpu"), None, 8, {}
        inp = (torch.ones(2),)
        assert c.get((2,), lambda *t: t[0], inp) is None
        assert c.get((2,), lambda *t: t[0], inp) is None
        assert calls["n"] == 1          # retried never for the same shape

    def test_replayer_copies_and_replays(self):
        class G:
            def __init__(self): self.n = 0
            def replay(self): self.n += 1
        g = G()
        static = (torch.zeros(2),)
        rep = graphs.Replayer(g, static, torch.tensor([9.0, 9.0]))
        output = rep(torch.tensor([1.0, 2.0]))
        assert g.n == 1 and torch.equal(static[0], torch.tensor([1.0, 2.0]))
        assert float(output[0]) == 9.0


class TestProbe:
    def test_worth_threshold(self):
        assert not probe.worth(1, 1, 1)
        assert probe.worth(1000, 4, 365)

    def test_signature_stable(self, physics_model):
        from nasa_farm.plan import enumerate_rotations
        from nasa_farm.forcing import synthetic_year
        from nasa_farm.forcing import climatology_ensemble
        plans = enumerate_rotations([0], 1)
        ensemble = climatology_ensemble([synthetic_year(42.0, seed=1)], 1)
        a = probe.signature(physics_model, plans, ensemble, 1)
        b = probe.signature(physics_model, plans, ensemble, 1)
        assert a == b and hash(a) == hash(b)

    def test_cache_roundtrip(self, monkeypatch, tmp_path):
        monkeypatch.setenv("NASA_FARM_CACHE", str(tmp_path))
        signal = (1, 2, 3)
        assert probe.lookup(signal, "cpu") is None
        probe.store(signal, "cpu", 42.0, True)
        assert probe.lookup(signal, "cpu") == (42.0, True)

    def test_measure_cpu(self):
        lane = schedule.Lane(name="cpu", device=torch.device("cpu"))
        lane.function = lambda l, s, blk: blk.float().sum()
        rate, g = probe.measure(lane, torch.ones(8, 2))
        assert rate > 0 and g is False


class TestThermo:
    def test_sampler_integration(self):
        watts = iter([100.0] * 10)

        def fake():
            return (next(watts, 0.0), 50.0, 0)

        m = Monitor(torch.device("cuda", 0), hz=200.0, sampler=fake)
        with m:
            time.sleep(0.05)
        st = m.stats()
        assert st["samples"] >= 1 and st["joules"] >= 0.0
        assert st["temp_max_c"] == 50.0
        assert not m.throttled.is_set()

    def test_throttle_flag(self):
        m = Monitor(torch.device("cuda", 0), hz=200.0,
                    sampler=lambda: (10.0, 90.0, 0x1))
        with m:
            time.sleep(0.05)
        assert m.throttled.is_set()
        assert m.stats()["throttle_events"] >= 1

    def test_no_sampler_no_thread(self):
        m = Monitor(torch.device("cpu"), sampler=None)
        with m:
            pass
        assert not m.available and m.stats()["samples"] == 0


class TestSched:
    def _lane(self, name="cpu", function=None, block=4, workers=1):
        l = schedule.Lane(name=name, device=torch.device("cpu"),
                       workers=workers, block_rows=block,
                       cap_rows=block)
        l.function = function or (lambda lane, s, blk: blk.to(DTYPE).unsqueeze(1))
        return l

    def test_run_assembles_in_order(self):
        """Two lanes pull from one cursor; slots assemble by row index —
        output identical to a single-lane run whatever the interleave."""
        plans = torch.arange(12).reshape(6, 2)
        output = torch.zeros(6, 1, 2, dtype=DTYPE)
        l1, l2 = self._lane("a", block=3), self._lane("b", block=2)
        schedule.run([l1, l2], plans, output, 1)
        exp = plans.to(DTYPE).unsqueeze(1)
        assert torch.equal(output, exp)

    def test_oom_lane_dies_survivor_absorbs(self):
        plans = torch.arange(8).reshape(4, 2)
        output = torch.zeros(4, 1, 2, dtype=DTYPE)
        bad = self._lane("bad", block=1,
                         function=lambda l, s, b: (_ for _ in ()).throw(
                             MemoryError("oom")))
        good = self._lane("good", block=2)
        schedule.run([bad, good], plans, output, 1)
        assert not bad.alive and good.alive
        assert torch.equal(output, plans.to(DTYPE).unsqueeze(1))

    def test_all_lanes_dead_raises(self):
        plans = torch.arange(8).reshape(4, 2)
        output = torch.zeros(4, 1, 2, dtype=DTYPE)
        dead = self._lane("d", block=1,
                          function=lambda l, s, b: (_ for _ in ()).throw(
                              MemoryError("oom")))
        with pytest.raises(MemoryError):
            schedule.run([dead], plans, output, 1)

    def test_non_oom_aborts(self):
        plans = torch.arange(8).reshape(4, 2)
        output = torch.zeros(4, 1, 2, dtype=DTYPE)
        bad = self._lane("x", function=lambda l, s, b: (_ for _ in ()).throw(
            ValueError("bug")))
        with pytest.raises(ValueError, match="bug"):
            schedule.run([bad], plans, output, 1)

    def test_is_oom_patterns(self):
        assert schedule.out_of_memory(MemoryError("x"))
        assert schedule.out_of_memory(RuntimeError("CUDA out of memory"))
        assert not schedule.out_of_memory(RuntimeError("illegal instruction"))

    def test_oom_rows_returned_to_cursor(self):
        """A lane that OOMs mid-sweep hands its rows back; total coverage
        is still exactly once per row."""
        plans = torch.arange(10).reshape(5, 2)
        output = torch.zeros(5, 1, 2, dtype=DTYPE)
        state = {"n": 0}

        def flaky(l, s, b):
            state["n"] += 1
            if state["n"] == 1:
                raise MemoryError("transient")
            return b.to(DTYPE).unsqueeze(1)

        l = self._lane("flaky", function=flaky, block=4)
        schedule.run([l], plans, output, 1)
        assert torch.equal(output, plans.to(DTYPE).unsqueeze(1))
        assert state["n"] >= 2
