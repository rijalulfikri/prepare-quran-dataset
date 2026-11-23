"""run_zipformer_training refuses a box too small for the dataloader workers (OOM on 2026-11-23)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vastai"))
import run_zipformer_training as rzt  # noqa: E402

GiB = 2**30


def test_the_8_core_30_gb_box_that_was_oom_killed_is_refused():
    problems = rzt.resource_problems(cpus=8, mem_bytes=31401705472, num_workers=24)
    assert len(problems) == 2
    assert "8 CPU" in problems[0] and "num_workers 24" in problems[0]
    assert "RAM" in problems[1]


def test_the_48_core_108_gb_box_that_trains_is_accepted():
    assert rzt.resource_problems(cpus=48, mem_bytes=111380791296, num_workers=24) == []


def test_ram_floor_is_absolute():
    assert rzt.resource_problems(cpus=32, mem_bytes=rzt.MIN_RAM_GB * GiB - 1, num_workers=24)
    assert rzt.resource_problems(cpus=32, mem_bytes=rzt.MIN_RAM_GB * GiB, num_workers=24) == []


def test_box_memory_takes_the_cgroup_limit_over_the_host_total():
    # nproc, cpu.max, memory.max, MemTotal; container limit "31401705472", host MemTotal much larger
    assert rzt.parse_box_resources("8\nmax 100000\n31401705472\n1056620544000\n") == (8, 31401705472)
    assert rzt.parse_box_resources("48\nmax 100000\nmax\n116000000000\n") == (48, 116000000000)


def test_box_cpus_take_the_cgroup_quota_over_nproc():
    # 2026-11-23 offer 40889255: nproc 192 (host), cpu.max quota 36.9 CPUs
    assert rzt.parse_box_resources("192\n3686399 100000\n322238000000\n500000000000\n")[0] == 36


def test_62_gib_box_listed_as_64_gb_is_accepted():
    assert rzt.resource_problems(cpus=32, mem_bytes=62 * GiB, num_workers=24) == []


def test_num_workers_read_from_the_config_text():
    assert rzt.config_num_workers("max_duration: 600\nnum_workers: 24                # fbank ...\n") == 24
    assert rzt.config_num_workers("max_duration: 600\n") is None


def test_repo_config_is_what_the_guard_checks_against():
    text = (Path(__file__).resolve().parents[1] / "configs" / "train_config.yml").read_text()
    assert rzt.config_num_workers(text) == 24


NOW = 1_795_400_000.0
OK_OFFER = {"end_date": NOW + 38 * 86400, "cpu_cores_effective": 38.4, "cpu_ram": 64305, "reliability2": 0.991}


def test_offer_with_a_long_rental_and_enough_cpu_ram_passes():
    assert rzt.offer_problems(OK_OFFER, num_workers=24, now=NOW) == []


def test_offer_whose_rental_ends_within_days_is_refused():
    # 2026-11-23: the California box was stopped by Vast.ai at its end_date mid-run
    problems = rzt.offer_problems({**OK_OFFER, "end_date": NOW + 0.2 * 86400}, num_workers=24, now=NOW)
    assert len(problems) == 1 and "sewa" in problems[0]


def test_offer_with_too_few_cpus_or_ram_is_refused():
    small = {**OK_OFFER, "cpu_cores_effective": 8.0, "cpu_ram": 31195}
    assert len(rzt.offer_problems(small, num_workers=24, now=NOW)) == 2


def test_offer_without_end_date_has_no_rental_limit():
    assert rzt.offer_problems({**OK_OFFER, "end_date": None}, num_workers=24, now=NOW) == []


class _Args:
    config = None
    offer_id = 1
    skip_resource_check = False


def test_new_instance_below_the_rules_is_destroyed_before_setup():
    destroyed = []
    short = {**OK_OFFER, "id": 7, "end_date": 1.0}  # rental already over
    try:
        rzt.check_new_instance(_Args(), 7, lambda: [short], destroyed.append)
    except SystemExit as exc:
        assert exc.code == 1
    assert destroyed == [7]


def test_new_instance_within_the_rules_is_kept():
    import time

    destroyed = []
    good = {**OK_OFFER, "id": 7, "end_date": time.time() + 30 * 86400}
    rzt.check_new_instance(_Args(), 7, lambda: [good], destroyed.append)
    assert destroyed == []


def test_offer_with_a_slow_cpu_is_refused():
    # 2026-11-23: EPYC 7K62 listed at 2.6 GHz trained ~1.7x slower (main process pinned at 100 % of one core)
    problems = rzt.offer_problems({**OK_OFFER, "cpu_ghz": 2.6}, num_workers=24, now=NOW)
    assert len(problems) == 1 and "GHz" in problems[0]


def test_offer_with_a_fast_or_unlisted_cpu_passes():
    assert rzt.offer_problems({**OK_OFFER, "cpu_ghz": 5.88}, num_workers=24, now=NOW) == []
    assert rzt.offer_problems({**OK_OFFER, "cpu_ghz": None}, num_workers=24, now=NOW) == []
