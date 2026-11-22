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
    # container limit "31401705472", host MemTotal much larger
    assert rzt.parse_box_resources("8\n31401705472\n1056620544000\n") == (8, 31401705472)
    assert rzt.parse_box_resources("48\nmax\n116000000000\n") == (48, 116000000000)


def test_num_workers_read_from_the_config_text():
    assert rzt.config_num_workers("max_duration: 600\nnum_workers: 24                # fbank ...\n") == 24
    assert rzt.config_num_workers("max_duration: 600\n") is None


def test_repo_config_is_what_the_guard_checks_against():
    text = (Path(__file__).resolve().parents[1] / "configs" / "train_config.yml").read_text()
    assert rzt.config_num_workers(text) == 24
