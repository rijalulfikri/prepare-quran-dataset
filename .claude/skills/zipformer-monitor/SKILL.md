---
name: zipformer-monitor
description: Monitor a Zipformer training run on Vast.ai (started by zipformer/vastai/run_zipformer_training.py) — per-epoch dev PER reports, crash diagnosis, and verifying the final destroy and Drive results. Use when the user asks to monitor, watch, pantau, or check zipformer training.
---

# Zipformer training monitor

You are the **reporter and auditor** of a training run. The box runs its own safety net: `zipformer_watchdog.sh` (early stop) and `--destroy-when-done` (upload + destroy). Stopping, restarting and destroying belong to that watchdog and to the user; you observe, report, diagnose, and verify.

All status comes from one command, run from the repo root:

```bash
python zipformer/vastai/zipformer_status.py            # one-shot status
python zipformer/vastai/zipformer_status.py --drive    # + SHA-256 check of the newest epoch-*.pt on Drive
python zipformer/vastai/zipformer_status.py --wait     # blocks until a new epoch or a STATUS change, then prints status
```

It finds the instance through the Vast.ai API (pass `--host/--port` if the user gives them). Its first line is `STATUS: <STATE>`; `--help` and the module docstring describe the rest. Reply to the user in their language.

## Loop

1. Run `zipformer_status.py --drive` and send the first report.
2. Start `zipformer_status.py --wait` as a **background** command and end your turn. You are re-invoked when it exits.
3. On each notification: read its output, act on the STATE (table below), send a report, and start the next `--wait` — until the run reaches a terminal state.

A **report** is: the last few rows of the dev PER table with the rekor marked, `Epoch tanpa rekor x/N`, minutes per epoch, the earliest early-stop epoch, and credit. Flag `CREDIT TIDAK CUKUP` prominently. Compare with earlier epochs you reported: say plainly when the curve is still improving and when it is flattening, and correct your own earlier reading when a new epoch contradicts it.

## States

| STATE | Action |
|---|---|
| `PREPARING` | Report setup progress; keep waiting. |
| `TRAINING` | Report the new epoch; keep waiting. Run `--drive` once every few epochs. |
| `WATCHDOG_STOPPING`, `FINISHING`, `UPLOADING` | Report the stage; keep waiting with `--wait --poll 60`. |
| `STOPPED` | Diagnose: read `ssh … 'tail -c 8000 /root/train_zipformer.log'` and the watchdog log, find the root cause, report it with the cost of the idle instance ($/hour from the status). Ask the user before any restart or destroy. |
| `ENDED` | The run succeeded but the instance is still up (no `--destroy-when-done`, or destroy failed). Report and ask the user. |
| `NO_INSTANCE` | Terminal. Verify the results the status lists on Drive: `averaged-*.pt`, `test_results.json`, `export/`. Report the final dev PER, test results, and the total epochs. Stop the loop. |
| `SSH_FAILED` | Retry once with `--wait --poll 60`; if it persists, report it (the box may be restarting or gone). |

When the watchdog acted, the final report also states which epochs were averaged (the last 3 epochs up to the stop epoch) and that `num_epochs` on the box was lowered to that epoch.
