"""Prepare, validate, launch and supervise the approved Codexa foundational run (no AI polling)."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from codexa_workspace import foundation_training as job
from codexa_workspace import training_supervisor as supervisor
from codexa_workspace import training_notifications
from codexa_workspace.training_notifications import deliver as startup_deliver


def deliver(*args, **kwargs):
    """Late-bound: load the notifier as it is on disk at the terminal event (it is not hash-bound),
    so notification fixes reach a long-running supervisor; fall back to the startup copy on error."""
    import importlib
    try:
        fresh = importlib.reload(training_notifications).deliver
    except Exception as error:   # a broken on-disk edit must never cost the notification
        print(f"notifier reload failed ({type(error).__name__}: {error}); using startup copy", file=sys.stderr)
        fresh = startup_deliver
    return fresh(*args, **kwargs)

SESSION_ENVIRONMENT = ("WAYLAND_DISPLAY", "DISPLAY", "DBUS_SESSION_BUS_ADDRESS", "XDG_RUNTIME_DIR",
                       "XDG_SESSION_TYPE", "PATH", "HOME", "LANG", "CODEXA_TRAINING_ENVIRONMENT")


def write_launch(run_dir: Path, *, provider: str, resume: bool = False, agent_window: bool = True) -> dict:
    command = [sys.executable, "-m", "codexa_workspace.foundation_training", "--run-dir", str(run_dir)]
    if resume:
        command.append("--resume")
    launch = dict(run_id=run_dir.name, command=command, cwd=str(ROOT),
                  environment=dict(PYTHONPATH=str(ROOT / "src"), LLM_INTEGRATION_ROOT=str(ROOT),
                                   CODEXA_TRAINING_ENVIRONMENT=os.environ["CODEXA_TRAINING_ENVIRONMENT"],
                                   # the trainer's required visible viewer is the 120x30 Rich dashboard
                                   LLM_TRAINING_VIEWER_COMMAND=json.dumps(
                                       [sys.executable, str(ROOT / "scripts/training_dashboard.py"), "--attach"])),
                  wakeup=dict(provider=provider), deadline_seconds=job.WALL_CLOCK_SECONDS,
                  agent_window=agent_window,  # interactive codex/claude Kitty window at the terminal event
                  execution_class="production",  # trusted: only this launcher writes production
                  policy={}, written_utc=supervisor.now())
    supervisor.atomic_document(run_dir / "supervisor/launch.json", launch)
    return launch


def validate(run_dir: Path, *, run_tests: bool) -> dict:
    state = supervisor.RunState(run_dir)
    state.transition("VALIDATING")
    record = job.validate_launch(ROOT, run_dir, run_tests=run_tests)
    if record["passed"]:
        state.transition("READY", failure_reason=None, launch_gates=str(run_dir / "config/launch-gates.json"))
    else:
        state.transition("BLOCKED", failure_reason="launch gates failed: " + ", ".join(record["failed"]))
    return record


def launch(run_dir: Path, *, provider: str, run_tests: bool, resume: bool = False, agent_window: bool = True) -> int:
    if os.environ.get("CODEXA_WAKEUP_AGENT"):
        raise SystemExit("A wakeup analysis agent may not launch training.")
    record = validate(run_dir, run_tests=run_tests) if not resume else dict(passed=True)
    if not record["passed"]:
        print(json.dumps(dict(state="BLOCKED", failed=record["failed"],
            details=[gate for gate in record["gates"] if not gate["passed"]]), indent=2))
        return 2
    write_launch(run_dir, provider=provider, resume=resume, agent_window=agent_window)
    script = Path(__file__).resolve()
    supervisor.install_reconcile_unit(ROOT, sys.executable, script)
    environment = {key: os.environ[key] for key in SESSION_ENVIRONMENT if os.environ.get(key)}
    unit = f"codexa-train-{run_dir.name}" + (f"-resume-{int(time.time())}" if resume else "")
    command = supervisor.systemd_launch(run_dir, ROOT, unit=unit, python=sys.executable, script=script,
                                        environment=environment)
    supervisor.RunState(run_dir).event("systemd_launch", unit=unit)
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        trainer = supervisor.read_json(run_dir / "supervisor/state.json").get("trainer")
        if trainer and supervisor.process_start_ticks(trainer["pid"]) == trainer["start_ticks"]:
            break
        time.sleep(1)
    state = supervisor.read_json(run_dir / "supervisor/state.json")
    print(json.dumps(dict(unit=unit, unit_active=supervisor.unit_active(unit + ".service"),
                          state=state.get("state"), trainer=state.get("trainer"),
                          supervisor=state.get("supervisor"), systemd_run=command[:6]), indent=2))
    if state.get("trainer"):
        print(f"Monitor (read-only): {sys.executable} {ROOT / 'scripts/training_dashboard.py'} --run-dir {run_dir}")
    return 0 if state.get("trainer") else 3


RESUMABLE_STOPS = ("MEMORY_HEADROOM_GATE", "DISK_HEADROOM", "DISK_RESERVE_WATCHDOG")


def resume_gates(run_dir: Path) -> dict:
    """Live resource and identity gates for an exact resume; thresholds identical to launch."""
    gates = []

    def gate(name, check):
        try:
            passed, detail = check()
        except Exception as error:
            passed, detail = False, f"{type(error).__name__}: {error}"
        gates.append(dict(name=name, passed=bool(passed), detail=detail))

    gate("approval_and_code_identity", lambda: (bool(job.verify_job(ROOT, run_dir)), "job hash and code identity unchanged"))
    latest = supervisor.recovery_role(run_dir, "latest")
    gate("latest_checkpoint_sha256", lambda: (bool(latest and latest["verified"]), latest))
    free = shutil.disk_usage(run_dir).free
    peak = 7 * job.NATIVE_PAYLOAD_BYTES + job.GIB
    gate("disk_capacity", lambda: (free >= job.DISK_FLOOR_BYTES and free - peak >= job.RECOVERY_RESERVE_BYTES,
                                   dict(free_gib=round(free / job.GIB, 2), floor_gib=180)))
    gate("host_ram", lambda: (job.host_available_bytes() >= job.HOST_RAM_BYTES,
                              dict(available_gib=round(job.host_available_bytes() / job.GIB, 2), required_gib=16)))
    snapshot = job.gpu_snapshot()
    gate("gpu_vram", lambda: (snapshot.get("available") and snapshot["free_mib"] >= job.VRAM_FREE_MIB,
                              dict(free_mib=snapshot.get("free_mib"), required_free_mib=job.VRAM_FREE_MIB,
                                   compute_apps=snapshot.get("compute_apps"))))
    gate("visible_viewer_environment", lambda: (bool(shutil.which("kitty")) and bool(
        os.environ.get("WAYLAND_DISPLAY") or os.environ.get("DISPLAY")), "kitty + display"))
    budget = supervisor.read_json(run_dir / "metrics/budget.json")
    remaining = min(budget["deadline_utc"] - time.time(),
                    budget["remaining_seconds"] - (time.time() - budget["last_recorded_utc"]))
    gate("original_budget_remaining", lambda: (remaining > budget["reserve_seconds"] + 900,
                                               dict(remaining_seconds=round(remaining), deadline_utc=budget["deadline_utc"])))
    record = dict(checked_utc=supervisor.now(), passed=all(g["passed"] for g in gates),
                  failed=[g["name"] for g in gates if not g["passed"]], gates=gates)
    with (run_dir / "config/resume-gates-history.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    return record


def resume(run_dir: Path, *, provider: str, confirm: str, agent_window: bool = True) -> int:
    """Manual exact resume: verified latest checkpoint, original deadline carried forward, new terminal event.

    Allowed after an interruption, or after an EARLY_STOPPED caused by a resource guard
    (never after a quality guard). Every launch resource threshold is re-checked unchanged.
    """
    if confirm != run_dir.name:
        raise SystemExit("--confirm-run must repeat the exact run id.")
    state = supervisor.read_json(run_dir / "supervisor/state.json") or {}
    reason = (supervisor.read_json(run_dir / "supervisor/completion.json") or {}).get("reason")
    if not (state.get("state") == "INTERRUPTED" or (state.get("state") == "EARLY_STOPPED" and reason in RESUMABLE_STOPS)):
        raise SystemExit(f"Not resumable: state {state.get('state')} ({reason}). Only interruptions and resource-guard stops resume.")
    record = resume_gates(run_dir)
    if not record["passed"]:
        print(json.dumps(dict(state="RESUME_BLOCKED", failed=record["failed"],
                              details=[g for g in record["gates"] if not g["passed"]]), indent=2))
        return 2
    return relaunch(run_dir, provider=provider, agent_window=agent_window, state=state, reason=reason)


def relaunch(run_dir: Path, *, provider: str, agent_window: bool, state: dict, reason: str | None,
             event: str = "manual_resume", **details) -> int:
    """Archive the finished segment's terminal records, then resume exactly from the latest checkpoint."""
    segment = run_dir / "supervisor/segments" / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    segment.mkdir(parents=True)
    for name in ("state.json", "completion.json", "notification-state.json", "desktop-notification.json",
                 "ai-wakeup.json", "agent-window.json", "ai-handoff.md", "launch.json", "notification-mock.jsonl"):
        if (run_dir / "supervisor" / name).exists():
            shutil.move(str(run_dir / "supervisor" / name), segment / name)
    if (run_dir / "final-report").exists():
        shutil.move(str(run_dir / "final-report"), segment / "final-report")
    shutil.copy2(run_dir / "metrics/report.json", segment / "trainer-report.json")
    latest = supervisor.recovery_role(run_dir, "latest")
    run_state = supervisor.RunState(run_dir)
    run_state.event(event, segment=str(segment), checkpoint_step=latest["optimizer_step"],
                    previous_state=state.get("state"), previous_reason=reason, **details)
    run_state.update(state="READY", resumed_from=latest, previous_segment=str(segment))
    return launch(run_dir, provider=provider, run_tests=False, resume=True, agent_window=agent_window)


MAX_EXTENSION_SECONDS = 7200


def extend(run_dir: Path, *, provider: str, confirm: str, seconds: int, statement: str,
           agent_window: bool = True) -> int:
    """User-granted budget extension: a run that hit its wall-clock deadline before the frozen
    step cap resumes exactly (same job, checkpoint, optimizer, LR schedule, data order) to finish
    it. Only the deadline moves; the grant is recorded and the previous budget is archived.
    """
    if os.environ.get("CODEXA_WAKEUP_AGENT"):
        raise SystemExit("A wakeup analysis agent may not extend or launch training.")
    if confirm != run_dir.name:
        raise SystemExit("--confirm-run must repeat the exact run id.")
    if not statement or not statement.strip():
        raise SystemExit("--statement must record the user's grant for the extra time.")
    if not 0 < seconds <= MAX_EXTENSION_SECONDS:
        raise SystemExit(f"--extend-seconds must be between 1 and {MAX_EXTENSION_SECONDS}.")
    state = supervisor.read_json(run_dir / "supervisor/state.json") or {}
    completion = supervisor.read_json(run_dir / "supervisor/completion.json") or {}
    steps, cap = completion.get("optimizer_steps"), completion.get("max_steps")
    if not (state.get("state") == "COMPLETED" and completion.get("reason") == "TRAINING_DEADLINE"
            and completion.get("final_checkpoint_verification") == "verified" and steps is not None and steps < cap):
        raise SystemExit(f"Not extendable: state {state.get('state')} ({completion.get('reason')}), "
                         f"{steps}/{cap} updates. Only a verified deadline stop before the step cap can be extended.")
    budget_path = run_dir / "metrics/budget.json"
    before = supervisor.read_json(budget_path)
    now, monotonic = time.time(), time.monotonic()
    after = dict(before, deadline_utc=now + seconds, remaining_seconds=float(seconds), last_recorded_utc=now,
                 deadline_monotonic=monotonic + seconds, seconds=now + seconds - before["started_utc"],
                 elapsed_seconds=now - before["started_utc"],
                 boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip())
    grant = dict(granted_utc=supervisor.now(), extend_seconds=seconds, statement=statement.strip(),
                 from_step=steps, step_cap=cap, completion_event_id=completion.get("event_id"),
                 previous_budget=before, new_deadline_utc=after["deadline_utc"])
    supervisor.atomic_document(budget_path, after)
    record = resume_gates(run_dir)
    if not record["passed"]:
        supervisor.atomic_document(budget_path, before)        # nothing launched: restore the original budget
        print(json.dumps(dict(state="EXTEND_BLOCKED", failed=record["failed"],
                              details=[g for g in record["gates"] if not g["passed"]]), indent=2))
        return 2
    (run_dir / "config").mkdir(exist_ok=True)
    with (run_dir / "config/budget-extensions.jsonl").open("a") as stream:
        stream.write(json.dumps(grant) + "\n")
    return relaunch(run_dir, provider=provider, agent_window=agent_window, state=state,
                    reason=completion.get("reason"), event="budget_extension",
                    extend_seconds=seconds, new_deadline_utc=after["deadline_utc"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "approve", "validate", "launch", "run", "reconcile", "resume", "extend"))
    parser.add_argument("--run-dir", type=Path, default=job.run_directory(ROOT))
    parser.add_argument("--job-sha256")
    parser.add_argument("--statement")
    parser.add_argument("--provider", default="codex", choices=("codex", "claude"))
    parser.add_argument("--skip-tests", action="store_true", help="validate without re-running focused suites")
    parser.add_argument("--confirm-run")
    parser.add_argument("--extend-seconds", type=int, help="extend: extra wall-clock seconds granted by the user")
    parser.add_argument("--no-agent-window", action="store_true",
                        help="do not open an interactive codex/claude Kitty window when the run ends")
    args = parser.parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    if args.command == "prepare":
        state = supervisor.RunState(run_dir)
        if state.load().get("state") in (None, "BLOCKED"):
            state.transition("PREPARING")
        frozen = job.prepare(ROOT, run_dir)
        print(json.dumps(dict(job_sha256=frozen["job_sha256"], spec_sha256=frozen["spec_sha256"],
                              derivation=frozen["derivation"]), indent=2))
    elif args.command == "approve":
        print(json.dumps(job.approve(run_dir, args.job_sha256, args.statement), indent=2))
    elif args.command == "validate":
        record = validate(run_dir, run_tests=not args.skip_tests)
        print(json.dumps(dict(passed=record["passed"], failed=record["failed"]), indent=2))
        return 0 if record["passed"] else 2
    elif args.command == "launch":
        return launch(run_dir, provider=args.provider, run_tests=not args.skip_tests,
                      agent_window=not args.no_agent_window)
    elif args.command == "run":
        result = supervisor.Supervisor(run_dir, ROOT, deliver=deliver).run()
        print(json.dumps(dict(action=result["action"]), indent=2))
    elif args.command == "reconcile":
        print(json.dumps(supervisor.reconcile_all(ROOT, deliver=deliver), indent=2, default=str))
    elif args.command == "resume":
        return resume(run_dir, provider=args.provider, confirm=args.confirm_run, agent_window=not args.no_agent_window)
    elif args.command == "extend":
        if not args.extend_seconds:
            parser.error("extend requires --extend-seconds and --statement")
        return extend(run_dir, provider=args.provider, confirm=args.confirm_run, seconds=args.extend_seconds,
                      statement=args.statement, agent_window=not args.no_agent_window)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
