"""Live read-only Codexa training dashboard. Closing it never affects training.

Keys: q quit · r refresh · d details · c compact · ? help
"""

import argparse
import io
import json
import os
from pathlib import Path
import select
import signal
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from codexa_workspace.foundation_training import run_directory
from codexa_workspace.training_dashboard import RunReader, frame, render_text
from codexa_workspace import dashboard_window

HELP = ("Read-only dashboard. q quit · r refresh now · d toggle raw details · c toggle compact · ? this help.\n"
        "Data comes from the run's small JSON status files; nothing here can stop, pause or notify.")


def open_kitty(run_dir: Path, interval: float) -> int:
    """Open the fixed 120x30 dashboard window in its own user unit, or point at the open one."""
    result = dashboard_window.open_window(run_dir, interval=interval)
    print(f"Opened dashboard window ({result['unit']})." if result["opened"]
          else f"Dashboard for this run is already open (pid {result['pid']}).")
    return 0


def mark_attached(marker: Path) -> None:
    temporary = marker.with_suffix(".attaching")
    temporary.write_text(json.dumps(dict(pid=os.getpid())))
    temporary.replace(marker)


def live(run_dir: Path, *, interval: float, details: bool, compact: bool, marker: Path | None = None) -> int:
    import termios
    import tty
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.console import Group

    console = Console()
    reader = RunReader(run_dir)
    measure = Console(file=io.StringIO())
    wake_read, wake_write = os.pipe()
    os.set_blocking(wake_read, False)
    os.set_blocking(wake_write, False)
    previous_wakeup = signal.set_wakeup_fd(wake_write)
    previous_winch = signal.signal(signal.SIGWINCH, lambda *_: None)
    stdin = sys.stdin.fileno()
    saved = termios.tcgetattr(stdin)
    show_help = False
    dashboard_window.register(run_dir)
    try:
        tty.setcbreak(stdin)
        with Live(console=console, screen=True, auto_refresh=False, redirect_stdout=False) as view:
            while True:
                width, height = console.size
                measure.width = width
                body = frame(reader.read(), width=width, height=height - (3 if show_help else 0),
                             details=details, compact=compact, measure=measure)
                view.update(Group(Panel(HELP, border_style="grey37"), body) if show_help else body, refresh=True)
                if marker is not None:      # first frame is on screen: the trainer may proceed
                    mark_attached(marker)
                    marker = None
                ready, _, _ = select.select([stdin, wake_read], [], [], interval)
                if wake_read in ready:
                    try:
                        os.read(wake_read, 512)
                    except BlockingIOError:
                        pass
                if stdin in ready:
                    key = os.read(stdin, 1).decode(errors="ignore").lower()
                    if key in ("q", "\x1b"):
                        return 0
                    details = not details if key == "d" else details
                    compact = not compact if key == "c" else compact
                    show_help = not show_help if key == "?" else show_help
    except KeyboardInterrupt:
        return 0
    finally:
        dashboard_window.unregister(run_dir)
        termios.tcsetattr(stdin, termios.TCSADRAIN, saved)
        signal.set_wakeup_fd(previous_wakeup)
        signal.signal(signal.SIGWINCH, previous_winch)
        os.close(wake_read)
        os.close(wake_write)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=run_directory(ROOT))
    parser.add_argument("--interval", type=float, default=2.0, help="refresh seconds (default 2)")
    parser.add_argument("--details", action="store_true", help="start with the raw technical panel")
    parser.add_argument("--compact", action="store_true", help="start in compact mode")
    parser.add_argument("--once", action="store_true", help="print one frame and exit")
    parser.add_argument("--plain", action="store_true", help="with --once: no colour")
    parser.add_argument("--width", type=int, help="with --once: render width")
    parser.add_argument("--kitty", action="store_true", help="open in a separate 120x30 Kitty window (or reuse it)")
    parser.add_argument("--attach", action="store_true", help=argparse.SUPPRESS)   # trainer viewer hook
    parser.add_argument("--status", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--marker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.attach:
        if not args.status or not args.marker:
            parser.error("--attach requires --status and --marker")
        dashboard_window.attach(args.status, args.marker)
        return 0
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    if args.kitty:
        return open_kitty(run_dir, args.interval)
    if args.once or not sys.stdin.isatty() or not sys.stdout.isatty():
        size = os.get_terminal_size() if sys.stdout.isatty() else os.terminal_size((120, 60))
        print(render_text(run_dir, width=args.width or size.columns, details=args.details, compact=args.compact,
                          color=not args.plain and sys.stdout.isatty()), end="")
        return 0
    return live(run_dir, interval=max(0.5, args.interval), details=args.details, compact=args.compact,
                marker=args.marker)


if __name__ == "__main__":
    raise SystemExit(main())
