"""Debug runner — captures errors to run_log.txt."""
import sys
import traceback
from pathlib import Path

log_path = Path(__file__).parent / "output" / "run_log.txt"
log_path.parent.mkdir(parents=True, exist_ok=True)

with open(log_path, "w") as log:
    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, s):
            for stream in self.streams:
                stream.write(s)
                stream.flush()
        def flush(self):
            for stream in self.streams:
                stream.flush()

    tee = Tee(sys.__stdout__, log)
    sys.stdout = tee
    sys.stderr = tee

    try:
        from main import cmd_normalize
        cmd_normalize()
    except Exception:
        traceback.print_exc()
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

print(f"Log: {log_path}")
