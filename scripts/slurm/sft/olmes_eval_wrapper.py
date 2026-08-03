"""Wrapper around OLMES that bypasses the ARG_MAX limitation.

OLMES's launch.py (line 391) serialises ~200 expanded task configs as JSON
CLI arguments into a single shell command, then runs it via
``subprocess.run(cmd, shell=True)``.  On Leonardo (Linux 5.15) this exceeds
the kernel ARG_MAX (~2 MiB) and fails with:
    OSError: [Errno 7] Argument list too long: '/bin/sh'

This wrapper monkey-patches ``subprocess.run`` inside ``oe_eval.launch`` so
that instead of passing the gigantic string to the shell, it:
  1. Writes the command string to a temporary shell script, and
  2. Executes that script with ``/bin/bash``, which has no ARG_MAX issue
     because the arguments live inside the script file, not on execve().

Usage (drop-in replacement for ``olmes``):
    python olmes_eval_wrapper.py [normal olmes arguments...]

Example:
    python olmes_eval_wrapper.py \
        --model /path/to/model \
        --model-type vllm \
        --model-args '{"trust_remote_code": true, "max_length": 32768}' \
        --task olmo3:adapt \
        --batch-size auto \
        --output-dir /path/to/output
"""

import os
import subprocess
import sys
import tempfile

# Monkey-patch subprocess.run BEFORE importing oe_eval.launch
_original_subprocess_run = subprocess.run


def _patched_subprocess_run(cmd, *args, **kwargs):
    """If shell=True and the command is a long string, write it to a temp
    script and execute that instead."""
    if kwargs.get("shell") and isinstance(cmd, str) and len(cmd) > 100_000:
        # Write the command to a temporary script file
        tmpdir = os.environ.get("TMPDIR", None)  # Use SLURM's per-job scratch if available
        fd, script_path = tempfile.mkstemp(suffix=".sh", prefix="olmes_eval_", dir=tmpdir)
        try:
            with os.fdopen(fd, "w") as f:
                f.write("#!/bin/bash\n")
                f.write(cmd)
                f.write("\n")
            os.chmod(script_path, 0o755)
            # Execute the script — arguments are inside the file, not on execve()
            kwargs_copy = dict(kwargs)
            kwargs_copy["shell"] = False
            return _original_subprocess_run(["/bin/bash", script_path], *args, **kwargs_copy)
        finally:
            try:
                os.unlink(script_path)
            except OSError:
                pass
    else:
        return _original_subprocess_run(cmd, *args, **kwargs)


subprocess.run = _patched_subprocess_run

# Now import and run OLMES
from oe_eval.launch import main  # noqa: E402

if __name__ == "__main__":
    main()
