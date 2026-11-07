"""Sandboxed `run_python` for rollouts.

Threat model (RL-specific): the policy is optimized against the reward, so any way to read labels is a reward
hack it will eventually find. The agent's code therefore must not be able to read anything except its own
working directory (anomaly.json + data.csv) and the Python installation, and must have no network.

Isolation backends (SandboxConfig.backend):
  "bwrap"        Linux (training nodes): bubblewrap with an allow-list of read-only system/python paths,
                 the workdir bound read-write, all namespaces unshared (no network).
  "sandbox-exec" macOS (local dev/calibration): Seatbelt profile denying reads under /Users except the
                 python prefixes + workdir, denying writes outside the workdir, denying network.
  "none"         plain subprocess (cwd isolation only). NOT safe for RL; allowed only for debugging and
                 flagged on every result.
  "auto"         bwrap on Linux if installed, sandbox-exec on macOS, else raises (never silently "none").
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

TOOL_OUT_MAX = 8000


@dataclass
class SandboxConfig:
    backend: str = "auto"
    timeout_s: int = 120
    out_max: int = TOOL_OUT_MAX
    python: str = os.path.abspath(sys.executable)   # interpreter with pandas/numpy (abspath, not resolved:
                                                    # resolving would bypass the venv's pyvenv.cfg)
    mem_bytes: int = 4 << 30              # RLIMIT_AS (Linux only; macOS ignores it)
    file_bytes: int = 256 << 20           # RLIMIT_FSIZE


@dataclass
class ToolResult:
    output: str
    seconds: float
    timed_out: bool
    error: bool
    backend: str


def _python_roots(python: str) -> list[str]:
    """Every directory the interpreter needs: venv + base prefixes, both as written and resolved, plus each
    hop of the executable's symlink chain (uv-managed pythons are symlinked several times)."""
    out = subprocess.run([python, "-I", "-c", "import sys;print(sys.prefix);print(sys.base_prefix)"],
                         capture_output=True, text=True, check=True).stdout.split()
    roots = set()
    for p in out:
        roots.update({os.path.abspath(p), str(Path(p).resolve())})
    hop = os.path.abspath(python)
    for _ in range(10):
        roots.add(str(Path(hop).parent.parent))
        if not os.path.islink(hop):
            break
        nxt = os.readlink(hop)
        hop = nxt if os.path.isabs(nxt) else os.path.normpath(os.path.join(os.path.dirname(hop), nxt))
    # parent-of-symlinked-dir case (e.g. .../python/cpython-3.12-... -> cpython-3.12.16-...)
    for r in list(roots):
        if os.path.islink(r):
            roots.add(str(Path(r).resolve()))
    return sorted(roots)


def resolve_backend(backend: str) -> str:
    if backend != "auto":
        return backend
    if platform.system() == "Linux" and shutil.which("bwrap"):
        return "bwrap"
    if platform.system() == "Darwin" and shutil.which("sandbox-exec"):
        return "sandbox-exec"
    raise RuntimeError("no sandbox backend available (install bubblewrap on Linux); "
                       "refusing to fall back to an unsandboxed run. Use backend='none' explicitly for debugging.")


def _seatbelt_profile(workdir: str, roots: list[str]) -> str:
    allow_read = " ".join(f'(subpath "{r}")' for r in roots + [workdir])
    return f"""(version 1)
(allow default)
(deny network*)
(deny file-read* (subpath "/Users") (subpath "/Volumes") (subpath "/private/tmp") (subpath "/private/var/folders"))
(allow file-read-metadata)
(allow file-read* {allow_read})
(deny file-write* (subpath "/Users") (subpath "/Volumes") (subpath "/private/tmp") (subpath "/private/var/folders"))
(allow file-write* (subpath "{workdir}"))
"""


def _bwrap_cmd(workdir: str, roots: list[str], argv: list[str]) -> list[str]:
    cmd = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--proc", "/proc", "--dev", "/dev",
           "--tmpfs", "/tmp"]
    for p in ["/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc/alternatives", "/etc/ld.so.cache",
              "/etc/ld.so.conf", "/etc/ld.so.conf.d", "/etc/localtime"] + roots:
        if os.path.exists(p):
            cmd += ["--ro-bind", p, p]
    cmd += ["--bind", workdir, "/work", "--chdir", "/work", "--setenv", "HOME", "/work"]
    return cmd + argv


def _limits(cfg: SandboxConfig):
    def f():
        import resource
        for lim, val in ((getattr(resource, "RLIMIT_AS", None), cfg.mem_bytes),
                         (resource.RLIMIT_FSIZE, cfg.file_bytes)):
            if lim is None:
                continue
            try:
                resource.setrlimit(lim, (val, val))
            except (ValueError, OSError):
                pass
    return f


class Sandbox:
    """One per rollout. Owns a fresh working directory; call close() when the rollout ends."""

    def __init__(self, cfg: SandboxConfig | None = None, workdir: Path | None = None):
        self.cfg = cfg or SandboxConfig()
        self.backend = resolve_backend(self.cfg.backend)
        base = Path(tempfile.mkdtemp(prefix="finops_rl_")) if workdir is None else Path(workdir)
        base.mkdir(parents=True, exist_ok=True)
        self.workdir = base.resolve()
        self._roots = _python_roots(self.cfg.python)

    def run_python(self, code: str) -> ToolResult:
        t0 = time.time()
        name = f"_tool_{uuid.uuid4().hex[:8]}.py"
        (self.workdir / name).write_text(code)
        env = {"PATH": "/usr/bin:/bin", "HOME": str(self.workdir), "MPLBACKEND": "Agg", "PYTHONHASHSEED": "0"}
        py = [self.cfg.python, "-I", name]
        if self.backend == "bwrap":
            argv, cwd = _bwrap_cmd(str(self.workdir), self._roots, py), None
            env["HOME"] = "/work"
        elif self.backend == "sandbox-exec":
            argv, cwd = ["sandbox-exec", "-p", _seatbelt_profile(str(self.workdir), self._roots)] + py, self.workdir
        elif self.backend == "none":
            argv, cwd = py, self.workdir
        else:
            raise ValueError(self.backend)
        timed_out = False
        try:
            p = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=self.cfg.timeout_s, env=env,
                               preexec_fn=_limits(self.cfg))
            out = (p.stdout or "") + (("\n[stderr]\n" + p.stderr) if p.stderr else "")
            if p.returncode != 0:
                out += f"\n[exit code {p.returncode}]"
            err = p.returncode != 0
        except subprocess.TimeoutExpired:
            out, timed_out, err = f"[tool timeout after {self.cfg.timeout_s}s]", True, True
        finally:
            (self.workdir / name).unlink(missing_ok=True)
        m = self.cfg.out_max
        if len(out) > m:
            out = out[:m // 2] + f"\n...[truncated {len(out) - m} chars]...\n" + out[-m // 2:]
        if self.backend == "none":
            out += "\n[sandbox: none — unsafe, debug only]"
        return ToolResult(out if out.strip() else "[no output — use print()]", time.time() - t0, timed_out, err,
                          self.backend)

    def close(self):
        shutil.rmtree(self.workdir, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


# Static audit of agent code: cheap tripwire for reward-hacking attempts, logged as a metric
# (it does not replace the sandbox). Matches attempts to look outside the working directory.
SUSPICIOUS = ("ground_truth", "label", "manifest", "finops_gym", "finops-eval", "/users/", "/home/", "../",
              "os.listdir('/')", 'os.listdir("/")', "glob('/", 'glob("/', "subprocess", "socket", "urllib",
              "requests", "os.environ")


def audit_code(code: str) -> list[str]:
    c = (code or "").lower()
    return [s for s in SUSPICIOUS if s in c]
