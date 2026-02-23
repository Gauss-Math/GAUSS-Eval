"""
Lean Mathlib Judge
==================
Verifies mathematical proofs by submitting Lean 4 code to a Lean verification
backend (remote server or local `lean` executable).

Backends
--------
- 'server'  : HTTP POST to a lean4web-compatible server (default)
- 'local'   : subprocess calling a locally-installed `lean` binary
- 'repl'    : subprocess calling `lean --stdin` (for quick single-expression checks)

Typical server endpoints
------------------------
- https://lean.math.hhu.de/api/compile  (lean4web, HHU Düsseldorf)
- https://live.lean-lang.org/            (official Lean playground)

Usage example
-------------
    from src.lean_judge import LeanJudge, extract_lean_code

    judge = LeanJudge(backend="server")
    code  = "import Mathlib\\n\\ntheorem add_comm_example : 1 + 2 = 3 := by norm_num"
    result = judge.judge(code)
    print(result)
    # {'score': 1.0, 'max_points': 1.0, 'success': True, 'errors': [], ...}
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# lean4web HHU server (open for public use, no auth required)
LEAN4WEB_API_URL = "https://lean.math.hhu.de/api/compile"

# Default Lean version tag understood by lean4web.
# Override via LeanJudge(lean_version="leanprover/lean4:v4.x.y") if needed.
DEFAULT_LEAN_VERSION = "leanprover/lean4:v4.15.0"

DEFAULT_TIMEOUT = 120  # seconds

# Standard header that gives Lean access to all of Mathlib
DEFAULT_IMPORT = "import Mathlib"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lean code utilities
# ---------------------------------------------------------------------------

def extract_lean_code(text: str) -> Optional[str]:
    """Extract Lean 4 code from a text string (e.g. a model's answer).

    Searches in order:
      1. Fenced ```lean / ```lean4 code blocks.
      2. Any fenced ``` block whose content contains Lean-like keywords.
      3. The entire text, if it starts with known Lean keywords.

    Returns the first matching block, or None if nothing is found.
    """
    # 1. Explicitly labelled fenced blocks
    fenced = re.compile(r"```(?:lean4?)\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
    matches = fenced.findall(text)
    if matches:
        return "\n\n".join(m.strip() for m in matches)

    # 2. Unlabelled fenced blocks that look like Lean
    lean_keywords = {"theorem", "lemma", "def ", "import Mathlib", "by ", "#check", "example :"}
    unlabeled = re.compile(r"```\s*\n(.*?)```", re.DOTALL)
    for block in unlabeled.findall(text):
        if any(kw in block for kw in lean_keywords):
            return block.strip()

    # 3. Entire text looks like Lean
    stripped = text.strip()
    leading_keywords = ("theorem", "lemma", "import Mathlib", "def ", "example :", "#check")
    if any(stripped.startswith(kw) for kw in leading_keywords):
        return stripped

    return None


def build_lean_file(
    code: str,
    imports: str = DEFAULT_IMPORT,
    preamble: str = "",
) -> str:
    """Assemble a complete Lean 4 source file.

    Args:
        code:     The main theorem / proof code.
        imports:  Import header (default: ``import Mathlib``).
        preamble: Optional declarations inserted after imports but before *code*.

    Returns:
        A string ready to be saved as a ``.lean`` file or sent to a server.
    """
    parts: List[str] = []
    if imports:
        parts.append(imports)
    if preamble:
        parts.append(preamble)
    if code:
        parts.append(code)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# LeanJudge
# ---------------------------------------------------------------------------

class LeanJudge:
    """Judge that verifies Lean 4 / Mathlib proofs and returns a numeric score.

    Parameters
    ----------
    backend:
        ``"server"`` (default) – POST code to a lean4web-compatible HTTP server.
        ``"local"``            – run a locally-installed ``lean`` binary.
    server_url:
        URL of the lean4web server.  Ignored when *backend* is ``"local"``.
    lean_version:
        Lean version tag forwarded to the lean4web server (e.g.
        ``"leanprover/lean4:v4.15.0"``).
    lean_executable:
        Path / name of the local ``lean`` binary.  Ignored when *backend* is
        ``"server"``.
    timeout:
        Maximum seconds to wait for a verification result.
    max_points:
        Score awarded for a successful proof.  Failed proofs always score 0.
    auto_add_imports:
        If True and the code does not already start with an ``import`` line,
        ``DEFAULT_IMPORT`` is prepended automatically.
    """

    def __init__(
        self,
        backend: str = "server",
        server_url: str = LEAN4WEB_API_URL,
        lean_version: str = DEFAULT_LEAN_VERSION,
        lean_executable: str = "lean",
        timeout: int = DEFAULT_TIMEOUT,
        max_points: float = 1.0,
        auto_add_imports: bool = True,
    ) -> None:
        if backend not in {"server", "local"}:
            raise ValueError(f"backend must be 'server' or 'local', got '{backend}'")
        self.backend = backend
        self.server_url = server_url
        self.lean_version = lean_version
        self.lean_executable = lean_executable
        self.timeout = timeout
        self.max_points = max_points
        self.auto_add_imports = auto_add_imports

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(self, lean_code: str) -> Dict[str, Any]:
        """Verify Lean 4 code and return structured results.

        Args:
            lean_code: Complete Lean 4 source (imports + proof).

        Returns:
            A dict with keys:
              - ``success``    (bool)       – True iff no errors.
              - ``errors``     (list[str])  – Error messages from Lean.
              - ``warnings``   (list[str])  – Warning messages from Lean.
              - ``raw_output`` (str)        – Full raw output text.
        """
        code = self._maybe_add_imports(lean_code)
        if self.backend == "server":
            return self._verify_via_server(code)
        else:
            return self._verify_via_local(code)

    def judge(self, lean_code: str) -> Dict[str, Any]:
        """Verify *lean_code* and return a score alongside verification details.

        Args:
            lean_code: Lean 4 source to judge.

        Returns:
            A dict with all keys from :meth:`verify`, plus:
              - ``score``      (float) – *max_points* on success, else 0.
              - ``max_points`` (float) – configured maximum.
        """
        result = self.verify(lean_code)
        score = self.max_points if result["success"] else 0.0
        return {
            "score": score,
            "max_points": self.max_points,
            **result,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_add_imports(self, code: str) -> str:
        """Prepend ``import Mathlib`` if the code has no import statements."""
        if self.auto_add_imports and not re.search(r"^\s*import\s", code, re.MULTILINE):
            return DEFAULT_IMPORT + "\n\n" + code
        return code

    # ---------- server backend ----------

    def _verify_via_server(self, lean_code: str) -> Dict[str, Any]:
        """POST Lean code to a lean4web-compatible HTTP server."""
        payload = {
            "code": lean_code,
            "lean_version": self.lean_version,
        }
        try:
            response = requests.post(
                self.server_url,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
        except requests.exceptions.Timeout:
            msg = f"Lean server request timed out after {self.timeout}s"
            logger.error(msg)
            return {"success": False, "errors": [msg], "warnings": [], "raw_output": ""}
        except requests.exceptions.RequestException as exc:
            msg = f"Lean server request failed: {exc}"
            logger.error(msg)
            return {"success": False, "errors": [msg], "warnings": [], "raw_output": ""}

        try:
            data = response.json()
        except ValueError:
            # Server returned non-JSON (e.g. plain-text error page)
            raw = response.text
            return {
                "success": False,
                "errors": [f"Non-JSON server response: {raw[:300]}"],
                "warnings": [],
                "raw_output": raw,
            }

        return self._parse_server_response(data)

    def _parse_server_response(self, data: Any) -> Dict[str, Any]:
        """Normalise different lean4web response formats into a common dict."""
        errors: List[str] = []
        warnings: List[str] = []
        raw_output: str = json.dumps(data) if not isinstance(data, str) else data

        # Format A: {"messages": [{"severity": "error"|"warning", "data": "..."}]}
        if isinstance(data, dict) and "messages" in data:
            for msg in data["messages"]:
                severity = msg.get("severity", "")
                text = msg.get("data", msg.get("message", msg.get("text", str(msg))))
                if severity == "error":
                    errors.append(text)
                elif severity == "warning":
                    warnings.append(text)

        # Format B: {"stdout": "...", "stderr": "..."}
        elif isinstance(data, dict) and ("stdout" in data or "stderr" in data):
            stderr = data.get("stderr", "")
            stdout = data.get("stdout", "")
            raw_output = (stdout + "\n" + stderr).strip()
            if stderr:
                errors_raw, warnings_raw = _parse_lean_output(stderr)
                errors.extend(errors_raw)
                warnings.extend(warnings_raw)
            if stdout:
                errors_raw, warnings_raw = _parse_lean_output(stdout)
                errors.extend(errors_raw)
                warnings.extend(warnings_raw)

        # Format C: explicit "success" boolean
        if isinstance(data, dict) and "success" in data:
            success = bool(data["success"]) and not errors
        else:
            success = not errors

        return {
            "success": success,
            "errors": errors,
            "warnings": warnings,
            "raw_output": raw_output,
        }

    # ---------- local backend ----------

    def _verify_via_local(self, lean_code: str) -> Dict[str, Any]:
        """Run Lean code through a locally-installed ``lean`` binary."""
        tmp_path: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".lean", delete=False, encoding="utf-8"
            ) as f:
                f.write(lean_code)
                tmp_path = f.name

            proc = subprocess.run(
                [self.lean_executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            combined = (proc.stdout + "\n" + proc.stderr).strip()
            errors, warnings = _parse_lean_output(combined)

            # If returncode != 0 but no structured errors were extracted,
            # treat the full stderr as one error.
            if proc.returncode != 0 and not errors:
                errors.append(proc.stderr.strip() or "Lean exited with a non-zero status")

            return {
                "success": proc.returncode == 0 and not errors,
                "errors": errors,
                "warnings": warnings,
                "raw_output": combined,
            }

        except subprocess.TimeoutExpired:
            msg = f"Lean process timed out after {self.timeout}s"
            logger.error(msg)
            return {"success": False, "errors": [msg], "warnings": [], "raw_output": ""}

        except FileNotFoundError:
            msg = (
                f"Lean executable '{self.lean_executable}' not found. "
                "Install Lean 4 from https://leanprover.github.io/lean4/doc/setup.html"
            )
            logger.error(msg)
            return {"success": False, "errors": [msg], "warnings": [], "raw_output": ""}

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Output parser helpers
# ---------------------------------------------------------------------------

def _parse_lean_output(text: str):
    """Return (errors, warnings) lists by scanning Lean compiler output."""
    errors: List[str] = []
    warnings: List[str] = []

    # Lean 4 lines look like:  foo.lean:5:0: error: ...
    error_re = re.compile(r"(?:error|Error):\s*(.+)")
    warning_re = re.compile(r"(?:warning|Warning):\s*(.+)")

    for line in text.splitlines():
        em = error_re.search(line)
        if em:
            errors.append(em.group(1).strip())
            continue
        wm = warning_re.search(line)
        if wm:
            warnings.append(wm.group(1).strip())

    return errors, warnings


# ---------------------------------------------------------------------------
# Dataset-level helper
# ---------------------------------------------------------------------------

def run_lean_judge_on_item(
    data_item: Dict[str, Any],
    judge: Optional[LeanJudge] = None,
    *,
    backend: str = "server",
    server_url: str = LEAN4WEB_API_URL,
    lean_version: str = DEFAULT_LEAN_VERSION,
    lean_executable: str = "lean",
    timeout: int = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """Run the Lean Mathlib judge on a single GAUSS-Eval dataset item.

    The function looks for Lean 4 code in the ``answer`` field of *data_item*,
    wraps it in a complete Lean file (adding ``import Mathlib`` if absent), and
    verifies it.  The result dict mirrors *data_item* with an extra
    ``lean_judge_result`` key.

    Args:
        data_item:        One entry from a GAUSS-Eval dataset.
        judge:            Optional pre-built :class:`LeanJudge` instance.
        backend:          Backend to use if *judge* is not provided.
        server_url:       Lean4web server URL.
        lean_version:     Lean version tag.
        lean_executable:  Local Lean binary path.
        timeout:          Verification timeout in seconds.

    Returns:
        A copy of *data_item* with ``lean_judge_result`` and
        ``lean_code_found`` fields added.
    """
    max_pts = float(data_item.get("max_points_judge_1", 1.0))

    if judge is None:
        judge = LeanJudge(
            backend=backend,
            server_url=server_url,
            lean_version=lean_version,
            lean_executable=lean_executable,
            timeout=timeout,
            max_points=max_pts,
        )
    else:
        # Override max_points with the item's value
        judge.max_points = max_pts

    answer = data_item.get("answer", "")
    lean_code = extract_lean_code(answer)

    result_item = dict(data_item)

    if lean_code is None:
        logger.warning("No Lean 4 code found in item id=%s", data_item.get("id", "?"))
        result_item["lean_code_found"] = False
        result_item["lean_judge_result"] = {
            "success": False,
            "score": 0.0,
            "max_points": max_pts,
            "errors": ["No Lean 4 code block found in the answer"],
            "warnings": [],
            "raw_output": "",
            "lean_code": None,
        }
    else:
        lean_file = build_lean_file(code=lean_code)
        judgment = judge.judge(lean_file)
        judgment["lean_code"] = lean_code
        result_item["lean_code_found"] = True
        result_item["lean_judge_result"] = judgment

    return result_item
