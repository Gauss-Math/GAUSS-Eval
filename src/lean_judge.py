"""
Lean Mathlib Judge
==================
Verifies mathematical proofs by submitting Lean 4 code to a Lean verification
backend (remote server or local `lean` executable).

Two-stage pipeline (optional)
------------------------------
When ``--decompose`` is requested, an LLM first translates the natural-language
proof into Lean 4 / Mathlib code (``LLMDecomposer``), then the resulting code
is verified by the Lean backend (``LeanJudge``).  The combined class is
``DecomposeAndVerifyPipeline``.

Backends
--------
- 'server'  : HTTP POST to a lean4web-compatible server (default)
- 'local'   : subprocess calling a locally-installed `lean` binary

Typical server endpoints
------------------------
- https://lean.math.hhu.de/api/compile  (lean4web, HHU Düsseldorf)
- https://live.lean-lang.org/            (official Lean playground)

Usage examples
--------------
    # Direct Lean verification (answer must already contain Lean code):
    from src.lean_judge import LeanJudge, extract_lean_code
    judge = LeanJudge(backend="server")
    result = judge.judge("import Mathlib\\n\\ntheorem t : 1 + 2 = 3 := by norm_num")

    # LLM decomposition → Lean verification (natural-language answer):
    from src.lean_judge import DecomposeAndVerifyPipeline
    pipeline = DecomposeAndVerifyPipeline(llm_model="openrouter/openai/gpt-4o-mini")
    result = pipeline.run(data_item)   # data_item has 'problem', 'answer', 'rubric'
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

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


# ===========================================================================
# LLM Decomposition prompts
# ===========================================================================

# The system prompt gives the LLM rich Mathlib context so it can map
# natural-language mathematical reasoning to specific Lean 4 tactics and lemmas.
DECOMPOSE_SYSTEM_PROMPT = """\
You are a world-class expert in Lean 4 formal mathematics with deep knowledge \
of Mathlib, the comprehensive Lean 4 mathematics library. Your task is to \
translate a student's natural-language mathematical proof into verified Lean 4 \
code, making full use of the Mathlib library.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LEAN 4 / MATHLIB REFERENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Imports & namespaces
Always start with `import Mathlib`.
Open useful namespaces as needed:
  open Nat Int Real Finset BigOperators

## Number / type hierarchy
  ℕ  (Nat)   ℤ  (Int)   ℚ  (Rat)   ℝ  (Real)   ℂ  (Complex)
  Fin n, ZMod n, GaussianInt

## Core automation tactics
  norm_num          – numerical goals with constants (1 + 1 = 2, 3 < 7, …)
  ring              – algebraic identities in comm rings / fields
  ring_nf           – normalize ring expressions without closing the goal
  linarith          – linear arithmetic over ordered fields/rings
  nlinarith         – polynomial / nonlinear arithmetic
  omega             – linear arithmetic over ℤ / ℕ (integers)
  decide            – decidable propositions on finite types
  simp              – rewrite with the global simp set
  simp [h1, h2]     – targeted simplification
  simp only [h]     – simp with exactly the listed lemmas
  field_simp        – clear denominators; combine with ring
  push_neg          – push ¬ inward (¬ ∀ → ∃ ¬, ¬ ∃ → ∀ ¬, …)
  contrapose!       – proof by contrapositive (also pushes neg)
  by_contra h       – proof by contradiction; h : ¬ goal
  positivity        – prove 0 ≤ x  or  0 < x
  gcongr            – congruence for ≤ / < goals
  aesop             – general-purpose automation (good fallback)
  tauto / trivial   – propositional / trivial goals
  norm_cast         – coercions between ℕ ℤ ℚ ℝ
  exact?            – search Mathlib for an exact closing lemma (hint mode)
  apply?            – search for applicable lemmas

## Common Mathlib lemma families

### Arithmetic / divisibility
  Nat.dvd_add, Nat.dvd_mul_right, Nat.dvd_sub'
  Int.dvd_add, Int.dvd_mul_right
  Nat.Coprime, Nat.gcd_dvd_left, Nat.gcd_dvd_right
  Nat.Prime, Nat.Prime.dvd_mul, Nat.prime_def_minFac
  Nat.Coprime.pow_dvd_of_pow_dvd

### Modular arithmetic
  Int.ModEq  (notation: a ≡ b [ZMOD n])
  Int.emod_emod_of_dvd, ZMod.val_natCast
  Nat.ModEq, Nat.add_mod, Nat.mul_mod

### Algebra / polynomials
  mul_comm, add_comm, mul_add, add_mul, sub_add_cancel
  sq_nonneg, mul_self_nonneg, abs_nonneg, abs_le
  Polynomial.eval, Polynomial.degree, Polynomial.roots

### Summation / products
  Finset.sum_range_succ, Finset.prod_range_succ
  Finset.sum_add_distrib, Finset.mul_sum
  Finset.sum_comm, Finset.sum_congr
  Finset.geom_sum_eq  (geometric series)
  BigOperators (∑ i in s, f i  and  ∏ i in s, f i notation)

### Inequalities
  le_of_eq, lt_of_le_of_lt, le_trans, lt_trans
  mul_le_mul_of_nonneg_right/left, pow_le_pow_left
  Real.sqrt_le_sqrt, Real.sq_sqrt (h : 0 ≤ x)
  sq_le_sq', abs_sub_lt_iff, dist_le_iff

### Real analysis
  Real.sqrt_nonneg, Real.sqrt_sq, Real.sqrt_mul
  Real.exp_pos, Real.log_pos, Real.rpow_natCast
  Real.inner_le_iff, Real.norm_eq_abs

### Combinatorics
  Finset.card_filter, Finset.card_range, Finset.card_Icc
  Fintype.card_prod, Nat.choose, Nat.factorial
  Finset.card_union_add_card_inter (inclusion-exclusion)
  Finset.sum_const (constant sum)

### Logic / structure
  And.intro / ⟨_, _⟩, Or.inl / Or.inr
  Iff.intro, Iff.mpr, iff_iff_eq
  Classical.byContradiction, Classical.em
  not_forall, not_exists, exists_prop

## Proof structure patterns

```lean
-- Conjunction:
constructor
· -- prove left
· -- prove right

-- Existential:
use <witness>
-- prove P <witness>

-- Induction on n : ℕ:
induction n with
| zero   => -- base case
| succ n ih => -- inductive step (ih : P n)

-- Case split on h : P ∨ Q:
rcases h with h1 | h2

-- Destructure ⟨a, ha⟩:
obtain ⟨a, ha⟩ := h

-- Intermediate steps:
have h1 : <claim> := by <tactic>
have h2 : <claim> := by
  <multi-line proof>
exact <conclusion using h1 h2>

-- Calc block (chain of equalities/inequalities):
calc a = b := by ring
     _ ≤ c := by linarith
     _ < d := by norm_num
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Output ONLY a single ```lean ... ``` code block.
• Do NOT include any prose outside the block.
• Use `sorry` + a `-- TODO:` comment for any step you cannot yet prove;
  do NOT use `sorry` without a comment.
• Name each `have` meaningfully (e.g. `have h_coprime`, `have h_bound`).
• Lean 4 syntax: use `·` (center dot) for focusing, not `{}`.\
"""

DECOMPOSE_USER_PROMPT = """\
## Problem
{problem}

## Grading rubric
{rubric}

## Student's proof (natural language)
{answer}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## Your task

1. **Identify** the main statement to prove (write it as the theorem signature).
2. **Decompose** the proof into sub-goals (`have` statements) that mirror the
   logical structure of the student's argument and correspond to rubric criteria.
3. **Select** the most specific Mathlib tactic or lemma for each step.
4. **Write** a complete, self-contained Lean 4 proof beginning with `import Mathlib`.

Output the proof in a single ```lean ... ``` code block. Nothing else.\
"""


# ===========================================================================
# LLMDecomposer
# ===========================================================================

class LLMDecomposer:
    """Translate a natural-language mathematical proof into Lean 4 / Mathlib code.

    Uses an LLM (via ``litellm``) with a carefully engineered system prompt
    that provides rich Mathlib knowledge, enabling the model to map
    competition-mathematics reasoning steps to concrete Lean 4 tactics and
    library lemmas.

    Parameters
    ----------
    model:
        Any ``litellm``-compatible model string, e.g.
        ``"openrouter/openai/gpt-4o-mini"`` or ``"anthropic/claude-3-5-sonnet"``.
    sampling_params:
        Extra keyword arguments forwarded to ``litellm.completion``
        (temperature, max_tokens, …).
    timeout:
        HTTP timeout in seconds for the LLM call.
    system_prompt:
        Override the built-in Mathlib-aware system prompt.
    user_prompt_template:
        Override the built-in user prompt (must contain ``{problem}``,
        ``{rubric}``, and ``{answer}`` placeholders).
    """

    def __init__(
        self,
        model: str = "openrouter/openai/gpt-4o-mini",
        sampling_params: Optional[Dict[str, Any]] = None,
        timeout: int = 300,
        system_prompt: str = DECOMPOSE_SYSTEM_PROMPT,
        user_prompt_template: str = DECOMPOSE_USER_PROMPT,
    ) -> None:
        self.model = model
        self.sampling_params = sampling_params or {
            "temperature": 0.1,
            "max_tokens": 4096,
        }
        self.timeout = timeout
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decompose(
        self,
        problem: str,
        answer: str,
        rubric: str = "",
    ) -> Dict[str, Any]:
        """Call the LLM to produce Lean 4 code from a natural-language proof.

        Args:
            problem: The mathematical problem statement.
            answer:  The student's natural-language solution / proof.
            rubric:  Optional grading rubric (helps align sub-goals with criteria).

        Returns:
            A dict with keys:
              - ``lean_code``       (str | None) – extracted Lean 4 code.
              - ``llm_response``    (str)         – raw LLM output.
              - ``llm_model``       (str)         – model used.
              - ``input_tokens``    (int)
              - ``output_tokens``   (int)
              - ``decompose_error`` (str | None)  – error message, if any.
        """
        user_prompt = self.user_prompt_template.format(
            problem=problem,
            rubric=rubric or "(no rubric provided)",
            answer=answer,
        )
        try:
            from litellm import completion  # lazy import – not always installed
        except ImportError:
            return {
                "lean_code": None,
                "llm_response": "",
                "llm_model": self.model,
                "input_tokens": 0,
                "output_tokens": 0,
                "decompose_error": (
                    "litellm is not installed. "
                    "Run: pip install litellm"
                ),
            }

        try:
            response = completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                timeout=self.timeout,
                **self.sampling_params,
            )
            raw_text: str = response["choices"][0]["message"]["content"]
            usage = response.get("usage", {})
            lean_code = extract_lean_code(raw_text)

            return {
                "lean_code": lean_code,
                "llm_response": raw_text,
                "llm_model": self.model,
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "decompose_error": None,
            }

        except Exception as exc:
            logger.error("LLM decomposition failed: %s", exc, exc_info=True)
            return {
                "lean_code": None,
                "llm_response": "",
                "llm_model": self.model,
                "input_tokens": 0,
                "output_tokens": 0,
                "decompose_error": str(exc),
            }


# ===========================================================================
# Partial-credit helper
# ===========================================================================

def _count_sorry(lean_code: str) -> Tuple[int, int]:
    """Return (sorry_count, have_count) for a Lean 4 source string.

    Used to estimate partial credit: proven sub-goals are `have` statements
    that are *not* followed by `sorry`.
    """
    # Count sorry occurrences (excluding sorry inside comments)
    code_no_comments = re.sub(r"--[^\n]*", "", lean_code)
    sorry_count = len(re.findall(r"\bsorry\b", code_no_comments))
    have_count = len(re.findall(r"\bhave\b", code_no_comments))
    return sorry_count, have_count


def _partial_score(
    lean_result: Dict[str, Any],
    lean_code: str,
    max_points: float,
) -> float:
    """Compute a score taking partial credit for sorry-free sub-goals into account.

    - Full proof verified (no sorry)  → max_points.
    - Proof contains sorry (unproven sub-goals):
        score = max_points × (proven_haves / total_haves),
        where proven_haves = have_count − sorry_count (clipped to 0).
    - Proof not verified at all (server error, no code, …) → 0.
    """
    if lean_result.get("success"):
        return max_points

    code_no_comments = re.sub(r"--[^\n]*", "", lean_code or "")
    sorry_count = len(re.findall(r"\bsorry\b", code_no_comments))
    have_count = len(re.findall(r"\bhave\b", code_no_comments))

    if have_count == 0 or sorry_count == 0:
        return 0.0

    proven = max(0, have_count - sorry_count)
    return round(max_points * proven / have_count, 4)


# ===========================================================================
# DecomposeAndVerifyPipeline
# ===========================================================================

class DecomposeAndVerifyPipeline:
    """Two-stage pipeline: LLM decomposition → Lean Mathlib verification.

    Stage 1 – **LLMDecomposer**
        Given the problem, rubric, and natural-language answer, an LLM
        (with a Mathlib-rich system prompt) generates Lean 4 code that
        formally formalises the proof, decomposing it into sub-goals aligned
        with the rubric criteria.

    Stage 2 – **LeanJudge**
        The generated Lean 4 code is compiled by a Lean verification backend.
        If the code contains ``sorry`` placeholders the pipeline assigns partial
        credit proportional to the fraction of proven ``have`` sub-goals.

    Parameters
    ----------
    llm_model:
        litellm model string for the decomposition step.
    llm_sampling_params:
        Extra kwargs for litellm (temperature, max_tokens, …).
    llm_timeout:
        Timeout in seconds for the LLM call.
    lean_backend:
        ``"server"`` or ``"local"``.
    lean_server_url:
        lean4web server URL (used when *lean_backend* is ``"server"``).
    lean_version:
        Lean version tag forwarded to the lean4web server.
    lean_executable:
        Local ``lean`` binary (used when *lean_backend* is ``"local"``).
    lean_timeout:
        Timeout in seconds for Lean compilation.
    max_points:
        Score ceiling for a fully proven answer.
    partial_credit:
        If True, award fractional credit for partly-proven proofs (sorry).
    system_prompt / user_prompt_template:
        Override the built-in LLM prompts.
    """

    def __init__(
        self,
        llm_model: str = "openrouter/openai/gpt-4o-mini",
        llm_sampling_params: Optional[Dict[str, Any]] = None,
        llm_timeout: int = 300,
        lean_backend: str = "server",
        lean_server_url: str = LEAN4WEB_API_URL,
        lean_version: str = DEFAULT_LEAN_VERSION,
        lean_executable: str = "lean",
        lean_timeout: int = DEFAULT_TIMEOUT,
        max_points: float = 1.0,
        partial_credit: bool = True,
        system_prompt: str = DECOMPOSE_SYSTEM_PROMPT,
        user_prompt_template: str = DECOMPOSE_USER_PROMPT,
    ) -> None:
        self.decomposer = LLMDecomposer(
            model=llm_model,
            sampling_params=llm_sampling_params,
            timeout=llm_timeout,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
        )
        self.lean_judge = LeanJudge(
            backend=lean_backend,
            server_url=lean_server_url,
            lean_version=lean_version,
            lean_executable=lean_executable,
            timeout=lean_timeout,
            max_points=max_points,
        )
        self.max_points = max_points
        self.partial_credit = partial_credit

    def run(self, data_item: Dict[str, Any]) -> Dict[str, Any]:
        """Run the full decompose-and-verify pipeline on one dataset item.

        Args:
            data_item: A GAUSS-Eval item dict (keys: problem, answer,
                       grading_details_judge_1, max_points_judge_1, …).

        Returns:
            A copy of *data_item* with extra keys:
              - ``decompose_result``   – output of :class:`LLMDecomposer`.
              - ``lean_judge_result``  – output of :class:`LeanJudge`.
              - ``pipeline_score``     – final numeric score (with partial credit).
              - ``pipeline_max``       – configured maximum score.
              - ``lean_code_found``    – whether the LLM produced valid Lean code.
        """
        max_pts = float(data_item.get("max_points_judge_1", self.max_points))
        self.lean_judge.max_points = max_pts

        problem = data_item.get("problem", "")
        answer  = data_item.get("answer", "")

        # Format rubric the same way eval_utils does
        rubric_raw = data_item.get("grading_details_judge_1", "")
        if isinstance(rubric_raw, list):
            rubric_lines = []
            for i, item in enumerate(rubric_raw):
                title   = item.get("title", "")
                pts     = item.get("max_points", "")
                content = item.get("grading_scheme_desc", "")
                rubric_lines.append(
                    f"{i+1}. [Title: {title}] [Max pts: {pts}] {content}"
                )
            rubric = "\n".join(rubric_lines)
        else:
            rubric = str(rubric_raw)

        result_item = dict(data_item)

        # ── Stage 1: LLM decomposition ──────────────────────────────────────
        logger.info(
            "Decomposing item id=%s with model=%s",
            data_item.get("id", "?"), self.decomposer.model,
        )
        decompose_result = self.decomposer.decompose(
            problem=problem,
            answer=answer,
            rubric=rubric,
        )
        result_item["decompose_result"] = decompose_result

        lean_code = decompose_result.get("lean_code")

        if lean_code is None:
            logger.warning(
                "LLM produced no Lean code for item id=%s. error=%s",
                data_item.get("id", "?"),
                decompose_result.get("decompose_error"),
            )
            result_item["lean_code_found"] = False
            result_item["lean_judge_result"] = {
                "success": False,
                "score": 0.0,
                "max_points": max_pts,
                "errors": [
                    decompose_result.get("decompose_error")
                    or "LLM did not produce a Lean code block"
                ],
                "warnings": [],
                "raw_output": decompose_result.get("llm_response", ""),
                "lean_code": None,
            }
            result_item["pipeline_score"] = 0.0
            result_item["pipeline_max"]   = max_pts
            return result_item

        # ── Stage 2: Lean verification ──────────────────────────────────────
        lean_file = build_lean_file(code=lean_code)
        logger.info(
            "Verifying Lean code for item id=%s (backend=%s)",
            data_item.get("id", "?"), self.lean_judge.backend,
        )
        lean_result = self.lean_judge.verify(lean_file)
        lean_result["lean_code"] = lean_code

        # ── Score (with optional partial credit) ────────────────────────────
        if self.partial_credit:
            score = _partial_score(lean_result, lean_code, max_pts)
        else:
            score = max_pts if lean_result["success"] else 0.0

        lean_result["score"]      = score
        lean_result["max_points"] = max_pts

        result_item["lean_code_found"]  = True
        result_item["lean_judge_result"] = lean_result
        result_item["pipeline_score"]   = score
        result_item["pipeline_max"]     = max_pts

        return result_item


# ===========================================================================
# Dataset-level helper for the decompose-and-verify pipeline
# ===========================================================================

def run_decompose_verify_on_item(
    data_item: Dict[str, Any],
    pipeline: Optional[DecomposeAndVerifyPipeline] = None,
    *,
    llm_model: str = "openrouter/openai/gpt-4o-mini",
    llm_sampling_params: Optional[Dict[str, Any]] = None,
    llm_timeout: int = 300,
    lean_backend: str = "server",
    lean_server_url: str = LEAN4WEB_API_URL,
    lean_version: str = DEFAULT_LEAN_VERSION,
    lean_executable: str = "lean",
    lean_timeout: int = DEFAULT_TIMEOUT,
    partial_credit: bool = True,
) -> Dict[str, Any]:
    """Convenience wrapper: run :class:`DecomposeAndVerifyPipeline` on one item.

    Creates a pipeline on the fly if *pipeline* is not provided.  All keyword
    arguments are forwarded to :class:`DecomposeAndVerifyPipeline`.

    Returns the dict produced by :meth:`DecomposeAndVerifyPipeline.run`.
    """
    max_pts = float(data_item.get("max_points_judge_1", 1.0))

    if pipeline is None:
        pipeline = DecomposeAndVerifyPipeline(
            llm_model=llm_model,
            llm_sampling_params=llm_sampling_params,
            llm_timeout=llm_timeout,
            lean_backend=lean_backend,
            lean_server_url=lean_server_url,
            lean_version=lean_version,
            lean_executable=lean_executable,
            lean_timeout=lean_timeout,
            max_points=max_pts,
            partial_credit=partial_credit,
        )
    else:
        pipeline.max_points = max_pts
        pipeline.lean_judge.max_points = max_pts

    return pipeline.run(data_item)
