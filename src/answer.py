"""Final-answer extraction and normalization.

Every curve in this project goes through `extract_answer` and `is_correct`, so the rules are
kept small and are unit tested in tests/test_answer.py.

Extraction rule (PLAN.md's rule plus a \\boxed{} step):
1. Find the last `Answer:` marker (case-insensitive, markdown bold allowed). Take the text after
   it, up to the end of the first non-empty line. If that text contains `=`, keep only the part
   after the last `=` (so "Answer: 5 + 3 = 8" gives 8). Then take the first number in it.
   `$`, `%`, thousands commas, and trailing periods are ignored. Simple fractions (`3/4`) are
   allowed here. Skipped when a `\\boxed{}` comes after the marker: in
   "**Final Answer:**\\nIn 2010 the car is worth \\boxed{23238}" the box is the answer.
2. Otherwise, the last `\\boxed{...}`, parsed the same way (`\\frac{a}{b}` allowed).
   Qwen2.5-Instruct ignores the requested `Answer:` format on ~94% of base-model responses and
   boxes its answer on ~56%. Without this step, "\\boxed{128} dollars for 16 glasses" scored as 16.
3. Otherwise, fall back to the last number anywhere in the response.
4. If there is no number at all, extraction fails (value None).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

# A number: comma-grouped thousands ("1,234,567.5") or plain digits ("1234.5", ".5").
# Commas only count as thousands separators when they group exactly three digits, so
# "3, 4" is two numbers and not 34.
_NUMBER = r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|-?\.\d+"
_FRACTION = r"-?\d+\s*/\s*\d+"
_NUMBER_RE = re.compile(_NUMBER)
_ANSWER_LINE_NUMBER_RE = re.compile(rf"(?P<frac>{_FRACTION})|(?P<num>{_NUMBER})")
# "Answer:", "answer :", "**Answer:**", "Final Answer:", full-width colon.
_ANSWER_MARKER_RE = re.compile(r"answer[ \t]*\**[ \t]*[:：][ \t]*\**", re.IGNORECASE)

_BOXED_RE = re.compile(r"\\boxed\s*\{")
_FRAC_CMD_RE = re.compile(r"\\d?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")

ANSWER_LINE = "answer_line"
BOXED = "boxed"
LAST_NUMBER = "last_number"


@dataclass(frozen=True)
class Extraction:
    value: float | None
    method: str | None  # ANSWER_LINE, BOXED, LAST_NUMBER, or None when nothing could be parsed

    @property
    def failed(self) -> bool:
        return self.value is None


def _clean(text: str) -> str:
    # Unicode minus sign and en dash used as minus; drop currency so "-$5" becomes "-5".
    # "- 5" stays two tokens on purpose. LaTeX thousands separator "1{,}000" -> "1,000".
    text = text.replace("\u2212", "-").replace("\u2013", "-").replace("$", "")
    return text.replace("{,}", ",")


def _last_boxed(text: str) -> tuple[int, str] | None:
    """(start offset, contents) of the last \\boxed{...}, with nested braces balanced."""
    matches = list(_BOXED_RE.finditer(text))
    if not matches:
        return None
    begin, start = matches[-1].start(), matches[-1].end()
    depth = 1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return begin, text[start:i]
    return begin, text[start:]  # unclosed (truncated); parse what is there


def _to_float(token: str) -> float | None:
    try:
        return float(token.replace(",", ""))
    except ValueError:
        return None


def _parse_answer_text(text: str) -> float | None:
    if "=" in text:
        text = text.rsplit("=", 1)[1]
    m = _ANSWER_LINE_NUMBER_RE.search(text)
    if m is None:
        return None
    if m.group("frac") is not None:
        num, den = (s.strip() for s in m.group("frac").split("/"))
        if int(den) == 0:
            return None
        return int(num) / int(den)
    return _to_float(m.group("num"))


def extract_answer(response: str) -> Extraction:
    text = _clean(response)
    boxed = _last_boxed(text)
    markers = list(_ANSWER_MARKER_RE.finditer(text))
    if markers and (boxed is None or boxed[0] < markers[-1].end()):
        # First non-empty line after the marker ("Answer:\n18" is allowed).
        rest = text[markers[-1].end():]
        line = next((ln for ln in rest.splitlines() if ln.strip()), "")
        if line:
            value = _parse_answer_text(_FRAC_CMD_RE.sub(r"\1/\2", line))
            if value is not None:
                return Extraction(value, ANSWER_LINE)
    if boxed is not None:
        value = _parse_answer_text(_FRAC_CMD_RE.sub(r"\1/\2", boxed[1]))
        if value is not None:
            return Extraction(value, BOXED)
    numbers = _NUMBER_RE.findall(text)
    if numbers:
        value = _to_float(numbers[-1])
        if value is not None:
            return Extraction(value, LAST_NUMBER)
    return Extraction(None, None)


def parse_gsm8k_answer(solution: str) -> float:
    """GSM8K ground truth: the number after '####' in the solution field."""
    if "####" not in solution:
        raise ValueError(f"no '####' in GSM8K solution: {solution[-80:]!r}")
    tail = solution.split("####")[-1].strip().replace(",", "").replace("$", "")
    return float(tail)


def is_correct(pred: float | None, gt: float) -> bool:
    if pred is None:
        return False
    return math.isclose(pred, gt, rel_tol=1e-6, abs_tol=1e-6)


def canonical(value: float | None) -> str | None:
    """Hashable key for majority voting: 18, 18.0 and 18.0000001 all map to '18'."""
    if value is None:
        return None
    s = f"{round(value, 6):.6f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s
