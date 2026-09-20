import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "report"))

import build_findings


def test_findings_builds_completely_from_results():
    text = build_findings.fill(build_findings.TEMPLATE.read_text())
    assert "{{" not in text and "}}" not in text
    # the sections Phase 9 must contain (docs/SPEC.md Phase 9 + the task brief)
    for needle in ["Dataset analysis", "The join, in both directions", "20,224", "Corrections to the original plan",
                   "Model comparison", "The RL result is negative", "Reward and quality disagree", "LinUCB's reward-hacking",
                   "k = 5 and α = 1.0", "no timestamps, no skips", "Why tabular RL and not DQN", "Why ALS with confidence"]:
        assert needle in text, needle
    assert not re.search(r"\b(rmse|mae)\b", text, flags=re.I) or "No RMSE or MAE" in text


def test_committed_findings_is_up_to_date():
    fresh = build_findings.fill(build_findings.TEMPLATE.read_text())
    assert build_findings.OUT.read_text() == fresh, "run: python report/build_findings.py"
