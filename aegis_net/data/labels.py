"""
labels.py - every rule about LABEL NAMES lives here.

* The "not an attack" class is always renamed to exactly BENIGN_NAME ("Benign") and always gets class code 0,
  whatever the dataset calls it (BenignTraffic, BENIGN, Normal, ...).
* The 34 raw CICIoT2023 labels are grouped into 8 families, but ONLY when the dataset really looks like CICIoT2023.
  Any other dataset keeps its own label names.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Optional, Sequence

BENIGN_NAME = "Benign"
BENIGN_PREFIXES = ("benign", "normal", "legitimate", "background")     # matched after removing every non-alphanumeric character

_PREFIX_RULES = (("BENIGN", "Benign"), ("DDOS", "DDoS"), ("DOS", "DoS"), ("MIRAI", "Mirai"), ("RECON", "Recon"))
_CONTAINS_RULES = (
    ("VULNERABILITYSCAN", "Recon"), ("BRUTEFORCE", "BruteForce"), ("SPOOF", "Spoofing"), ("MITM", "Spoofing"),
    ("SQLINJECTION", "Web-based"), ("XSS", "Web-based"), ("COMMANDINJECTION", "Web-based"),
    ("BACKDOOR", "Web-based"), ("BROWSERHIJACKING", "Web-based"), ("UPLOADING", "Web-based"),
)


def _squash(label: str) -> str:
    return re.sub(r"[^a-z0-9]", "", label.lower())


def is_benign_raw(label: str) -> bool:
    """Is this RAW dataset label a non-attack label?"""
    s = _squash(label)
    return any(s.startswith(p) for p in BENIGN_PREFIXES)


def is_benign(class_name: str) -> bool:
    """Is this already-normalised class name the non-attack class?"""
    return class_name == BENIGN_NAME


def _family_rule(label: str) -> Optional[str]:
    u = label.strip().upper().replace(" ", "_")
    for prefix, family in _PREFIX_RULES:
        if u.startswith(prefix):
            return family
    for key, family in _CONTAINS_RULES:
        if key in u:
            return family
    return None


def looks_like_ciciot(raw_names: Iterable[str], min_matched: float = 0.8) -> bool:
    """True if at least 80% of the attack labels match the CICIoT2023 grouping rules."""
    attacks = [n for n in raw_names if not is_benign_raw(n)]
    return bool(attacks) and sum(_family_rule(n) is not None for n in attacks) / len(attacks) >= min_matched


def class_order(names: Iterable[str]) -> List[str]:
    """Benign first (code 0), then alphabetical."""
    return sorted(set(names), key=lambda g: (not is_benign(g), g))


def resolve_class_names(raw_names: Sequence[str], group_labels: str = "auto") -> List[str]:
    """Raw labels -> modelling class names (same order as raw_names). group_labels: 'auto' | 'yes' | 'no'."""
    if group_labels not in ("auto", "yes", "no"):
        raise ValueError(f"group_labels must be auto|yes|no, got {group_labels!r}")
    group = group_labels == "yes" or (group_labels == "auto" and looks_like_ciciot(raw_names))
    out = []
    for n in raw_names:
        if is_benign_raw(n):
            out.append(BENIGN_NAME)
        else:
            out.append((_family_rule(n) or n.strip()) if group else n.strip())
    return out
