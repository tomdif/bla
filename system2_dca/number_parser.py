"""Robust problem-number extraction.

Naive `re.findall(r"-?\d+(?:\.\d+)?", problem)` has two failure modes:

  1. "$80,000" splits into ["80", "000"]
  2. "three apples" yields zero numbers — the spelled-out "three" is
     lost, so downstream perturbation tests can't perturb it.

This module fixes both:
  - merge comma-separated digits before regex
  - normalize common spelled-out numerals to digit strings
"""
from __future__ import annotations

import re


SPELLED = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000",
    "million": "1000000",
    "half": "0.5", "third": "0.333", "quarter": "0.25",
    "double": "2", "triple": "3", "twice": "2", "thrice": "3",
    "dozen": "12",
}


def extract_problem_numbers(text: str) -> list[str]:
    """Return all numeric quantities mentioned in the problem text.

    Combines:
      * digit literals (with optional decimals, commas stripped)
      * spelled-out numerals (mapped to their digit representation)
    """
    # 1. Strip dollar signs and commas inside numbers ("$80,000" -> "80000")
    cleaned = re.sub(r"\$", "", text)
    cleaned = re.sub(r"(\d),(\d{3})\b", r"\1\2", cleaned)
    cleaned = re.sub(r"(\d),(\d{3})\b", r"\1\2", cleaned)  # repeat for 1,000,000

    nums: list[str] = []

    # 2. Digit literals
    for m in re.finditer(r"\b(\d+(?:\.\d+)?)\b", cleaned):
        nums.append(m.group(1))

    # 3. Spelled-out
    for w in re.findall(r"[A-Za-z]+", cleaned):
        lw = w.lower()
        if lw in SPELLED:
            nums.append(SPELLED[lw])

    # Deduplicate preserving order
    seen: set[str] = set()
    out: list[str] = []
    for n in nums:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


# Backward-compat aliases
def problem_numbers(text: str) -> list[str]:
    return extract_problem_numbers(text)
