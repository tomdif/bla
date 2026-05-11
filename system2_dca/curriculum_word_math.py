"""GSM8K-style word-math curriculum source.

Produces synthetic word problems with chain-of-thought reasoning and a
standardized 'Answer: N' final line. Designed to bridge the gap between
our procedural curriculum and standard benchmark formats.

Each example has the shape:
  prompt: "Solve this math problem step by step. End with 'Answer: <number>'.
           Problem: <story>
           Solution:"
  target: "Step 1: <op1>. <intermediate>. Step 2: <op2>. ... Answer: <N>"

Templates cover:
  - 1-step add/sub/mul/div
  - 2-step composite arithmetic
  - rate problems (distance, time, money)
  - percentage discount
  - unit conversion (multiples)
"""

from __future__ import annotations

import random
from typing import Iterator

from .curriculum import CurriculumExample, CurriculumSource


NAMES = ["Alice", "Bob", "Carol", "Dan", "Eve", "Frank", "Grace", "Henry",
         "Iris", "Jack", "Karen", "Leo", "Maya", "Noah", "Olivia"]
ITEMS = ["apples", "books", "marbles", "coins", "stickers", "pencils",
         "cookies", "cards", "bottles", "candies", "tickets"]
PLACES = ["the store", "the market", "the park", "school", "the library"]


def _name(rng): return rng.choice(NAMES)
def _item(rng): return rng.choice(ITEMS)


# ---------- 1-step templates ----------

def _t_add(rng):
    n1, n2 = rng.randint(2, 50), rng.randint(2, 50)
    a, b, item = _name(rng), _name(rng), _item(rng)
    while b == a:
        b = _name(rng)
    story = f"{a} has {n1} {item}. {b} gives {a} {n2} more {item}. How many {item} does {a} have now?"
    sol = (f"Step 1: {a} starts with {n1} {item}.\n"
           f"Step 2: {a} receives {n2} more, so total is {n1} + {n2}.\n"
           f"Step 3: {n1} + {n2} = {n1 + n2}.\n"
           f"Answer: {n1 + n2}")
    return story, sol, n1 + n2


def _t_sub(rng):
    n1 = rng.randint(10, 80)
    n2 = rng.randint(1, n1 - 1)
    a, item = _name(rng), _item(rng)
    story = f"{a} has {n1} {item}. {a} gives away {n2} {item}. How many {item} does {a} have left?"
    sol = (f"Step 1: {a} starts with {n1} {item}.\n"
           f"Step 2: {a} loses {n2}, so remaining is {n1} - {n2}.\n"
           f"Step 3: {n1} - {n2} = {n1 - n2}.\n"
           f"Answer: {n1 - n2}")
    return story, sol, n1 - n2


def _t_mul(rng):
    boxes = rng.randint(2, 12)
    per_box = rng.randint(2, 12)
    item = _item(rng)
    a = _name(rng)
    story = f"{a} has {boxes} boxes of {item}. Each box contains {per_box} {item}. How many {item} does {a} have in total?"
    sol = (f"Step 1: There are {boxes} boxes.\n"
           f"Step 2: Each box has {per_box} {item}, so total = {boxes} * {per_box}.\n"
           f"Step 3: {boxes} * {per_box} = {boxes * per_box}.\n"
           f"Answer: {boxes * per_box}")
    return story, sol, boxes * per_box


def _t_div(rng):
    per = rng.randint(2, 12)
    groups = rng.randint(2, 10)
    total = per * groups
    item = _item(rng)
    a = _name(rng)
    story = f"{a} has {total} {item} and wants to divide them equally among {groups} friends. How many {item} does each friend get?"
    sol = (f"Step 1: There are {total} {item} total.\n"
           f"Step 2: Divided equally among {groups} friends, each gets {total} / {groups}.\n"
           f"Step 3: {total} / {groups} = {per}.\n"
           f"Answer: {per}")
    return story, sol, per


# ---------- 2-step templates ----------

def _t_buy_remainder(rng):
    """Two-step: total cost, then money remaining."""
    n = rng.randint(2, 10)
    price = rng.randint(2, 10)
    budget = n * price + rng.randint(5, 30)
    cost = n * price
    left = budget - cost
    a = _name(rng)
    item = _item(rng)
    story = (f"{a} buys {n} {item} at ${price} each. {a} started with ${budget}. "
             f"How much money does {a} have left?")
    sol = (f"Step 1: Cost of {n} {item} at ${price} each = {n} * {price} = ${cost}.\n"
           f"Step 2: Money left = ${budget} - ${cost}.\n"
           f"Step 3: {budget} - {cost} = {left}.\n"
           f"Answer: {left}")
    return story, sol, left


def _t_rate_distance(rng):
    """Rate × time = distance."""
    speed = rng.randint(20, 80)
    hours = rng.randint(2, 8)
    dist = speed * hours
    a = _name(rng)
    story = f"{a} drives at {speed} miles per hour for {hours} hours. How many miles does {a} travel?"
    sol = (f"Step 1: Distance = speed * time = {speed} * {hours}.\n"
           f"Step 2: {speed} * {hours} = {dist}.\n"
           f"Answer: {dist}")
    return story, sol, dist


def _t_percent_off(rng):
    """Percentage discount on a price."""
    price = rng.randint(20, 200)
    pct = rng.choice([10, 20, 25, 50, 75])
    discount = price * pct // 100
    final = price - discount
    item = _item(rng)
    story = (f"A bag of {item} costs ${price}. There is a {pct}% discount. "
             f"What is the discounted price in dollars?")
    sol = (f"Step 1: Discount amount = {price} * {pct} / 100 = {discount}.\n"
           f"Step 2: Discounted price = {price} - {discount}.\n"
           f"Step 3: {price} - {discount} = {final}.\n"
           f"Answer: {final}")
    return story, sol, final


def _t_two_groups(rng):
    """Two groups with different rates, sum them."""
    g1 = rng.randint(2, 10)
    p1 = rng.randint(2, 10)
    g2 = rng.randint(2, 10)
    p2 = rng.randint(2, 10)
    total = g1 * p1 + g2 * p2
    a, b = _name(rng), _name(rng)
    while b == a:
        b = _name(rng)
    item = _item(rng)
    story = (f"{a} has {g1} bags with {p1} {item} each. {b} has {g2} bags with {p2} {item} each. "
             f"How many {item} do they have in total?")
    sol = (f"Step 1: {a}'s total = {g1} * {p1} = {g1 * p1}.\n"
           f"Step 2: {b}'s total = {g2} * {p2} = {g2 * p2}.\n"
           f"Step 3: Combined = {g1 * p1} + {g2 * p2} = {total}.\n"
           f"Answer: {total}")
    return story, sol, total


def _t_difference(rng):
    """Comparison: A has X, B has Y, how many more does A have?"""
    n1 = rng.randint(20, 80)
    n2 = rng.randint(1, n1 - 1)
    a, b = _name(rng), _name(rng)
    while b == a:
        b = _name(rng)
    item = _item(rng)
    story = f"{a} has {n1} {item}. {b} has {n2} {item}. How many more {item} does {a} have than {b}?"
    sol = (f"Step 1: {a} has {n1}, {b} has {n2}.\n"
           f"Step 2: Difference = {n1} - {n2}.\n"
           f"Step 3: {n1} - {n2} = {n1 - n2}.\n"
           f"Answer: {n1 - n2}")
    return story, sol, n1 - n2


TEMPLATES = [_t_add, _t_sub, _t_mul, _t_div, _t_buy_remainder,
             _t_rate_distance, _t_percent_off, _t_two_groups, _t_difference]


class WordMathSource(CurriculumSource):
    """Synthetic GSM8K-style word problems with chain-of-thought solutions
    and standardized 'Answer: N' format."""

    name = "word_math"

    def __init__(self, n_examples: int = 10_000, seed: int = 0):
        self.n_examples = n_examples
        self.seed = seed

    def __iter__(self) -> Iterator[CurriculumExample]:
        rng = random.Random(self.seed)
        produced = 0
        while produced < self.n_examples:
            tmpl = rng.choice(TEMPLATES)
            try:
                story, sol, ans = tmpl(rng)
            except Exception:
                continue
            prompt = (f"Solve this math problem step by step. End with: 'Answer: <number>'.\n"
                      f"Problem: {story}\nSolution:")
            yield CurriculumExample(
                prompt=prompt, target=sol, source=self.name,
                metadata={"template": tmpl.__name__, "answer": ans},
            )
            produced += 1
