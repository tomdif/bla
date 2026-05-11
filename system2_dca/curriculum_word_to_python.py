"""Word-problem → Python translation curriculum source.

Each example is (prompt: math problem, target: short Python program that
prints the answer). Designed to teach the model the 'simulate' routing
skill — translate a word problem into a deterministic computation that
an interpreter can run.

This pairs with the PAL eval (phase6_eval_pal.py) which executes the
generated Python at inference time. The 'asymmetric scaling' bet is:
a small model that learns this translation can leverage a calculator
(Python) to do math far better than a large model trying arithmetic
in its weights.

Templates mirror curriculum_word_math but emit Python instead of CoT.
"""

from __future__ import annotations

import random
from typing import Iterator

from .curriculum import CurriculumExample, CurriculumSource


NAMES = ["Alice", "Bob", "Carol", "Dan", "Eve", "Frank", "Grace", "Henry",
         "Iris", "Jack", "Karen", "Leo", "Maya", "Noah", "Olivia",
         "Pete", "Quinn", "Ruth", "Sam", "Tara", "Uma", "Victor"]
ITEMS = ["apples", "books", "marbles", "coins", "stickers", "pencils",
         "cookies", "cards", "bottles", "candies", "tickets", "stamps",
         "pebbles", "rings", "stones", "shells", "berries", "buttons"]


def _name(rng): return rng.choice(NAMES)
def _item(rng): return rng.choice(ITEMS)


def _t_add(rng):
    n1, n2 = rng.randint(2, 99), rng.randint(2, 99)
    a, b, item = _name(rng), _name(rng), _item(rng)
    while b == a:
        b = _name(rng)
    q = f"{a} has {n1} {item}. {b} gives {a} {n2} more {item}. How many {item} does {a} have now?"
    code = f"start = {n1}\nadded = {n2}\nanswer = start + added\nprint(answer)"
    return q, code, n1 + n2


def _t_sub(rng):
    n1 = rng.randint(10, 99)
    n2 = rng.randint(1, n1 - 1)
    a, item = _name(rng), _item(rng)
    q = f"{a} has {n1} {item}. {a} gives away {n2} {item}. How many {item} does {a} have left?"
    code = f"start = {n1}\ngiven_away = {n2}\nanswer = start - given_away\nprint(answer)"
    return q, code, n1 - n2


def _t_mul(rng):
    boxes = rng.randint(2, 20)
    per_box = rng.randint(2, 20)
    item = _item(rng)
    a = _name(rng)
    q = f"{a} has {boxes} boxes of {item}. Each box contains {per_box} {item}. How many {item} does {a} have in total?"
    code = f"boxes = {boxes}\nper_box = {per_box}\nanswer = boxes * per_box\nprint(answer)"
    return q, code, boxes * per_box


def _t_div(rng):
    per = rng.randint(2, 20)
    groups = rng.randint(2, 12)
    total = per * groups
    item = _item(rng)
    a = _name(rng)
    q = f"{a} has {total} {item} and wants to divide them equally among {groups} friends. How many {item} does each friend get?"
    code = f"total = {total}\ngroups = {groups}\nanswer = total // groups\nprint(answer)"
    return q, code, per


def _t_buy_remainder(rng):
    n = rng.randint(2, 12)
    price = rng.randint(2, 20)
    cost = n * price
    budget = cost + rng.randint(5, 50)
    left = budget - cost
    a = _name(rng)
    item = _item(rng)
    q = (f"{a} buys {n} {item} at ${price} each. {a} started with ${budget}. "
         f"How much money does {a} have left?")
    code = (f"n = {n}\nprice = {price}\nbudget = {budget}\n"
            f"cost = n * price\nanswer = budget - cost\nprint(answer)")
    return q, code, left


def _t_rate_distance(rng):
    speed = rng.randint(20, 100)
    hours = rng.randint(2, 12)
    dist = speed * hours
    a = _name(rng)
    q = f"{a} drives at {speed} miles per hour for {hours} hours. How many miles does {a} travel?"
    code = f"speed = {speed}\nhours = {hours}\nanswer = speed * hours\nprint(answer)"
    return q, code, dist


def _t_percent_off(rng):
    price = rng.randint(20, 500)
    pct = rng.choice([10, 20, 25, 30, 40, 50, 60, 75])
    discount = price * pct // 100
    final = price - discount
    item = _item(rng)
    q = (f"A bag of {item} costs ${price}. There is a {pct}% discount. "
         f"What is the discounted price in dollars?")
    code = (f"price = {price}\npct = {pct}\n"
            f"discount = price * pct // 100\nanswer = price - discount\nprint(answer)")
    return q, code, final


def _t_two_groups(rng):
    g1 = rng.randint(2, 15)
    p1 = rng.randint(2, 15)
    g2 = rng.randint(2, 15)
    p2 = rng.randint(2, 15)
    total = g1 * p1 + g2 * p2
    a, b = _name(rng), _name(rng)
    while b == a:
        b = _name(rng)
    item = _item(rng)
    q = (f"{a} has {g1} bags with {p1} {item} each. {b} has {g2} bags with {p2} {item} each. "
         f"How many {item} do they have in total?")
    code = (f"a_bags = {g1}\na_per = {p1}\nb_bags = {g2}\nb_per = {p2}\n"
            f"answer = a_bags * a_per + b_bags * b_per\nprint(answer)")
    return q, code, total


def _t_difference(rng):
    n1 = rng.randint(20, 200)
    n2 = rng.randint(1, n1 - 1)
    a, b = _name(rng), _name(rng)
    while b == a:
        b = _name(rng)
    item = _item(rng)
    q = f"{a} has {n1} {item}. {b} has {n2} {item}. How many more {item} does {a} have than {b}?"
    code = f"a = {n1}\nb = {n2}\nanswer = a - b\nprint(answer)"
    return q, code, n1 - n2


def _t_total_cost(rng):
    """Total cost of multiple item types."""
    n1 = rng.randint(2, 10)
    p1 = rng.randint(2, 20)
    n2 = rng.randint(2, 10)
    p2 = rng.randint(2, 20)
    total = n1 * p1 + n2 * p2
    a = _name(rng)
    item1, item2 = _item(rng), _item(rng)
    while item2 == item1:
        item2 = _item(rng)
    q = (f"{a} buys {n1} {item1} at ${p1} each and {n2} {item2} at ${p2} each. "
         f"How much does {a} spend in total?")
    code = (f"item1_n = {n1}\nitem1_p = {p1}\nitem2_n = {n2}\nitem2_p = {p2}\n"
            f"answer = item1_n * item1_p + item2_n * item2_p\nprint(answer)")
    return q, code, total


def _t_average(rng):
    """Sum then divide. Integer average via //."""
    n = rng.randint(3, 8)
    vals = [rng.randint(10, 100) for _ in range(n)]
    s = sum(vals)
    avg = s // n
    item = _item(rng)
    q = (f"Over {n} days, a shop sold the following number of {item}: {vals}. "
         f"What is the average number of {item} sold per day (rounded down)?")
    vals_str = "[" + ", ".join(str(v) for v in vals) + "]"
    code = (f"sales = {vals_str}\nn = {n}\n"
            f"answer = sum(sales) // n\nprint(answer)")
    return q, code, avg


TEMPLATES = [_t_add, _t_sub, _t_mul, _t_div, _t_buy_remainder, _t_rate_distance,
             _t_percent_off, _t_two_groups, _t_difference, _t_total_cost, _t_average]


class WordToPythonSource(CurriculumSource):
    """Synthetic word problem → Python solution curriculum.

    Each target is a short executable script that prints the answer.
    Format matches what the PAL eval expects: 'Python:\\n{code}\\nAnswer: N'.
    """

    name = "word_to_python"

    def __init__(self, n_examples: int = 10_000, seed: int = 0):
        self.n_examples = n_examples
        self.seed = seed

    def __iter__(self) -> Iterator[CurriculumExample]:
        rng = random.Random(self.seed)
        produced = 0
        while produced < self.n_examples:
            tmpl = rng.choice(TEMPLATES)
            try:
                q, code, ans = tmpl(rng)
            except Exception:
                continue
            prompt = (
                f"Write a Python program that prints the answer to this math problem.\n"
                f"End with: print(answer)\n"
                f"Problem: {q}\n"
                f"Python:\n"
            )
            target = f"{code}\nAnswer: {ans}"
            yield CurriculumExample(
                prompt=prompt, target=target, source=self.name,
                metadata={"template": tmpl.__name__, "answer": ans},
            )
            produced += 1
