"""Which residue classes c mod 7 of the Hecke–Rogers series are 'single-orbit' at ζ7?"""
from collections import defaultdict
p = 7
def orbit(j): return min(j % p, (-j) % p)
# (2.19): Σ_{n≥0} Σ_{|j|≤n} (-1)^j z^j q^{n(3n+1)/2 - j^2} (1 - q^{2n+1})
h19 = defaultdict(set)
for n in range(0, 60):
    for j in range(-n, n + 1):
        for e in (n * (3 * n + 1) // 2 - j * j, n * (3 * n + 1) // 2 - j * j + 2 * n + 1):
            h19[e % p].add(orbit(j))
print("(2.19) class -> z-exponent orbits (|j| mod 7):", dict(sorted(h19.items())))
# (2.18): z^{n-3j} q^{(n^2-3j^2)/2+(n-j)/2}, and z^{n-3j+1} q^{(n^2-3j^2)/2+(n+j)/2}, j ranges, both signs symmetric
h18 = defaultdict(set)
for n in range(0, 80):
    for j in range(0, n // 2 + 1):
        e = (n * n - 3 * j * j) // 2 + (n - j) // 2
        if (n * n - 3 * j * j + n - j) % 2 == 0: h18[e % p].add(orbit(n - 3 * j))
    for j in range(1, n // 2 + 1):
        e = (n * n - 3 * j * j + n + j) // 2
        h18[e % p].add(orbit(n - 3 * j + 1))
print("(2.18) class -> orbits:", dict(sorted(h18.items())))
# base-q^2 version of (2.18): classes of F(q^2)R(ζ;q^2) -> class c in q^2-exponent 2e
h18b = defaultdict(set)
for c, o in h18.items(): h18b[(2 * c) % p] |= o
print("(2.18) at q^2: class ->", dict(sorted(h18b.items())))
