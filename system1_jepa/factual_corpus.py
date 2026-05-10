"""A small, curated factual corpus for Phase 3 retrieval audits.

Static data only — no network dependency, no LLM calls. Real Wikidata
ingestion (10⁵-10⁹ entities) is Phase 7's job. Phase 3 just needs enough
structured facts to validate the retrieval + provenance pipeline.

Three relation types covered:
  * "capital_of"   country → city
  * "orbits"       moon/planet → parent body
  * "atomic_number" element → integer (literal value)
"""

from __future__ import annotations

COUNTRY_CAPITALS = [
    ("France", "Paris", "wikidata:Q142#capital"),
    ("Germany", "Berlin", "wikidata:Q183#capital"),
    ("Italy", "Rome", "wikidata:Q38#capital"),
    ("Spain", "Madrid", "wikidata:Q29#capital"),
    ("Portugal", "Lisbon", "wikidata:Q45#capital"),
    ("United Kingdom", "London", "wikidata:Q145#capital"),
    ("Ireland", "Dublin", "wikidata:Q27#capital"),
    ("Netherlands", "Amsterdam", "wikidata:Q55#capital"),
    ("Belgium", "Brussels", "wikidata:Q31#capital"),
    ("Switzerland", "Bern", "wikidata:Q39#capital"),
    ("Austria", "Vienna", "wikidata:Q40#capital"),
    ("Sweden", "Stockholm", "wikidata:Q34#capital"),
    ("Norway", "Oslo", "wikidata:Q20#capital"),
    ("Finland", "Helsinki", "wikidata:Q33#capital"),
    ("Denmark", "Copenhagen", "wikidata:Q35#capital"),
    ("Iceland", "Reykjavik", "wikidata:Q189#capital"),
    ("Poland", "Warsaw", "wikidata:Q36#capital"),
    ("Czech Republic", "Prague", "wikidata:Q213#capital"),
    ("Slovakia", "Bratislava", "wikidata:Q214#capital"),
    ("Hungary", "Budapest", "wikidata:Q28#capital"),
    ("Romania", "Bucharest", "wikidata:Q218#capital"),
    ("Bulgaria", "Sofia", "wikidata:Q219#capital"),
    ("Greece", "Athens", "wikidata:Q41#capital"),
    ("Croatia", "Zagreb", "wikidata:Q224#capital"),
    ("Serbia", "Belgrade", "wikidata:Q403#capital"),
    ("Slovenia", "Ljubljana", "wikidata:Q215#capital"),
    ("Russia", "Moscow", "wikidata:Q159#capital"),
    ("Ukraine", "Kyiv", "wikidata:Q212#capital"),
    ("Belarus", "Minsk", "wikidata:Q184#capital"),
    ("Estonia", "Tallinn", "wikidata:Q191#capital"),
    ("Latvia", "Riga", "wikidata:Q211#capital"),
    ("Lithuania", "Vilnius", "wikidata:Q37#capital"),
    ("Turkey", "Ankara", "wikidata:Q43#capital"),
    ("Egypt", "Cairo", "wikidata:Q79#capital"),
    ("Morocco", "Rabat", "wikidata:Q1028#capital"),
    ("South Africa", "Pretoria", "wikidata:Q258#capital"),
    ("Nigeria", "Abuja", "wikidata:Q1033#capital"),
    ("Kenya", "Nairobi", "wikidata:Q114#capital"),
    ("Ethiopia", "Addis Ababa", "wikidata:Q115#capital"),
    ("Ghana", "Accra", "wikidata:Q117#capital"),
    ("China", "Beijing", "wikidata:Q148#capital"),
    ("Japan", "Tokyo", "wikidata:Q17#capital"),
    ("South Korea", "Seoul", "wikidata:Q884#capital"),
    ("North Korea", "Pyongyang", "wikidata:Q423#capital"),
    ("Mongolia", "Ulaanbaatar", "wikidata:Q711#capital"),
    ("India", "New Delhi", "wikidata:Q668#capital"),
    ("Pakistan", "Islamabad", "wikidata:Q843#capital"),
    ("Bangladesh", "Dhaka", "wikidata:Q902#capital"),
    ("Sri Lanka", "Colombo", "wikidata:Q854#capital"),
    ("Nepal", "Kathmandu", "wikidata:Q837#capital"),
    ("Thailand", "Bangkok", "wikidata:Q869#capital"),
    ("Vietnam", "Hanoi", "wikidata:Q881#capital"),
    ("Indonesia", "Jakarta", "wikidata:Q252#capital"),
    ("Philippines", "Manila", "wikidata:Q928#capital"),
    ("Malaysia", "Kuala Lumpur", "wikidata:Q833#capital"),
    ("Singapore", "Singapore", "wikidata:Q334#capital"),
    ("Saudi Arabia", "Riyadh", "wikidata:Q851#capital"),
    ("Iran", "Tehran", "wikidata:Q794#capital"),
    ("Iraq", "Baghdad", "wikidata:Q796#capital"),
    ("Israel", "Jerusalem", "wikidata:Q801#capital"),
    ("Lebanon", "Beirut", "wikidata:Q822#capital"),
    ("Syria", "Damascus", "wikidata:Q858#capital"),
    ("Jordan", "Amman", "wikidata:Q810#capital"),
    ("United Arab Emirates", "Abu Dhabi", "wikidata:Q878#capital"),
    ("Qatar", "Doha", "wikidata:Q846#capital"),
    ("Kuwait", "Kuwait City", "wikidata:Q817#capital"),
    ("United States", "Washington", "wikidata:Q30#capital"),
    ("Canada", "Ottawa", "wikidata:Q16#capital"),
    ("Mexico", "Mexico City", "wikidata:Q96#capital"),
    ("Cuba", "Havana", "wikidata:Q241#capital"),
    ("Brazil", "Brasilia", "wikidata:Q155#capital"),
    ("Argentina", "Buenos Aires", "wikidata:Q414#capital"),
    ("Chile", "Santiago", "wikidata:Q298#capital"),
    ("Peru", "Lima", "wikidata:Q419#capital"),
    ("Colombia", "Bogota", "wikidata:Q739#capital"),
    ("Venezuela", "Caracas", "wikidata:Q717#capital"),
    ("Ecuador", "Quito", "wikidata:Q736#capital"),
    ("Bolivia", "La Paz", "wikidata:Q750#capital"),
    ("Paraguay", "Asuncion", "wikidata:Q733#capital"),
    ("Uruguay", "Montevideo", "wikidata:Q77#capital"),
    ("Australia", "Canberra", "wikidata:Q408#capital"),
    ("New Zealand", "Wellington", "wikidata:Q664#capital"),
    ("Fiji", "Suva", "wikidata:Q712#capital"),
]


PLANET_ORBITS = [
    ("Mercury", "Sun", "wikidata:Q308#orbits"),
    ("Venus", "Sun", "wikidata:Q313#orbits"),
    ("Earth", "Sun", "wikidata:Q2#orbits"),
    ("Mars", "Sun", "wikidata:Q111#orbits"),
    ("Jupiter", "Sun", "wikidata:Q319#orbits"),
    ("Saturn", "Sun", "wikidata:Q193#orbits"),
    ("Uranus", "Sun", "wikidata:Q324#orbits"),
    ("Neptune", "Sun", "wikidata:Q332#orbits"),
    ("Pluto", "Sun", "wikidata:Q339#orbits"),
    ("Moon", "Earth", "wikidata:Q405#orbits"),
    ("Phobos", "Mars", "wikidata:Q3169#orbits"),
    ("Deimos", "Mars", "wikidata:Q3303#orbits"),
    ("Io", "Jupiter", "wikidata:Q3123#orbits"),
    ("Europa", "Jupiter", "wikidata:Q3134#orbits"),
    ("Ganymede", "Jupiter", "wikidata:Q3169#orbits"),
    ("Callisto", "Jupiter", "wikidata:Q3134#orbits"),
    ("Titan", "Saturn", "wikidata:Q2565#orbits"),
    ("Enceladus", "Saturn", "wikidata:Q2565#orbits"),
    ("Triton", "Neptune", "wikidata:Q2565#orbits"),
]


ELEMENT_ATOMIC_NUMBERS = [
    ("Hydrogen", 1), ("Helium", 2), ("Lithium", 3), ("Beryllium", 4),
    ("Boron", 5), ("Carbon", 6), ("Nitrogen", 7), ("Oxygen", 8),
    ("Fluorine", 9), ("Neon", 10), ("Sodium", 11), ("Magnesium", 12),
    ("Aluminum", 13), ("Silicon", 14), ("Phosphorus", 15), ("Sulfur", 16),
    ("Chlorine", 17), ("Argon", 18), ("Potassium", 19), ("Calcium", 20),
    ("Scandium", 21), ("Titanium", 22), ("Vanadium", 23), ("Chromium", 24),
    ("Manganese", 25), ("Iron", 26), ("Cobalt", 27), ("Nickel", 28),
    ("Copper", 29), ("Zinc", 30), ("Gallium", 31), ("Germanium", 32),
    ("Arsenic", 33), ("Selenium", 34), ("Bromine", 35), ("Krypton", 36),
    ("Silver", 47), ("Gold", 79), ("Mercury (element)", 80), ("Lead", 82),
    ("Uranium", 92), ("Plutonium", 94),
]


def all_facts() -> dict:
    """Returns the full corpus as a dict of relation → list of triples."""
    return {
        "capital_of": [
            {"subject": cap, "object": cou, "source": src.replace("#capital", "#capital_of")}
            for cou, cap, src in COUNTRY_CAPITALS
        ],
        "country_capital_is": [
            {"subject": cou, "object": cap, "source": src}
            for cou, cap, src in COUNTRY_CAPITALS
        ],
        "orbits": [
            {"subject": s, "object": o, "source": src}
            for s, o, src in PLANET_ORBITS
        ],
        "atomic_number": [
            {"subject": elem, "value": z, "source": f"periodic_table#{elem}"}
            for elem, z in ELEMENT_ATOMIC_NUMBERS
        ],
    }


def total_facts() -> int:
    return sum(len(v) for v in all_facts().values())
