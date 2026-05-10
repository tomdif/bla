"""Ingest the canned factual corpus into a SymbolicMemory store.

Phase 3 ingestion path. Wikidata-100K (Phase 7) will replace this with a
real RDF dump pipeline; the contract stays the same:
  * Each entity carries a type and an embedding.
  * Each triple carries a source_ref pointing back to its origin.
  * The graph layer auto-handles contradictions.

Run:
    python3 scripts/ingest_facts.py --db runs/phase3_facts/symbolic.db
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from system1_jepa.factual_corpus import all_facts, total_facts
from system2_dca.symbolic_memory import SymbolicMemory


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="runs/phase3_facts/symbolic.db")
    p.add_argument("--limit", type=int, default=None,
                   help="Only ingest first N triples per relation (smoke).")
    p.add_argument("--no-embed", action="store_true",
                   help="Skip computing embeddings (faster, no similarity retrieval).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.db)), exist_ok=True)
    if os.path.exists(args.db):
        os.remove(args.db)
    sm = SymbolicMemory(db_path=args.db)
    facts = all_facts()
    t0 = time.time()
    entity_cache: dict[tuple[str, str], str] = {}

    def get_or_create(name: str, etype: str) -> str:
        key = (name, etype)
        if key in entity_cache:
            return entity_cache[key]
        emb = None if args.no_embed else sm.embed_text(name)
        ent_id = sm.add_typed_entity(name, entity_type=etype, embedding=emb)
        entity_cache[key] = ent_id
        return ent_id

    triple_count = 0
    contradictions = 0

    for relation, triples in facts.items():
        items = triples if args.limit is None else triples[: args.limit]
        for tr in items:
            subj_type = _entity_type_for(relation, role="subject")
            obj_type = _entity_type_for(relation, role="object")
            subj_id = get_or_create(tr["subject"], subj_type)
            if "object" in tr:
                obj_id = get_or_create(tr["object"], obj_type)
                tid, contras = sm.add_typed_triple(
                    subject_id=subj_id, predicate=relation,
                    object_id=obj_id, source_ref=tr["source"],
                )
            else:
                tid, contras = sm.add_typed_triple(
                    subject_id=subj_id, predicate=relation,
                    object_value=str(tr["value"]), source_ref=tr["source"],
                )
            triple_count += 1
            contradictions += len(contras)

    elapsed = time.time() - t0
    print(json.dumps({
        "event": "ingest_done",
        "db": args.db,
        "total_triples_input": total_facts(),
        "triples_inserted": triple_count,
        "entities": len(entity_cache),
        "contradictions_detected": contradictions,
        "elapsed_s": round(elapsed, 1),
        "with_embeddings": not args.no_embed,
    }, indent=2))


def _entity_type_for(relation: str, role: str) -> str:
    table = {
        "capital_of": {"subject": "city", "object": "country"},
        "country_capital_is": {"subject": "country", "object": "city"},
        "orbits": {"subject": "celestial_body", "object": "celestial_body"},
        "atomic_number": {"subject": "element", "object": None},
    }
    return table.get(relation, {}).get(role) or "entity"


if __name__ == "__main__":
    main()
