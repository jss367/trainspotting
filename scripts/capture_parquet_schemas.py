"""Re-capture the Parquet schemas that tests/test_grep.py checks against.

One schema per (model, stage) in the registry, read from the first shard of the
datasets-server's Parquet conversion of each mix.

These are what makes an upstream column rename a test failure rather than a
silently smaller count. `trainspotting grep` maps columns to the part of the
example they belong to by name: a mix that renames `ground_truth`, or adds a
text column the mapping has never seen, would keep returning a number — just a
number over less text than the caller thinks. The saved schemas pin the mapping
against real shapes, and the goldens below record what each one resolved to.

Refresh deliberately (`python scripts/capture_parquet_schemas.py`) and read the
diff. Reading footers only; no dataset rows are fetched.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trainspotting import grep, registry  # noqa: E402

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "schemas"


def main():
    FIXTURES.mkdir(parents=True, exist_ok=True)
    con = grep.connect()
    for model_name, model in registry.MODELS.items():
        for stage in registry.post_training_stages(model):
            dataset = stage["hf_dataset"]
            listing = grep.parquet_listing(dataset)
            schema = grep.schema(con, listing["urls"][0])
            exprs, leaves, unsearched = grep.text_fields(schema)
            _, source_column = grep.source_expr(schema, stage["source_columns"])
            path = FIXTURES / f"{model_name}.{stage['stage']}.json"
            path.write_text(
                json.dumps(
                    {
                        "model": model_name,
                        "stage": stage["stage"],
                        "dataset": dataset,
                        # Not the revision: it moves whenever the server
                        # reconverts, and pinning it here would turn every
                        # reconversion into a failing test about nothing.
                        "shards": len(listing["urls"]),
                        "partial": listing["partial"],
                        "schema": schema,
                        "groups": {g: len(e) for g, e in exprs.items()},
                        "leaves": [[c, sub] for c, sub in leaves],
                        "unsearched": unsearched,
                        "source_column": source_column,
                    },
                    indent=2,
                )
                + "\n"
            )
            print(f"{path.name}: {len(schema)} columns, groups {sorted(exprs)}")


if __name__ == "__main__":
    main()
