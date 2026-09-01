"""Recompute `bytes_read` for every committed grep run, from the shard footers.

`bytes_read` is not a product of the scan: byte_cost() reads only Parquet
footers, so the value a re-run would write can be reproduced without moving the
data again. Everything else in these files is left exactly as the scan wrote it.
Pinned to each run's recorded revision so the figure belongs to the tree the run
actually read.
"""
import glob
import json
import sys
sys.path.insert(0, ".")
from trainspotting import grep

con = grep.connect()
listings = {}
changed = []
paths = sorted(glob.glob("results/*.grep-*.json"))
for path in paths:
    run = json.loads(open(path).read())
    # The Parquet-branch revision, not `revision`. Since the merge with main,
    # `revision` is the *dataset* commit that `_stamp` records, and substituting
    # it into a Parquet-branch URL 404s — which is how this was caught, loudly,
    # rather than by silently pricing the wrong tree.
    ds, rev = run["dataset"], run["parquet_revision"]
    if ds not in listings:
        listings[ds] = grep.parquet_listing(ds)
    live = listings[ds]
    urls = [u.replace(live["revision"], rev) for u in live["urls"]]
    assert len(urls) == run["shards"], f"{path}: shard count moved"
    schema = grep.schema(con, urls[0])
    # Same function the CLI prices its plan with. This script used to compute the
    # leaves itself and the two drifted: the CLI learned to clamp the source
    # leaf's multiplicity and this kept appending, so a maintenance run would
    # have written back the inflated figure the CLI had just stopped producing.
    leaves = grep.plan_leaves(schema, run["fields"], run["source_column"])
    got = grep.byte_cost(con, urls, leaves)
    if got != run["bytes_read"]:
        changed.append((path, run["bytes_read"], got))
        run["bytes_read"] = got
        open(path, "w").write(json.dumps(run, indent=2))
for p, old, new in changed:
    print(f"{p.split('/')[-1]:58s} {old} -> {new}  (+{new-old})")
print(f"\n{len(changed)} of {len(paths)} updated")
