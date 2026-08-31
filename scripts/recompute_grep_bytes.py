"""Recompute `bytes_read` for every committed grep run, from the shard footers.

`bytes_read` is not a product of the scan: byte_cost() reads only Parquet
footers, so the value a re-run would write can be reproduced without moving the
data again. Everything else in these files is left exactly as the scan wrote it.
Pinned to each run's recorded revision so the figure belongs to the tree the run
actually read.
"""
import json, sys, glob
sys.path.insert(0, ".")
from trainspotting import grep

con = grep.connect()
listings = {}
changed = []
for path in sorted(glob.glob("results/*.grep-*.json")):
    run = json.loads(open(path).read())
    ds, rev = run["dataset"], run["revision"]
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
print(f"\n{len(changed)} of 12 updated")
