# GitHub Actions data collection

> Active protocol: `iac26-10d-v2`, 2026-07-24T00:17:00Z through
> 2026-08-03T00:17:00Z. Bundles use manifest schema 3 and are stored beneath
> `experiments/iac26-10d-v2/collections/`. The older root-level
> `collections/` tree is the isolated v1 pilot archive and is never imported
> into v2. See `docs/EXPERIMENT_10D_V2.md` for the frozen scientific contract.

`.github/workflows/collect_observations.yml` offers delivery opportunities at
minutes `17` and `47` of every UTC hour. An archive-backed slot guard permits
at most one actual CelesTrak poll in each frozen two-hour scientific slot and
also requires two elapsed hours since the latest archived fetch. The latter
protects CelesTrak's one-download-per-update policy when a delayed fetch lands
just before a slot boundary. The redundant triggers tolerate delayed GitHub
cron delivery without changing sample weighting. Results are stored on the
repository's separate `data-collection` branch, so generated observations do
not clutter the source-code history on `main`.

## Archive layout

Each workflow execution gets an immutable directory:

```text
experiments/iac26-10d-v2/collections/github-run-RUN_ID-attempt-ATTEMPT/
  experiment/config.json
  history/conjunction_observations_iac26_75_v2.csv
  runs/TIMESTAMP/conjunction_dataset.csv
  runs/TIMESTAMP/identified_conjunctions.csv
  tle/tles_TIMESTAMP.txt
  tle/tles_TIMESTAMP.json
  environment/runtime.json
  manifest.json
```

`manifest.json` contains the source Git commit, GitHub run identifiers, byte
sizes, and SHA-256 digest of every file. `environment/runtime.json` records the
exact Python, platform and installed-package versions used by that numerical
run. The workflow verifies that TLE, history, and simulation outputs exist
before publishing anything.

Every claim-eligible v2 bundle uses manifest schema 3. Manifest generation
fails closed unless a valid `environment/runtime.json` and the exact embedded
experiment config are present. The manifest binds the experiment ID, canonical
config SHA-256, catalogue, simulation settings, source commit, runtime and
every payload byte. Schema 1/2 acceptance exists only in the isolated v1
compatibility path; the v2 workflow never produces or imports those schemas.

The retained first three v1 bundles contain the same TLE text SHA-256. They are
pilot/audit artifacts, not three independent orbital-element updates and not
members of the v2 cohort. The v2 scientific-progress report records both
occupied bins and unique TLE-input hashes instead of treating raw snapshot
count as information diversity.

The first successful run creates `data-collection` automatically. Later runs
append a new directory and never rewrite an earlier bundle. The workflow uses
`concurrency` to ensure that only one collector writes at a time and fails
closed if a run/attempt destination already exists.

## Verified local import

Check out the generated branch into a separate worktree and run the importer:

```powershell
git fetch origin data-collection
git worktree add ..\space-debris-data origin/data-collection
python src/import_collection_archive.py ..\space-debris-data --config config/experiment_10_days_v2.json
python src/resimulate_snapshots.py --config config/experiment_10_days_v2.json --workers 4
```

The importer first requires the archive path to be the root of a clean Git
checkout/worktree and records its exact HEAD revision. It then recomputes every
manifest size/hash, rejects unlisted files, unsafe paths, partial/wrong
catalogue cohorts and duplicate collection IDs, then atomically copies only
TLE and per-run simulation artifacts into the flat canonical paths from
`experiment_10_days_v2.json`. Existing byte-identical files are an idempotent
no-op; a same-name/different-content collision aborts before new files are
copied. Per-bundle live history files are not concatenated: the publication
history is rebuilt from corrected-TCA re-simulation. Before claim-eligible
training, every imported bundle is bound one-to-one to the resimulation report
by `collection_id` and the verified TLE-file SHA-256; verify-only, unspecified
revision, missing, extra, or hash-mismatched inputs stop the evidence pipeline.

The claim-eligible catalogue is `iac26-leo-mixed-75-v2`. Older v1 and local
5/43-object snapshots remain audit data and are excluded by experiment ID,
manifest schema, catalogue version and exact sorted-ID SHA-256. They must not
be mixed into the final model history. The authoritative collector requires
the complete frozen 75-object cohort.

The canonical half-open window is frozen in
`config/experiment_10_days_v2.json` as `2026-07-24T00:17:00Z` through
`2026-08-03T00:17:00Z`. Scheduled invocations outside that window exit
successfully without fetching or publishing data. Evaluation counts occupied
two-hour cadence slots (so retries do not inflate coverage), requires at least
108 of 120 slots, rejects an endpoint-inclusive actual snapshot-time gap above
six hours, requires at least 30% unique TLE hashes among occupied bins and
rejects more than six consecutive bins with one hash.

## Repository setting

In **Settings > Actions > General > Workflow permissions**, select
**Read and write permissions**. The workflow declares `contents: write`, but
the repository setting must also permit `GITHUB_TOKEN` to push the data branch.
No cloud account, payment card, access key, or repository secret is required.

The default `--provider auto` mode uses CelesTrak. If a credentialed fallback
is desired, add repository secrets `SPACETRACK_IDENTITY` and
`SPACETRACK_PASSWORD`, then expose them as environment variables in the
collection step. A failed/partial provider response still fails the workflow;
it is never published as a successful bundle.

If branch protection or a ruleset blocks bot pushes to `data-collection`, add a
narrow exception for this workflow or exclude only that generated-data branch.
Do not weaken protection on `main`.

## First run

1. Commit and push the workflow to the default branch.
2. Open **Actions > Collect TLE observations > Run workflow**.
3. Confirm the job succeeds.
4. Confirm that the new `data-collection` branch contains a collection folder
   and that its `manifest.json` file lists every uploaded file.
5. Leave the scheduled trigger enabled.

GitHub scheduled workflows can be delayed under load. Each bundle's recorded
`snapshot_utc` is authoritative for the physical observation. Coverage divides
the frozen interval into half-open two-hour bins anchored at the configured v2
start, `2026-07-24T00:17:00Z`; a delayed
run stays in the bin where the observation actually occurred, and retries in
the same bin count only once. Public repositories can also
have scheduled workflows disabled after 60 days without repository activity;
check the Actions page periodically.

The separate **Check collection cadence health** workflow checks out both the
default branch and `data-collection` every hour at `07` minutes past the hour,
after the most recent `:47` collector opportunity.
During the active window, it fails when the newest schema-3 manifest is more
than the configured
`max_snapshot_gap_hours` old. This failure is an early operational alert; it
does not substitute manifest time for `snapshot_utc` in the scientific
coverage calculation.

The same job emits a separate `scientific_progress` object derived only from
fully verified schema-3 bundles. During collection it reports total and
currently-due slot coverage, endpoint-inclusive observed gap, unique TLE hash
fraction, longest identical-hash run, and whether the frozen 90% final coverage
is still mathematically reachable. At the window end these fields are checked
against the same publication-window implementation in regression tests.
Operational manifest liveness and scientific snapshot quality remain distinct.

After the window closes, use the resumable finalizer from the source checkout:

```powershell
.\scripts\finalize_iac_experiment.ps1 `
  -ArchiveRoot ..\space-debris-data `
  -Workers 4
```

It refuses pre-window finalization, imports only a clean committed archive,
re-simulates immutable snapshots in parallel, and then runs the fail-closed
training/publication entry point. A checkpoint journal under `outputs/history`
binds completed import and resimulation stages to the exact config SHA-256,
archive commit and complete manifest set. A later invocation reuses a stage
only while every recorded output hash/tree hash still matches. A changed
archive or config is rejected instead of silently resuming stale work.

## Existing local archive

The current local archive should be imported once under a distinct prefix such
as `historical/local-import-20260712/`, with its own SHA-256 manifest. Keep the
local copy until file counts, total bytes, and sample hashes have been verified
on GitHub. Do not append the growing history CSV on every scheduled run;
the scheduled workflow intentionally stores independent per-run bundles.

## Failure policy

The collector validates the TLE line numbers, matching catalogue identifiers,
69-character width, modulo-10 checksums, response type, response size, and at
least 90% catalogue coverage. Files and accumulated history are replaced
atomically only after validation. In one-shot/GitHub mode any error produces a
non-zero exit code, so verification and publishing do not run. This cannot
prevent an external service outage, but it prevents an outage or malformed
response from silently contaminating the archive.
