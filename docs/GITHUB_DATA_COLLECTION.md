# GitHub Actions data collection

`.github/workflows/collect_observations.yml` runs one CelesTrak collection at
minute 17 of every second UTC hour. It stores the results on the repository's
separate `data-collection` branch, so generated observations do not clutter the
source-code history on `main`.

## Archive layout

Each workflow execution gets an immutable directory:

```text
collections/github-run-RUN_ID-attempt-ATTEMPT/
  history/conjunction_observations_iac26_75_v1.csv
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

After these changes are merged to the default branch, every newly generated
authoritative bundle uses manifest schema 2. Manifest generation fails closed
unless a valid `environment/runtime.json` is present; it never downgrades a new
bundle to schema 1. The three bundles present as of 2026-07-17 were created by
source commit `1d7e272` before runtime capture was deployed, so their schema-1
manifests remain importable for backward compatibility and are explicitly
reported as `legacy_runtime_missing`. Schema 1 acceptance exists only in the
importer as a backward-compatible path for pre-runtime bundles; it is not
produced by the updated workflow.

Those first three bundles contain the same TLE text SHA-256. They occupy three
different snapshot-time bins because their propagation windows differ, but
they are not three independent orbital-element updates. The resimulation
quality report therefore records both occupied bins and unique TLE-input hashes
instead of treating raw snapshot count as information diversity.

The first successful run creates `data-collection` automatically. Later runs
append a new directory and never rewrite an earlier bundle. The workflow uses
`concurrency` to ensure that only one collector writes at a time and fails
closed if a run/attempt destination already exists.

## Verified local import

Check out the generated branch into a separate worktree and run the importer:

```powershell
git fetch origin data-collection
git worktree add ..\space-debris-data origin/data-collection
python src/import_collection_archive.py ..\space-debris-data
python src/resimulate_snapshots.py --config config/experiment_60_days.json
```

The importer first requires the archive path to be the root of a clean Git
checkout/worktree and records its exact HEAD revision. It then recomputes every
manifest size/hash, rejects unlisted files, unsafe paths, partial/wrong
catalogue cohorts and duplicate collection IDs, then atomically copies only
TLE and per-run simulation artifacts into the flat canonical paths from
`experiment_60_days.json`. Existing byte-identical files are an idempotent
no-op; a same-name/different-content collision aborts before new files are
copied. Per-bundle live history files are not concatenated: the publication
history is rebuilt from corrected-TCA re-simulation. Before claim-eligible
training, every imported bundle is bound one-to-one to the resimulation report
by `collection_id` and the verified TLE-file SHA-256; verify-only, unspecified
revision, missing, extra, or hash-mismatched inputs stop the evidence pipeline.

The final IAC experiment begins with catalogue cohort
`iac26-leo-mixed-75-v1`. Older local 5/43-object snapshots remain an audit
archive but are excluded from this 60-day cohort by the sidecar
`catalog_version` and exact catalog-ID-set SHA-256; they must not be mixed into
the final model history. The authoritative collector rejects partial coverage
even though non-frozen exploratory fetches may use the general 90% availability
floor.

The canonical half-open collection window is frozen in
`config/experiment_60_days.json` as `2026-07-16T16:17:00Z` through
`2026-09-14T16:17:00Z`, anchored to the pre-specified `17 */2 * * *` UTC cron
grid. Scheduled invocations outside that window exit
successfully without fetching or publishing data. Evaluation counts occupied
two-hour cadence slots (so retries do not inflate coverage), requires at least
90% of the 720 expected slots, rejects an actual snapshot-time gap above six
hours (including start/end boundaries), requires at least 50% unique TLE hashes
among occupied bins, and rejects more than 12 consecutive bins with one hash.

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
the frozen interval into half-open two-hour bins anchored at `16:17Z`; a delayed
run stays in the bin where the observation actually occurred, and retries in
the same bin count only once. Public repositories can also
have scheduled workflows disabled after 60 days without repository activity;
check the Actions page periodically.

The separate **Check collection cadence health** workflow checks out both the
default branch and `data-collection` every two hours. During the active window,
it fails when the newest schema-2 manifest is more than the configured
`max_snapshot_gap_hours` old. This failure is an early operational alert; it
does not substitute manifest time for `snapshot_utc` in the scientific
coverage calculation.

After the window closes, use the resumable finalizer from the source checkout:

```powershell
.\scripts\finalize_iac_experiment.ps1 `
  -ArchiveRoot ..\space-debris-data `
  -Workers 4
```

It refuses pre-window finalization, imports only a clean committed archive,
re-simulates immutable snapshots in parallel, and then runs the fail-closed
training/publication entry point.

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
