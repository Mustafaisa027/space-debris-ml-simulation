# GitHub Actions data collection

`.github/workflows/collect_observations.yml` runs one CelesTrak collection at
minute 17 of every second UTC hour. It stores the results on the repository's
separate `data-collection` branch, so generated observations do not clutter the
source-code history on `main`.

## Archive layout

Each workflow execution gets an immutable directory:

```text
collections/github-run-RUN_ID-attempt-ATTEMPT/
  history/conjunction_observations_v3.csv
  runs/TIMESTAMP/conjunction_dataset.csv
  runs/TIMESTAMP/identified_conjunctions.csv
  tle/tles_TIMESTAMP.txt
  tle/tles_TIMESTAMP.json
  manifest.json
```

`manifest.json` contains the source Git commit, GitHub run identifiers, byte
sizes, and SHA-256 digest of every file. The workflow verifies that TLE,
history, and simulation outputs exist before publishing anything.

The first successful run creates `data-collection` automatically. Later runs
append a new directory and never rewrite an earlier bundle. The workflow uses
`concurrency` to ensure that only one collector writes at a time.

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

GitHub scheduled workflows can be delayed under load, so the timestamps stored
inside each collection—not the nominal cron time—are authoritative. Public
repositories can also have scheduled workflows disabled after 60 days without
repository activity; check the Actions page periodically.

## Existing local archive

The current local archive should be imported once under a distinct prefix such
as `historical/local-import-20260712/`, with its own SHA-256 manifest. Keep the
local copy until file counts, total bytes, and sample hashes have been verified
on GitHub. Do not append the growing 17.5 MB history CSV on every scheduled run;
the scheduled workflow intentionally stores independent per-run bundles.

## Failure policy

The collector validates the TLE line numbers, matching catalogue identifiers,
69-character width, modulo-10 checksums, response type, response size, and at
least 90% catalogue coverage. Files and accumulated history are replaced
atomically only after validation. In one-shot/GitHub mode any error produces a
non-zero exit code, so verification and publishing do not run. This cannot
prevent an external service outage, but it prevents an outage or malformed
response from silently contaminating the archive.
