# Orbital-element source policy

## Decision

For the catalogue used by paper 114764, use CelesTrak GP as the primary source
and Space-Track `gp` as an optional authenticated fallback. Both paths enter
the same strict TLE validator before any file is committed. Provider fallback
improves availability; it does not make two independently measured datasets,
because both ultimately distribute public US catalogue products.

The authoritative `leo_mixed` preset contains 75 explicit five-digit NORAD
IDs. Its 32 Cosmos 2251 fragments were selected from CelesTrak's official
`COSMOS 2251 Debris` table on 2026-07-16 only when a current GP solution and
perigee above 400 km were present. This dated selection must be revalidated
before a later final collection; decayed/missing objects are not silently
replaced with unrelated catalogue members. Frozen-cohort collection requires
all 75 IDs exactly and records/verifies the sorted ID-set SHA-256
`64c3d2329281da0e246226cb552aafe3b961c553c9487d7cfd9c96374aaa1f4c`.

## Sources considered

| Source | Role in this project | Decision |
| --- | --- | --- |
| CelesTrak GP | Current public GP elements, group and CATNR queries | Primary |
| Space-Track `gp` | Authenticated current GP catalogue; explicit CATNR batch | Fallback |
| SatNOGS DB | Satellite/transmitter/community metadata | Metadata cross-check only |
| ESA DISCOS | Object characteristics and mission metadata | Metadata only; it does not distribute surveillance ephemerides |

Do not scrape web pages and do not mix unverifiable TLE mirrors into the
training archive. Adding more mirrors can increase uptime while decreasing
traceability and consistency.

## Format choice

The current SGP4 implementation consumes classic 3-line TLE blocks because
that is the representation named in the abstract and all curated NORAD IDs are
below 100000. CelesTrak and Space-Track also expose OMM. OMM XML should be the
next ingestion format when the catalogue expands: it carries named fields,
units/schema context, and supports catalogue identifiers that classic TLE's
fixed-width field cannot represent. CSV/JSON provider defaults must not be
trusted implicitly; every request must state its desired format.

## Operational contract

1. Fetch at most once every two hours, with bounded retries and jitter.
2. Request an explicit curated CATNR list for reproducible scientific runs.
3. Reject HTML, oversized payloads, malformed lines, checksum failures,
   mismatched IDs, and catalogue coverage below 90%.
4. Write through a temporary file and atomically replace only after validation.
5. Record provider, returned catalogue IDs, requested count, timestamp, commit,
   and SHA-256 manifest in provenance.
6. In GitHub Actions, fail the job on any collection error and publish only
   after output verification and manifest generation.
7. Never append a dataset whose columns differ from the history header.
8. Exclude station-attached/docked modules and vehicles from independent-object
   conjunction labels; preserve and count filtered historical rows for audit.

This contract guarantees archive integrity under detected failures. It cannot
guarantee that an external provider or GitHub will always be online; a missed
cycle is deliberately preferred to corrupted scientific data.
