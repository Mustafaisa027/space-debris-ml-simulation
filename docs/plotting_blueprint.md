# Scientific Plotting Blueprint

This project should prioritize conjunction-analysis plots that map directly to
observable or derived orbital quantities:

1. TCA-centered miss distance history.
   Shows whether the minimum is isolated, periodic, or sensitive to sampling.

2. Encounter-plane projection.
   Projects relative motion into the plane perpendicular to relative velocity at
   TCA. This is closer to collision-avoidance practice than generic scatter
   plots because it focuses on close-approach geometry.

3. RIC/RTN relative position components.
   Shows radial, in-track, and cross-track separation around TCA. This makes the
   orbital geometry interpretable to space-flight readers.

4. 3D ECI trajectory context.
   Useful for presentation, but should not be the only figure because the TCA
   separation can be visually hidden at Earth-orbit scale.

5. Candidate ranking table/plot.
   Useful for engineering triage, especially when covariance data is not
   available and the project uses transparent proxy risk labels.

6. Model metrics only after enough samples exist.
   For small smoke-test datasets, model plots must be marked preliminary or
   suppressed. Final IAC figures should use the verified, corrected-TCA
   `iac26-10d-v2` accumulated history and its pre-registered pair-held-out,
   chronological split.

Avoid:

- Heatmaps on fewer than roughly 30 samples.
- Smooth density plots that imply statistical support not present in the data.
- Claiming physical Pc without covariance, hard-body radius, and uncertainty
  modeling.
