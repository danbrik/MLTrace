# Data Quality Analysis

Uses the immutable CSV resources and uploads shared with Redundancy Analysis.
Results are project-local and keyed by source SHA-256, canonical parameters and
engine version. Display thresholds do not affect the calculation cache.

The source-wide earliest timestamp anchors the expected grid. Range bounds are
inclusive. Off-grid rows are diagnosed, never rounded. Duplicate timestamps are
resolved to the first valid observation **per sensor**, in source order; distinct
conflicting values are reported. Missingness includes absent CSV rows.

Longest gap is the number of consecutive missing expected observations multiplied
by the sampling interval. Three missing minute-grid observations equal 3 minutes,
including at either range boundary. Numeric statistics use only deduplicated
finite observations, linear quantiles and sample standard deviation (ddof=1).
Text sensors retain completeness, gaps, unique counts and constant detection.

Observation spools use temporary files and statistics process one sensor at a
time. Missingness is persisted as half-open grid-index intervals. Heatmap queries
return at most 1,000 bins; aggregated cells contain missing fractions rather than
binary flags. Zooming requests a narrower viewport without rereading the CSV.

API: `/api/data-quality/analyses` (GET/POST), `/lookup` (POST), `/{id}`
(GET/DELETE), `/{id}/cancel`, `/{id}/retry` (POST), `/{id}/heatmap` (GET,
optional inclusive `start`/`end` in CSV-local time). Job responses exclude raw
missing intervals. Background jobs report step, progress, elapsed time and an
estimated remaining time. Startup marks interrupted jobs as retryable failures.

Migration: `0055_data_quality`; the normal application startup migrates projects.
