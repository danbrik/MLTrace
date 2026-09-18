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

## Statistical & Temporal Characterization

The second tab shares the immutable quality-analysis parameters. It has a
separate versioned background job (`0056_characterization` migration), so existing
quality results remain valid. CSV normalization is shared through
`normalized_csv`; summary rows retain CSV header order. Numeric observations are
stored as compressed per-sensor index/value arrays, published atomically when the
job completes. Sensor histograms and dynamics are created lazily and cached next
to those arrays. Deleting an analysis removes all characterization files; active
jobs block deletion and interrupted jobs become retryable on startup.

The overview uses linear quantiles and sample standard deviation. Text sensors
have N valid only. Differences are taken strictly between adjacent valid grid
indices, never across a gap. `% unchanged` counts exact zero differences divided
by the number of valid pairs. Histogram classes use Freedman–Diaconis (1–100
classes), with a square-root fallback for zero IQR. Boxplots use Tukey whiskers
and report outlier counts without transmitting raw outlier arrays.

Time views return raw observations at <=2,000 expected grid points, otherwise at
most 1,000 bins with median and Q25–Q75. Connection flags prevent lines and bands
across missing observations. Internally discontinuous bins are isolated and show
vertical IQR whiskers. Zoom changes only the temporal view, not distributions or
dynamics over the full analysis period.

API prefix: `/api/data-quality/analyses/{id}/characterization`: GET for state,
POST to start, `/cancel` and `/retry` POST, `/detail?sensor=...`,
`/series?sensor=...&start=...&end=...`, and `/export` GET. Export accepts `search`,
`data_type`, `sort` and `descending` and includes the entire filtered table.
