# Cardinality Audit Protocol

## Objective
Evaluate whether multi-view anomaly aggregation performance can be explained by view-count differences.

## Registered controls

- Size-only null control: preserve group size and labels while replacing scores with IID noise.
- Equal-view evaluation: compare groups using the same number of sampled views.
- Fixed-order aggregation: use constant-order subset kernels.

## Registered aggregators

mean, max, top-2, top-3, quantile, log-sum-exp, U2, U3, U4.

## Boundaries

This protocol evaluates aggregation validity. It does not claim detector superiority.
