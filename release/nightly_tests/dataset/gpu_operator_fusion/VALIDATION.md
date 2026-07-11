# GPU operator fusion validation

The implementation was validated at commit
`e05da7b3132296c9420fa1f77efbd8cb68cae422` against stock Ray commit
`2741c6461d2bd3e5ff114af67be7a1190453dadd`.

## Source and GPU tests

- 44 public API, contract, composition, physical-fusion, provider, planner, and
  malformed-fragment tests passed.
- 4 Parquet range-planning and modular-runtime tests passed.
- 1 local-Ray one-GPU end-to-end test passed.
- 110 pre-existing physical-optimizer and operator-fusion tests passed, with 1
  pre-existing test skipped.

## Stock/modified acceptance result

Every arm produced 1,229,748 `uint16[4096]` rows with canonical digest
`cd440f692ed6e69f3980600a73afdb730b29ec67a8725bfbb484cc0b15edec85`.

| Arm | Pipeline seconds | Physical evidence |
| --- | ---: | --- |
| stock legacy | 123.13 | stock hash shuffle; no GPU fusion node |
| modified legacy, fusion disabled | 122.03 | same stock hash shuffle; no GPU fusion node |
| modified V1 | 13.59 | one fused GPU operator; no exchange |

Modified legacy differed from stock by -0.89%. Modified V1 was 9.06x faster
than stock and passed the 17.15-second maximum and 7x minimum-speedup gates.

The repeated benchmark also reproduced a known stock-path large-batch issue:
Arrow-to-cuDF conversion can raise `cudaErrorInvalidValue` at the legacy
1,048,576-row batch size. All three modular V1 trials completed (12.35s,
11.19s, and 12.27s), while an untouched stock or fusion-disabled legacy arm
eventually encountered that pre-existing failure. The benchmark does not retry
or silently change batch sizes.

`validation/acceptance-summary.json` is a path-sanitized machine-readable
record. Raw outputs, host paths, logs, generated data, environments, and wheels
are intentionally excluded from Git.
