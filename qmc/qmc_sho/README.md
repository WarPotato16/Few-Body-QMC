# Complete QMC, with/without Simple Harmonic Oscillator

Files:

- `harmonic_potential.hpp`: header-only helper that constructs the site potential
  `V_i = 0.5 * m * omega^2 * |r_i - r_c|^2` and the positive SSE shift.
- `qmc_sho.cpp`: full edited version of the supplied QMC program.
- `qmc_sho.patch`: unified diff against the supplied source.

## Import

Place `harmonic_potential.hpp` beside the QMC source and add this after the
standard-library includes:

```cpp
#include "harmonic_potential.hpp"
```

The edited source already contains this include.

## Build

```bash
g++ -std=c++17 -O3 -fopenmp qmc_sho.cpp -o qmc_sho
```

## Example

```bash
./qmc_sho \
  --model boson \
  --boundary open \
  --L 4 \
  --N 2 \
  --beta 1 \
  --t 1 \
  --trap \
  --mass 1 \
  --omega 0.5 \
  --a 1 \
  --M 260 \
  --therm 100000 \
  --sweeps 1000000 \
  --chains 12
```

The default trap center is `((L - 1)/2, (L - 1)/2)`. Override it with
`--center_x` and `--center_y`.

## New output

- `<P>`: mean number of harmonic-potential operators in the SSE string.
- `<V>`: sign-reweighted imaginary-time average of the physical trap energy
  `sum_i V_i n_i`.
- `QMC E`: total physical energy after adding the auxiliary SSE shift back.

## Why a shifted operator is used

The physical trap is positive:

```text
H_V = sum_i V_i n_i.
```

Directly placing `-V_i n_i` into `-H` would create negative diagonal sampling
weights. The helper instead chooses `C_i = V_i + epsilon` and rewrites

```text
H_V = sum_i C_i - sum_i (C_i - V_i n_i).
```

The sampled site-operator matrix element `C_i - V_i n_i` is strictly positive.
The code then adds `sum_i C_i` back to the expansion-order estimator:

```text
E = -<n> / beta + sum_i C_i.
```

## Practical notes

A harmonic trap is normally paired with open boundaries. Periodic hopping is
allowed, but it introduces a wraparound bond across the spatial edge even
though the potential rises toward that edge.

The trap raises the average expansion order by approximately
`beta * sum_i C_i`. Increase `--M` if the cutoff warning appears. Stronger
`omega`, larger mass, larger lattice spacing, or a larger lattice all require a
larger operator string.
