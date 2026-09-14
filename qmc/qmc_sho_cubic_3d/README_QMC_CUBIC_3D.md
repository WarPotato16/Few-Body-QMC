# QMC on a true 4 x 4 x 4 cubic lattice

This version changes the physical lattice from an `L x L` square to an `L x L x L` cube. With `--L 4`, the simulation has 64 physical sites rather than 16 sites displayed above a plane.

## Model changes

The site index is

```text
site = x + L * (y + L * z)
```

with `x,y,z` in `0,...,L-1`.

The cubic bond builder adds nearest-neighbor hopping in all three directions. For `L=4`:

- periodic boundaries: `B = 3 L^3 = 192` bonds;
- open boundaries: `B = 3 L^2 (L-1) = 144` bonds.

The 3D harmonic potential is

```text
V(x,y,z) = 0.5 * m * omega^2 * a^2
           * [(x-xc)^2 + (y-yc)^2 + (z-zc)^2]
```

The default center is `(1.5,1.5,1.5)` for `L=4`.

## 64-site basis support

A 4 x 4 x 4 lattice has exactly 64 sites. The code therefore uses `std::uint64_t` and a recursive fixed-particle-number basis generator rather than calculating `1ULL << 64`, which is invalid.

Basis sizes include:

```text
N=2: C(64,2) = 2,016
N=3: C(64,3) = 41,664
N=4: C(64,4) = 635,376
```

The current implementation rejects bases larger than one million states, so run `N=2`, `N=3`, or `N=4` separately. `--all` is disabled for the cubic model.

## Compile

```bash
g++ -std=c++17 -O3 -march=native -fopenmp \
    qmc_sho_cubic_3d.cpp \
    -o qmc_sho_cubic_3d
```

Windows/MinGW:

```powershell
g++ -std=c++17 -O3 -march=native -fopenmp `
    .\qmc_sho_cubic_3d.cpp `
    -o .\qmc_sho_cubic_3d.exe
```

## Initial 12-chain run

Start with a moderate benchmark before committing to a very large sweep count:

```bash
export OMP_NUM_THREADS=12

./qmc_sho_cubic_3d \
    --model boson \
    --boundary periodic \
    --L 4 \
    --N 2 \
    --beta 1 \
    --t 1 \
    --M 800 \
    --therm 1000 \
    --sweeps 10000 \
    --measure_every 1 \
    --chains 12 \
    --trap \
    --mass 1 \
    --omega 0.5 \
    --a 1 \
    --output cubic_sho_4x4x4
```

The run writes one post-aggregation result:

```text
cubic_sho_4x4x4.csv
cubic_sho_4x4x4.html
```

Keep `plotly.min.js` in the same folder as the HTML file.

## Interactive map

The HTML viewer uses the actual lattice coordinates `(x,y,z)`.

- Drag to rotate around the cube.
- Scroll to zoom.
- Hover over a site for index, coordinates, occupation, single-particle probability, and potential.
- Click a site to pin its information and add a bright selection halo.
- Use the Isometric, Top, Front, and Side buttons.

The black background, Inferno color scale, translucent outer markers, and bright yellow high-value markers create the requested glow effect. Marker color and marker size both encode the pooled occupation probability.

## Twelve-chain aggregation

Each chain accumulates signed site occupations. Only after all chains finish does `combineChainResults()` calculate

```text
<n_i> = sum_c rawSignedOccupationSum[c][i]
        / sum_c rawSignSum[c]
```

The HTML and CSV exporters are called once from `main()` after `runParallelChains()` returns.

## String length and runtime

The 3D periodic lattice has 192 bonds instead of the 32 bonds in the 2D periodic 4 x 4 lattice. The default operator-string cutoff is therefore raised to `M=800`.

Watch:

```text
<Nop>
cutoffHitFraction
```

Increase `M` if the active expansion order approaches `0.85 M` or the cutoff warning appears.

The current update algorithm replays the full operator string for many proposals, so its cost is approximately quadratic in `M`. Moving from `M=260` to `M=800` can make each chain roughly `(800/260)^2`, or about 9.5 times, more expensive per sweep before other 3D overhead. The 12 independent chains still run in parallel across 12 cores.

## Smoke-test files

`cubic_4x4x4_smoke.html` and `.csv` only verify the 64-site viewer and post-aggregation export. They use too few sweeps to represent a converged physical distribution.
