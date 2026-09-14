# Lattice ground-state potential and occupation heatmap

`lattice_potential_heatmap.cpp` exactly diagonalizes the finite fixed-particle Hamiltonian used by the QMC project, then writes a site-resolved potential/occupation heatmap.

## Build

```bash
g++ -std=c++17 -O3 lattice_potential_heatmap.cpp -o lattice_heatmap
```

No external C++ libraries are required.

## Reproduce the periodic 4x4, N=2 SHO result

```bash
./lattice_heatmap \
  --L 4 \
  --N 2 \
  --model boson \
  --boundary periodic \
  --potential sho \
  --mass 1 \
  --omega 0.5 \
  --a 1 \
  --t 1 \
  --output periodic_sho_heatmap
```

Expected main results:

```text
Ground-state energy: -62.9672420519
Occupation sum: 2.0000000000
```

Expected occupation grid:

```text
0.103250  0.124116  0.124116  0.103250
0.124116  0.148518  0.148518  0.124116
0.124116  0.148518  0.148518  0.124116
0.103250  0.124116  0.124116  0.103250
```

Outputs:

- `periodic_sho_heatmap.svg`: side-by-side potential and occupation heatmaps.
- `periodic_sho_heatmap.csv`: site index, coordinates, potential, site occupation, and normalized one-particle probability.

## Standard periodic comparison

```bash
./lattice_heatmap --L 4 --N 2 --boundary periodic \
  --potential none --output periodic_standard
```

This produces occupation `0.125` on every site.

## Other included onsite potentials

```text
none
sho
quartic
gaussian_well
gaussian_barrier
double_well_x
linear_x
checkerboard
disorder
impurity
```

Examples:

```bash
# Attractive Gaussian well
./lattice_heatmap --potential gaussian_well --strength 2 --sigma 0.8 \
  --output gaussian_well

# Double well along x, harmonic confinement along y
./lattice_heatmap --potential double_well_x --strength 0.15 \
  --separation 1 --omega 0.5 --output double_well

# Staggered checkerboard superlattice
./lattice_heatmap --potential checkerboard --strength 0.5 \
  --output checkerboard

# Reproducible onsite disorder in [-W/2,W/2]
./lattice_heatmap --potential disorder --disorder_width 1 \
  --seed 12345 --output disorder

# Single repulsive impurity at (1,1)
./lattice_heatmap --potential impurity --impurity_x 1 --impurity_y 1 \
  --strength 2 --output impurity
```

Run `./lattice_heatmap --help` for the full option list.

## Important scaling limitation

This is exact diagonalization, not stochastic QMC sampling. The basis dimension is

```text
choose(L^2, N)
```

so it is ideal for validating small systems such as `L=4, N=2`, but becomes expensive quickly. The occupation estimator itself can later be added to the QMC code for larger lattices.
