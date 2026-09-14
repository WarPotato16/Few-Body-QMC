@echo off
set OMP_NUM_THREADS=12
qmc_sho_cubic_3d.exe ^
  --model boson ^
  --boundary periodic ^
  --L 4 --N 2 ^
  --beta 1 --t 1 --M 800 ^
  --therm 1000 --sweeps 10000 --measure_every 1 ^
  --chains 12 ^
  --trap --mass 1 --omega 0.5 --a 1 ^
  --output cubic_sho_4x4x4
pause
