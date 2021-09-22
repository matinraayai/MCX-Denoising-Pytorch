% Script to apply GPU ANLM provided by the MCX software to 3D test volumes.
% The filter parameters are the same as the following paper:
% "Graphics processing units-accelerated adaptive nonlocal means filter for denoising three-dimensional
% Monte Carlo photon transport simulations," by Yuan et al.
benchmark_anlm("../data/test/3D/absorb/64x64x64/", "../results/3D/gpu_anlm/absorb/64x64x64/");
benchmark_anlm("../data/test/3D/homo/64x64x64/", "../results/3D/gpu_anlm/homo/64x64x64/");
benchmark_anlm("../data/test/3D/refractive/64x64x64/", "../results/3D/gpu_anlm/refractive/64x64x64/");
benchmark_anlm("../data/test/3D/absorb/128x128x128/", "../results/3D/gpu_anlm/absorb/128x128x128/");
benchmark_anlm("../data/test/3D/homo/128x128x128/", "../results/3D/gpu_anlm/homo/128x128x128/");
benchmark_anlm("../data/test/3D/refractive/128x128x128/", "../results/3D/gpu_anlm/refractive/128x128x128/");
benchmark_anlm("../data/test/3D/colin27/", "../results/3D/gpu_anlm/colin27/");
benchmark_anlm("../data/test/3D/USE195/", "../results/3D/gpu_anlm/use195/");
benchmark_anlm("../data/test/3D/digimouse/", "../results/3D/gpu_anlm/digimouse/");
