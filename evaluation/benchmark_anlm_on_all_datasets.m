% Script to apply GPU ANLM provided by the MCX software to 3D test volumes.
% The filter parameters are the same as the following paper:
% "Graphics processing units-accelerated adaptive nonlocal means filter for denoising three-dimensional
% Monte Carlo photon transport simulations," by Yuan et al.
benchmark_anlm("../data/3D/test/absorb/64x64x64/", "../results/3D/gpu_anlm/absorb/64x64x64/");
benchmark_anlm("../data/3D/test/homo/64x64x64/", "../results/3D/gpu_anlm/homo/64x64x64/");
benchmark_anlm("../data/3D/test/refractive/64x64x64/", "../results/3D/gpu_anlm/refractive/64x64x64/");
benchmark_anlm("../data/3D/test/absorb/128x128x128/", "../results/3D/gpu_anlm/absorb/128x128x128/");
benchmark_anlm("../data/3D/test/homo/128x128x128/", "../results/3D/gpu_anlm/homo/128x128x128/");
benchmark_anlm("../data/3D/test/refractive/128x128x128/", "../results/3D/gpu_anlm/refractive/128x128x128/");
benchmark_anlm("../data/3D/test/colin27/", "../results/3D/gpu_anlm/colin27/");
benchmark_anlm("../data/3D/test/USE195/", "../results/3D/gpu_anlm/use195/");
benchmark_anlm("../data/3D/test/digimouse/", "../results/3D/gpu_anlm/digimouse/");
