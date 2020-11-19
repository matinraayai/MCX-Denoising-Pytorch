% Generates data sets used in OSA
% Input arguments:
clear all;
TOP_FOLDER_NAME = 'osa';
MCXLAB_PATH = "./mcx/mcxlab";
NUM_SIMULATIONS = 1000;
X = 100;
Y = 100;
Z = 100;
PHOTON_COUNTS = [1e5, 1e6, 1e7, 1e8, 1e9];

% set path for mcxlab
addpath(genpath(MCXLAB_PATH))

% Create dataset folder if already not present
if ~exist(TOP_FOLDER_NAME, 'dir')
    mkdir(TOP_FOLDER_NAME);
end

volume = uint8(ones(X,Y,Z));


for phtn_cnt = 1:length(PHOTON_COUNTS)
    % Generate unique random seeds for Monte Carlo simulation
    are_seed_unique = 1;
	while are_seed_unique ~= 0
	    rand_seed = randi([1 2^31 - 1], 1, NUM_SIMULATIONS);
	    are_seed_unique = length(unique(rand_seed)) < length(rand_seed);
    end

	dir_phn = sprintf('./%s/%1.0e', TOP_FOLDER_NAME, PHOTON_COUNTS(phtn_cnt));
    if ~exist(dir_phn, 'dir')
        mkdir(dir_phn);
    end

	for sim_id = 1:NUM_SIMULATIONS
		dir_phn_test = sprintf('%s/%d', dir_phn, sim_id);
        if ~exist(dir_phn_test, 'dir')
            mkdir(dir_phn_test);
        end

		clear cfg
		cfg.nphoton = PHOTON_COUNTS(phtn_cnt);
		cfg.vol = volume;
		cfg.srcpos = [50 50 1];
		cfg.srcdir = [0 0 1];
		%cfg.gpuid = 1;
		cfg.gpuid='11'; % use two GPUs together
		cfg.autopilot = 1;
		cfg.prop = [0 0 1 1; 0.005 1 0 1.37];
		cfg.tstart = 0;
		cfg.tend = 5e-8;
		cfg.tstep = 5e-8;
		cfg.seed = rand_seed(sim_id); % each random seed will have different pattern

		% calculate the flux distribution with the given config
		[flux,detpos]=mcxlab(cfg);

		volume=flux.data;
		%%% export each image in 3D volume
		fname = sprintf('%s/osa_phn%1.0e_test%d.mat', dir_phn_test, PHOTON_COUNTS(phtn_cnt), sim_id);
		save(fname, 'volume');
	end
end