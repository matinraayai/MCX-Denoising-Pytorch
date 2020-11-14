%% generating data sets used in OSA

clear all

% set path for mcxlab
addpath(genpath('./mcx/mcxlab'))

% Top-level Dir
topFolderName='osa_data_inclusion';
if ~exist('osa_data_inclusion', 'dir')  mkdir(topFolderName); end

N = 100; % run N times simulation for each test, 
x = 100;
y = 100;
z = 100;

pho_cnt = [1e5, 1e6, 1e7, 1e8];  % use 10 x 1e8 for 1e9 (as the ground truth)
volume = uint8(ones(x,y,z));


% Generate new random seed for Monte Carlo simulation
% do it twice, assume the 2nd time will have different nums than the prev
rand_seed = randi([1 2^31-1], 1, N);
rand_seed = randi([1 2^31-1], 1, N); 
if (length(unique(rand_seed)) < length(rand_seed)) ~= 0
    error('There are repeated random seeds!')
end


% folder for 1e5
dir_phn = sprintf('./%s/%1.0e', topFolderName, pho_cnt(1));
if ~exist(dir_phn, 'dir')  mkdir(dir_phn); end


% using each rand number to generate each 3D results
for tid =1:N
		dir_phn_test = sprintf('%s/%d', dir_phn, tid);
		if ~exist(dir_phn_test, 'dir')  mkdir(dir_phn_test); end

		clear cfg
		cfg.nphoton=pho_cnt(1); % 1e5
		cfg.vol= volume;
		cfg.vol(40:60, 40:60, 20:50) = 2;   % inclusion
		cfg.srcpos=[50 50 1];
		cfg.srcdir=[0 0 1];
		cfg.gpuid=1;
		% cfg.gpuid='11'; % use two GPUs together
		cfg.autopilot=1;
		%cfg.prop=[0 0 1 1;0.005 1 0 1.37];
		cfg.prop=[0 0 1 1;0.01 1 0 1.3; 0.1 2 0 1.3];   % inclusion

		cfg.tstart=0;
		cfg.tend=5e-8;
		cfg.tstep=5e-8;
		cfg.seed = rand_seed(tid); % each random seed will have different pattern 

		% calculate the flux distribution with the given config
		[flux,detpos]=mcxlab(cfg);

		image3D=flux.data;

		%%% export each image in 3D volume
		for imageID=1:y
			fname = sprintf('%s/osa_phn%1.0e_test%d_img%d.mat', dir_phn_test, pho_cnt(1), tid, imageID);
			fprintf('Generating %s\n',fname);
			currentImage = squeeze(image3D(:,imageID,:));
			feval('save', fname, 'currentImage');
		end
end
