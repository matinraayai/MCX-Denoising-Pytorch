%% generating data sets used in OSA

clear all

% Top-level Dir
topFolderName='osa_data';
if ~exist('osa_data', 'dir')  mkdir(topFolderName); end

N = 100; % run N times simulation for each test, 
x = 100;
y = 100;
z = 100;
time = zeros(N,5);
%pho_cnt = [1e5, 1e6, 1e7, 1e8];  % use 10 x 1e8 for 1e9 (as the ground truth)
pho_cnt = [1e9];  % use 10 x 1e8 for 1e9 (as the ground truth)
volume = uint8(ones(x,y,z));


% Generate new random seed for Monte Carlo simulation
rand_seed = randi([1 2^31-1], 1, N);

if (length(unique(rand_seed)) < length(rand_seed)) ~= 0
	error('There are repeated random seeds!')
end

dir_phn = sprintf('./%s/%1.0e', topFolderName, pho_cnt(1));
if ~exist(dir_phn, 'dir')  mkdir(dir_phn); end


data = zeros(x, y, z);

% run 10 x 1e8 = 1e9
for tid =1:10
	clear cfg
	cfg.nphoton=1e8; % run 10x 1e8
	cfg.vol= volume;
	cfg.srcpos=[50 50 1];
	cfg.srcdir=[0 0 1];
	cfg.gpuid=1;
	% cfg.gpuid='11'; % use two GPUs together
	cfg.autopilot=1;
	cfg.prop=[0 0 1 1;0.005 1 0 1.37];
	cfg.tstart=0;
	cfg.tend=5e-8;
	cfg.tstep=5e-8;
	cfg.seed = rand_seed(tid); % each random seed will have different pattern 

	% calculate the flux distribution with the given config
	[flux,detpos]=mcxlab(cfg);

	image3D=flux.data;
	data = data + image3D;
end

data = data * 0.1; % avg


%%% export each image in 3D volume
for imageID=1:y
	fname = sprintf('%s/osa_1e9_img%d.mat', dir_phn, imageID);
	fprintf('Generating %s\n',fname);
	currentImage = squeeze(data(:,imageID,:));
	feval('save', fname, 'currentImage');
end
