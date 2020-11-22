clear all;

PHOTON_LIST=[1e5 1e6 1e7 1e8 1e9]
N=1e4;
MAX_PROPS=0
DATA_DIMs=[100 100]
% Top-level Dir
TOP_FOLDER_NAME='./data/rand2d';
if ~exist(TOP_FOLDER_NAME, 'dir')
    mkdir(TOP_FOLDER_NAME);
end

addpath('./mcxlab')

% Generate new unique random seed for Monte Carlo simulation
is_seed_unique=0
while is_seed_unique~=0
    rand_seed=randi([1 2^31-1], 1, N);
    is_seed_unique=length(unique(rand_seed)) < length(rand_seed)
end


testID=0;
for i=0:MAX_PROPS
    for j=1:N
        fname=sprintf('%s/%d.mat', TOP_FOLDER_NAME, testID);
        % skip if already generated
        if ~exist(fname, 'file')
            continue;
        end
        fprintf('Generating %s\n',fname);
        for phtn_count=1:length(PHOTON_LIST)
            rand_sd=rand_seed(j);
            label = sprintf("x1.0e", phtn_count);
            [image, ~, ~]=rand_2d_mcx(phtn_count, i, DATA_DIMs, rand_sd);
            eval(sprintf("%s=image", label));
            save(fname, label, '-append');
        end
        testID=testID+1;
    end
end
