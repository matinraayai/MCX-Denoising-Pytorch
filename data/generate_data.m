clear all;
% Variables for the script
PHOTON_LIST = [1e5 1e6 1e7 1e8 1e9];
N = 1e4;
MAX_NUM_PROPS = 10;
MIN_NUM_PROPS= 2;
DATA_DIMs = [100 100];
GPU_IDs = '11';
% Top level directory
TOP_FOLDER_NAME = './rand2d';

if ~exist(TOP_FOLDER_NAME, 'dir')
    mkdir(TOP_FOLDER_NAME);
end

addpath('./mcxlab');

% Generate new unique random seed for Monte Carlo simulation
is_seed_unique = 1;
while is_seed_unique ~= 0
    rand_seed = randi([1 2^31 - 1], 1, N);
    is_seed_unique = length(unique(rand_seed)) < length(rand_seed);
end


file_id = 0;
for i = 1 : N
    % Keep already generated data intact
    fname = sprintf('%s/%d.mat', TOP_FOLDER_NAME, file_id);
    while exist(fname, 'file')
        file_id = file_id + 1;
        fname = sprintf('%s/%d.mat', TOP_FOLDER_NAME, file_id);
    end
    fprintf('Generating file %s\n',fname);

    for phtn_count = 1 : length(PHOTON_LIST)
        rand_sd = rand_seed(i);
        % Log 10 for creating a variable name to be saved in the mat file.
        label = sprintf("x1e%d", log10(PHOTON_LIST(phtn_count)));
        [image, ~, ~] = rand_2d_mcx(PHOTON_LIST(phtn_count), MIN_NUM_PROPS, MAX_NUM_PROPS, DATA_DIMs, rand_sd, GPU_IDs);
        eval(sprintf("%s = image;", label));
        % create the file in the first run since it doesn't exist.
        if phtn_count == 1
            save(fname, label);
        else
            save(fname, label, '-append');
        end
    end
end
