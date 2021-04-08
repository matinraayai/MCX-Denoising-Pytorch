clear cfg cfgs
PHOTON_LIST = [1e5 1e6 1e7];
N = 100;
GPU_IDs = '11';
DATA_DIMs = [100 100];
% Top level directory
TOP_FOLDER_NAME = './absorb';
TYPE = 'absorb';
if ~exist(TOP_FOLDER_NAME, 'dir')
    mkdir(TOP_FOLDER_NAME);
end

addpath('../../octave');

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
        % Log 10 for creating a variable name to be saved in the mat file.
        label = sprintf("x1e%d", log10(PHOTON_LIST(phtn_count)));
        [image, ~, ~] = benchmark(PHOTON_LIST(phtn_count), DATA_DIMs, GPU_IDs, TYPE);
        eval(sprintf("%s = image;", label));
        % create the file in the first run since it doesn't exist.
        if phtn_count == 1
            save(fname, label);
        else
            save(fname, label, '-append');
        end
    end
end