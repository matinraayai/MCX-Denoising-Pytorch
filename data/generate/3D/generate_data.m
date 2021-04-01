clear all;
% Variables for the script
PHOTON_LIST = [1e5 1e6 1e7 1e8 1e9];
N = 1;
MAX_NUM_PROPS = 10;
MIN_NUM_PROPS= 2;
DATA_DIMs = [128 128 128];
GPU_IDs = '11';
START_FILE_ID = 0;
% Top level directory
TOP_FOLDER_NAME = './rand3d';

if ~exist(TOP_FOLDER_NAME, 'dir')
    mkdir(TOP_FOLDER_NAME);
end

addpath('../../../octave');

file_id = START_FILE_ID;
for i = 1 : N
    % Keep already generated data intact
    fname = sprintf('%s/%d.mat', TOP_FOLDER_NAME, file_id);
    while exist(fname, 'file')
        file_id = file_id + 1;
        fname = sprintf('%s/%d.mat', TOP_FOLDER_NAME, file_id);
    end
    fprintf('Generating file %s\n',fname);
    cfg = generate_random_cfg(DATA_DIMs, randi([MIN_NUM_PROPS, MAX_NUM_PROPS]), GPU_IDs, file_id);
    % Save the configuration first for possible future inspection
    save(fname, 'cfg');
    for phtn_count = 1 : length(PHOTON_LIST)
        cfg.nphoton = PHOTON_LIST(phtn_count);
        % Log 10 for creating a variable name to be saved in the mat file.
        label = sprintf("x1e%d", log10(PHOTON_LIST(phtn_count)));
        flux = mcxlab(cfg);
        cw = squeeze(sum(flux.data, 4));
        eval(sprintf("%s = cw;", label));
        save(fname, label, '-append');
        end
end
