function generate_data_using_cfg_lambda(cfg_lambda, output_dir, photon_list, n, overwrite)
    % Enforce default arguments
    if nargin == 1
        photon_list = [1e5 1e6 1e7 1e8 1e9];
    end
    if nargin <= 2
        output_dir = './generated_data';
    end
    if nargin <= 3
        n = 1000;
    end
    if nargin <= 4
        overwrite = false;
    end

    % Import MCX Lab Mexfile from the top folder of the repository
    addpath('../matlab')
    % Create the output directory if it doesn't already exist
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    file_id = 0;
    for i = 1 : n
        fname = sprintf('%s/%d.mat', output_dir, file_id);
        if ~overwrite
            % Keep already generated data intact if overwrite is false
            while exist(fname, 'file')
                file_id = file_id + 1;
                fname = sprintf('%s/%d.mat', output_dir, file_id);
            end
        end
        fprintf('Generating file %s\n',fname);
        % Configuration used for all the simulations in the file
        cfg = cfg_lambda();
        % save the configuration in file as well for future analysis
        save(fname, 'cfg');

        for phtn_count = 1 : length(photon_list)
            cfg.nphoton = photon_list(phtn_count);
            % Run the simulation
            flux = mcxlab(cfg);
            simulation = squeeze(sum(flux.data, 4));
            % Generate a label to save in the mat file. Labels use the log10 of the photon count for convenience
            label = sprintf("x1e%d", log10(photon_list(phtn_count)));
            eval(sprintf("%s = simulation;", label));
            % Append to the already created file
            save(fname, label, '-append');
        end
    end