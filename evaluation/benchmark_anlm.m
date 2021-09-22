function benchmark_anlm(input_dir, output_dir, v, f1, f2, rician, gpuid, bw)
    %
    % Applies the MCX filter on the benchmark datasets and saves them to an output file.
    % input_path and output_path should be in C-formatted strings to point to correct files
    %
    % Default parameters used in the GPU-ANLM paper
    if nargin == 2
        v = 3;
        f1 = 1;
        f2 = 2;
        rician = 0;
        gpuid = 0;
        bw = 8;

    end
    % Add MCX filter mex file to path
    addpath('../matlab');
    % Get a cell array of all the files located in the input_dir
    path_list = make_path_list(input_dir);
    % Create the output directory if it doesn't exist already
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    % Apply the filter to each input file and save it in the output
    for i = 1 : length(path_list)

        current_file_name = path_list{i};
        fprintf("Processing file %s\n", current_file_name);
        input_file_path = strcat(input_dir, '/', current_file_name);
        output_file_path = strcat(output_dir, '/', current_file_name);
        % Each input file will have a simulation for a particular number of photons. We will have to identify and
        % preserve each label
        labels = {whos('-file', input_file_path).name};
        % Load all the simulations into the workspace
        input_file = load(input_file_path);
        % Apply the filter on each simulation
        for j = 1 : length(labels)
            current_label = labels{j};
            if strcmp(current_label, 'cfg')
                cfg = getfield(input_file, 'cfg');
                save(output_file_path, 'cfg');
            else
                fprintf("Applying filter to %s\n", current_label);
                current_simulation = getfield(input_file, current_label);
                current_output = mcxfilter(current_simulation, v, f1, f2, rician, gpuid, bw);
                eval(sprintf("%s = current_output;", current_label));
                % Save the results to the output directory
                if j == 1
                    save(output_file_path, current_label);
                else
                    save(output_file_path, current_label, '-append');
                end
            end
        end
    end










