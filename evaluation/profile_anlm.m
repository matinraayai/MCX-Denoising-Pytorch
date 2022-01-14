function time = profile_anlm(input_dims, num_iterations, v, f1, f2, rician, gpuid, bw)
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
    % Apply the filter to each input file and save it in the output
    simulation = rand(input_dims);
    tic;
    for i = 1 : num_iterations
        current_output = mcxfilter(simulation, v, f1, f2, rician, gpuid, bw);
    end
    time = toc;










