function path_list = make_path_list(path)
    % Creates a list of files located in the path as a CS list
    listing = dir(path);
    path_list = {listing.name};
    path_list = {path_list{3:end}}
