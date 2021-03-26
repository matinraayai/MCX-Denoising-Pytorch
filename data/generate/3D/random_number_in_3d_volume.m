function output = random_number_in_3d_volume(volumesize)
    % Generate a random 2D character
    addpath('../2D');
    char_2d = random_number_in_2d_volume(volumesize(1:2));
    % Randomly select a girth for it and stack it over the z-axis
    z_start = randi(volumesize(3));
    z_end = randi([z_start, volumesize(3)]);
    char_girth = z_end - z_start;
    char_3d = char_2d;
    for i = 1 : char_girth
        char_3d = cat(3, char_3d, char_2d);
    end
    % Insert it inside the ouput
    output = zeros(volumesize);
    output(:, :, z_start: z_end) = char_3d;

