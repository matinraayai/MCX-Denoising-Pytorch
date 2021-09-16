function binary_volume = random_char_in_3d_volume(volume_size)
    % Places a random ASCII character with a random thickness randomly inside a 3D volume. The character is also
    % randomly rotated around each axis.
    % Input:
    %       volume_size: shape of the volume
    % Output:
    %       binary_volume: a binary 3D array, where 1 is part of the character and 0 is absence of character

    % Generate a random 2D character
    addpath(genpath('../../2D'));
    char_2d = random_char_in_2d_volume(volume_size(1:2));
    % Randomly select a z-start and z-end to give volume to the letter
    z_start = randi(volume_size(3));
    z_end = randi([z_start, volume_size(3)]);
    char_girth = z_end - z_start;
    char_3d = char_2d;
    for i = 1 : char_girth
        char_3d = cat(3, char_3d, char_2d);
    end
    %  Place the slices containing the character into the output
    binary_volume = zeros(volume_size);
    binary_volume(:, :, z_start: z_end) = char_3d;
    % Random rotation around the x and y-axis
    binary_volume = imrotate3(binary_volume, rand() * 360, [1 0 0], 'crop');
    binary_volume = imrotate3(binary_volume, rand() * 360, [0 1 0], 'crop');
    % Some interpolation is done in the rotation, we need to make them part of the object
    binary_volume = ceil(binary_volume);
