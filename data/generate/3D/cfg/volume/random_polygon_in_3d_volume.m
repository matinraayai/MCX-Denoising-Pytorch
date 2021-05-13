function binary_volume = random_polygon_in_3d_volume(num_sides, ctr_to_vtx_range, volume_size)
    % Starter code from: https://au.mathworks.com/matlabcentral/answers/uploaded_files/201505/shape_recognition_demo1.m
	% Create a random polygon around the origin with given number of sides and ctr_to_vtx_range and randomly
    % rotates and places it in a 3D volume specified by volume_size.
    % 3D point placement logic used from https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
    % Inputs:
    %       num_sides: Number of sides for the polygon. Must be greater than 4.
    %       ctr_to_vtx_range: In form of [min_distance, max_distance], specifies the distance range from a
    %                         vertex to the centroid.
    %       volume_size: Size of the output volume, in the form of [rows, columns, sheets]
    % Output:
    %       binary_volume: A binary volume with the size of volume_size, with 1 indicating the polygon and 0 indicating
    %                      the absence of the polygon.
    % Create a polygon around the origin
    v = 0 : num_sides - 1;
    % Randomly generate distances for each vertex if a range is specified
    if length(ctr_to_vtx_range) == 2
        centroid_to_vertex_dist = randi(ctr_to_vtx_range, [1, num_sides]);
    else
        centroid_to_vertex_dist = ctr_to_vtx_range;
    end
    z = (v / num_sides) * 2 .* centroid_to_vertex_dist - centroid_to_vertex_dist;
    phi = (v / num_sides) * 2 * pi;
    x = sqrt(centroid_to_vertex_dist.^2 - z.^2) .* cos(phi);
    y = sqrt(centroid_to_vertex_dist.^2 - z.^2) .* sin(phi);
    % Randomly rotate the vertices around each axis in 3D
	rot_x_angle = 2 * pi * rand();
	rot_y_angle = 2 * pi * rand();
	rot_z_angle = 2 * pi * rand();
	[x, y, z] = rotate_3d(x, y, z, rot_x_angle, rot_y_angle, rot_z_angle);


	max_distance = max(centroid_to_vertex_dist);
	x_center = max_distance + (volume_size(1) - 2 * max_distance) * rand();
	y_center = max_distance + (volume_size(2) - 2 * max_distance) * rand();
	z_center = max_distance + (volume_size(3) - 2 * max_distance) * rand();
	% Translate the image so that the center is at (x_center, y_center, z_center) rather than at (0,0, 0).
	x = x + x_center;
	y = y + y_center;
	z = z + z_center;
	alpha_shape = alphaShape(x', y', z');
	[xx, yy, zz] = meshgrid(1: volume_size(1), 1: volume_size(2), 1: volume_size(3));
    binary_volume = inShape(alpha_shape, xx, yy, zz);

function [x, y, z] = rotate_3d(x, y, z, rot_x_angle, rot_y_angle, rot_z_angle)
    % Rotates the given coordinates around each axis by the given angle in radians.
    rot_x_matrix = [1, 0, 0; 0 cos(rot_x_angle) -sin(rot_x_angle); 0 sin(rot_x_angle) cos(rot_x_angle)];
	rot_y_matrix = [cos(rot_y_angle) 0 sin(rot_y_angle); 0 1 0; -sin(rot_y_angle) 0 cos(rot_y_angle)];
    rot_z_matrix = [cos(rot_z_angle) -sin(rot_z_angle) 0; sin(rot_z_angle) cos(rot_z_angle) 0; 0 0 1];
	% Do the actual rotation
	xyz = [x', y', z'];
	xyz_rotated = xyz * rot_x_matrix * rot_y_matrix * rot_z_matrix;
	x = xyz_rotated(:, 1)';
	y = xyz_rotated(:, 2)';
	z = xyz_rotated(:, 3)';
