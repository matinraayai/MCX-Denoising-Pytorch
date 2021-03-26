function binary_volume = random_polygon_in_3d_volume(num_sides, centroid_to_vertex_dist, volume_size)
% Starter code from: https://au.mathworks.com/matlabcentral/answers/uploaded_files/201505/shape_recognition_demo1.m
try
	% Create a polygon around the origin
    % Logic used from https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
    v = 1 : num_sides;
    z = (v / num_sides) * 2 * centroid_to_vertex_dist - centroid_to_vertex_dist;
    phi = (v / num_sides) * 2 * pi;
    x = sqrt(centroid_to_vertex_dist^2 - z.^2) .* cos(phi);
    y = sqrt(centroid_to_vertex_dist^2 - z.^2) .* sin(phi);
    % Randomly rotate the vertices around each axis in 3D
	rot_x_angle = 2 * pi * rand();
	rot_y_angle = 2 * pi * rand();
	rot_z_angle = 2 * pi * rand();
	rot_x_matrix = [1, 0, 0; 0 cos(rot_x_angle) -sin(rot_x_angle); 0 sin(rot_x_angle) cos(rot_x_angle)];
	rot_y_matrix = [cos(rot_y_angle) 0 sin(rot_y_angle); 0 1 0; -sin(rot_y_angle) 0 cos(rot_y_angle)];
    rot_z_matrix = [cos(rot_z_angle) -sin(rot_z_angle) 0; sin(rot_z_angle) cos(rot_z_angle) 0; 0 0 1];
	% Do the actual rotation
	xyz = [x', y', z'];
	xyz_rotated = xyz * rot_x_matrix * rot_y_matrix * rot_z_matrix; % A numSides*2 matrix times a 2*2 = a numSides*2 matrix
	x = xyz_rotated(:, 1);
	y = xyz_rotated(:, 2);
	z = xyz_rotated(:, 3);

	% Get a random center location between centroidToVertexDistance and (columns - centroidToVertexDistance).
	% This will ensure it's always in the image.
	x_center = centroid_to_vertex_dist + (volume_size(1) - 2 * centroid_to_vertex_dist) * rand();
	y_center = centroid_to_vertex_dist + (volume_size(2) - 2 * centroid_to_vertex_dist) * rand();
	z_center = centroid_to_vertex_dist + (volume_size(3) - 2 * centroid_to_vertex_dist) * rand();
	% Translate the image so that the center is at (x_center, y_center, z_center) rather than at (0,0, 0).
	x = x + x_center;
	y = y + y_center;
	z = z + z_center;
	alpha_shape = alphaShape(x, y, z);
	[xx, yy, zz] = meshgrid(1: volume_size(1), 1: volume_size(2), 1: volume_size(3));
    binary_volume = inShape(alpha_shape, xx, yy, zz);
catch ME
	errorMessage = sprintf('Error in function %s() at line %d.\n\nError Message:\n%s', ...
		ME.stack(1).name, ME.stack(1).line, ME.message);
	fprintf(1, '%s\n', errorMessage);
	uiwait(warndlg(errorMessage));
end