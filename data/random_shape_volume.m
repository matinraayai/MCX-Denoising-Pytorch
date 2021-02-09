function binaryImage = random_shape_volume(imsize, num_shapes, max_sides, num_max_attempts)
% Create an image and add in some triangles at various angles and various sizes.
rows = imsize(1);
columns = imsize(2);
binaryImage = zeros(rows, columns);

shapesPlacedSoFar = 0;
while shapesPlacedSoFar < num_shapes
    overlap = true;
    numberOfAttempts = 0;
    while overlap && numberOfAttempts < num_max_attempts
        centroidToVertexDistance = [randi([round(rows / 8), round(rows / 3)]), randi([round(columns / 6), round(columns / 3)])];
        thisBinaryImage = (shapesPlacedSoFar + 1) * CreatePolygon(randi(max_sides - 3) + 3, centroidToVertexDistance, rows, columns);
        % Sometimes two polygons will be next to each other but not overlapping.
        % However bwlabel() and bwconncomp() would consider those two regions as being the same region.
        % To check for and prevent that kind of situation (which happened to me once),
        % we need to dilate the binary image by one layer before checking for overlap.
        dilatedImage = imdilate(thisBinaryImage, ones(9));
        % See if any pixels in this binary image overlap any existing pixels.
        overlapImage = binaryImage & dilatedImage;
        if ~any(overlapImage(:))
            overlap = false;
            binaryImage = binaryImage + thisBinaryImage;
        else
            fprintf('Skipping attempt %d because of overlap.\n', numberOfAttempts);
        end
        numberOfAttempts = numberOfAttempts + 1;
    end
    shapesPlacedSoFar = shapesPlacedSoFar + 1;
end

function binaryImage = CreatePolygon(numSides, centroidToVertexDistance, rows, columns)
try
	% Get the range for the size from the center to the vertices.
	if length(centroidToVertexDistance) > 1
		% Random size between a min and max distance.
		minDistance = centroidToVertexDistance(1);
		maxDistance = centroidToVertexDistance(2);
	else
		% All the same size.
		minDistance = centroidToVertexDistance;
		maxDistance = centroidToVertexDistance;
	end
	thisDistance = (maxDistance - minDistance) * rand(1) + minDistance;

	% Create a polygon around the origin
	for v = 1 : numSides
		angle = v * 360 / numSides;
		x(v) = thisDistance * cosd(angle);
		y(v) = thisDistance * sind(angle);
	end
	% Make last point the same as the first
	x(end+1) = x(1);
	y(end+1) = y(1);
	% 	plot(x, y, 'b*-', 'LineWidth', 2);
	% 	grid on;
	% 	axis image;

	% Rotate the coordinates by a random angle between 0 and 360
	angleToRotate = 360 * rand(1);
	rotationMatrix = [cosd(angleToRotate), sind(angleToRotate); -sind(angleToRotate), cosd(angleToRotate)];
	% Do the actual rotation
	xy = [x', y']; % Make a numSides*2 matrix;
	xyRotated = xy * rotationMatrix; % A numSides*2 matrix times a 2*2 = a numSides*2 matrix.
	x = xyRotated(:, 1); % Extract out the x as a numSides*2 matrix.
	y = xyRotated(:, 2); % Extract out the y as a numSides*2 matrix.

	% Get a random center location between centroidToVertexDistance and (columns - centroidToVertexDistance).
	% This will ensure it's always in the image.
	xCenter = thisDistance + (columns - 2 * thisDistance) * rand(1);
	% Get a random center location between centroidToVertexDistance and (rows - centroidToVertexDistance).
	% This will ensure it's always in the image.
	yCenter = thisDistance + (rows - 2 * thisDistance) * rand(1);
	% Translate the image so that the center is at (xCenter, yCenter) rather than at (0,0).
	x = x + xCenter;
	y = y + yCenter;
	binaryImage = poly2mask(x, y, rows, columns);
catch ME
	errorMessage = sprintf('Error in function %s() at line %d.\n\nError Message:\n%s', ...
		ME.stack(1).name, ME.stack(1).line, ME.message);
	fprintf(1, '%s\n', errorMessage);
	uiwait(warndlg(errorMessage));
end