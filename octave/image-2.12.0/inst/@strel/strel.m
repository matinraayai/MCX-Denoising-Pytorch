## Copyright (C) 2012, 2013 Roberto Metere <roberto@metere.it>
## Copyright (C) 2012, 2013 Carnë Draug <carandraug@octave.org>
##
## This program is free software; you can redistribute it and/or modify it under
## the terms of the GNU General Public License as published by the Free Software
## Foundation; either version 3 of the License, or (at your option) any later
## version.
##
## This program is distributed in the hope that it will be useful, but WITHOUT
## ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
## FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
## details.
##
## You should have received a copy of the GNU General Public License along with
## this program; if not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*-
## @deftypefn {Function File} {} strel (@var{shape}, @var{parameters})
## Create a strel (structuring element) object for morphology operations.
##
## The structuring element can have any type of shape as specified by
## @var{shape}, each one with its @var{parameters}.
##
## @end deftypefn
## @deftypefn  {Function File} {} strel ("arbitrary", @var{nhood})
## @deftypefnx {Function File} {} strel ("arbitrary", @var{nhood}, @var{height})
## Create arbitrary shaped structuring elements.
##
## @var{nhood} must be a matrix of 0's and 1's.  Any number with of dimensions
## are possible.  To create a non-flat SE, the @var{height} can be specified.
## See individual functions that use the strel object for an interpretation of
## non-flat SEs.
##
## Note that if an arbitrary shape is used, it may not be possible to guess
## it which may prevent shape-based optimizations (structuring element
## decomposition, see @@strel/getsequence for details).
##
## @end deftypefn
## @deftypefn {Function File} {} strel ("ball", @var{radius}, @var{height})
## Create ball shaped @var{nonflat} structuring element.  @var{radius} must be a
## nonnegative integer that specifies the ray of a circle in X-Y plane.  @var{height}
## is a real number that specifies the height of the center of the circle.
##
## @end deftypefn
## @deftypefn {Function File} {} strel ("cube", @var{edge})
## Create cube shaped @var{flat} structuring element.  @var{edge} must be a
## positive integer that specifies the length of its edges.  This shape meant to
## perform morphology operations in volumes, see the square shape for 2
## dimensional images.
##
## @end deftypefn
## @deftypefn {Function File} {} strel ("diamond", @var{radius})
## Create diamond shaped flat structuring element.  @var{radius} must be a
## positive integer.
##
## @end deftypefn
## @deftypefn  {Function File} {} strel ("disk", @var{radius})
## @deftypefnx {Function File} {} strel ("disk", @var{radius}, @var{n})
## Create disk shaped flat structuring element.  @var{radius} must be a positive
## integer.
##
## The optional @var{n} argument *must* have a value of zero but the default
## value is 4.  This is to prevent future backwards incompatibilty since
## @sc{Matlab} default is also 4 but at the moment only 0 has been
## implemented.
##
## @end deftypefn
## @deftypefn {Function File} {} strel ("hypercube", @var{n}, @var{edge})
## Create @var{n} dimensional cube (n-cube) shaped @var{flat} structuring
## element.  @var{edge} must be a positive integer that specifies the length
## of its edges.
##
## @end deftypefn
## @deftypefn {Function File} {} strel ("hyperrectangle", @var{dimensions})
## Create @var{n} dimensional hyperrectangle (or orthotope) shaped flat
## structuring element.  @var{dimensions} must be a vector of positive
## integers with its lengtht at each of the dimensions.
##
## @end deftypefn
## @deftypefn {Function File} {} strel ("line", @var{len}, @var{deg})
## Create line shaped flat structuring element.  @var{len} must be a positive
## real number.  @var{deg} must be a 1 or 2 elements real number, for a line in
## in 2D or 3D space.  The first element of @var{deg} is the angle from X-axis
## to X-Y projection of the line while the second is the angle from Z-axis to
## the line.
##
## @end deftypefn
## @deftypefn {Function File} {} strel ("octagon", @var{apothem})
## Create octagon shaped flat structuring element.  @var{apothem} must be a
## non-negative integer, multiple of 3, that specifies the distance from the
## origin to the sides of the octagon.
##
## @end deftypefn
## @deftypefn {Function File} {} strel ("pair", @var{offset})
## Create flat structuring element with two members.  One member is placed
## at the origin while the other is placed with @var{offset} in relation to the
## origin.  @var{offset} must then be a 2 element vector for the coordinates.
##
## @end deftypefn
## @deftypefn {Function File} {} strel ("periodicline", @var{p}, @var{v})
## Create periodic line shaped flat structuring element.  A periodic line will
## be built with 2*@var{p}+1 points around the origin included. These points will
## be displaced in accordance with the offset @var{v} at distances: 1*@var{v},
## -1*@var{v}, 2*@var{v}, -2*@var{v}, ..., @var{p}*@var{v}, -@var{p}*@var{v}.
##   Therefore @var{v} must be a 2 element vector for the coordinates.
##
## @end deftypefn
## @deftypefn {Function File} {} strel ("rectangle", @var{dimensions})
## Create rectangular shaped flat structuring element.  @var{dimensions} must
## be a two element vector of positive integers with the number of rows and
## columns of the rectangle.
##
## @end deftypefn
## @deftypefn {Function File} {} strel ("square", @var{edge})
## Create square shaped flat structuring element.  @var{edge} must be a positive
## integer that specifies the length of its edges.  For use in volumes, see the
## cube shape.
##
## The actual structuring element neighborhood, the logical matrix used for the
## operations, can be accessed with the @code{getnhood} method.  However, most
## morphology functions in the image package will have an improved performance
## if the actual strel object is used, and not its element neighborhood.
##
## @example
## @group
## se = strel ("square", 5);
## getnhood (se)
##     @result{}
##         1  1  1  1  1
##         1  1  1  1  1
##         1  1  1  1  1
##         1  1  1  1  1
##         1  1  1  1  1
## @end group
## @end example
##
## @seealso{imdilate, imerode}
## @end deftypefn

function SE = strel (shape, varargin)

  if (nargin < 1 || nargin > 4 || (ischar (shape) && nargin < 2))
    print_usage ();
  elseif (! ischar (shape))
    varargin = horzcat ({shape}, varargin);
    shape = "arbitrary";
  endif
  nvar = numel (varargin);

  ## because the order that these are created matters, we make them all here
  SE        = struct;
  SE.shape  = tolower (shape);
  SE.nhood  = false;
  SE.flat   = true;
  SE.height = [];
  SE.seq    = cell;
  SE.opt    = struct;

  switch (SE.shape)
    case "arbitrary"
      if (numel (varargin) == 1)
        nhood   = varargin{1};
        SE.flat = true;
      elseif (numel (varargin) == 2)
        nhood     = varargin{1};
        SE.height = varargin{2};
        SE.flat   = false;
      else
        error ("strel: an arbitrary shape takes 1 or 2 arguments");
      endif
      ## don't use isbw because we also want to allow empty nhood
      if (any ((nhood(:) != 1) & (nhood(:) != 0)))
        error ("strel: NHOOD must be a matrix with only 0 and 1 values")
      endif

      SE.nhood = logical (nhood); # we need this as logical for the height tests

      if (! SE.flat && ! (isnumeric (SE.height) && isreal (SE.height) &&
                          ndims (SE.height) == ndims (nhood)          &&
                          all (size (SE.height) == size (nhood))      &&
                          all (isfinite (SE.height(:)))))
        error ("strel: HEIGHT must be a finite real matrix of the same size as NHOOD");
      endif

      if (nnz (SE.height) == 0)
        SE.flat = true;
      endif

    case "ball"
      if (numel (varargin) == 2)
        radius = varargin{1};
        height = varargin{2};
      else
        ## TODO implement third option for number of periodic lines approximation
        error ("strel: a ball shape needs 2 arguments");
      endif
      if (! is_positive_integer (radius))
        error ("strel: RADIUS must be a positive integer");
      elseif (! (isscalar (height) && isnumeric (height)))
        error ("strel: HEIGHT must be a real number");
      endif

      # Ellipsoid: (x/radius)^2 + (y/radius)^2 + (z/height)^2 = 1
      # We need only the 1 cells of SE.nhood
      [x, y] = meshgrid (-radius:radius, -radius:radius);
      SE.nhood = ((x.^2 + y.^2) <= radius^2); # X-Y circle
      SE.height = height / radius * SE.nhood .* sqrt (radius^2 - x .^2 - y.^2);
      SE.flat = false;

    case "cube"
      if (numel (varargin) == 1)
        SE.opt.edge = varargin{1};
      else
        error ("strel: no EDGE specified for cube shape");
      endif
      if (! is_positive_integer (SE.opt.edge))
        error ("strel: EDGE value must be a positive integer");
      endif

      SE.nhood = true (SE.opt.edge, SE.opt.edge, SE.opt.edge);
      SE.flat  = true;

    case "diamond"
      if (numel (varargin) == 1)
        radius = varargin{1};
      else
        error ("strel: no RADIUS specified for diamond shape");
      endif
      if (! is_positive_integer (radius))
        error ("strel: RADIUS must be a positive integer");
      endif

      corner   = tril (true (radius+1, radius), -1);
      SE.nhood = [rot90(tril(true(radius+1))) corner;
                  corner' rot90(triu(true(radius),1))];
      SE.flat  = true;

    case "disk"
      if (nvar < 1 || nvar > 2)
        error ("strel: disk shape takes 1 or 2 arguments");
      endif

      radius = varargin{1};
      if (! is_positive_integer (radius))
        error ("strel: RADIUS must be a positive integer");
      endif

      n = 4;
      if (nvar > 1)
        n = varargin{2};
        if (! isnumeric (n) && ! isscalar (n) && any (n != [0 4 6 8]))
          error ("strel: N for disk shape must be 0, 4, 6, or 8");
        endif
      endif
      ## TODO implement approximation by periodic lines
      if (n != 0)
        error ("strel: N for disk shape not yet implemented, use N of 0");
      endif

      [x, y] = meshgrid (-radius:radius, -radius:radius);
      r = sqrt (x.^2 + y.^2);
      SE.nhood = r <= radius;
      SE.flat  = true;

    case "hypercube"
      if (numel (varargin) == 2)
        SE.opt.n    = varargin{1};
        SE.opt.edge = varargin{2};
      else
        error ("strel: an hypercube shape needs 2 arguments");
      endif
      if (! is_positive_integer (SE.opt.n))
        error ("strel: N value must be a positive integer");
      elseif (! is_positive_integer (SE.opt.edge))
        error ("strel: EDGE value must be a positive integer");
      endif

      SE.nhood = true (repmat (SE.opt.edge, 1, SE.opt.n));
      SE.flat  = true;

    case "hyperrectangle"
      if (numel (varargin) == 1)
        SE.opt.dimensions = varargin{1};
      else
        error ("strel: no DIMENSIONS specified for rectangle shape");
      endif
      if (! isnumeric (SE.opt.dimensions))
        error ("strel: DIMENSIONS must be a 2 element vector");
      elseif (! all (arrayfun (@is_positive_integer, SE.opt.dimensions(:))))
        error ("strel: DIMENSIONS values must be positive integers");
      endif

      SE.nhood = true (SE.opt.dimensions(:));
      SE.flat  = true;

    case "line"
      if (numel (varargin) == 2)
        linelen = varargin{1};
        degrees = varargin{2};
      else
        error ("strel: a line shape needs 2 arguments");
      endif
      if (! (isscalar (linelen) && isnumeric (linelen) && linelen > 0))
        error ("strel: LEN must be a positive real number");
      elseif (! isnumeric (degrees))
        error ("strel: DEG must be numeric");
      endif
      ## 2d or 3d line
      dimens = numel (degrees) +1;
      if (dimens == 2)
        degrees = degrees(1);
      elseif (dimens == 3)
        alpha = degrees(1);
        phi   = degrees(2);
      else
        error ("strel: DEG must be a 1 or 2 elements matrix");
      endif

      ## TODO this was the 3dline and line options, which have separate code
      ##      but a proper merge should be made.

      if (dimens == 2)
        ## Line length are always odd, to center strel at the middle of the line.
        ## We look it as a diameter of a circle with given slope
        # It computes only lines with angles between 0 and 44.9999
        deg90 = mod (degrees, 90);
        if (deg90 > 45)
          alpha = pi * (90 - deg90) / 180;
        else
          alpha = pi * deg90 / 180;
        endif
        ray = (linelen - 1)/2;

        ## We are interested only in the discrete rectangle which contains the diameter
        ## However we focus our attention to the bottom left quarter of the circle,
        ## because of the central symmetry.
        c = round (ray * cos (alpha)) + 1;
        r = round (ray * sin (alpha)) + 1;

        ## Line rasterization
        line = false (r, c);
        m = tan (alpha);
        x = [1:c];
        y = r - fix (m .* (x - 0.5));
        indexes = sub2ind ([r c], y, x);
        line(indexes) = true;

        ## We view the result as 9 blocks.
        # Preparing blocks
        linestrip = line(1, 1:c - 1);
        linerest = line(2:r, 1:c - 1);
        z = false (r - 1, c);

        # Assemblying blocks
        SE.nhood =  vertcat (
                      horzcat (z, linerest(end:-1:1,end:-1:1)),
                      horzcat (linestrip, true, linestrip(end:-1:1,end:-1:1)),
                      horzcat (linerest, z(end:-1:1,end:-1:1))
                    );

        # Rotate/transpose/flip?
        sect = fix (mod (degrees, 180) / 45);
        switch (sect)
          case 1, SE.nhood = transpose (SE.nhood);
          case 2, SE.nhood = rot90 (SE.nhood, 1);
          case 3, SE.nhood = fliplr (SE.nhood);
          otherwise, # do nothing
        endswitch

      elseif (dimens == 3)
        ## This is a first implementation
        ## Stroke line from cells (x1, y1, z1) to (x2, y2, z2)
        alpha *= pi / 180;
        phi *= pi / 180;
        x1 = y1 = z1 = 0;
        x2 = round (linelen * sin (phi) * cos (alpha));
        y2 = round (linelen * sin (phi) * sin (alpha));
        z2 = round (linelen * cos (phi));
        # Adjust x2, y2, z2 to have one central cell
        x2 += (! mod (x2, 2)) * sign0positive (x2);
        y2 += (! mod (y2, 2)) * sign0positive (y2);
        z2 += (! mod (z2, 2)) * sign0positive (z2);
        # Invert x
        x2 = -x2;

        # Tanslate parallelepiped to be in positive quadrant
        if (x2 < 0)
          x1 -= x2;
          x2 -= x2;
        endif
        if (y2 < 0)
          y1 -= y2;
          y2 -= y2;
        endif
        if (z2 < 0)
          z1 -= z2;
          z2 -= z2;
        endif

        # Compute index2es
        dim = abs ([(x2 - x1) (y2 - y1) (z2 - z1)]);
        m = max (dim);
        base = meshgrid (0:m - 1,1) .+ 0.5;
        a = floor ((x2 - x1)/m .* base);
        b = floor ((y2 - y1)/m .* base);
        c = floor ((z2 - z1)/m .* base);
        # Adjust indexes to be valid
        a -= min (a) - 1;
        b -= min (b) - 1;
        c -= min (c) - 1;
        indexes = sub2ind (dim, a, b, c);

        SE.nhood = false (dim);
        SE.nhood(indexes) = true;
      endif

      SE.flat = true;

    case "octagon"
      if (numel (varargin) == 1)
        SE.opt.apothem = apothem = varargin{1};
      else
        error ("strel: no APOTHEM specified for octagon shape");
      endif
      if (! is_nonnegative_integer (apothem) || mod (apothem, 3) != 0)
        error ("strel: APOTHEM must be a positive integer multiple of 3");
      endif

      ## we look at it as 9 blocks. North AND South are the same and West TO
      ## East as well. We make the corner for NorthEast and rotate it for the
      ## other corners
      if (apothem == 0)
        SE.nhood = true (1);
      else
        cwide    = apothem/3*2 + 1;
        iwide    = apothem/3*2 - 1;
        N_and_S  = true ([cwide iwide]);
        corner   = tril (true (cwide));
        SE.nhood = [rotdim(corner), N_and_S, corner;
                    true([iwide (2*apothem + 1)]);
                    transpose(corner), N_and_S, rotdim(corner, -1)];
      endif
      SE.flat  = true;

    case "pair"
      if (numel (varargin) == 1)
        offset = varargin{1};
      else
        error ("strel: no OFFSET specified for pair shape");
      endif
      if (! isnumeric (offset) || numel (offset) != 2)
        error ("strel: OFFSET must be a 2 element vector");
      elseif (any (fix (offset) != offset))
        error ("strel: OFFSET values must be integers");
      endif

      lengths  = abs (2*offset) + 1;
      SE.nhood = false (lengths);
      origin   = (lengths + 1)/2;
      SE.nhood(origin(1), origin(2)) = true;
      SE.nhood(origin(1) + offset(1), origin(2) + offset(2)) = true;

      SE.flat = true;

    case "periodicline"
      if (numel (varargin) == 2)
        p = varargin{1};
        v = varargin{2};
      else
        error ("strel: a periodic line shape needs 2 arguments");
      endif
      if (! is_positive_integer (p))
        error ("strel: P must be a positive integer");
      elseif (! isnumeric (v) || numel (v) != 2)
        error ("strel: V must be a 2 element vector");
      elseif (any (fix (v) != v))
        error ("strel: values of V must be integers");
      endif

      lengths  = abs (2*p*v) + 1;
      SE.nhood = false (lengths);
      origin   = (lengths + 1)/2;
      for i = -p:p
        point = i*v + origin;
        SE.nhood(point(1), point(2)) = true;
      endfor

    case "rectangle"
      if (numel (varargin) == 1)
        SE.opt.dimensions = varargin{1};
      else
        error ("strel: no DIMENSIONS specified for rectangle shape");
      endif
      if (! isnumeric (SE.opt.dimensions) || numel (SE.opt.dimensions) != 2)
        error ("strel: DIMENSIONS must be a 2 element vector");
      elseif (! is_positive_integer (SE.opt.dimensions(1)) ||
              ! is_positive_integer (SE.opt.dimensions(2)))
        error ("strel: DIMENSIONS values must be positive integers");
      endif

      SE.nhood = true (SE.opt.dimensions);
      SE.flat  = true;

    case "square"
      if (numel (varargin) == 1)
        SE.opt.edge = varargin{1};
      else
        error ("strel: no EDGE specified for square shape");
      endif
      if (! is_positive_integer (SE.opt.edge))
        error ("strel: EDGE value must be positive integers");
      endif

      SE.nhood = true (SE.opt.edge);
      SE.flat  = true;

    otherwise
      error ("strel: unknown SHAPE `%s'", shape);
  endswitch

  SE = class (SE, "strel");
endfunction

function retval = is_positive_integer (val)
  retval = isscalar (val) && isnumeric (val) && val > 0 && fix (val) == val;
endfunction

function retval = is_nonnegative_integer (val)
  retval = isscalar (val) && isnumeric (val) && val >= 0 && fix (val) == val;
endfunction

function retval = sign0positive (val)
  if (sign (val) == -1)
    retval = -1;
  else
    retval = 1;
  endif
endfunction

%!test
%! shape  = logical ([0 0 0 1]);
%! assert (getnhood (strel (shape)), shape);
%! assert (getnhood (strel ("arbitrary", shape)), shape);
%!
%! height = [0 0 0 3];
%! assert (getnhood (strel ("arbitrary", shape, height)), shape);
%! assert (getheight (strel ("arbitrary", shape, height)), height);

%!test
%! shape = logical ([0 0 1]);
%! height = [-2 1 3];  ## this works for matlab compatibility
%! assert (getnhood (strel ("arbitrary", shape, height)), shape);
%! assert (getheight (strel ("arbitrary", shape, height)), height);

%!test
%! shape = logical ([0 0 0 1 0 0 0
%!                   0 1 1 1 1 1 0
%!                   0 1 1 1 1 1 0
%!                   1 1 1 1 1 1 1
%!                   0 1 1 1 1 1 0
%!                   0 1 1 1 1 1 0
%!                   0 0 0 1 0 0 0]);
%! height = [ 0.00000   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000
%!            0.00000   0.33333   0.66667   0.74536   0.66667   0.33333   0.00000
%!            0.00000   0.66667   0.88192   0.94281   0.88192   0.66667   0.00000
%!            0.00000   0.74536   0.94281   1.00000   0.94281   0.74536   0.00000
%!            0.00000   0.66667   0.88192   0.94281   0.88192   0.66667   0.00000
%!            0.00000   0.33333   0.66667   0.74536   0.66667   0.33333   0.00000
%!            0.00000   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000];
%! assert (getnhood (strel ("ball", 3, 1)), shape);
%! assert (getheight (strel ("ball", 3, 1)), height, 0.0001);

%!test
%! shape = logical ([0 0 0 1 0 0 0
%!                   0 0 1 1 1 0 0
%!                   0 1 1 1 1 1 0
%!                   1 1 1 1 1 1 1
%!                   0 1 1 1 1 1 0
%!                   0 0 1 1 1 0 0
%!                   0 0 0 1 0 0 0]);
%! assert (getnhood (strel ("diamond", 3)), shape);

%!test
%! shape = logical ([0 0 0 1 0 0 0
%!                   0 1 1 1 1 1 0
%!                   0 1 1 1 1 1 0
%!                   1 1 1 1 1 1 1
%!                   0 1 1 1 1 1 0
%!                   0 1 1 1 1 1 0
%!                   0 0 0 1 0 0 0]);
%! assert (getnhood (strel ("disk", 3, 0)), shape);

%!test
%! shape = logical ([1 1 1]);
%! assert (getnhood (strel ("line", 3.9, 20.17)), shape);
%! shape = logical ([0 0 1
%!                   0 1 0
%!                   1 0 0]);
%! assert (getnhood (strel ("line", 3.9, 20.18)), shape);
%! shape = logical ([1 0 0 0 0 0 0 0 0
%!                   0 1 0 0 0 0 0 0 0
%!                   0 0 1 0 0 0 0 0 0
%!                   0 0 1 0 0 0 0 0 0
%!                   0 0 0 1 0 0 0 0 0
%!                   0 0 0 0 1 0 0 0 0
%!                   0 0 0 0 0 1 0 0 0
%!                   0 0 0 0 0 0 1 0 0
%!                   0 0 0 0 0 0 1 0 0
%!                   0 0 0 0 0 0 0 1 0
%!                   0 0 0 0 0 0 0 0 1]);
%! assert (getnhood (strel ("line", 14, 130)), shape);

%!test
%! se = strel ("octagon", 0);
%! seq = getsequence (se);
%! assert (getnhood (se), true (1));
%! assert (getnhood (seq(1)), true (1));
%!
%! se = strel ("octagon", 3);
%! seq = getsequence (se);
%! shape = logical ([0 0 1 1 1 0 0
%!                   0 1 1 1 1 1 0
%!                   1 1 1 1 1 1 1
%!                   1 1 1 1 1 1 1
%!                   1 1 1 1 1 1 1
%!                   0 1 1 1 1 1 0
%!                   0 0 1 1 1 0 0]);
%! assert (getnhood (se), shape);
%! assert (size (seq), [4 1]);
%!
%! templ1 = logical ([0 0 0; 1 1 1; 0 0 0]);
%! templ2 = logical ([0 1 0; 0 1 0; 0 1 0]);
%! templ3 = logical ([1 0 0; 0 1 0; 0 0 1]);
%! templ4 = logical ([0 0 1; 0 1 0; 1 0 0]);
%! assert ({getnhood(seq(1)) getnhood(seq(2)) getnhood(seq(3)) getnhood(seq(4))},
%!         {templ1 templ2 templ3 templ4});
%!
%! seq = getsequence (strel ("octagon", 21));
%! assert (size (seq), [28 1]);
%! assert (arrayfun (@(x) getnhood (seq(x)), 1:4:25, "UniformOutput", false),
%!         repmat ({templ1}, 1, 7));
%! assert (arrayfun (@(x) getnhood (seq(x)), 2:4:26, "UniformOutput", false),
%!         repmat ({templ2}, 1, 7));
%! assert (arrayfun (@(x) getnhood (seq(x)), 3:4:27, "UniformOutput", false),
%!         repmat ({templ3}, 1, 7));
%! assert (arrayfun (@(x) getnhood (seq(x)), 4:4:28, "UniformOutput", false),
%!         repmat ({templ4}, 1, 7));

%!test
%! shape = logical ([1 1 0]');
%! assert (getnhood (strel ("pair", [-1 0])), shape);
%! shape = logical ([1 0 0 0 0 0 0
%!                   0 0 0 1 0 0 0
%!                   0 0 0 0 0 0 0]);
%! assert (getnhood (strel ("pair", [-1 -3])), shape);
%! shape = logical ([0 0 0 0 0 0 0
%!                   0 0 0 0 0 0 0
%!                   0 0 0 1 0 0 0
%!                   0 0 0 0 0 0 0
%!                   0 0 0 0 0 0 1]);
%! assert (getnhood (strel ("pair", [2 3])), shape);

%!test
%! assert (getnhood (strel ("rectangle", [10 5])), true (10, 5));
%! assert (getnhood (strel ("square", 5)), true (5));

## test how @strel/getsequence and indexing works fine
%!shared se, seq
%! se = strel ("square", 5);
%! seq = getsequence (se);
%! assert (class (se(1)),  "strel")
%! assert (class (se(1,1)),"strel")
%! assert (class (seq),    "strel")
%! assert (class (seq(1)), "strel")
%! assert (class (seq(2)), "strel")
%! assert (numel (se), 1)
%! assert (numel (seq), 2)
%! assert (getnhood (seq(1)), true (5, 1))
%! assert (getnhood (seq(2)), true (1, 5))
%! assert (size (se),  [1 1])
%! assert (size (seq), [2 1])
%! assert (isscalar (se),  true)
%! assert (isscalar (seq), false)
%!error <index out of bounds> se(2);
%!error <out of bound> seq(3);

## test reflection
%!test
%! se = strel ("arbitrary", [1 0 0; 1 1 0; 0 1 0], [2 0 0; 3 1 0; 0 3 0]);
%! ref = reflect (se);
%! assert (getnhood (ref), logical([0 1 0; 0 1 1; 0 0 1]));
%! assert (getheight (ref), [0 3 0; 0 1 3; 0 0 2]);

## test input validation
%!error strel()
%!error strel("nonmethodthing", 2)
%!error strel("arbitrary", "stuff")
%!error strel("arbitrary", [0 0 1], [2 0 1; 4 5 1])
%!error strel("arbitrary", [0 0 1], "stuff")
%!error strel("ball", -3, 1)
%!error strel("diamond", -3)
%!error strel("disk", -3)
%!error strel("line", 0, 45)
%!error <positive integer multiple of 3> strel("octagon", 3.5)
%!error <positive integer multiple of 3> strel("octagon", 4)
%!error <positive integer multiple of 3> strel("octagon", -1)
%!error strel("pair", [45 67 90])
%!error strel("rectangle", 2)
%!error strel("rectangle", [2 -5])
%!error strel("square", [34 1-2])
