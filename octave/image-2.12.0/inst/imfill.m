## Copyright (C) 2016 Hartmut Gimpel <hg_code@gmx.de>
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*-
## @deftypefn  {Function File} {@var{bw2} =} imfill (@var{bw}, "holes")
## @deftypefnx {Function File} {@var{bw2} =} imfill (@var{bw}, @var{conn}, "holes")
## @deftypefnx {Function File} {@var{bw2} =} imfill (@var{bw}, @var{locations})
## @deftypefnx {Function File} {@var{bw2} =} imfill (@var{bw}, @var{locations}, @var{conn})
## @deftypefnx {Function File} {@var{I2} =} imfill (@var{I})
## @deftypefnx {Function File} {@var{I2} =} imfill (@var{I}, "holes")
## @deftypefnx {Function File} {@var{I2} =} imfill (@var{I}, @var{conn})
## @deftypefnx {Function File} {@var{I2} =} imfill (@var{I}, @var{conn}, "holes")
## Fill image holes or regions.
##
## The image to be filled can be binary @var{bw} or grayscale @var{I}.
## Regions are a series of connected elements whose values are lower than
## at least one of its neighbor elements.
##
## The option @qcode{"holes"}, default for grayscale images, can be used
## to fill all holes, i.e., regions without border elements.
##
## Alternatively, the argument @var{locations} are coordinates for elements
## in the regions to be filled.  It can be a a @var{n}-by-1 vector of linear
## indices or a @var{n}-by-@var{d} matrix of subscript indices where @var{n}
## is the number of points and @var{d} is the number of dimensions in the
## image.
##
## The argument @var{conn} specifies the connectivity used for filling
## the region and defaults to
## @code{conndef (ndims (@var{img}, @qcode{"minimal"})} which is 4 for
## the case of 2 dimensional images.
##
## @example
## im = [0  1  0
##       1  0  1
##       0  1  0];
##
## imfill (im, "holes")
## ans =
##
##  0  1  0
##  1  1  1
##  0  1  0
## @end example
##
## @seealso{bwfill, bwselect, imreconstruct}
## @end deftypefn

function filled = imfill (img, varargin)

  if (nargin < 1 || nargin > 3)
    print_usage ();
  endif

  if (! isimage (img))
    error ("imfill: IMG and BW must be a binary or grayscale image");
  endif

  ## Default parameter values
  conn = conndef (ndims (img), "minimal");
  fill_holes = false;

  if (nargin () == 1)
    if (islogical (img))
      ## imfill (BW)
      error ("imfill: interactive usage is not yet supported");
    else
      ## syntax: imfill (img)
      fill_holes = true;
    endif

  elseif (nargin () == 2)
    opt2 = varargin{1};
    if (ischar (opt2))
      ## syntax: imfill (BW, "holes") or imfill (IMG, "holes")
      validatestring (opt2, {"holes"}, "imfill", "OPTION");
      fill_holes = true;
    elseif (! islogical (img))
      ## syntax: imfill (IMG, CONN)
      fill_holes = true;
      iptcheckconn (opt2, "imfill", "CONN");
      conn = opt2;
    elseif (islogical (img) && isnumeric (opt2) && isindex (opt2))
      ## syntax: imfill (BW, LOCATIONS)
      locations = check_loc (opt2, img);
    else
      error ("imfill: second argument must be 'holes', a connectivity specification, or an index array");
    endif

  elseif (nargin () == 3)
    opt2 = varargin{1};
    opt3 = varargin{2};
    if (ischar (opt3))
      ## syntax: imfill (BW, CONN, "holes") or imfill (IMG, CONN, "holes")
      validatestring (opt3, {"holes"}, "imfill", "OPTION");
      iptcheckconn (opt2, "imfill", "CONN");
      conn = opt2;
      fill_holes = true;
    elseif (islogical (img) && isnumeric (opt2) && isindex (opt2))
      ## syntax: imfill (BW, LOCATIONS, CONN)
      iptcheckconn(opt3, "imfill", "CONN");
      conn = opt3;
      locations = check_loc (opt2, img);
    elseif (islogical (img) && isscalar (opt2) && opt2 == 0)
        ## syntax: imfill (BW, 0, CONN)
        error ("imfill: interactive usage is not yet supported");
    else
      print_usage ();
    endif
  endif


  if (fill_holes)
    ##  Hole filling algorithm adapted from the book Gonzalez & Woods
    ## "Digital Image Processing" (3rd Edition), section 9.5.9
    ## "Morphological Reconstruction", subsection "Filling holes" (page 682):
    ##
    ##      "Let I(x,y) denote a binary image and suppose that we form a
    ##      marker image F that is 0 everywhere, except at the image border
    ##      where it is set to 1-I; that is,
    ##
    ##                   / 1 - I(x,y) if (x,y) is on the border of I
    ##          F(x,y) = |
    ##                   \ 0          otherwise
    ##
    ##      Then
    ##
    ##          H = complement (reconstruct (F, complement (I)))
    ##
    ##      is a binary image equal to I with all holes filled."
    ##
    ## This is generalized to grayscale images with:
    ##
    ##      * (1-I) -> complement(I)
    ##      * 0 value -> -Inf value if non logical
    ##
    ##   step 1: define the mask and marker as complement of input
    ##   step 2: set border elements of marker as false/-Inf
    ##   step 3: do morphological reconstruction
    ##   step 4: complement of reconstruction has the holes filled.
    ##
    ## Beware of unusual connectivity definition which changes what is
    ## defined as border pixels.  For example, [0 0 0; 1 1 1; 0 0 0],
    ## means that top and bottom row of an image are not border pixels.

    mask = imcomplement (img);

    if (islogical (img))
      marker = set_nonborder_pixels (mask, conn, false);
    else
      marker = set_nonborder_pixels (mask, conn, -Inf);
    endif

    filled = imreconstruct (marker, mask, conn);
    filled = imcomplement (filled);

  else
    ## Using explicitly given marker pixels instead of "holes" option
    ## should only be used for logical images
    ## no border padding necessary in this case

    mask = imcomplement (img);

    marker = false (size (img));
    marker (locations) = mask (locations);

    filled = imreconstruct (marker, mask, conn);

    ## Adjusted step 4: add the filled hole(s) FILLED to the original image IMG
    filled = img | filled;
  endif

endfunction

## Helper function for LOCATIONS input checking.
## Drops locations outside of the image and proceeds with a warning.
## Additionally transforms matrix indices to linear indices.
function loc_lin_idx = check_loc (loc, img)
  sz = size (img);
  if (ndims (loc) != 2)
    error ("imfill: LOCATIONS must be a n-by-1 or n-by-d sized index array");
  elseif (columns (loc) == 1)  # linear indices given
    idx_outside = loc > numel (img);
    loc(idx_outside) = [];
    loc_lin_idx = loc;
  elseif (columns (loc) == ndims (img)) # subscript indices given
    idx_outside = any (loc > sz, 2);
    loc(idx_outside,:) = [];
    loc_lin_idx = sub2ind (sz, num2cell (loc, 1){:});
  else
    error ("imfill: LOCATIONS must be a n-by-1 or n-by-d sized index array");
  endif

  if (any (idx_outside))
    warning ("imfill: ignored LOCATIONS outside of image borders");
  endif
endfunction

## Set non border of pixels of IMG according to connectivity CONN to VAL.
## The only tricky part is handling unusual connectivity such as
## [0 0 0; 1 1 1; 0 0 0] which define top and bottom row as non border.
## Another case is 8 connectivity in 3d images.  In such case, the first
## row and column of each page is border elements, but the first and last
## page itself are not.
function marker = set_nonborder_pixels (img, conn, val)
  sz = size (img);

  nonborder_idx = repmat ({":"}, ndims (img), 1);

  conn_idx_tmp = repmat ({":"}, ndims (conn), 1);
  for dim = 1:ndims (conn)
    conn_idx = conn_idx_tmp;
    ## Because connectivity is symmetric by definition, we only need
    ## to check the first slice of that dimension.
    conn_idx{dim} = [1];
    if (any (conn(conn_idx{:})(:)))
      nonborder_idx{dim} = 2:(sz(dim) -1);
    endif
  endfor

  marker = img;
  marker(nonborder_idx{:}) = val;
endfunction


## test the possible INPUT IMAGE TYPES
%!test
%! I = uint8 (5.*[1 1 1; 1 0 1; 1 1 1]);
%! bw = logical ([1 1 1; 1 0 1; 1 1 1]);
%! I2 = uint8 (5.*ones (3));
%! bw2 = logical (ones (3));
%!
%! assert (imfill (int8 (I)), int8 (I2))
%! assert (imfill (int16 (I)), int16 (I2))
%! assert (imfill (int32 (I)), int32 (I2))
%! assert (imfill (int64 (I)), int64 (I2))
%! assert (imfill (uint8 (I)), uint8 (I2))
%! assert (imfill (uint16 (I)), uint16 (I2))
%! assert (imfill (uint32 (I)), uint32 (I2))
%! assert (imfill (uint64 (I)), uint64 (I2))
%! assert (imfill (single (I)), single (I2))
%! assert (imfill (double (I)), double (I2))
%! assert (imfill (bw, "holes"), bw2)
%! assert (imfill (uint8 (bw)), uint8 (bw2))

%!error <must be a binary or grayscale image>
%!  imfill (i + ones (3, 3));                 # complex input

%!error <must be a binary or grayscale image>
%!  imfill (sparse (double (I)));   # sparse input

%!error
%!  imfill ();

%!error
%! imfill (true (3), 4, "holes", 5)

%!error <LOCATIONS must be a n-by-1 or n-by-d sized index array>
%! imfill (false (3), ones (2, 3))

%!error <LOCATIONS must be a n-by-1 or n-by-d sized index array>
%! imfill (false (3), ones (2, 3), 4)

%!error <interactive usage is not yet supported>
%! imfill (false (3))

%!error <interactive usage is not yet supported>
%! imfill (false (3), 0, 4)

%!warning <ignored LOCATIONS outside of image borders>
%! bw = logical ([1 1 1; 1 0 1; 1 1 1]);
%! assert (imfill (bw, [5 5]), bw)
%! assert (imfill (bw, 15), bw)
%!
%! bw = repmat (bw, [1 1 3]);
%! assert (imfill (bw, 30), bw)
%! assert (imfill (bw, [2 2 5]), bw)

## test BINARY hole filling and binary filling from starting point
%!test
%! bw = logical ([1 0 0 0 0 0 0 0
%!                1 1 1 1 1 0 0 0
%!                1 0 0 0 1 0 1 0
%!                1 0 0 0 1 1 1 0
%!                1 1 1 1 0 1 1 1
%!                1 0 0 1 1 0 1 0
%!                1 0 0 0 1 0 1 0
%!                1 0 0 0 1 1 1 0]);
%! bw2 = logical ([1 0 0 0 0 0 0 0
%!                 1 1 1 1 1 0 0 0
%!                 1 1 1 1 1 0 1 0
%!                 1 1 1 1 1 1 1 0
%!                 1 1 1 1 1 1 1 1
%!                 1 0 0 1 1 1 1 0
%!                 1 0 0 0 1 1 1 0
%!                 1 0 0 0 1 1 1 0]);
%! bw3 = logical ([1 0 0 0 0 0 0 0
%!                 1 1 1 1 1 0 0 0
%!                 1 1 1 1 1 0 1 0
%!                 1 1 1 1 1 1 1 0
%!                 1 1 1 1 0 1 1 1
%!                 1 0 0 1 1 0 1 0
%!                 1 0 0 0 1 0 1 0
%!                 1 0 0 0 1 1 1 0]);
%! assert (imfill (bw, "holes"), bw2)
%! assert (imfill (bw, 8, "holes"), bw2)
%! assert (imfill (bw, 4, "holes"), bw2)
%! assert (imfill (bw, [3 3]), bw3)
%! assert (imfill (bw, 19), bw3)
%! assert (imfill (bw, [3 3], 4), bw3)
%! assert (imfill (bw, 19, 4), bw3)
%! assert (imfill (bw, [3 3], 8), bw2)
%! assert (imfill (bw, 19, 8), bw2)
%! assert (imfill (bw, [19; 20]), bw3)
%! assert (imfill (bw, [19; 20], 4), bw3)
%! assert (imfill (bw, [19; 20], 8), bw2)

%!warning <ignored LOCATIONS outside of image borders>
%! bw = logical ([1 1 1 1 1 1 1
%!                1 0 0 0 0 0 1
%!                1 0 1 1 1 0 1
%!                1 0 1 0 1 0 1
%!                1 0 1 1 1 0 1
%!                1 0 0 0 0 0 1
%!                1 1 1 1 1 1 1]);
%! bw44 = logical ([1 1 1 1 1 1 1
%!                  1 0 0 0 0 0 1
%!                  1 0 1 1 1 0 1
%!                  1 0 1 1 1 0 1
%!                  1 0 1 1 1 0 1
%!                  1 0 0 0 0 0 1
%!                  1 1 1 1 1 1 1]);
%! bw9 = logical ([1 1 1 1 1 1 1
%!                 1 1 1 1 1 1 1
%!                 1 1 1 1 1 1 1
%!                 1 1 1 0 1 1 1
%!                 1 1 1 1 1 1 1
%!                 1 1 1 1 1 1 1
%!                 1 1 1 1 1 1 1]);
%! assert (imfill (bw, "holes"), logical (ones (7)))
%! assert (imfill (bw, [4 4]), bw44)
%! assert (imfill (bw, 9), bw9)
%! assert (imfill (bw, [4 4; 10 10]), bw44)

%!test
%! bw = logical ([1 1 0 1 1]);
%! assert (imfill (bw, "holes"), bw)
%! bw = logical([1 1 0 1 1; 1 1 1 1 1]);
%! assert (imfill (bw, "holes"), bw)

## test hole filling with extravagant connectivity definitions
%!test
%! I = zeros (5);
%! I(:, [2 4]) = 1;
%! I2_expected = [0   1   1   1   0
%!                0   1   1   1   0
%!                0   1   1   1   0
%!                0   1   1   1   0
%!                0   1   1   1   0];
%! I2 = imfill (I, [0 0 0; 1 1 1; 0 0 0], "holes");
%! assert (I2, I2_expected)

%!test
%! I = zeros (5);
%! I(:, [2 4]) = 1;
%! I2_expected = I;
%! I2 = imfill (I, [0 1 0; 0 1 0; 0 1 0], "holes");
%! assert (I2, I2_expected)

%!test  # this test is Matlab compatible
%! I = zeros (5);
%! I(:, [2 4]) = 1;
%! I2_expected = inf .* ones (5);
%! I2 =  imfill (I, [0 0 0; 0 1 0; 0 0 0], "holes");
%! assert (I2, I2_expected)

%!test
%! I = false (5);
%! I(:, [2 4]) = true;
%! I2_expected = true (5);
%! I2 = imfill (I, [0 0 0; 0 1 0; 0 0 0], "holes");
%! assert (I2, I2_expected)

## test GRAYSCALE hole filling
%!test
%! I  = uint8 ([10 20 80 85 20
%!              15 90 03 25 88
%!              05 85 02 50 83
%!              90 04 03 80 80
%!             10 81 83 85 30]);
%! I2 = uint8 ([10 20 80 85 20
%!              15 90 80 80 88
%!              05 85 80 80 83
%!              90 80 80 80 80
%!             10 81 83 85 30]);
%! I3  = uint8 ([10 20 80 85 20
%!               15 90 05 25 88
%!               05 85 05 50 83
%!               90 05 05 80 80
%!               10 81 83 85 30]);
%! assert (imfill (I), I2)
%! assert (imfill (I, 4), I2)
%! assert (imfill (I, 4, "holes"), I2)
%! assert (imfill (I, 8), I3)
%! assert (imfill (I, "holes"), I2)

## Test dimensions of length 1 whose extremes may or may not be on the
## border due to less typical connectivity.
%!test
%! v_line = [0 1 0; 0 1 0; 0 1 0];
%! h_line = [0 0 0; 1 1 1; 0 0 0];
%! im = [0 1 0 0 1 0];
%!
%! assert (imfill (im, h_line, "holes"), [0 1 1 1 1 0])
%! assert (imfill (im, v_line, "holes"), [0 1 0 0 1 0])
%! assert (imfill (im', h_line, "holes"), [0 1 0 0 1 0]')
%! assert (imfill (im', v_line, "holes"), [0 1 1 1 1 0]')
%!
%! im = repmat (im, [1 1 5]);
%! assert (imfill (im, h_line, "holes"), repmat ([0 1 1 1 1 0], [1 1 5]))
%! assert (imfill (im, v_line, "holes"), im)
%!
%! im = permute (im, [2 1 3]);
%! assert (imfill (im, h_line, "holes"), im)
%! assert (imfill (im, v_line, "holes"), repmat ([0 1 1 1 1 0]', [1 1 5]))

%!test
%! im = logical ([0 0 0 0 0 0
%!                0 1 1 1 1 0
%!                0 1 0 0 1 0
%!                0 1 1 1 1 0
%!                0 0 0 0 0 0]);
%! fi = logical ([0 0 0 0 0 0
%!                0 1 1 1 1 0
%!                0 1 1 1 1 0
%!                0 1 1 1 1 0
%!                0 0 0 0 0 0]);
%!
%! assert (imfill (cat (3, im, im, im), 8, 'holes'), cat (3, fi, fi, fi))
%! assert (imfill (cat (3, im, im, im), 'holes'), cat (3, im, im, im))
%! assert (imfill (cat (3, fi, im, fi), 'holes'), cat (3, fi, fi, fi))

%!test
%! emp = false (5, 6);
%!  im = logical ([0 0 0 0 0 0
%!                 0 1 1 1 1 0
%!                 0 1 0 1 0 1
%!                 0 1 1 1 1 0
%!                 0 0 0 0 0 0]);
%!  fi = logical ([0 0 0 0 0 0
%!                 0 1 1 1 1 0
%!                 0 1 1 1 1 1
%!                 0 1 1 1 1 0
%!                 0 0 0 0 0 0]);
%!  fi1 = logical ([0 0 0 0 0 0
%!                  0 1 1 1 1 0
%!                  0 1 1 1 0 1
%!                  0 1 1 1 1 0
%!                  0 0 0 0 0 0]);
%!  fi2 = logical ([0 0 0 0 0 0
%!                  0 1 1 1 1 0
%!                  0 1 0 1 1 1
%!                  0 1 1 1 1 0
%!                  0 0 0 0 0 0]);
%!
%! assert (imfill (cat (3, im, im, im), [3 3 2]), cat (3, fi1, fi1, fi1))
%! assert (imfill (cat (3, im, im, im), [3 5 2]), cat (3, fi2, fi2, fi2))
%! assert (imfill (cat (3, im, im, im), [3 3 2; 3 5 2]), cat (3, fi, fi, fi))
%! assert (imfill (cat (3, emp, im, emp), [3 3 2]), true (5, 6, 3))
