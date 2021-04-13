## Copyright (C) 2014 Carnë Draug <carandraug@octave.org>
##
## This program is free software; you can redistribute it and/or
## modify it under the terms of the GNU General Public License as
## published by the Free Software Foundation; either version 3 of the
## License, or (at your option) any later version.
##
## This program is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
## General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program; if not, see
## <http://www.gnu.org/licenses/>.

## -*- texinfo -*-
## @deftypefn  {Function File} {} imregionalmin (@var{img})
## @deftypefnx {Function File} {} imregionalmin (@var{img}, @var{conn})
## Compute regional minima.
##
## Returns a logical matrix, same size as the input @var{img}, with the
## regional minima.
##
## The optional argument @var{conn}, defines the connectivity.  It can
## be a scalar value or a boolean matrix (see @code{conndef} for details).
## Defaults to @code{conndef (ndims (@var{img}), "maximal")}
##
## Regional minima should not be mistaken with local minima.  Local minima
## are pixels whose value is less or equal to all of its neighbors.
## A regional minima is the connected component of pixels whose values are
## all less than the neighborhood of the minima (the connected component,
## not its individual pixels).
## All pixels belonging to a regional minima are local minima, but the
## inverse is not true.
##
## @seealso{imreconstruct, imregionalmax}
## @end deftypefn

function bw = imregionalmin (img, conn)
  if (nargin < 1 || nargin > 2)
    print_usage ();
  endif

  if (nargin < 2)
    conn = conndef (ndims (img), "maximal");
  else
    conn = conndef (conn);
  endif

  if (isfloat (img))
    img = - img;
  else
    img = imcomplement (img);
  endif
  bw = imregionalmax (img, conn);
endfunction

%!test
%! a = [
%!    7    3    9    3   10    3
%!    4    2    3   10    1    3
%!    1    4    6    9    4   10
%!    8    7    9    3    4    8
%!    5    9    3    3    8    9
%!    3    6    9    4    1   10];
%!
%! a4 = logical ([
%!    0    0    0    1    0    0
%!    0    1    0    0    1    0
%!    1    0    0    0    0    0
%!    0    0    0    1    0    0
%!    0    0    1    1    0    0
%!    1    0    0    0    1    0]);
%! assert (imregionalmin (a, 4), a4)
%! assert (imregionalmin (uint8 (a), 4), a4)
%! assert (imregionalmin (int8 (a), 4), a4)
%!
%! a8 = logical ([
%!    0    0    0    0    0    0
%!    0    0    0    0    1    0
%!    1    0    0    0    0    0
%!    0    0    0    0    0    0
%!    0    0    0    0    0    0
%!    1    0    0    0    1    0]);
%! assert (imregionalmin (a), a8)
%! assert (imregionalmin (a, 8), a8)
%! assert (imregionalmin (uint8 (a), 8), a8)
%! assert (imregionalmin (int8 (a), 8), a8)

%!test
%! a = [
%!   4   8   5  -1   8   7
%!  -1   4   0   7   1   1
%!   6   1   2   6   7   0
%!   6   1   5  -2   5   9
%!   1   4  -1   0   0   2
%!   4   6   1   0   7   1];
%!
%! a4 = logical ([
%!   0   0   0   1   0   0
%!   1   0   1   0   0   0
%!   0   1   0   0   0   1
%!   0   1   0   1   0   0
%!   1   0   1   0   0   0
%!   0   0   0   0   0   1]);
%! assert (imregionalmin (a, 4), a4)
%! assert (imregionalmin (int8 (a), 4), a4)
%!
%! a8 = logical ([
%!   0   0   0   1   0   0
%!   1   0   0   0   0   0
%!   0   0   0   0   0   1
%!   0   0   0   1   0   0
%!   0   0   0   0   0   0
%!   0   0   0   0   0   0]);
%! assert (imregionalmin (a), a8)
%! assert (imregionalmin (a, 8), a8)
%! assert (imregionalmin (int8 (a), 8), a8)

%!test
%! ## test float input images <bug #51724>
%! im0 = peaks ();
%! im1 = im0 ./ 100;
%! max_pos_expected = [1; 49; 664; 1286; 1302; 2401];
%! max0 = imregionalmin (im0);
%! max0_pos = find (max0);
%! max1 = imregionalmin (im1);
%! assert (max1, max0)
%! assert (max0_pos, max_pos_expected)
