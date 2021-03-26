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
## @deftypefn  {Function File} {} imregionalmax (@var{img})
## @deftypefnx {Function File} {} imregionalmax (@var{img}, @var{conn})
## Compute regional maxima.
##
## Returns a logical matrix, same size as the input @var{img}, with the
## regional maxima.
##
## The optional argument @var{conn}, defines the connectivity.  It can
## be a scalar value or a boolean matrix (see @code{conndef} for details).
## Defaults to @code{conndef (ndims (@var{img}), "maximal")}
##
## Regional maxima should not be mistaken with local maxima.  Local maxima
## are pixels whose value is greater or equal to all of its neighbors.
## A regional maxima is the connected component of pixels whose values are
## all higher than the neighborhood of the maxima (the connected component,
## not its individual pixels).
## All pixels belonging to a regional maximum are local maxima, but the
## inverse is not true.
##
## @seealso{immaximas, imreconstruct, imregionalmin}
## @end deftypefn

function bw = imregionalmax (img, conn)
  if (nargin < 1 || nargin > 2)
    print_usage ();
  endif

  if (nargin < 2)
    conn = conndef (ndims (img), "maximal");
  else
    conn = conndef (conn);
  endif

  if (islogical (img))
    bw = img;
  else
    ## we could probably still make this more efficient
    if (isfloat (img))
      recon = imreconstruct (img, img + __eps__ (img), conn);
    else
      recon = imreconstruct (img, img + 1, conn);
    endif
    bw = (recon == img);
  endif
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
%! a4 = [
%!    1    0    1    0    1    0
%!    0    0    0    1    0    0
%!    0    0    0    0    0    1
%!    1    0    1    0    0    0
%!    0    1    0    0    0    0
%!    0    0    1    0    0    1];
%! assert (imregionalmax (a, 4), logical (a4))
%! a8 = [
%!    1    0    0    0    1    0
%!    0    0    0    1    0    0
%!    0    0    0    0    0    1
%!    0    0    0    0    0    0
%!    0    0    0    0    0    0
%!    0    0    0    0    0    1];
%! assert (imregionalmax (a, 8), logical (a8))
%! assert (imregionalmax (a), logical (a8))

%!test
%! ## test float input images <bug #51724>
%! im0 = peaks ();
%! im1 = im0 ./ 100;
%! max_pos_expected = [1000; 1214; 1691; 2353];
%! max0 = imregionalmax (im0);
%! max0_pos = find (max0);
%! max1 = imregionalmax (im1);
%! assert (max1, max0)
%! assert (max0_pos, max_pos_expected)
