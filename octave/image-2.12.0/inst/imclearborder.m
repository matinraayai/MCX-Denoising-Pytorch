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
## <http:##www.gnu.org/licenses/>.

## -*- texinfo -*-
## @deftypefn  {Function File} {} imclearborder (@var{im})
## @deftypefnx {Function File} {} imclearborder (@var{im}, @var{conn})
## Clear borders of objects or ligher structures.
##
## On the simplest case of binary images, this function removes objects
## that touch the image borders.  In the case of grayscale images, lighter
## regions (higher intensity values) that touch the image border get removed.
##
## To be more exact, this is equivalent to use @code{imreconstruct} using
## the @var{im} borders as marker, and setting all unchanged elements to
## a background value.
##
## Element connectivity @var{conn}, to define the size of objects, can be
## specified with a numeric scalar (number of elements in the neighborhood):
##
## @table @samp
## @item 4 or 8
## for 2 dimensional matrices;
## @item 6, 18 or 26
## for 3 dimensional matrices;
## @end table
##
## or with a binary matrix representing a connectivity array.  Defaults to
## @code{conndef (ndims (@var{bw}), "maximal")} which is equivalent to
## @var{conn} of 8 and 26 for 2 and 3 dimensional matrices respectively.
##
## @seealso{imreconstruct}
## @end deftypefn

function im = imclearborder (im, conn)

  if (nargin < 1 || nargin > 2)
    print_usage ();
  elseif (! isimage (im))
    error ("imclearborder: IM must be an image");
  endif

  if (nargin < 2)
    conn = conndef (ndims (im), "maximal");
  else
    conn = conndef (conn);
  endif

  bg_val  = cast (getrangefromclass (im)(1), class (im));
  marker  = get_borders (im, conn, bg_val);
  border_elems = imreconstruct (marker, im, conn) == im;
  im(border_elems) = bg_val;

endfunction

function borders = get_borders (im, conn, val)
  im_size = size (im);
  borders = repmat (val, im_size);

  tmp_idx       = repmat ({":"}, [1 ndims(im)]);
  tmp_conn_idx  = repmat ({":"}, [1 ndims(conn)]);
  for dim = 1:min (ndims (im), ndims (conn))
    conn_idx = tmp_conn_idx;
    conn_idx{dim} = [1 3];
    if (im_size(dim) == 1 || ! any (conn(conn_idx{:})(:)))
      continue
    endif

    idx = tmp_idx;
    idx{dim} = [1 im_size(dim)];
    borders(idx{:}) = im(idx{:});
  endfor
endfunction

## TODO check what exactly does Matlab do with in grayscale images, specially
##      in the case of signed integers and floating point with negative values.
##      We are different from the Matlab documentation suggests but that's
##      because Matlab documentation sounds wrong to me.

%!test
%! a = logical ([
%!    0   1   0   0   1   0   0   0   0   1
%!    1   0   0   0   0   1   0   0   0   0
%!    0   1   0   0   0   0   0   0   0   0
%!    1   0   1   0   1   0   1   0   0   1
%!    0   0   0   0   0   0   0   1   1   0
%!    0   0   1   0   0   1   0   1   0   0
%!    0   1   0   1   0   1   1   0   0   0
%!    0   0   0   1   0   0   0   0   0   0
%!    0   0   0   1   0   1   1   0   0   0
%!    0   0   0   1   1   0   0   0   1   0]);
%!
%! a4 = logical ([
%!    0   0   0   0   0   0   0   0   0   0
%!    0   0   0   0   0   1   0   0   0   0
%!    0   1   0   0   0   0   0   0   0   0
%!    0   0   1   0   1   0   1   0   0   0
%!    0   0   0   0   0   0   0   1   1   0
%!    0   0   1   0   0   1   0   1   0   0
%!    0   1   0   0   0   1   1   0   0   0
%!    0   0   0   0   0   0   0   0   0   0
%!    0   0   0   0   0   1   1   0   0   0
%!    0   0   0   0   0   0   0   0   0   0]);
%!
%! a8 = logical ([
%!    0   0   0   0   0   0   0   0   0   0
%!    0   0   0   0   0   0   0   0   0   0
%!    0   0   0   0   0   0   0   0   0   0
%!    0   0   0   0   1   0   0   0   0   0
%!    0   0   0   0   0   0   0   0   0   0
%!    0   0   0   0   0   0   0   0   0   0
%!    0   0   0   0   0   0   0   0   0   0
%!    0   0   0   0   0   0   0   0   0   0
%!    0   0   0   0   0   0   0   0   0   0
%!    0   0   0   0   0   0   0   0   0   0]);
%!
%! assert (imclearborder (a, 4), a4)
%! assert (imclearborder (a, [0 1 0; 1 1 1; 0 1 0]), a4)
%! assert (imclearborder (a), a8)
%! assert (imclearborder (a, 8), a8)
%! assert (imclearborder (a, ones (3)), a8)

%!test
%! a = false (5, 5, 3);
%! a(2:4,2:4,:) = true;
%! assert (imclearborder (a, 4), a)
%!
%! a(1,2) = true;
%! a4 = a;
%! a4(:,:,1) = false;
%! assert (imclearborder (a, 4), a4)
