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
## @deftypefn  {Function File} {} bwareafilt (@var{bw}, @var{range})
## @deftypefnx {Function File} {} bwareafilt (@var{bw}, @var{n})
## @deftypefnx {Function File} {} bwareafilt (@var{bw}, @var{n}, @var{keep})
## @deftypefnx {Function File} {} bwareafilt (@dots{}, @var{conn})
## Filter objects from image based on their sizes.
##
## Returns a logical matrix with the objects of @var{bw} filtered based
## on their area (defined by thei number of pixels).  This function is
## equivalent to @code{bwpropfilt (@var{bw}, "Area", @dots{})}.
##
## To filter objects with a value on a specific interval, @var{range} must be
## a two-element vector with the interval @code{[@var{low} @var{high}]}
## (values are inclusive).
##
## Alternatively, a scalar @var{n} will select the objects with the N highest
## values.  The @var{keep} option defaults to @qcode{"largest"} but can also
## be set to @qcode{"smallest"} to select the N objects with lower values.
##
## The last optional argument, @var{conn}, can be a connectivity matrix, or
## the number of elements connected to the center (see @command{conndef}).
##
## @seealso{bwareaopen, bwlabel, bwlabeln, bwconncomp, bwpropfilt, regionprops}
## @end deftypefn

function bwfiltered = bwareafilt (bw, varargin)
  if (nargin < 2 || nargin > 4)
    print_usage ();
  endif
  bwfiltered = bwpropfilt (bw, "Area", varargin{:});
endfunction

%!shared a2d, a3d
%! a2d = [1   0   0   0   0   0   1   0   0   1
%!        1   0   0   1   0   1   0   1   0   1
%!        1   0   1   0   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   1   0   0   0   0   0   0   0   0
%!        1   1   0   1   1   1   0   0   0   0
%!        1   1   0   1   0   0   0   1   0   0
%!        1   1   0   0   0   0   1   0   1   0
%!        1   1   0   0   0   0   0   0   0   0
%!        1   1   0   0   0   1   1   0   0   1];
%!
%! a3d = a2d;
%! a3d(:,:,2) = [
%!        0   0   0   0   0   0   0   0   0   0
%!        1   0   0   1   1   0   0   1   0   0
%!        0   0   0   1   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   1   0   0   0   0   0   0   0   0
%!        1   1   0   0   1   1   0   0   0   0
%!        1   1   0   1   0   0   0   0   0   0
%!        1   0   0   0   0   0   1   0   0   0
%!        0   1   0   0   0   0   0   0   0   1
%!        1   1   0   0   0   0   1   0   0   0];
%!
%! a3d(:,:,3) = [
%!        1   0   0   0   0   0   0   0   0   0
%!        0   1   0   1   1   0   0   1   0   0
%!        0   0   0   1   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   0   0   1   1   1   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        1   0   0   0   0   0   0   0   0   0
%!        1   1   0   0   0   0   0   0   0   1
%!        1   1   0   0   0   0   0   0   0   0];

%!test
%! f2d = [0   0   0   0   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   1   0   0   0   0   0   0   0   0
%!        1   1   0   1   1   1   0   0   0   0
%!        1   1   0   1   0   0   0   0   0   0
%!        1   1   0   0   0   0   0   0   0   0
%!        1   1   0   0   0   0   0   0   0   0
%!        1   1   0   0   0   0   0   0   0   0];
%! assert (bwareafilt (a2d, 2), logical (f2d));
%! assert (bwareafilt (a2d, 2, 8), logical (f2d));
%! assert (bwareafilt (a2d, 2, 4), logical (f2d));

%!test
%! f2d = [1   0   0   0   0   0   1   0   0   0
%!        1   0   0   0   0   1   0   1   0   0
%!        1   0   0   0   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   1   0   0   0   0   0   0   0   0
%!        1   1   0   1   1   1   0   0   0   0
%!        1   1   0   1   0   0   0   1   0   0
%!        1   1   0   0   0   0   1   0   1   0
%!        1   1   0   0   0   0   0   0   0   0
%!        1   1   0   0   0   0   0   0   0   0];
%! assert (bwareafilt (a2d, 5), logical (f2d));
%! assert (bwareafilt (a2d, 5, 8), logical (f2d));

%!test
%! f2d = [0   0   0   0   0   0   1   0   0   1
%!        0   0   0   1   0   1   0   1   0   1
%!        0   0   1   0   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   1   0   0
%!        0   0   0   0   0   0   1   0   1   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   0   0   0   0   1   1   0   0   1];
%! assert (bwareafilt (a2d, 11, "smallest", 4), logical (f2d));

%!test
%! f2d = [1   0   0   0   0   0   1   0   0   0
%!        1   0   0   0   0   1   0   1   0   0
%!        1   0   0   0   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   0   0   1   1   1   0   0   0   0
%!        0   0   0   1   0   0   0   1   0   0
%!        0   0   0   0   0   0   1   0   1   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0];
%! assert (bwareafilt (a2d, [3 5]), logical (f2d));
%! assert (bwareafilt (a2d, [3 5], 8), logical (f2d));

%!test
%! f2d = [1   0   0   0   0   0   0   0   0   0
%!        1   0   0   0   0   0   0   0   0   0
%!        1   0   0   0   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   0   0   1   1   1   0   0   0   0
%!        0   0   0   1   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0];
%! assert (bwareafilt (a2d, [3 4], 4), logical (f2d));
%! assert (bwareafilt (a2d, [3 4], [0 1 0; 1 1 1; 0 1 0]), logical (f2d));

%!test
%! f2d = [1   0   0   0   0   0   1   0   0   1
%!        1   0   0   1   0   1   0   1   0   1
%!        1   0   1   0   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   0   0   1   1   1   0   0   0   0
%!        0   0   0   1   0   0   0   1   0   0
%!        0   0   0   0   0   0   1   0   1   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   0   0   0   0   1   1   0   0   0];
%! assert (bwareafilt (a2d, [2 4]), logical (f2d));
%! assert (bwareafilt (a2d, [2 4], 8), logical (f2d));
%! assert (bwareafilt (a2d, [2 4], ones (3)), logical (f2d));

%!test
%! f3d = [0   0   0   0   0   0   1   0   0   0
%!        0   0   0   1   0   1   0   1   0   0
%!        0   0   1   0   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   1   0   0   0   0   0   0   0   0
%!        1   1   0   0   0   0   0   0   0   0
%!        1   1   0   0   0   0   0   0   0   0
%!        1   1   0   0   0   0   0   0   0   0
%!        1   1   0   0   0   0   0   0   0   0
%!        1   1   0   0   0   0   0   0   0   0];
%!
%! f3d(:,:,2) = [
%!        0   0   0   0   0   0   0   0   0   0
%!        0   0   0   1   1   0   0   1   0   0
%!        0   0   0   1   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   1   0   0   0   0   0   0   0   0
%!        1   1   0   0   0   0   0   0   0   0
%!        1   1   0   0   0   0   0   0   0   0
%!        1   0   0   0   0   0   0   0   0   0
%!        0   1   0   0   0   0   0   0   0   0
%!        1   1   0   0   0   0   0   0   0   0];
%!
%! f3d(:,:,3) = [
%!        0   0   0   0   0   0   0   0   0   0
%!        0   0   0   1   1   0   0   1   0   0
%!        0   0   0   1   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        1   0   0   0   0   0   0   0   0   0
%!        1   1   0   0   0   0   0   0   0   0
%!        1   1   0   0   0   0   0   0   0   0];
%! assert (bwareafilt (a3d, 2), logical (f3d));
%! assert (bwareafilt (a3d, 2, 26), logical (f3d));
%! assert (bwareafilt (a3d, 2, ones (3, 3, 3)), logical (f3d));

%!test
%! f3d = [0   0   0   0   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   1   0   0   0   0   0   0   0   0
%!        1   1   0   1   1   1   0   0   0   0
%!        1   1   0   1   0   0   0   0   0   0
%!        1   1   0   0   0   0   0   0   0   0
%!        1   1   0   0   0   0   0   0   0   0
%!        1   1   0   0   0   0   0   0   0   0];
%!
%! f3d(:,:,2) = [
%!        0   0   0   0   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   1   0   0   0   0   0   0   0   0
%!        1   1   0   0   1   1   0   0   0   0
%!        1   1   0   1   0   0   0   0   0   0
%!        1   0   0   0   0   0   0   0   0   0
%!        0   1   0   0   0   0   0   0   0   0
%!        1   1   0   0   0   0   0   0   0   0];
%!
%! f3d(:,:,3) = [
%!        0   0   0   0   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   0   0   1   1   1   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        1   0   0   0   0   0   0   0   0   0
%!        1   1   0   0   0   0   0   0   0   0
%!        1   1   0   0   0   0   0   0   0   0];
%! assert (bwareafilt (a3d, 2, 6), logical (f3d));
%! assert (bwareafilt (a3d, 2, conndef (3, "minimal")), logical (f3d));

