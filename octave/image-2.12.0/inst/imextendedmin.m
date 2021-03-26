## Copyright (C) 2017 Hartmut Gimpel <hg_code@gmx.de>
##
## This program is free software; you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
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
## @deftypefn  {Function File} {} @ imextendedmin (@var{im}, @var{h})
## @deftypefnx {Function File} {} @ imextendedmin (@var{im}, @var{h}, @var{conn})
## Caculate the (morphological) extended minima of an image @var{im}.
##
## This function returns a binary image that marks the extended minima
## of the input image @var{im}. Those extended  minima are definded as the
## regional minima of the h-minima transform of the input image (which removed
## all regional minima of a depth less then h beforehand).
##
## The input image @var{im} needs to be a real and nonsparse numeric array (of any dimension),
## and the height parameter @var{h} a non-negative scalar number.
##
## The definition of "neighborhood" for this morphological operation can be set
## with the connectivity parameter @var{conn},
## which defaults to 8 for 2D images, to 26 for 3D images and to
## @code{conn(ndims(n), "maximal")} in general. @var{conn} can be given as scalar value
## or as a boolean matrix (see @code{conndef} for details).
##
## The output is a binary image of same shape as the input image @var{im}.
##
## @seealso{imextendedmax, imhmin, imregionalmin, imreconstruct}
## @end deftypefn

## Algorithm:
## * The 'classical' reference for this morphological "extended maximum" function
##    is the book "Morphological Image Analysis" by P. Soille
##    (Springer, 2nd edition, 2004), chapter 6.3.4 "Extended and h-extrema".
##    It says: "The extended minima EMIN are definded as the regional minima
##                  of the corresponding h-minima transformation, the extended maxima
##                  EMAX being definded by duality:
##                     EMAX_h(f) = RMAX[HMAX_h(f)],
##                     EMIN_h(f) = RMIN[HMIN_h(f)].".
## * A more easily accessible reference is for example the following
##    web page by Régis Clouard:
##    https://clouard.users.greyc.fr/Pantheon/experiments/morphology/index-en.html#extremum
##    It says: "The extended minima EMIN are definded as the regional minima of
##                  the corresponding h-minima transformation:
##                  EMIN_h(f) = RMIN( HMIN_h(f) )"
##   (We will call the grayscale image im instead of f.)

function bw = imextendedmin (im, h, varargin)

  ## retrieve input parameters, set default value:
  if (nargin == 3)
    conn = varargin{1};
    iptcheckconn (conn, "imextendedmin", "CONN");
  elseif (nargin == 2)
    conn = conndef (ndims (im), "maximal");
  else
    print_usage ();
  endif

  ## check input parameters:
  if (! isnumeric (im)  || ! isreal (im) || issparse (im) )
    error ("imextendedmin: IM must be a real and nonsparse numeric array");
  endif

  if (! isnumeric (h) || ! isscalar (h) || ! isreal (h) || (h<0) )
    error ("imextendedmin: H must be a non-negative scalar number");
  endif

  ## do the actual calculation:
  bw = imregionalmin (imhmin (im, h, conn), conn);

endfunction

%!shared im0, bw0_h2_out
%! im0 = uint8 ([5 5 5 5 5;
%!               5 4 3 4 5;
%!               5 3 0 3 5;
%!               5 4 3 4 5;
%!               5 5 5 5 5]);
%! bw0_h2_out = false (5);
%! bw0_h2_out(3,3) = true;

## test input syntax:
%!error imextendedmin ()
%!error imextendedmin (im0)
%!error imextendedmin ("hello", 2)
%!error imextendedmin (i.*im0, 2)
%!error imextendedmin (sparse (im0), 2)
%!error imextendedmin (im0, -2)
%!error imextendedmin (im0, 'a')
%!error imextendedmin (im0, ones (2))
%!error imextendedmin (im0, 2*i)

%!assert (imextendedmin (im0, 2), bw0_h2_out)
%!assert (imextendedmin (double (im0), 2), bw0_h2_out)
%!assert (imextendedmin (im0, 2, 8), bw0_h2_out)
%!assert (imextendedmin (im0, 2, 4), bw0_h2_out)
%!assert (imextendedmin (im0, 2, true (3)), bw0_h2_out)

## test output class and shape:
%!test
%! out = imextendedmin (im0, 2);
%! assert (size (out), size (im0))
%! assert (class (out), "logical")

%!test
%! out = imextendedmin (single (im0), 2);
%! assert (size (out), size (im0))
%! assert (class (out), "logical")

%!test
%! out = imextendedmin (uint8 (im0), 2);
%! assert (size (out), size (im0))
%! assert (class (out), "logical")

%!test
%! out = imextendedmin (uint16 (im0), 2);
%! assert (size (out), size (im0))
%! assert (class (out), "logical")

%!test
%! im = cat (3, im0, im0, im0, im0);
%! out = imextendedmin (im, 2);
%! assert (size (out), size (im))

## test calculation result:
%!test
%! im = 10 .* ones (10);
%! im(2:4, 2:4) = 7;
%! im(6:8, 6:8) = 2;
%! expected_4 = false (10);
%! expected_4(6:8, 6:8) = true;
%! expected_2 = expected_4;
%! expected_2(2:4, 2:4) = true;
%! out = imextendedmin (im, 4);
%! assert (out, expected_4, eps)
%! out = imextendedmin (0.1.*im, 0.4);
%! assert (out, expected_4, eps)
%! out = imextendedmin (im, 2);
%! assert (out, expected_2, eps)

%!test
%! im2 = 10 .* ones (10);
%! im2(2:4, 2:4) = 7;
%! im2(6:9, 6:9)=2;
%! im2(5, 5)=2;
%! im2(6, 7)=10;
%! im2(7, 8)=10;
%! expected_8 = false (10);
%! expected_8(6:9, 6:9) = true;
%! expected_8(5, 5) = true;
%! expected_8(6, 7) = false;
%! expected_8(7, 8) = false;
%! expected_4 = expected_8;
%! expected_4(2:4, 2:4) = true;
%! out2 = imextendedmin (im2, 2);
%! assert (out2, expected_8, eps)
%! out2 = imextendedmin (im2, 2, 4);
%! assert (out2, expected_4, eps)
%! out2 = imextendedmin (im2, 2, 8);
%! assert (out2, expected_8, eps)
