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
## @deftypefn  {Function File} {} @ imhmax (@var{im}, @var{h})
## @deftypefnx {Function File} {} @ imhmax (@var{im}, @var{h}, @var{conn})
## Caculate the morphological h-maximum transform of an image @var{im}.
##
## This function removes all regional maxima in the grayscale image @var{im} whose 
## height is lower or equal to the given threshold level @var{h}, and it decreases the height
## of the remaining regional maxima by the value of @var{h}. (A "regional maximum"
## is defined as a connected component of pixels with an equal pixel value
## that is higher than the value of all its neighboring pixels. And the
## "height" of a regional maximum can be thought of as minimum pixel value difference
## between the regional maximum and its neighboring minima.)
##
## The input image @var{im} needs to be a real and nonsparse numeric array (of any dimension),
## and the height parameter @var{h} a non-negative scalar number.
##
## The definition of "neighborhood"  for this morphological operation
## can be set with the connectivity parameter @var{conn},
## which defaults to 8 for 2D images, to 26 for 3D images and to
## @code{conn(ndims(n), "maximal")} in general. @var{conn} can be given as scalar value
## or as a boolean matrix (see @code{conndef} for details).
##
## The output is a transformed grayscale image of same type and
## shape as the input image @var{im}.
##
## @seealso{imhmin, imregionalmax, imextendedmax, imreconstruct}
## @end deftypefn

## Algorithm:
## * The 'classical' reference for this morphological h-maximum function
##    is the book "Morphological Image Analysis" by P. Soille
##    (Springer, 2nd edition, 2004), chapter 6.3.4 "Extended and h-extrema".
##    It says: "This is achieved by performing the reconstruction by dilation
##                  of [a grayscale image] f from f-h:
##                  HMAX_h(f) = R^delta_f (f - h)".
## * A more easily accessible reference is for example the following
##    web page by Régis Clouard:
##    https://clouard.users.greyc.fr/Pantheon/experiments/morphology/index-en.html#extremum
##    It says: "It is defined as the [morphological] reconstruction by dilation
##                  of [a grayscale image] f subtracted by a height h."
##   (We will call the grayscale image im instead of f.)

function im2 = imhmax (im, h, varargin)

  ## retrieve input parameters, set default value:
  if (nargin == 3)
    conn = varargin{1};
    iptcheckconn (conn, "imhmax", "CONN");
  elseif (nargin == 2)
    conn = conndef (ndims (im), "maximal");
  else
    print_usage ();
  endif

  ## check input parameters:
  if (! isnumeric (im)  || ! isreal (im) || issparse (im) )
    error ("imhmax: IM must be a real and nonsparse numeric array");
  endif

  if (! isnumeric (h) || ! isscalar (h) || ! isreal (h) || (h<0) )
    error ("imhmax: H must be a non-negative scalar number");
  endif

  ## do the actual calculation:
  im2 = imreconstruct ((im-h), im, conn);

endfunction

%!shared im0, im0_h2_out
%! im0 = uint8 ([0 0 0 0 0;
%!               0 1 2 1 0;
%!               0 2 5 2 0;
%!               0 1 2 1 0;
%!               0 0 0 0 0]);
%! im0_h2_out = uint8 ([0 0 0 0 0;
%!                      0 1 2 1 0;
%!                      0 2 3 2 0;
%!                      0 1 2 1 0;
%!                      0 0 0 0 0]);

## test input syntax:
%!error imhmax ()
%!error imhmax (im0)
%!error imhmax ("hello", 2)
%!error imhmax (i.*im0, 2)
%!error imhmax (sparse (im0), 2)
%!error imhmax (im0, -2)
%!error imhmax (im0, 'a')
%!error imhmax (im0, ones (2))
%!error imhmax (im0, 2*i)

%!assert (imhmax (im0, 2), im0_h2_out)
%!assert (imhmax (double (im0), 2), double (im0_h2_out))
%!assert (imhmax (im0, 2, 8), im0_h2_out)
%!assert (imhmax (im0, 2, 4), im0_h2_out)
%!assert (imhmax (im0, 2, true (3)), im0_h2_out)

## test output class and shape:
%!test
%! out = imhmax (double (im0), 2);
%! assert (size (out), size (im0))
%! assert (class (out), "double")

%!test
%! out = imhmax (single (im0), 2);
%! assert (size (out), size (im0))
%! assert (class (out), "single")

%!test
%! out = imhmax (uint8 (im0), 2);
%! assert (size (out), size (im0))
%! assert (class (out), "uint8")

%!test
%! out = imhmax (uint16 (im0), 2);
%! assert (size (out), size (im0))
%! assert (class (out), "uint16")

%!test
%! im = cat (3, im0, im0, im0, im0);
%! out = imhmax (im, 2);
%! assert (size (out), size (im))

## test calculation result:
%!test
%! im = zeros (10);
%! im(2:4, 2:4) = 3;
%! im(6:8, 6:8) = 8;
%! expected_4 = zeros (10);
%! expected_4(6:8, 6:8) = 4;
%! expected_2 = zeros (10);
%! expected_2(2:4, 2:4) = 1;
%! expected_2(6:8, 6:8) = 6;
%! out = imhmax (im, 4);
%! assert (out, expected_4, eps)
%! out = imhmax (im, 2);
%! assert (out, expected_2, eps)
%! out = imhmax (0.1 .* im, 0.4);
%! assert (out, 0.1 .* expected_4, eps)

%!test
%! im2 = zeros (10);
%! im2(2:4, 2:4) = 3;
%! im2(6:9, 6:9)=8;
%! im2(5, 5)=8;
%! im2(6, 7)=0;
%! im2(7, 8)=0;
%! expected_4 = zeros (10);
%! expected_4(6:9, 6:9) = 4;
%! expected_4(5, 5) = 4;
%! expected_4(6, 7) = 0;
%! expected_4(7, 8) = 0;
%! expected_8 = expected_4;
%! expected_8(2:4, 2:4) = 3;
%! out2 = imhmax (im2, 4);
%! assert (out2, expected_8, eps)
%! out2 = imhmax (im2, 4, 4);
%! assert (out2, expected_4, eps)
%! out2 = imhmax (im2, 4, 8);
%! assert (out2, expected_8, eps)
