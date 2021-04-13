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
## @deftypefn  {Function File} {} @ imimposemin (@var{im}, @var{bw})
## @deftypefnx {Function File} {} @ imimposemin (@var{im}, @var{bw}, @var{conn})
## Modify the input impage @var{im} to only have regional minima at
## the marker positions given by the nonzero pixels of @var{bw}.
##
## This function returns a grayscale image that is similar to @var{im} but
## only has regional minima at pixel positions where the marker image @var{bw} is nonzero.
##
## The input image @var{im} needs to be a real and nonsparse numeric array
## (of any dimension). And the marker image @var{bw} needs to be a real or
## logical nonsparse array of identical size.
## (The values in @var{bw} will first be converted to logical values.)
##
## The definition of "neighborhood" for this morphological operation can be set
## with the connectivity parameter @var{conn},
## which defaults to 8 for 2D images, to 26 for 3D images and to
## @code{conn(ndims(n), "maximal")} in general. @var{conn} can be given as scalar value
## or as a boolean matrix (see @code{conndef} for details).
##
## @seealso{imextendedmin, imhmin, imregionalmin, imreconstruct}
## @end deftypefn

## Algorithm:
## * The 'classical' reference for this morphological "minima imposition" function
##    is the book "Morphological Image Analysis" by P. Soille
##    (Springer, 2nd edition, 2004), chapter 6.4.6 "Minima imposition".
##    It says (own translation from the german book, two typos corrected):
##                 "The marker image f_m is defined for every pixel as follows:
##                           f_m(x) = 0 if x is part of a marker
##                                      = t_max else.
##                   The minima imposition of the input image f is performed
##                   in two steps. First, the pointwise minimum between the input
##                   image and the marker image is calculated: f /\ f_m. [...]
##                   it is necessary to rather consider (f+1) /\ f_m instead of f /\ f_m.
##                   The second step consists of a morphological reconstruction
##                   by erosion of (f+1) /\ f_m from the marker image fm."
##   (We will call the grayscale image im instead of f.)
## * A more easily accessible reference is for example the following
##    web page by Régis Clouard:
##    https://clouard.users.greyc.fr/Pantheon/experiments/morphology/index-en.html#ch4-D
##    It says: "IV.D.2. Maxima/Minima Imposition
##                  [..., it follows an example script with Pandore functions,  the input image is in.pan, 
##                  the generated marker image is i1.pan and the output is swamping.pan]
##                  padcst 1 in.pan i2.pan                                                   [i2 = in + 1]
##                  pmin i1.pan i2.pan i3.pan                                              [i3 = min(i1, i2)]
##                  perosionreconstruction 8 i1.pan i3.pan swamping.pan [out = erosionreconstruction(i1, i3, 8)]

function im2 = imimposemin (im, bw, varargin)

  ## retrieve input parameters, set default value:
  if (nargin == 3)
    conn = varargin{1};
    iptcheckconn (conn, "imimposemin", "CONN");
  elseif (nargin == 2)
    ## Buggy Matlab doc claims "minimum" connectivity instead,
    ## but defaults of 8 and 26, which are "maximal" connectivities.
    conn = conndef (ndims (im), "maximal");
  else
    print_usage ();
  endif

  ## check input parameters:
  if (! isnumeric (im)  || ! isreal (im) || issparse (im) )
    error ("imimposemin: IM must be a real and nonsparse numeric array");
  endif

  if (((! isnumeric (bw)  || ! isreal (bw)) && (! islogical (bw))) || issparse (bw) )
    error ("imimposemin: BW must be a logical or numeric nonsparse array");
  endif

  if (! (ndims (im) == ndims (bw)) || ! all (size (im) == size (bw)))
    error ("imimposemin: BW must have the same size as IM");
  endif

  ## do the actual calculation
  ## convert bw to class logical if necessary:
  if (! islogical (bw))
    bw = logical (bw);
  endif

  ## define the marker image fm (see algorithm above):
  fm = zeros (size (im), class  (im));
  fm(bw) = -Inf; # min value of class(im)
  fm(!bw) = Inf; # max value of class(im)

  ## define the difference delta:
  ## for integer images this value is 1 (see algorithm above)
  ## for float images this is done in analogy
  ## (A possible value for float images would be delta = eps (im),
  ## see bug #51724, but Matlab seems to do the following instead.)
  if isfloat (im)
    delta = ( max (im(:)) - min (im(:)) ) / 1000;
  else   # integer images
    delta = 1;
  endif

  ## calculate pointwise minimum (see algorithm above):
  min_im = min (im + delta, fm);

  ## do the morphological reconstruction by erosion (see algorithm above):
  ## (Calculate dilations of the inverse images, instead of erosions of the
  ##  original images, because this is what imreconstruct can do.)
  im2 = imreconstruct (imcomplement (fm), imcomplement (min_im), conn);
  im2 = imcomplement (im2);

endfunction

%!shared im0, bw0, out0, out0_4
%! im0 = uint8 ([5 5 5 5 5;
%!               5 4 3 4 5;
%!               5 3 0 3 5;
%!               5 4 3 4 5;
%!               5 5 5 5 5]);
%! bw0 = false (5);
%! bw0(4, 4) = true;
%! out0 = im0 + 1;
%! out0(4, 4) = 0;
%! out0_4 = out0;
%! out0_4(3, 3) = 4;

## test input syntax:
%!error imimposemin ()
%!error imimposemin (im0)
%!error imimposemin ("hello", bw0)
%!error imimposemin (i.*im0, bw0)
%!error imimposemin (sparse (im0), bw0)
%!error imimposemin (im0, ones (2))
%!error imimposemin (im0, 'hello')
%!error imimposemin (im0, i .* double (bw0))
%!error imimposemin (im0, sparse (bw0))
%!error imimposemin (im0, bw0, 'hello')
%!error imimposemin (im0, bw0, 3)

%!assert (imimposemin (im0, bw0), out0)
%!assert (imimposemin (im0, bw0, 8), out0)
%!assert (imimposemin (im0, bw0, 4), out0_4)
%!assert (imimposemin (im0, bw0, true (3)), out0)

## test output class and shape:
%!test
%! out = imimposemin (im0, bw0);
%! assert (size (out), size (im0))
%! assert (class (out), "uint8")

%!test
%! out = imimposemin (double (im0), bw0);
%! assert (size (out), size (im0))
%! assert (class (out), "double")

%!test
%! out = imimposemin (single (im0), bw0);
%! assert (size (out), size (im0))
%! assert (class (out), "single")

%!test
%! out = imimposemin (uint16 (im0), bw0);
%! assert (size (out), size (im0))
%! assert (class (out), "uint16")

%!test
%! im = cat (3, im0, im0, im0, im0);
%! bw = cat (3, bw0, bw0, bw0, bw0);
%! out = imimposemin (im, bw);
%! assert (size (out), size (im))

## test calculation result:
%!test
%! expected_double = double (im0);
%! expected_double += 0.005;
%! expected_double(4, 4) = -inf;
%! out = imimposemin (double (im0), bw0);
%! assert (out, expected_double, eps)

%!test
%! im = uint8 (10 .* ones (10));
%! im(6:8, 6:8) = 2;
%! im(2:4, 2:4) = 7;
%! im(3, 3) = 5;
%! im(2, 9) = 9;
%! im(3, 8) = 9;
%! im(9, 2) = 9;
%! im(8, 3) = 9;
%! bw = false (10);
%! bw(3, 3) = true;
%! bw(6:8, 6:8) = true;
%! expected = uint8 (11 .* ones(10));
%! expected(2:4, 2:4) = 8;
%! expected(3, 3) = 0;
%! expected(6:8, 6:8) = 0;
%! expected_double = double (expected);
%! expected_double -= 0.992;
%! expected_double (expected_double < 0) = -inf;
%! out = imimposemin (im, bw);
%! assert (out, expected, eps)
%! out = imimposemin (double (im), bw);
%! assert (out, expected_double, eps)
