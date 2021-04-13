## Copyright (C) 2015 Hartmut Gimpel
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
## @deftypefn  {Function File} {@var{lab} =} rgb2lab (@var{rgb})
## @deftypefnx {Function File} {@var{lab_map} =} rgb2lab (@var{rgb_map})
## Transform a colormap or image from sRGB to CIE L*a*b* color space.
##
## A color in the RGB space consists of red, green, and blue intensities.
## The input RGB values are interpreted as nonlinear sRGB values
## with the white point D65. This means the input values are assumed to
## be in the colorimetric (sRGB) colorspace.
##
## A color in the CIE L*a*b* (or CIE Lab) space consists of lightness L* and
## two color-opponent dimensions a* and b*. The whitepoint is taken as D65.
## The CIE L*a*b* colorspace is also a colorimetric colorspace. It is designed
## to incorporate the human perception of color differences.
##
## Input values of class double, single, uint8 or uint16 are accepted.
## Output class is generally of type double, only input type single will
## result in an output type of single. The shape of the input is
## conserved.
##
## note: This function returns slightly different values than the Matlab
##            version. But it has a better "round trip accuracy" (<2e-5)
##            for RGB -> Lab -> RGB.
##
## @seealso{lab2rgb, rgb2xyz, rgb2hsv, rgb2ind, rgb2ntsc}
## @end deftypefn

## Author: Hartmut Gimpel <hg_code@gmx.de>
## algorithm taken from the following book:
## Burger, Burge "Digitale Bildverarbeitung", 3rd edition  (2015)

function lab = rgb2lab (rgb)

  if (nargin != 1)
    print_usage ();
  endif

  [rgb, cls, sz, is_im, is_nd, is_int] ...
    = colorspace_conversion_input_check ("rgb2lab", "RGB", rgb, 0);

  ## transform from non-linear sRGB values to CIE XYZ values
  xyz = rgb2xyz (rgb);

  ## transform from CIE XYZ values to CIE L*a*b* values
  lab = xyz2lab (xyz);

  # always return values of type double for Matlab compatibility (exception: type single)
  lab = colorspace_conversion_revert (lab, cls, sz, is_im, is_nd, is_int, 1);

endfunction


## Test pure colors, gray and some other colors
## (This set of test values is taken from the book by Burger.)
%!assert (rgb2lab ([0 0 0]), [0, 0, 0], 1e-2)
%!assert (rgb2lab ([1 0 0]), [53.24, 80.09, 67.20], 1e-2)
%!assert (rgb2lab ([1 1 0]), [97.14, -21.55, 94.48], 1e-2)
%!assert (rgb2lab ([0 1 0]), [87.74, -86.18, 83.18], 1e-2)
%!assert (rgb2lab ([0 1 1]), [91.11, -48.09, -14.13], 1e-2)
%!assert (rgb2lab ([0 0 1]), [32.30, 79.19, -107.86], 1e-2)
%!assert (rgb2lab ([1 0 1]), [60.32, 98.24, -60.83], 1e-2)
%!assert (rgb2lab ([1 1 1]), [100, 0.00, 0.00], 1e-2)
%!assert (rgb2lab ([0.5 0.5 0.5]), [53.39, 0.00, 0.00], 1e-2)
%!assert (rgb2lab ([0.75 0 0]), [39.77, 64.51, 54.13], 1e-2)
%!assert (rgb2lab ([0.5 0 0]), [25.42, 47.91, 37.91], 1e-2)
%!assert (rgb2lab ([0.25 0 0]), [9.66, 29.68, 15.24], 1e-2)
%!assert (rgb2lab ([1 0.5 0.5]), [68.11, 48.39, 22.83], 1e-2)

## Test tolarant input checking on floats
%!assert (rgb2lab ([1.5 1 1]), [111.47, 43.42, 17.98], 1e-2)

%!test
%! rgb_map = rand (64, 3);
%! assert (lab2rgb (rgb2lab (rgb_map)), rgb_map, 2e-5);

%!test
%! rgb_img = rand (64, 64, 3);
%! assert (lab2rgb (rgb2lab (rgb_img)), rgb_img, 2e-5);

## support sparse input
%!assert (rgb2lab (sparse ([0 0 1])), sparse ([32.30, 79.19, -107.86]), 1e-2)
%!assert (rgb2lab (sparse ([0 1 1])), sparse ([91.11, -48.09, -14.13]), 1e-2)
%!assert (rgb2lab (sparse ([1 1 1])), sparse ([100, 0.00, 0.00]), 1e-2)

## support integer input (and double output)
%!assert (rgb2lab (uint8([255 255 255])), [100, 0.00, 0.00], 1e-2)

## conserve class of single input
%!assert (class (rgb2lab (single([1 1 1]))), 'single')

## Test input validation
%!error rgb2lab ()
%!error rgb2lab (1,2)
%!error <invalid data type 'cell'> rgb2lab ({1})
%!error <RGB must be a colormap or RGB image> rgb2lab (ones (2,2))

## Test ND input
%!test
%! rgb = rand (16, 16, 3, 5);
%! lab = zeros (size (rgb));
%! for i = 1:5
%!   lab(:,:,:,i) = rgb2lab (rgb(:,:,:,i));
%! endfor
%! assert (rgb2lab (rgb), lab)
