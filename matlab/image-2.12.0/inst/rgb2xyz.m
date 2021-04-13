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
## @deftypefn  {Function File} {@var{xyz} =} rgb2xyz (@var{rgb})
## @deftypefnx {Function File} {@var{xyz_map} =} rgb2xyz (@var{rgb_map})
## Transform a colormap or image from sRGB to CIE XYZ color space.
##
## A color in the RGB space consists of red, green, and blue intensities.
## The input RGB values are interpreted as nonlinear sRGB values
## with the white point D65. This means the input values are assumed to
## be in the colorimetric (sRGB) colorspace.
##
## A color in the CIE XYZ color space consists of three values X, Y and Z.
## Those values are designed to be colorimetric, meaning that their values
## do not depend on the display device hardware.
##
## Input values of class double, single, uint8 or uint16 are accepted.
## Output class is generally of type double, only input type single will
## result in an output type of single. The shape of the input is
## conserved.
##
## note: This function returns slightly different values than the Matlab
##            version. But it has a better "round trip accuracy" (<2e-5)
##            for RGB -> XYZ -> RGB.
##
## @seealso{xyz2rgb, rgb2lab, rgb2hsv, rgb2ind, rgb2ntsc}
## @end deftypefn

## Author: Hartmut Gimpel <hg_code@gmx.de>
## algorithm taken from the following book:
## Burger, Burge "Digitale Bildverarbeitung", 3rd edition  (2015)

function xyz = rgb2xyz (rgb)

  if (nargin != 1)
    print_usage ();
  endif

  [rgb, cls, sz, is_im, is_nd, is_int] ...
    = colorspace_conversion_input_check ("rgb2xyz", "RGB", rgb, 0);

  ## transform from non-linear sRGB values to linear sRGB values
  ##  (inverse modified gamma transform)
  rgb_lin = rgb;
  mask = rgb <= 0.04045;
  rgb_lin(mask) = rgb(mask) ./ 12.92;
  rgb_lin(! mask) = ((rgb(! mask) + 0.055) ./ 1.055) .^2.4;

  ## transform from linear sRGB values to CIE XYZ with whitepoint D65
  ## (source of this matrix: book of Burger)
  matrix_rgb2xyz_D65 = ...
    [0.412453, 0.357580, 0.180423;
    0.212671, 0.715160, 0.072169;
    0.019334, 0.119193, 0.950227];

  # Matlab uses the following slightly different conversion matrix.
  # matrix_rgb2xyz_D65 = ...
  #  [0.4124, 0.3576, 0.1805;
  #  0.2126, 0.7152, 0.0722;
  #  0.0193, 0.1192, 0.9505];
  # open source of this transformation matrix:
  # https://de.wikipedia.org/wiki/CIE-Normvalenzsystem#Umrechnung_der_Farbr.C3.A4ume
  # on July 30, 2015

  xyz = rgb_lin * matrix_rgb2xyz_D65';

  # always return values of type double for Matlab compatibility (exception: type single)
  xyz = colorspace_conversion_revert (xyz, cls, sz, is_im, is_nd, is_int, 1);

endfunction

## Test pure colors, gray and some other colors
## (This set of test values is taken from the book by Burger.)
%!assert (rgb2xyz ([0 0 0]), [0, 0, 0], 1e-3)
%!assert (rgb2xyz ([1 0 0]), [0.4125, 0.2127, 0.0193], 1e-3)
%!assert (rgb2xyz ([1 1 0]), [0.7700, 0.9278, 0.1385], 1e-3)
%!assert (rgb2xyz ([0 1 0]), [0.3576, 0.7152, 0.1192], 1e-3)
%!assert (rgb2xyz ([0 1 1]), [0.5380, 0.7873, 1.0694], 1e-3)
%!assert (rgb2xyz ([0 0 1]), [0.1804, 0.0722, 0.9502], 1e-3)
%!assert (rgb2xyz ([1 0 1]), [0.5929, 0.2848, 0.9696], 1e-3)
%!assert (rgb2xyz ([1 1 1]), [0.9505, 1.0000, 1.0888], 1e-3)
%!assert (rgb2xyz ([0.5 0.5 0.5]), [0.2034, 0.2140, 0.2330], 1e-3)
%!assert (rgb2xyz ([0.75 0 0]), [0.2155, 0.1111, 0.0101], 1e-3)
%!assert (rgb2xyz ([0.5 0 0]), [0.0883, 0.0455, 0.0041], 1e-3)
%!assert (rgb2xyz ([0.25 0 0]), [0.0210, 0.0108, 0.0010], 1e-3)
%!assert (rgb2xyz ([1 0.5 0.5]), [0.5276, 0.3812, 0.2482], 1e-3)

## Test tolarant input checking on floats
%!assert (rgb2xyz ([1.5 1 1]), [1.5845, 1.3269, 1.1185], 1e-3)

%!test
%! rgb_map = rand (64, 3);
%! assert (xyz2rgb (rgb2xyz (rgb_map)), rgb_map, 2e-5);

%!test
%! rgb_img = rand (64, 64, 3);
%! assert (xyz2rgb (rgb2xyz (rgb_img)), rgb_img, 2e-5);

## support sparse input
%!assert (rgb2xyz (sparse ([0 0 0])), [0 0 0], 1e-3)
%!assert (rgb2xyz (sparse ([0 0 1])), [0.1804, 0.0722, 0.9502], 1e-3)

## support integer input (and double output)
%!assert (rgb2xyz (uint8([255 255 255])), [0.9505, 1.0000, 1.0888], 1e-3)

## conserve class of single input
%!assert (class (rgb2xyz (single([1 1 1]))), 'single')

## Test input validation
%!error rgb2xyz ()
%!error rgb2xyz (1,2)
%!error <invalid data type 'cell'> rgb2xyz ({1})
%!error <RGB must be a colormap or RGB image> rgb2xyz (ones (2,2))

## Test ND input
%!test
%! rgb = rand (16, 16, 3, 5);
%! xyz = zeros (size (rgb));
%! for i = 1:5
%!   xyz(:,:,:,i) = rgb2xyz (rgb(:,:,:,i));
%! endfor
%! assert (rgb2xyz (rgb), xyz)
