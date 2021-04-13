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
## @deftypefn  {Function File} {@var{lab} =} xyz2lab (@var{xyz})
## @deftypefnx {Function File} {@var{lab_map} =} xyz2lab (@var{xyz_map})
## Transform a colormap or image from CIE XYZ to CIE L*a*b* color space.
##
## A color in the CIE XYZ color space consists of three values X, Y and Z.
## Those values are designed to be colorimetric, meaning that their values
## do not depend on the display device hardware.
##
## A color in the CIE L*a*b* (or CIE Lab) space consists of lightness L* and
## two color-opponent dimensions a* and b*. The whitepoint is taken as D65.
## The CIE L*a*b* colorspace is also a colorimetric colorspace. It is designed
## to incorporate the human perception of color differences.
##
## Input values of class single and double are accepted.
## The shape and the class of the input are conserved.
##
## The return values of L* are normally in the inteval [0, 100]
## and the values of a* and b* in the interval [-127, 127]
##
## @seealso{lab2xyz, rgb2lab, rgb2hsv, rgb2ind, rgb2ntsc}
## @end deftypefn

## Author: Hartmut Gimpel <hg_code@gmx.de>
## algorithm taken from the following book:
## Burger, Burge "Digitale Bildverarbeitung", 3rd edition (2015)

function lab = xyz2lab (xyz)

  if (nargin != 1)
    print_usage ();
  endif

  [xyz, cls, sz, is_im, is_nd, is_int] ...
    = colorspace_conversion_input_check ("xyz2lab", "XYZ", xyz, 1);
  #  only accept single and double inputs because valid xyz values can be >1

  ## normalize with the whitepoint D65
  ## (reference: en.wikipedia.org/wiki/Illuminant_D65)
  D65 = [0.95047, 1, 1.08883];
  xyz_D65 = xyz ./ D65;
  # Matlab truncates to D65_Matlab = [0.9504, 1.0000, 1.0888];

  ## transformation xyz -> xyz'
  epsilon = (6/29)^3;
  kappa = 1/116 * (29/3)^3;
  xyz_prime = xyz_D65;
  mask = xyz_D65 <= epsilon;
  xyz_prime(mask) = kappa .* xyz_D65(mask) + 16/116;
  xyz_prime(! mask) = xyz_D65(! mask) .^(1/3);
  x_prime = xyz_prime(:,1);
  y_prime = xyz_prime(:,2);
  z_prime = xyz_prime(:,3);

  ## transformation xyz' -> lab
  L = 116 .* y_prime - 16;
  a = 500 .* (x_prime - y_prime);
  b = 200 .* (y_prime -  z_prime);

  lab = [L, a, b];

  # always return values of type double for Matlab compatibility (exception: type single)
  lab = colorspace_conversion_revert (lab, cls, sz, is_im, is_nd, is_int, 1);

endfunction

## Test pure colors, gray and some other colors
## (This set of test values is taken from the book by Burger.)
%!assert (xyz2lab ([0, 0, 0]), [0 0 0], 5e-2)
%!assert (xyz2lab ([0.4125, 0.2127, 0.0193]), [53.24, 80.09, 67.20], 5e-2)
%!assert (xyz2lab ([0.7700, 0.9278, 0.1385]), [97.14, -21.55, 94.48], 5e-2)
%!assert (xyz2lab ([0.3576, 0.7152, 0.1192]), [87.74, -86.18, 83.18], 5e-2)
%!assert (xyz2lab ([0.5380, 0.7873, 1.0694]), [91.11, -48.09, -14.13], 5e-2)
%!assert (xyz2lab ([0.1804, 0.07217, 0.9502]), [32.30, 79.19, -107.86], 5e-2)
%!assert (xyz2lab ([0.5929, 0.28484, 0.9696]), [60.32, 98.24, -60.83], 5e-2)
%!assert (xyz2lab ([0.9505, 1.0000, 1.0888]), [100, 0.00, 0.00], 5e-2)
%!assert (xyz2lab ([0.2034, 0.2140, 0.2330]), [53.39, 0.00, 0.00], 5e-2)
%!assert (xyz2lab ([0.2155, 0.1111, 0.0101]), [39.77, 64.51, 54.13], 5e-2)
%!assert (xyz2lab ([0.0883, 0.0455, 0.0041]), [25.42, 47.91, 37.91], 5e-2)
%!assert (xyz2lab ([0.02094, 0.0108, 0.00098]), [9.66, 29.68, 15.24], 5e-2)
%!assert (xyz2lab ([0.5276, 0.3812, 0.2482]), [68.11, 48.39, 22.83], 5e-2)

## Test tolarant input checking on floats
%!assert (xyz2lab ([1.5 1 1]), [100, 82.15, 5.60], 5e-2)

%! xyz_map = rand (64, 3);
%! assert (lab2xyz (xyz2lab (xyz_map)), xyz_map, 1e-5);

%!test
%! xyz_img = rand (64, 64, 3);
%! assert (lab2xyz (xyz2lab (xyz_img)), xyz_img, 1e-5);

## support sparse input  (the only useful xyz value with zeros is black)
%!assert (xyz2lab (sparse ([0 0 0])), [0 0 0], 5e-2)

## conserve class of single input
%!assert (class (xyz2lab (single([0.5 0.5 0.5]))), 'single')

## Test input validation
%!error xyz2lab ()
%!error xyz2lab (1,2)
%!error <invalid data type 'cell'> xyz2lab ({1})
%!error <XYZ must be a colormap or XYZ image> xyz2lab (ones (2,2))

## Test ND input
%!test
%! xyz = rand (16, 16, 3, 5);
%! lab = zeros (size (xyz));
%! for i = 1:5
%!   lab(:,:,:,i) = xyz2lab (xyz(:,:,:,i));
%! endfor
%! assert (xyz2lab (xyz), lab)
