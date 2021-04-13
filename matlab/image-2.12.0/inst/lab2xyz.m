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
## @deftypefn  {Function File} {@var{xyz} =} lab2xyz (@var{lab})
## @deftypefnx {Function File} {@var{xyz_map} =} lab2xyz (@var{lab_map})
## Transform a colormap or image from CIE L*a*b* to CIE XYZ color space.
##
## A color in the CIE L*a*b* (or CIE Lab) space consists of lightness L* and
## two color-opponent dimensions a* and b*. The whitepoint is taken as D65.
## The CIE L*a*b* colorspace is a colorimetric colorspace, meaning that their values
## do not depend on the display device hardware. This colorspace is designed
## to incorporate the human perception of color differences.
##
## A color in the CIE XYZ color space consists of three values X, Y and Z.
## Those values are also designed to be colorimetric.
##
## Input values of class single and double are accepted.
## The shape and the class of the input are conserved.
##
## The input values of L* are normally in the inteval [0, 100]
## and the values of a* and b* in the interval [-127, 127].
##
## @seealso{xyz2lab, rgb2lab, rgb2hsv, rgb2ind, rgb2ntsc}
## @end deftypefn

## Author: Hartmut Gimpel <hg_code@gmx.de>
## algorithm taken from the following book:
## Burger, Burge "Digitale Bildverarbeitung", 3rd edition (2015)

function xyz = lab2xyz (lab)

  if (nargin != 1)
    print_usage ();
  endif

  [lab, cls, sz, is_im, is_nd, is_int] ...
    = colorspace_conversion_input_check ("lab2xyz", "Lab", lab, 1);
  #  currently only accept single and double inputs (as Matlab does)
  # (Integer types would be possible, but would need an explanation in the
  #  help text how to scale them.)

  ## use the whitepoint D65 (reference: en.wikipedia.org/wiki/Illuminant_D65)
  D65 = [0.95047, 1, 1.08883];
  # Matlab truncates to D65_Matlab = [0.9504, 1.0000, 1.0888];

  ## transformation Lab -> XYZ
  L = lab(:,1);
  a = lab(:,2);
  b = lab(:,3);

  L_prime = (L + 16) ./ 116;

  x = D65(1) .* f (L_prime + a./500);
  y = D65(2) .* f (L_prime);
  z = D65(3) .* f (L_prime - b./200);

  xyz = [x, y, z];

  # always return values of type double for Matlab compatibility (exception: type single)
  xyz = colorspace_conversion_revert (xyz, cls, sz, is_im, is_nd, is_int, 1);

endfunction

function out = f (in)
  epsilon = (6/29)^3;
  kappa = 1/116 * (29/3)^3;

  out = in;
  mask = in.^3 > epsilon;
  out(mask) = in(mask).^3;
  out(! mask) = (in(! mask) - 16/116)./kappa;
endfunction


## Test pure colors, gray and some other colors
## (This set of test values is taken from the book by Burger.)
%!assert (lab2xyz ([0, 0, 0]), [0 0 0], 1e-3)
%!assert (lab2xyz ([53.24, 80.09, 67.20]), [0.4125, 0.2127, 0.0193], 1e-3)
%!assert (lab2xyz ([97.14, -21.55, 94.48]), [0.7700, 0.9278, 0.1385], 1e-3)
%!assert (lab2xyz ([87.74, -86.18, 83.18]), [0.3576, 0.7152, 0.1192], 1e-3)
%!assert (lab2xyz ([91.11, -48.09, -14.13]), [0.5380, 0.7873, 1.0694], 1e-3)
%!assert (lab2xyz ([32.30, 79.19, -107.86]), [0.1804, 0.07217, 0.9502], 1e-3)
%!assert (lab2xyz ([60.32, 98.24, -60.83]), [0.5929, 0.28484, 0.9696], 1e-3)
%!assert (lab2xyz ([100, 0.00, 0.00]), [0.9505, 1.0000, 1.0888], 1e-3)
%!assert (lab2xyz ([53.39, 0.00, 0.00]), [0.2034, 0.2140, 0.2330], 1e-3)
%!assert (lab2xyz ([39.77, 64.51, 54.13]), [0.2155, 0.1111, 0.0101], 1e-3)
%!assert (lab2xyz ([25.42, 47.91, 37.91]), [0.0883, 0.0455, 0.0041], 1e-3)
%!assert (lab2xyz ([9.66, 29.68, 15.24]), [0.02094, 0.0108, 0.00098], 1e-3)
%!assert (lab2xyz ([68.11, 48.39, 22.83]), [0.5276, 0.3812, 0.2482], 1e-3)

## Test tolarant input checking on floats
%!assert (lab2xyz ([150 130 130]), [4.596, 2.931, 0.519], 1e-3)

%!test
%! lab_map = rand (64, 3);
%! lab_map(:,1) = lab_map(:,1) .* 100;
%! lab_map(:,2) = lab_map(:,2) .* 254 - 127;
%! lab_map(:,3) = lab_map(:,3) .* 254 - 127;
%! assert (xyz2lab (lab2xyz (lab_map)), lab_map, 1e-5);

%!test
%! lab_img = rand (64, 64, 3);
%! lab_img(:,:,1) = lab_img(:,:,1) .* 100;
%! lab_img(:,:,2) = lab_img(:,:,2) .* 254 - 127;
%! lab_img(:,:,3) = lab_img(:,:,3) .* 254 - 127;
%! assert (xyz2lab (lab2xyz (lab_img)), lab_img, 1e-5);

## support sparse input
%!assert (lab2xyz (sparse ([0 0 0])), [0 0 0], 1e-3)
%!assert (lab2xyz (sparse ([100, 0.00, 0.00])), [0.9505, 1.0000, 1.0888], 1e-3)

## conserve class of single input
%!assert (class (lab2xyz (single([50 50 50]))), 'single')

## Test input validation
%!error lab2xyz ()
%!error lab2xyz (1,2)
%!error <invalid data type 'cell'> lab2xyz ({1})
%!error <Lab must be a colormap or Lab image> lab2xyz (ones (2,2))

## Test ND input
%!test
%! lab = rand (16, 16, 3, 5);
%! lab(:,:,1,:) = lab(:,:,1,:) .* 100;
%! lab(:,:,2,:) = lab(:,:,2,:) .* 254 - 127;
%! lab(:,:,3,:) = lab(:,:,3,:) .* 254 - 127;
%! xyz = zeros (size (lab));
%! for i = 1:5
%!   xyz(:,:,:,i) = lab2xyz (lab(:,:,:,i));
%! endfor
%! assert (lab2xyz (lab), xyz)
