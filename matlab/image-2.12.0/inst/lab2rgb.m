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
## @deftypefn  {Function File} {@var{rgb} =} lab2rgb (@var{lab})
## @deftypefnx {Function File} {@var{rgb_map} =} lab2rgb (@var{lab_map})
## Transform a colormap or image from CIE L*a*b* to sRGB  color space.
##
## A color in the CIE L*a*b* (or CIE Lab) space consists of lightness L* and
## two color-opponent dimensions a* and b*. The whitepoint is taken as D65.
## The CIE L*a*b* colorspace is a colorimetric colorspace. It is additionally
## designed to incorporate the human perception of color differences.
##
## A color in the RGB space consists of red, green, and blue intensities.
## The input RGB values are interpreted as nonlinear sRGB values
## with the white point D65. This means the input values are assumed to
## be in a colorimetric (sRGB) colorspace.
##
## Input values of class single and double are accepted.
## The shape and the class of the input are conserved.
##
## The input values of L* are normally in the inteval [0, 100]
## and the values of a* and b* in the interval [-127, 127].
##
## note: This function returns slightly different values than the Matlab
##            version. But it has a better "round trip accuracy" (<2e-5)
##            for RGB -> Lab -> RGB.
##
## @seealso{rgb2lab, rgb2xyz, rgb2hsv, rgb2ind, rgb2ntsc}
## @end deftypefn

## Author: Hartmut Gimpel <hg_code@gmx.de>
## algorithm taken from the following book:
## Burger, Burge "Digitale Bildverarbeitung", 3rd edition  (2015)

function rgb = lab2rgb (lab)

  if (nargin != 1)
    print_usage ();
  endif

  [lab, cls, sz, is_im, is_nd, is_int] ...
    = colorspace_conversion_input_check ("lab2rgb", "Lab", lab, 1);
  #  currently only accept single and double inputs (as Matlab does)
  # (Integer types would be possible, but would need an explanation in the
  #  help text how to scale them.)

  ## transform from CIE L*a*b* to CIE XYZ values
  xyz = lab2xyz (lab);

  ## transform from CIE XYZ to non-linear sRGB values
  rgb = xyz2rgb (xyz);

  # always return values of type double for Matlab compatibility (exception: type single)
  rgb = colorspace_conversion_revert (rgb, cls, sz, is_im, is_nd, is_int, 1);

endfunction

## Test pure colors, gray and some other colors
## (This set of test values is taken from the book by Burger.)
%!assert (lab2rgb ([0 0 0]), [0, 0, 0], 1e-3)
%!assert (lab2rgb ([53.24, 80.09, 67.20]), [1 0 0], 1e-3)
%!assert (lab2rgb ([97.14, -21.55, 94.48]), [1 1 0], 1e-3)
%!assert (lab2rgb ([87.74, -86.18, 83.18]), [0 1 0], 1e-3)
%!assert (lab2rgb ([91.11, -48.09, -14.13]), [0 1 1], 1e-3)
%!assert (lab2rgb ([32.30, 79.19, -107.86]), [0 0 1], 1e-3)
%!assert (lab2rgb ([60.32, 98.24, -60.83]), [1 0 1], 1e-3)
%!assert (lab2rgb ([100, 0.00, 0.00]), [1 1 1], 1e-3)
%!assert (lab2rgb ([53.39, 0.00, 0.00]), [0.5 0.5 0.5], 1e-3)
%!assert (lab2rgb ([39.77, 64.51, 54.13]), [0.75 0 0], 1e-3)
%!assert (lab2rgb ([25.42, 47.91, 37.91]), [0.5 0 0], 1e-3)
%!assert (lab2rgb ([9.66, 29.68, 15.24]), [0.25 0 0], 1e-3)
%!assert (lab2rgb ([68.11, 48.39, 22.83]), [1 0.5 0.5], 1e-3)

## Test tolarant input checking
%!assert (lab2rgb ([150 130 130]), [2.714, 1.028, 0.492], 1e-3)

%!test
%! lab_map = rand (64, 3);
%! lab_map(:,1) = lab_map(:,1) .* 100;
%! lab_map(:,2) = lab_map(:,2) .* 254 - 127;
%! lab_map(:,3) = lab_map(:,3) .* 254 - 127;
%! assert (rgb2lab (lab2rgb (lab_map)), lab_map, 5e-3);

%!test
%! lab_img = rand (64, 64, 3);
%! lab_img(:,:,1) = lab_img(:,:,1) .* 100;
%! lab_img(:,:,2) = lab_img(:,:,2) .* 254 - 127;
%! lab_img(:,:,3) = lab_img(:,:,3) .* 254 - 127;
%! assert (rgb2lab (lab2rgb (lab_img)), lab_img, 5e-3);

## support sparse input
%!assert (lab2rgb (sparse ([0 0 0])), [0 0 0], 1e-3)
%!assert (lab2rgb (sparse ([100, 0.00, 0.00])), [1 1 1], 1e-3)

## conserve class of single input
%!assert (class (lab2rgb (single([50 50 50]))), 'single')

## Test input validation
%!error lab2rgb ()
%!error lab2rgb (1,2)
%!error <invalid data type 'cell'> lab2rgb ({1})
%!error <Lab must be a colormap or Lab image> lab2rgb (ones (2,2))

## Test ND input
%!test
%! lab = rand (16, 16, 3, 5);
%! lab(:,:,1,:) = lab(:,:,1,:) .* 100;
%! lab(:,:,2,:) = lab(:,:,2,:) .* 254 - 127;
%! lab(:,:,3,:) = lab(:,:,3,:) .* 254 - 127;
%! rgb = zeros (size (lab));
%! for i = 1:5
%!   rgb(:,:,:,i) = lab2rgb (lab(:,:,:,i));
%! endfor
%! assert (lab2rgb (lab), rgb)
