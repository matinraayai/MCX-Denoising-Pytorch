## Copyright (C) 2013 Carnë Draug <carandraug@octave.org>
##
## This program is free software; you can redistribute it and/or modify it under
## the terms of the GNU General Public License as published by the Free Software
## Foundation; either version 3 of the License, or (at your option) any later
## version.
##
## This program is distributed in the hope that it will be useful, but WITHOUT
## ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
## FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
## details.
##
## You should have received a copy of the GNU General Public License along with
## this program; if not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*-
## @deftypefn  {Function File} {@var{cmap} =} ycbcr2rgb (@var{YCbCrmap})
## @deftypefnx {Function File} {@var{RGB} =} ycbcr2rgb (@var{YCbCr})
## @deftypefnx {Function File} {@dots{} =} ycbcr2rgb (@dots{}, [@var{Kb} @var{Kr}])
## @deftypefnx {Function File} {@dots{} =} ycbcr2rgb (@dots{}, @var{standard})
## Convert YCbCr color space to RGB.
##
## The conversion changes the image @var{YCbCr} or colormap @var{YCbCrmap},
## from the YCbCr (luminance, chrominance blue, and chrominance red)
## color space to RGB values.  @var{YCbCr} must be of class double, single,
## uint8, or uint16.
##
## The formula used for the conversion is dependent on two constants, @var{Kb}
## and @var{Kr} which can be specified individually, or according to existing
## standards:
##
## @table @asis
## @item "601" (default)
## According to the ITU-R BT.601 (formerly CCIR 601) standard.  Its values
## of @var{Kb} and @var{Kr} are 0.114 and 0.299 respectively.
## @item "709"
## According to the ITU-R BT.709 standard.  Its values of @var{Kb} and
## @var{Kr} are 0.0722 and 0.2116 respectively.
## @item "2020"
## According to the ITU-R BT.2020 standard.  Its values of @var{Kb} and
## @var{Kr} are 0.0593 and 0.2627 respectively.
## @end table
##
## @seealso{hsv2rgb, ntsc2rgb, rgb2hsv, rgb2ntsc, rgb2ycbcr}
## @end deftypefn

function rgb = ycbcr2rgb (ycbcr, standard = "601")
  if (nargin < 1 || nargin > 2)
    print_usage ();
  endif
  rgb = ycbcrfunc ("ycbcr2rgb", ycbcr, standard);
endfunction

%!assert (ycbcr2rgb (rgb2ycbcr (jet (10))), jet (10), 0.00001);

%!assert (class (ycbcr2rgb (single (rand (5, 5, 3)))), "single")
