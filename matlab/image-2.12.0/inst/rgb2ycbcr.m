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
## @deftypefn  {Function File} {@var{YCbCrmap} =} rgb2ycbcr (@var{cmap})
## @deftypefnx {Function File} {@var{YCbCr} =} rgb2ycbcr (@var{RGB})
## @deftypefnx {Function File} {@dots{} =} rgb2ycbcr (@dots{}, [@var{Kb} @var{Kr}])
## @deftypefnx {Function File} {@dots{} =} rgb2ycbcr (@dots{}, @var{standard})
## Convert RGB values to YCbCr.
##
## The conversion changes the image @var{RGB} or colormap @var{cmap}, from
## the RGB color model to YCbCr (luminance, chrominance blue, and chrominance
## red).  @var{RGB} must be of class double, single, uint8, or uint16.
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
## @seealso{hsv2rgb, ntsc2rgb, rgb2hsv, rgb2ntsc}
## @end deftypefn

function ycbcr = rgb2ycbcr (rgb, standard = "601")
  if (nargin < 1 || nargin > 2)
    print_usage ();
  endif
  ycbcr = ycbcrfunc ("rgb2ycbcr", rgb, standard);
endfunction

%!test
%! in(:,:,1) = magic (5);
%! in(:,:,2) = magic (5);
%! in(:,:,3) = magic (5);
%! out(:,:,1) = [31  37  17  23  29
%!               36  20  22  28  30
%!               19  21  27  33  35
%!               25  26  32  34  19
%!               25  31  37  18  24];
%! out(:,:,2) = 128;
%! out(:,:,3) = 128;
%! assert (rgb2ycbcr (uint8 (in)), uint8 (out));

%!shared cbcr
%! cbcr = 0.5019607843137255;
%! out(1:10, 1)  = linspace (16/255, 235/255, 10);
%! out(:, [2 3]) = cbcr;
%! assert (rgb2ycbcr (gray (10)), out, 0.00001);

%!assert (rgb2ycbcr ([1 1 1]), [0.92157 cbcr cbcr], 0.0001);

%!assert (class (rgb2ycbcr (single (rand (5, 5, 3)))), "single")
