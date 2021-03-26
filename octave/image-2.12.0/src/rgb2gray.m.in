## Copyright (C) 2000, 2001 Kai Habel <kai.habel@gmx.de>
## Copyright (C) 2012 Carnë Draug <carandraug+dev@gmail.com>
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
## @deftypefn {Function File} @var{gray}= rgb2gray (@var{rgb})
## Convert RGB image or colormap to grayscale.
##
## If @var{rgb} is an RGB image, the conversion to grayscale is weighted based
## on the luminance values (see @code{rgb2ntsc}).  Supported classes are single,
## double,  uint8 and uint16.
##
## If @var{rgb} is a colormap it is converted into the YIQ space of ntsc.  The
## luminance value (Y) is taken to create a gray colormap.
##
## @seealso{mat2gray, ntsc2rgb, rgb2ind, rgb2ntsc, rgb2ycbcr}
## @end deftypefn

function gray = rgb2gray (rgb)

  if (nargin != 1)
    print_usage();
  endif

  if (iscolormap (rgb))
    gray = rgb2ntsc (rgb) (:, 1) * ones (1, 3);

  elseif (isimage (rgb) && ndims (rgb) == 3)
    cls = class (rgb);
    if (! isfloat (rgb))
      rgb = im2double (rgb);
    endif

    ## multiply each color by the luminance factor (this is also matlab compatible)
    ##      0.29894 * red + 0.58704 * green + 0.11402 * blue
    gray = rgb .* permute ([0.29894, 0.58704, 0.11402], [1, 3, 2]);
    gray = sum (gray, 3);
    gray = imcast (gray, cls);
  else
    error ("rgb2gray: the input must either be an RGB image or a colormap");
  endif
endfunction

# simplest test, double image, each channel on its own and then all maxed
%!shared img
%! img = zeros (2, 2, 3);
%! img(:,:,1) = [1 0; 0 1];
%! img(:,:,2) = [0 1; 0 1];
%! img(:,:,3) = [0 0; 1 1];
%! img = rgb2gray (img);
%!assert ([img(1,1) img(1,2) img(2,1) img(2,2)], [0.29894 0.58704 0.11402 1]);

%!assert (class (rgb2gray (single (ones (5, 5, 3)))), "single")
%!assert (class (rgb2gray (uint8 (ones (5, 5, 3)))), "uint8")
