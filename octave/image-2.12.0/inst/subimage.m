## Copyright (C) 2014 Carnë Draug <carandraug@octave.org>
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
## @deftypefn  {Function File} {} subimage (@var{bw})
## @deftypefnx {Function File} {} subimage (@var{img})
## @deftypefnx {Function File} {} subimage (@var{rgb})
## @deftypefnx {Function File} {} subimage (@var{ind}, @var{cmap})
## @deftypefnx {Function File} {} subimage (@var{x}, @var{y}, @dots{})
## @deftypefnx {Function File} {@var{h} =} subimage (@dots{})
## Display images in subplots.
##
## A single figure, even with multiple subplots, is limited to a single
## colormap.  With the exception of truecolor images, images will use the
## figure colormap which make it impossible to have multiple images with
## different display.  This function transforms any image in truecolor
## to workaround this limitation.
##
## The new subimage is displayed as if using @code{image}.  The optional
## arguments @var{x} and @var{y} are passed to @code{image} to specify the
## range of the axis labels.
##
## @seealso{image, imagesc, imshow, subplot}
## @end deftypefn

function h = subimage (varargin)
  if (nargin < 1 || nargin > 4)
    print_usage ();
  endif

  if (nargin < 3)
    alternative_xy = false;
    im_ind = 1;
  else
    alternative_xy = true;
    im_ind = 3;
    if (numel (varargin{1}) == 2 && numel (varargin{2}) == 2)
      x = varargin{1};
      y = varargin{2};
    else
      error ("subimage: X and Y must be two element vectors each");
    endif
  endif

  im = varargin{im_ind};
  if (numel (varargin) > im_ind)
    rgb = ind2rgb (im, varargin{im_ind +1});
  elseif (isbw (im))
    rgb = repmat (im2uint8 (im), [1 1 3 1]);
  elseif (isgray (im))
    rgb = repmat (im, [1 1 3 1]);
  elseif (isrgb (im))
    rgb = im;
  else
    error ("subimage: no valid BW, IMG, IND, or RGB images in input arguments");
  endif

  if (alternative_xy)
    tmp_h = image (x, y, rgb);
  else
    tmp_h = image (rgb);
  endif

  if (nargout > 0)
    h = tmp_h;
  endif
endfunction

