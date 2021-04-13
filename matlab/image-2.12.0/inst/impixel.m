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
## @deftypefn  {Function File} {} impixel ()
## @deftypefnx {Function File} {} impixel (@var{img}, @var{x}, @var{y})
## @deftypefnx {Function File} {} impixel (@var{ind}, @var{map}, @var{x}, @var{y})
## @deftypefnx {Function File} {} impixel (@var{xdata}, @var{ydata}, @var{img}, @var{x}, @var{y})
## @deftypefnx {Function File} {} impixel (@var{xdata}, @var{ydata}, @var{ind}, @var{map}, @var{x}, @var{y})
## @deftypefnx {Function File} {[@var{x}, @var{y}, @var{p}] =} impixel (@dots{})
## Get pixel values.
##
## For any image @var{img}, or indexed image @var{ind} with colormap @var{map},
## returns the pixel values at the image coordinates @var{x} and @var{y}.
##
## The 2 element vectors @var{xdata} and @var{ydata} can be used to set an
## alternative coordinate system.
##
## If more than one output argument is requested, also returns the @var{x} and
## @var{y} coordinates for the image.
##
## @itemize @bullet
## @item
## The pixel values are always returned in RGB style triples, even when
## @var{img} is a grayscale image.
##
## @item
## The value for pixel coordinates outside the image limits is NaN.
##
## @item
## Because a floating-point is required to represent a NaN, the pixel
## values will be of class double if input is double, and single otherwise.
## @end itemize
##
## @end deftypefn

function varargout = impixel (varargin)

  if (nargin > 6)
    print_usage ();

  ## interactive usage
  elseif (nargin <= 2)
    ## FIXME not yet implemented
    print_usage ();
    if (nargin == 0)
      ## If using the current image, it is possible that xData and yData
      ## were changed? We will confirm later is they were tampered with.
      xData = get (gcf (), "xData");
      yData = get (gcf (), "yData");
    else
      ## with given image, otherwise we will use current image
      [img, map, is_indexed] = get_image (varargin{:});
    endif

    ## If only 2 output arguments are requested in interactive mode, then
    ## only the coordinates are required, no need to do anything else.
    if (nargout <= 2)
      varargout(1:2) = {x y};
      return
    endif

  ## non-interactive usage
  else
    x = varargin{end-1};
    y = varargin{end};
    if (! isnumeric (x) || ! isreal (x) || ! isnumeric (y) || ! isreal (y))
      error ("impixel: X and Y must be real numbers");
    endif
    x = x(:);
    y = y(:);

    if (nargin >= 5)
      [img, map, is_indexed] = get_image (varargin{3:end-2});
      xData = varargin{1};
      yData = varargin{2};
      if (! isnumeric (xData) || ! isnumeric (yData))
        ## For Matlab compatibility we do not check if there's
        ## only 2 elements, or if they are real numbers
        error ("impixel: XDATA and YDATA must be numeric");
      endif
    else
      [img, map, is_indexed] = get_image (varargin{1:end-2});
      xData = 1:columns (img);
      yData = 1:rows (img);
    endif

  endif

  ## We need to return NaN if the requested pixels are outside the image
  ## limits. interp2() will respect the input class, which means it will
  ## return a 0 instead of NaN if the image is an integer class. Because
  ## of that, we convert it to single. If the input image was double, then
  ## we let it be.
  if (isinteger (img))
    img = single (img);
    if (is_indexed)
      ## There's an offset in indexed images depending on their class. An
      ## indexed image from integer class, matches the value 0 to row 1 of the
      ## colormap. An indexed image from a float class, matches value 1 to
      ## row 1. Since we are changing the class, we need to readjust it.
      img++;
    endif
  endif

  xx   = linspace (min (xData), max (xData), columns (img));
  yy   = linspace (min (yData), max (yData), rows (img));
  data = interp2 (xx, yy, img(:,:,1), x, y, "nearest");
  if (ndims (img) == 3 && size (img, 3) == 3)
    ## We can't use interp3() because XI and YI will be used to select entire
    ## columns and vectors instead of matched coordinates
    for ch = 2:3
      data(:,ch) = interp2 (xx, yy, img(:,:,ch), x, y, "nearest");
    endfor
  endif

  if (is_indexed)
    bad       = isnan (data);
    data(bad) = 1;
    data      = map(data(:),:);
    data([bad bad bad]) = NA;
  elseif (isvector (data))
    ## If we have a vector but the image was not indexed, it must have
    ## been a grayscale image. We need to repeat the values into a Nx3
    ## matrix as if they were RGB values.
    data = [data(:) data(:) data(:)];
  endif

  if (nargout > 1)
    varargout(1:3) = {x y data}
  else
    varargout(1)   = {data};
  endif

endfunction

function [img, map, is_indexed] = get_image (img, map = [])

  if (! isimage (img))
    error ("impixel: invalid image");
  endif

  is_indexed = false;
  if (nargin > 2)
    error ("impixel: too many input arguments");
  elseif (nargin == 2)
    is_indexed = true;
    if (! iscolormap (map))
      error ("impixel: invalid colormap");
    elseif (! isind (img))
      error ("impixel: invalid indexed image");
    endif
  endif

endfunction

%!shared img2d, img3d
%! img2d = uint8 (magic (10));
%! img3d(:,:,1) = img2d;
%! img3d(:,:,2) = img2d + 1;
%! img3d(:,:,3) = img2d + 2;
%! img3d = uint8 (img3d);
%!
%!assert (impixel (img2d, 2, 2), single ([80 80 80]));
%!assert (impixel (img2d, -2, 2), single ([NA NA NA]));
%!
%!assert (impixel (img2d, [1 10], [1 10]), single ([92 92 92; 59 59 59]));
%!assert (impixel (img3d, [1 10], [1 10]), single ([92 93 94; 59 60 61]));
%!assert (impixel (double (img2d), [1 10], [1 10]), [92 92 92; 59 59 59]);
%!
%!assert (impixel ([1 10], [1 10], img2d, [1 10], [1 10]), single ([92 92 92; 59 59 59]));
%!assert (impixel ([3 12], [-4 12], img2d, [1 10], [1 10]), single ([NA NA NA; 44 44 44]));
%!assert (impixel ([3 5], [-4 3], img2d, [1 10], [1 10]), single ([NA NA NA; NA NA NA]));
%!
%! ## the following returns double because it's an indexed image
%!assert (impixel ([3 12], [-4 12], img2d, gray (100), [1 10], [1 10]), [NA NA NA; 4/9 4/9 4/9]);

