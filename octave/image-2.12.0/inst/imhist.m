## Copyright (C) 2011, 2012 Carnë Draug <carandraug+dev@gmail.com>
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
## @deftypefn  {Function File} {} imhist (@var{I})
## @deftypefnx {Function File} {} imhist (@var{I}, @var{n})
## @deftypefnx {Function File} {} imhist (@var{X}, @var{cmap})
## @deftypefnx {Function File} {[@var{counts}, @var{x}] =} imhist (@dots{})
## Produce histogram counts of image @var{I}.
##
## The second argument can either be @var{n}, a scalar that specifies the number
## of bins; or @var{cmap}, a colormap in which case @var{X} is expected to be
## an indexed image. If not specified, @var{n} defaults to 2 for binary images,
## and 256 for grayscale images.
##
## If output is requested, @var{counts} is the number of counts for each bin and
## @var{x} is a range for the bins so that @code{stem (@var{x}, @var{counts})} will
## show the histogram.
##
## @emph{Note:} specially high peaks that may prevent an overview of the histogram
## may not be displayed.  To avoid this, use @code{axis "auto y"} after the call
## to @code{imhist}.
##
## @seealso{hist, histc, histeq}
## @end deftypefn

function [varargout] = imhist (img, b)

  ## "img" can be a normal or indexed image. We need to check "b" to find out
  indexed = false;

  if (nargin < 1 || nargin > 2)
    print_usage;

  elseif (nargin == 1)
    if (islogical (img))
      b = 2;
    else
      b = 256;
    endif

  elseif (nargin == 2)
    if (iscolormap (b))
      if (!isind(img))
        error ("imhist: second argument is a colormap but first argument is not an indexed image.");
      endif
      indexed = true;
      ## an indexed image reads differently wether it's uint8/16 or double
      ## If uint8/16, index+1 is the colormap row number (0 on the image
      ## corresponds to the 1st column on the colormap).
      ## If double, index is the colormap row number (no offset).
      ## isind above already checks for double/uint8/uint16 so we can use isinteger
      ## and isfloat safely
      if ( (isfloat   (img) && max (img(:)) > rows(b)  ) ||
           (isinteger (img) && max (img(:)) > rows(b)-1) )
        warning ("imhist: largest index in image exceeds length of colormap.");
      endif
    elseif (isnumeric (b) && isscalar (b) && fix(b) == b && b > 0)
      if (islogical (img) && b != 2)
        error ("imhist: there can only be 2 bins when input image is binary")
      endif
    else
      error ("imhist: second argument must be a positive integer scalar or a colormap");
    endif
  endif

  ## prepare bins and image
  if (indexed)
    if (isinteger (img))
      bins = 0:rows(b)-1;
    else
      bins = 1:rows(b);
    endif
  else
    if (isinteger (img))
      bins  = linspace (intmin (class (img)), intmax (class (img)), b);
    elseif (islogical (img))
      bins = 0:1;
    else
      ## image must be single or double
      bins = linspace (0, 1, b);
    endif
    ## we will use this bins with histc() where their values will be edges for
    ## each bin. However, what we actually want is for their values to be the
    ## center of each bin. To do this, we decrease their values by half of bin
    ## width and will increase it back at the end of the function. We could do
    ## it on the image and it would be a single step but it would be an heavier
    ## operation since images are likely to be much longer than the bins.
    ## The use of hist() is also not simple for this since values right in the
    ## middle of two bins will go to the bottom bin (4.5 will be placed on the
    ## bin 4 instead of 5 and we must keep matlab compatibility).
    ## Of course, none of this needed for binary images.
    if (!islogical (img))
      bins_adjustment = ((bins(2) - bins(1))/2);
      bins -= bins_adjustment;
    endif
    ## matlab returns bins as one column instead of a row but only for non
    ## indexed images
    bins = bins';

    ## histc does not counts values outside the edges of the bins so we need to
    ## truncate their values.

    ## truncate the minimum... integers could in no way have a value below the
    ## minimum of their class so truncation on this side is only required for
    if (isfloat (img) && min (img(:)) < 0)
      img(img < 0) = 0;
    endif

    ## truncating the maximum... also adjusts floats above 1. We might need
    if (max (img(:)) > bins(end))
      ## bins (end) is probably a decimal number. If an image is an int, we
      ## can't assign the new value since it will be fix(). So we need to change
      ## the image class to double but that will take more memory so let's
      ## avoid it if we can
      if (fix (bins(end)) != bins(end))
        img = double (img);
      endif
      img(img > bins(end)) = bins(end);
    endif
  endif

  [nn] = histc (img(:), bins);
  if (!indexed && !islogical(img))
    bins += bins_adjustment;
  endif

  if (nargout != 0)
    varargout{1} = nn;
    varargout{2} = bins;
  else
    stem (bins, nn, "marker", "none");
    xlim ([bins(1) bins(end)]);
    box off;     # remove the box to see bar for the last bin

    ## If we have a few very high peaks, it prevents the overview of the
    ## histogram since the axis are set automatically. So we consider the
    ## automatic y axis bad if it's 10 times above the median of the
    ## histogram.
    ## The (ylimit != 0) is for cases when most of the bins is zero. In
    ## such cases, the median is zero and we'd get an error trying to set
    ## "ylim ([0 0])". We could adjust it to [0 1] but in such cases, it's
    ## probably important to show how high those few peaks are.
    ylimit = round (median (nn) * 10);
    if (ylim()(2) > ylimit && ylimit != 0)
      ylim ([0 ylimit]);
    endif
    if (indexed)
      colormap (b);
    else
      colormap (gray (b));
    endif
    colorbar ("xticklabel", [], "SouthOutside");
  endif
endfunction

%!shared nn, bb, enn, ebb
%! [nn, bb] = imhist(logical([0 1 0 0 1]));
%!assert({nn, bb}, {[3 2]', [0 1]'})
%! [nn, bb] = imhist([0 0.2 0.4 0.9 1], 5);
%!assert({nn, bb}, {[1 1 1 0 2]', [0 0.25 0.5 0.75 1]'})
%! [nn, bb] = imhist([-2 0 0.2 0.4 0.9 1 5], 5);
%!assert({nn, bb}, {[2 1 1 0 3]', [0 0.25 0.5 0.75 1]'})
%! [nn, bb] = imhist(uint8([0 32 255]), 256);
%! enn = zeros(256, 1); enn([1, 33, 256]) = 1;
%! ebb = 0:255;
%!assert({nn, bb}, {enn, ebb'})
%! [nn, bb] = imhist(int8([-50 0 100]), 31);
%! enn = zeros(31, 1); enn([10, 16, 28]) = 1;
%! ebb = -128:8.5:127;
%!assert({nn, bb}, {enn, ebb'})
