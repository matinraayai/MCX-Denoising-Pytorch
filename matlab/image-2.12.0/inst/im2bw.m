## Copyright (C) 2000 Kai Habel <kai.habel@gmx.de>
## Copyright (C) 2012-2016 Carnë Draug <carandraug@octave.org>
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
## @deftypefn  {Function File} {} im2bw (@var{img})
## @deftypefnx {Function File} {} im2bw (@var{X}, @var{cmap})
## @deftypefnx {Function File} {} im2bw (@dots{}, @var{threshold})
## @deftypefnx {Function File} {} im2bw (@dots{}, @var{method})
## Convert image to binary, black and white, by threshold.
##
## The input image @var{img} can either be a grayscale or RGB image.  In the later
## case, @var{img} is first converted to grayscale with @code{rgb2gray}.  Input
## can also be an indexed image @var{X} in which case the colormap @var{cmap}
## needs to be specified.
##
## The value of @var{threshold} should be in the range [0,1] independently of the
## class of @var{img}.  Values from other classes can be converted to the correct
## value with @code{im2double}:
##
## @example
## bw = im2bw (img_of_class_uint8, im2double (thresh_of_uint8_class));
## @end example
##
## For an automatic threshold value, consider using @code{graythresh}.
## The argument @var{method} is a string that specifies a valid algorithm
## available in @code{graythresh}.  The following are equivalent:
##
## @example
## bw = im2bw (img, "moments");
## bw = im2bw (img, graythresh (img, "moments"));
## @end example
##
## @seealso{graythresh, ind2gray, otsuthresh, rgb2gray}
## @end deftypefn

function BW = im2bw (img, cmap, thresh = 0.5)

  if (nargin < 1 || nargin > 3)
    print_usage ();
  elseif (nargin == 3 && ! isind (img))
    error ("im2bw: IMG must be an indexed image when are 3 input arguments");
  elseif (nargin == 3 && ! iscolormap (cmap))
    error ("im2bw: CMAP must be a colormap");
  elseif (nargin == 2)
    thresh = cmap;
  endif

  if (! isimage (img))
    error ("im2bw: IMG must be an image");
  elseif (! ischar (thresh) && ! (isnumeric (thresh) && isscalar (thresh)
                                  && thresh >= 0 && thresh <= 1))
    error ("im2bw: THRESHOLD must be a string or a scalar in the interval [0 1]");
  endif

  if (islogical (img))
    warning ("im2bw: IMG is already binary so nothing is done");
    tmp = img;

  else
    ## Convert img to gray scale
    if (nargin == 3)
      ## indexed image (we already checked that is indeed indexed earlier)
      img = ind2gray (img, cmap);
    elseif (isrgb (img))
      img = rgb2gray (img);
    else
      ## Everything else, we do nothing, no matter how many dimensions
    endif

    if (ischar (thresh))
      thresh = graythresh (img, thresh);
    endif

    ## Convert the threshold value to same class as the image which
    ## is faster and saves more memory than the opposite.
    if (isinteger (img))
      ## We do the conversion from double to int ourselves (instead
      ## of using im2uint* functions), because those functions round
      ## during the conversion but we need thresh to be the limit.
      ## See bug #46390.
      cls = class(img);
      I_min = double (intmin (cls));
      I_range = double (intmax (cls)) - I_min;
      thresh = cast (floor ((thresh * I_range) + I_min), cls);
    elseif (isfloat (img))
      ## do nothing
    else
      ## we should have never got here in the first place anyway
      error ("im2bw: unsupported image of class '%s'", class (img));
    endif

    tmp = (img > thresh);
  endif

  if (nargout > 0)
    BW = tmp;
  else
    imshow (tmp);
  endif

endfunction

%!assert(im2bw ([0 0.4 0.5 0.6 1], 0.5), logical([0 0 0 1 1])); # basic usage
%!assert(im2bw (uint8 ([0 100 255]), 0.5), logical([0 0 1]));   # with a uint8 input

## We use "bw = im2bw (...)" because otherwise it would display a figure
%!warning <is already binary so nothing is done> bw = im2bw (logical ([0 1 0]));
%!warning <is already binary so nothing is done> bw = im2bw (logical ([0 1 0]), 1);

%!test
%! warning ("off", "all", "local");
%! assert (im2bw (logical ([0 1 0])),    logical ([0 1 0]))
%! assert (im2bw (logical ([0 1 0]), 0), logical ([0 1 0]))
%! assert (im2bw (logical ([0 1 0]), 1), logical ([0 1 0]))

## bug #46390 (on the rounding/casting of the threshold value)
%!assert (nnz (im2bw (uint8 ([0:255]), 0.9)), 26)

%!test
%! img = uint8 ([0:255]);
%! s = 0;
%! for i=0:.1:1
%!   s += nnz (im2bw (img, i));
%! endfor
%! assert (s, 1405)

## threshold may be a negative value in the image class so care must
## taken when casting and rounding it.
%!assert (nnz (im2bw (int16 ([-128:127]), 0.499)), 194)
%!assert (nnz (im2bw (int16 ([-128:127]), 0.500)), 128)
%!assert (nnz (im2bw (int16 ([-128:127]), 0.501)), 62)

%!test
%! img = uint16 ([0:intmax("uint16")]);
%! s = 0;
%! for i=0:.1:1
%!   s += nnz (im2bw (img, i));
%! endfor
%! assert (s, 360445)

%!test
%! img = int16 ([intmin("int16"):intmax("int16")]);
%! s = 0;
%! for i=0:.1:1
%!   s += nnz (im2bw (img, i));
%! endfor
%! assert (s, 360445)

%!test
%! im = [((randn(10)/10)+.3) ((randn(10)/10)+.7)];
%! assert (im2bw (im, "Otsu"), im2bw (im, graythresh (im, "Otsu")))
%! assert (im2bw (im, "moments"), im2bw (im, graythresh (im, "moments")))

%!test
%! im = [((randn(10)/10)+.3) ((randn(10)/10)+.7)];
%! im = reshape (im, [10 10 1 2]);
%! assert (im2bw (im, "Otsu"), im2bw (im, graythresh (im, "Otsu")))
%! assert (im2bw (im, "moments"), im2bw (im, graythresh (im, "moments")))
