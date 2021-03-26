## Copyright (C) 2007 Søren Hauberg <soren@hauberg.org>
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
## @deftypefn {Function File} @var{J} = imfilter(@var{I}, @var{f})
## @deftypefnx{Function File} @var{J} = imfilter(@var{I}, @var{f}, @var{options}, @dots{})
## Computes the linear filtering of the image @var{I} and the filter @var{f}.
## The computation is performed using double precision floating point numbers,
## but the class of the input image is preserved as the following example shows.
## @example
## I = 255*ones(100, 100, "uint8");
## f = fspecial("average", 3);
## J = imfilter(I, f);
## class(J)
## @result{} ans = uint8
## @end example
##
## The function also accepts a number of optional arguments that control the
## details of the filtering. The following options is currently accepted
## @table @samp
## @item S
## If a scalar input argument is given, the image is padded with this scalar
## as part of the filtering. The default value is 0.
## @item "symmetric"
## The image is padded symmetrically. 
## @item "replicate"
## The image is padded using the border of the image.
## @item "circular"
## The image is padded by circular repeating of the image elements.
## @item "same"
## The size of the output image is the same as the input image. This is the default
## behaviour.
## @item "full"
## Returns the full filtering result.
## @item "corr"
## The filtering is performed using correlation. This is the default behaviour.
## @item "conv"
## The filtering is performed using convolution.
## @end table
## @seealso{conv2, filter2, fspecial, padarray}
## @end deftypefn

function retval = imfilter(im, f, varargin)

  if (nargin < 2)
    print_usage ();
  endif

  ## Check image
  if (! isimage (im))
    error("imfilter: IM must be an image");
  endif
  [imrows, imcols, imchannels, tmp] = size(im);
  if (tmp != 1 || (imchannels != 1 && imchannels != 3))
    error("imfilter: first input argument must be an image");
  endif
  C = class(im);

  ## Check filter (XXX: matlab support 3D filter, but I have no idea what they do with them)
  if (! isnumeric (f))
    error("imfilter: F must be a numeric array");
  endif
  [frows, fcols, tmp] = size(f);
  if (tmp != 1)
    error("imfilter: second argument must be a 2-dimensional matrix");
  endif

  ## Parse options
  res_size = "same";
  res_size_options = {"same", "full"};
  pad = 0;
  pad_options = {"symmetric", "replicate", "circular"};
  ftype = "corr";
  ftype_options = {"corr", "conv"};
  for i = 1:length(varargin)
    v = varargin{i};
    if (any(strcmpi(v, pad_options)) || isscalar(v))
      pad = v;
    elseif (any(strcmpi(v, res_size_options)))
      res_size = v;
    elseif (any(strcmpi(v, ftype_options)))
      ftype = v;
    else
      warning("imfilter: cannot handle input argument number %d", i+2);
    endif
  endfor

  ## Pad the image
  im = padarray(im, floor([frows/2, fcols/2]), pad);
  if (mod(frows,2) == 0)
    im = im(2:end, :, :);
  endif
  if (mod(fcols,2) == 0)
    im = im(:, 2:end, :);
  endif

  ## Do the filtering
  if (strcmpi(res_size, "same"))
    res_size = "valid";
  else # res_size == "full"
    res_size = "same";
  endif
  if (strcmpi(ftype, "corr"))
    for i = imchannels:-1:1
      retval(:,:,i) = filter2(f, im(:,:,i), res_size);
    endfor
  else
    for i = imchannels:-1:1
      retval(:,:,i) = conv2(im(:,:,i), f, res_size);
    endfor
  endif

  ## Change the class of the output to the class of the input
  ## (the filtering functions returns doubles)
  retval = cast(retval, C);

endfunction

%!test
%!  img = [
%!   8   2   6   7   4   3   7   8   4   1
%!   9   9   1   1   4   7   3   3   8   1
%!   2   9   8   3   7   6   5   8   6   5
%!   9   5   9   1   8   2   7   3   5   8
%!   6   8   7   1   2   2   9   9   9   9
%!   1   2   7   8   5   5   9   4   3   2
%!   3   4   7   7   5   9   5   2   7   6
%!   5   9   4   3   6   4   2   3   7   5
%!   9   8   6   9   7   6   2   6   4   1
%!   9   9   2   1   7   3   3   5   6   4];
%!
%! expected_corr = [
%!  46  53  30  34  44  42  40  51  42  19
%!  48  66  57  42  46  50  59  58  49  34
%!  48  67  55  54  44  58  50  50  64  39
%!  44  77  52  43  28  55  57  75  70  50
%!  29  51  65  51  42  50  60  62  55  42
%!  23  44  58  59  63  59  55  57  50  36
%!  36  50  52  56  56  47  48  45  47  39
%!  51  64  70  62  56  50  40  38  41  31
%!  58  72  50  49  58  45  41  42  49  28
%!  27  37  27  21  19  26  16  23  24  17];
%! assert (imfilter (img, [0 1 0; 2 1 1; 1 2 2]), expected_corr)
%!
%! ## test order of options (and matching with defaults)
%! assert (imfilter (img, [0 1 0; 2 1 1; 1 2 2], 0), expected_corr)
%! assert (imfilter (img, [0 1 0; 2 1 1; 1 2 2], "corr"), expected_corr)
%! assert (imfilter (img, [0 1 0; 2 1 1; 1 2 2], "corr", 0), expected_corr)
%! assert (imfilter (img, [0 1 0; 2 1 1; 1 2 2], 0, "corr"), expected_corr)
%!
%! expected_conv = [
%!  21  31  23  22  21  28  29  26  22   6
%!  47  55  43  43  51  44  49  64  44  24
%!  56  69  53  34  47  50  57  48  52  37
%!  38  70  60  56  41  57  54  61  66  44
%!  46  67  53  48  32  54  59  65  63  46
%!  28  56  63  50  36  54  58  66  63  47
%!  20  43  55  62  67  57  52  53  44  28
%!  42  51  54  61  57  53  44  46  48  39
%!  53  70  63  50  57  42  38  38  43  33
%!  53  62  50  54  52  44  38  40  40  20];
%! assert (imfilter (img, [0 1 0; 2 1 1; 1 2 2], "conv"), expected_conv)
%!
%! ## alternative class
%! assert (imfilter (single (img), [0 1 0; 2 1 1; 1 2 2]),
%!         single (expected_corr))
%! assert (imfilter (int8 (img), [0 1 0; 2 1 1; 1 2 2]),
%!         int8 (expected_corr))
%! assert (imfilter (uint8 (img), [0 1 0; 2 1 1; 1 2 2]),
%!         uint8 (expected_corr))
%!
%! assert (imfilter (single (img), [0 1 0; 2 1 1; 1 2 2], "conv"),
%!         single (expected_conv))
%! assert (imfilter (int8 (img), [0 1 0; 2 1 1; 1 2 2], "conv"),
%!         int8 (expected_conv))
%! assert (imfilter (uint8 (img), [0 1 0; 2 1 1; 1 2 2], "conv"),
%!         uint8 (expected_conv))
%!

## test padding with even sized filters (bug #45568)
%!test
%! I = zeros (6);
%! I(2:3,2:3) = 1;
%! F = zeros (4);
%! F(2,2:3) = 1;
%! result = [0 0 0 0 0 0
%!           1 2 1 0 0 0
%!           1 2 1 0 0 0
%!           0 0 0 0 0 0
%!           0 0 0 0 0 0
%!           0 0 0 0 0 0];
%! assert (imfilter (I, F), result)
