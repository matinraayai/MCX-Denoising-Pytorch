## Copyright (C) 2017 Avinoam Kalma <a.kalma@gmail.com>
## Copyright (C) 2017 Carnë Draug <carandraug@octave.org>
##
## This program is free software; you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*-
## @deftypefn  {Function File} {} imsharpen (@var{im})
## @deftypefnx {Function File} {} imsharpen (@var{im}, @var{option}, @var{value}, @dots{})
## Sharpen image using unsharp masking.
##
## @var{im} must be a grayscale or RGB image.  The unsharp masking can
## be controlled with @var{option}-@var{value} pairs.
##
## The unsharp masking technique is equivalent to:
##
## @example
## @var{im} + @var{k} * (@var{im} - smooth (@var{im}))
## @end example
##
## where @var{im} is a grayscale image and @command{smooth} performs
## gaussian smoothing.  RGB images are transformed to Lab colorspace,
## the L channel is sharpen to L', and L'ab is transformed back to RGB.
## See @url{https://en.wikipedia.org/wiki/Unsharp_masking,
## "Unsharp masking" in Wikipedia}
##
## The following options control the unsharp masking:
##
## @table @asis
## @item @qcode{"Radius"}
## Sigma of Gaussian Filter for the smoothing stage.  Must be a
## positive number.  Defaults to 1.
##
## @item @qcode{"Amount"}
## Magnitude of the overshoot @var{k}.   Must be a non-negative
## number.  Defaults to 0.8.
##
## @item @qcode{"Threshold"}
## Minimum brightness change that will be sharpened.  Must be in the
## range [0 1].  Defaults to 0.
##
## @end table
##
## Examples:
##
## @example
## @group
## out = imsharpen (im);              # Using imsharpen with default values
## out = imsharpen (im, "Radius", 1.5);
## out = imsharpen (im, "Amount", 1.2);
## out = imsharpen (im, "Threshold", 0.5);
## out = imsharpen (im, "Radius", 1.5, "Amount", 1.2, "Threshold", 0.5);
## @end group
## @end example
##
## @seealso{imfilter, fspecial}
## @end deftypefn

function [sharp] = imsharpen (im, varargin)

  if (nargin == 0)
    print_usage ();
  elseif (! isnumeric (im) && ! isbool (im))
    error ("imsharpen: IM must be numeric or logical");
  elseif (ndims (im) > 4 || all (size (im, 3) != [1 3]))
    error ("imsharpen: IM must be a grayscale or RGB image");
  endif

  p = inputParser ();
  p.addParamValue ("Radius", 1, @(x) isnumeric (x) && isscalar (x));
  p.addParamValue ("Amount", 0.8, @(x) isnumeric (x) && isscalar (x));
  p.addParamValue ("Threshold", 0, @(x) isnumeric (x) && isscalar (x));
  p.parse (varargin{:});

  if (p.Results.Radius <= 0)
    error ("imsharpen: RADIUS should be positive");
  elseif (p.Results.Amount < 0)
    error ("imsharpen: AMOUNT should be non-negative");
  elseif (p.Results.Threshold < 0 || p.Results.Threshold > 1)
    error ("imsharpen: THRESHOLD should be in the range [0:1]");
  endif

  imsharpen_size = ceil (max (4 * p.Results.Radius +1, 3));
  if (mod (imsharpen_size, 2) == 0)
    imsharpen_size += 1;
  endif

  if (size (im, 3) == 1)
    sharp = USMGray (im, imsharpen_size, p.Results.Radius,
                     p.Results.Amount, p.Results.Threshold);
  else
    sharp = USMColor (im, imsharpen_size, p.Results.Radius,
                      p.Results.Amount, p.Results.Threshold);
  endif
  sharp = imcast (sharp, class (im));
endfunction

## UnSharp Masking of gray images
function [sharp] = USMGray (im, hsize, sigma, amount, thresh)
  f = fspecial ("gaussian", hsize, sigma);
  sharp = im2double (im);
  filtered = imfilter (sharp, f, "replicate");
  g = sharp - filtered;
  if (thresh > 0)
    absg = abs (g);
    thresh *= max (absg(:));
    g(absg <= thresh) = 0;
  endif
  sharp += amount*g;
endfunction

## UnSharp Masking of color images
## Transform image to CIELab color space, perform UnSharp Masking on L channel,
## and transform back to RGB.
function [sharp] = USMColor (im, hsize, sigma, amount, thresh)
  lab = rgb2lab (im);
  lab(:,:,1) = USMGray (lab(:,:,1), hsize, sigma, amount, thresh);
  sharp = lab2rgb (lab);
endfunction

%!test
%! A = zeros (7, 7);
%! A(4,4) = 1;
%! B = [
%!  0.00000   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000
%!  0.00000  -0.00238  -0.01064  -0.01755  -0.01064  -0.00238   0.00000
%!  0.00000  -0.01064  -0.04771  -0.07866  -0.04771  -0.01064   0.00000
%!  0.00000  -0.01755  -0.07866   1.67032  -0.07866  -0.01755   0.00000
%!  0.00000  -0.01064  -0.04771  -0.07866  -0.04771  -0.01064   0.00000
%!  0.00000  -0.00238  -0.01064  -0.01755  -0.01064  -0.00238   0.00000
%!  0.00000   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000];
%! assert (imsharpen (A), B, 5e-6)

%!test
%! A = zeros (7, 7);
%! A(4,4) = 1;
%! B = [
%!  -0.0035147  -0.0065663  -0.0095539  -0.0108259  -0.0095539  -0.0065663  -0.0035147
%!  -0.0065663  -0.0122674  -0.0178490  -0.0202255  -0.0178490  -0.0122674  -0.0065663
%!  -0.0095539  -0.0178490  -0.0259701  -0.0294280  -0.0259701  -0.0178490  -0.0095539
%!  -0.0108259  -0.0202255  -0.0294280   1.7666538  -0.0294280  -0.0202255  -0.0108259
%!  -0.0095539  -0.0178490  -0.0259701  -0.0294280  -0.0259701  -0.0178490  -0.0095539
%!  -0.0065663  -0.0122674  -0.0178490  -0.0202255  -0.0178490  -0.0122674  -0.0065663
%!  -0.0035147  -0.0065663  -0.0095539  -0.0108259  -0.0095539  -0.0065663  -0.0035147];
%! assert (imsharpen (A, "radius", 2), B, 5e-8)

%!test
%! A = zeros (7, 7);
%! A(4,4) = 1;
%! assert (imsharpen (A, "radius", 0.01), A)

%!test
%! A = zeros (7, 7);
%! A(4,4) = 1;
%! B = A;
%! B(3:5,3:5) = -0.000000000011110;
%! B(3:5,4)   = -0.000002981278097;
%! B(4,3:5)   = -0.000002981278097;
%! B(4,4)     =  1.000011925156828;
%! assert (imsharpen (A, "radius", 0.2), B, eps*10)

%!test
%! A = zeros (7, 7);
%! A(4,4) = 1;
%! B = [
%!   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000
%!   0.00000  -0.00297  -0.01331  -0.02194  -0.01331  -0.00297   0.00000
%!   0.00000  -0.01331  -0.05963  -0.09832  -0.05963  -0.01331   0.00000
%!   0.00000  -0.02194  -0.09832   1.83790  -0.09832  -0.02194   0.00000
%!   0.00000  -0.01331  -0.05963  -0.09832  -0.05963  -0.01331   0.00000
%!   0.00000  -0.00297  -0.01331  -0.02194  -0.01331  -0.00297   0.00000
%!   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000];
%! assert (imsharpen (A, "amount", 1), B, 5e-6)

%!test
%! A = zeros (7, 7);
%! A(4,4) = 1;
%! B = zeros (7, 7);
%! B(4,4) =  1.670317742690299;
%! B(4,3) = -0.078656265079077;
%! B(3,4) = -0.078656265079077;
%! B(4,5) = -0.078656265079077;
%! B(5,4) = -0.078656265079077;
%! assert (imsharpen (A, "Threshold", 0.117341762), B, eps*10)

%!test
%! A = zeros (7, 7);
%! A(4,4) = 1;
%! B = zeros (7, 7);
%! B(4,4) = 1.670317742690299;
%! assert (imsharpen (A, "Threshold", 0.117341763), B, eps*10)

## uint8 test
%!test
%! A = zeros (7, 7, "uint8");
%! A(3:5,3:5) = 150;
%! B = zeros (7, 7, "uint8");
%! B(3:5,3:5) = 211;
%! B(4,3:5) = 195;
%! B(3:5,4) = 195;
%! B(4,4) = 175;
%! assert (imsharpen (A), B)

## uint8 test
%!test
%! A = zeros (7, 7, "uint8");
%! A(3:5,3:5) = 100;
%! B = zeros (7, 7, "uint8");
%! B(3:5,3:5) = 173;
%! assert (imsharpen (A, "radius", 4), B)

## color image test #1
%!test
%! A = zeros (7, 7, 3, "uint8");
%! A(4,4,:) = 255;
%! assert (imsharpen (A), A)

## Matlab result is different by 1 grayscale
%!xtest
%! A = zeros(7,7,3, "uint8");
%! A(4,4,1) = 255;
%! B = A;
%! B(4,4,2) = 146;   # Octave result is 145;
%! B(4,4,3) = 100;   # Octave result is 99;
%! assert (imsharpen (A), B)

## Matlab result is different by 1 grayscale
%!xtest
%! A = zeros (7, 7, 3, "uint8");
%! A(3:5,3:5,1) = 100;
%! A(3:5,3:5,2) = 150;
%! B = A;
%! B(3:5,3:5,1) = 164;
%! B(3:5,4,1)   = 146;     # Octave result is 147
%! B(4,3:5,1)   = 146;     # Octave result is 145
%! B(4,4,1)     = 125;     # Octave result is 126
%! B(3:5,3:5,2) = 213;
%! B(3:5,4,2)   = 195;     # Octave result is 196
%! B(4,3:5,2)   = 195;     # Octave result is 196
%! B(4,4,2)     = 175;
%! B(3:5,3:5,3) = 79;
%! B(3:5,4,3)   = 62;
%! B(4,3:5,3)   = 62;
%! B(4,4,3)     = 40;      # Octave result is 39
%! assert (imsharpen (A), B)

## Test input validation
%!error imsharpen ()
%!error imsharpen (ones (3, 3), "Radius")
%!error imsharpen (ones (3, 3), "Radius", 0)
%!error imsharpen (ones (3, 3), "Amount", -1)
%!error imsharpen (ones (3, 3), "Threshold", 1.5)
%!error imsharpen (ones (3, 3), "Threshold", -1)
%!error imsharpen (ones (3, 3), "foo")
%!error imsharpen ("foo")
