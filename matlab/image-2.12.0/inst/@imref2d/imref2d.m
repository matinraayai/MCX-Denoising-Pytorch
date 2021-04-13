## Copyright (C) 2018 Martin Janda <janda.martin1@gmail.com>
## 
## This program is free software: you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see
## <https://www.gnu.org/licenses/>.

## -*- texinfo -*- 
## @deftypefn {} {@var{r} =} imref2d
## @deftypefnx {} {@var{r} =} imref2d (@var{imageSize})
## @deftypefnx {} {@var{r} =} imref2d (@var{imageSize}, @var{pixelExtentInWorldX}, @var{pixelExtentInWorldY})
## @deftypefnx {} {@var{r} =} imref2d (@var{imageSize}, @var{xWorldLimits}, @var{yWorldLimits})
## Reference 2-D image to world coordinates.
##
## Creates an imref2d object referencing a 2-D m-by-n image with the size
## @var{imageSize} to world coordinates.  The world extent is either given
## by @var{xWorldLimits} and @var{yWorldLimits} or computed from
## @var{pixelExtentInWorldX} and @var{pixelExtentInWorldY}.
## @var{imageSize} is [2, 2] by default.
##
## Intrinsic coordinates are x = 1.0, y = 1.0 in the center of the top left
## pixel and x = n, y = m in the center of the bottom right pixel.
## Spatial resolution in each dimension can be different.
##
## imref2d object has the following properties:
##
## ImageSize - two element integer vector with image height and width
## in pixels.
##
## XWorldLimits - limits of the image along the x-axis in world units
## specified as a two element real vector @code{[xMin, xMax]}.
##
## YWorldLimits - limits of the image along the y-axis in world units
## specified as a two element real vector @code{[yMin, yMax]}.
##
## PixelExtentInWorldX - pixel extent along the x-axis in world units
## specified as a real scalar.
##
## PixelExtentInWorldY - pixel extent along the y-axis in world units
## specified as a real scalar.
##
## ImageExtentInWorldX - image extent along the x-axis in world units
## specified as a real scalar.
##
## ImageExtentInWorldY - image extent along the y-axis in world units
## specified as a real scalar.
##
## XIntrinsicLimits - limits of the image along the x-axis in intrinsic
## units, equals to @code{[n - 0.5, n + 0.5]}.
##
## YIntrinsicLimits - limits of the image along the y-axis in intrinsic
## units, equals to @code{[m - 0.5, m + 0.5]}.
##
## @seealso{imref3d}
## @end deftypefn

function r = imref2d (imageSize, varargin)
  if (nargin > 1 && nargin != 3)
    print_usage();
  endif

  if (nargin == 0)
    imageSize = [2, 2];
  endif

  if (nargin > 0)
    if (length (imageSize) < 2)
      error ("Octave:invalid-input-arg", ...
      "ImageSize must have at least two elements");
    endif

    validateattributes (imageSize, {"numeric"}, ...
    {"positive", "integer", "vector"}, "imref2d", "imageSize");
  endif

  imageSize = imageSize(1:2);
  m = imageSize(1);
  n = imageSize(2);

  if (numel (varargin) == 0)
    xWorldLimits = [0.5, n + 0.5];
    yWorldLimits = [0.5, m + 0.5];
  elseif (numel (varargin) == 2)
    if (isscalar (varargin{1}))
      validateattributes (varargin{1}, {"numeric"}, ...
      {"real", "positive", "scalar"}, "imref2d", "pixelExtentInWorldX");
      validateattributes (varargin{2}, {"numeric"}, ...
      {"real", "positive", "scalar"}, "imref2d", "pixelExtentInWorldY");
      pixelExtentInWorldX = varargin{1};
      pixelExtentInWorldY = varargin{2};
    else
      validateattributes (varargin{1}, {"numeric"}, ...
      {"real", "increasing", "vector", "size", [1, 2]}, "imref2d", ...
      "xWorldLimits");
      validateattributes (varargin{2}, {"numeric"}, ...
      {"real", "increasing", "vector", "size", [1, 2]}, "imref2d", ...
      "yWorldLimits");
      xWorldLimits = varargin{1};
      yWorldLimits = varargin{2};
    endif
  endif

  if (exist ("pixelExtentInWorldX") && exist ("pixelExtentInWorldY"))
    imageExtentInWorldX = pixelExtentInWorldX * n;
    imageExtentInWorldY = pixelExtentInWorldY * m;
    xWorldLimits = [pixelExtentInWorldX / 2, imageExtentInWorldX + ...
    pixelExtentInWorldX / 2];
    yWorldLimits = [pixelExtentInWorldY / 2, imageExtentInWorldY + ...
    pixelExtentInWorldY / 2];
  elseif (exist ("xWorldLimits") && exist ("yWorldLimits"))
    imageExtentInWorldX = xWorldLimits(2) - xWorldLimits(1);
    imageExtentInWorldY = yWorldLimits(2) - yWorldLimits(1);
    pixelExtentInWorldX = imageExtentInWorldX / n;
    pixelExtentInWorldY = imageExtentInWorldY / m;
  endif

  xIntrinsicLimits = [0.5, n + 0.5];
  yIntrinsicLimits = [0.5, m + 0.5];

  r.ImageSize = imageSize;
  r.XWorldLimits = xWorldLimits;
  r.YWorldLimits = yWorldLimits;
  r.PixelExtentInWorldX = pixelExtentInWorldX;
  r.PixelExtentInWorldY = pixelExtentInWorldY;
  r.ImageExtentInWorldX = imageExtentInWorldX;
  r.ImageExtentInWorldY = imageExtentInWorldY;
  r.XIntrinsicLimits = xIntrinsicLimits;
  r.YIntrinsicLimits = yIntrinsicLimits;
  r = class (r, "imref2d");
endfunction

%!error id=Octave:invalid-fun-call imref2d (1, 2, 3, 4)
%!error id=Octave:invalid-input-arg imref2d (42)
%!error id=Octave:invalid-input-arg imref2d ([42])
%!error id=Octave:expected-integer imref2d ([4.2, 42])
%!error id=Octave:expected-positive imref2d ([0, 0])
%!error id=Octave:expected-positive imref2d ([-4, 2])
%!error id=Octave:expected-positive imref2d ([4, 2], 0, 2)
%!error id=Octave:expected-positive imref2d ([4, 2], 2, 0)
%!error id=Octave:expected-real imref2d ([4, 2], j, 2)
%!error id=Octave:expected-real imref2d ([4, 2], 2, j)
%!error id=Octave:expected-real imref2d ([4, 2], [j, 2], [3, 4])
%!error id=Octave:expected-real imref2d ([4, 2], [1, 2], [j, 4])
%!error id=Octave:expected-vector imref2d ([4, 2], [], [])
%!error id=Octave:expected-vector imref2d ([4, 2], [], [1])
%!error id=Octave:expected-scalar imref2d ([4, 2], [1], [])
%!error id=Octave:incorrect-size imref2d ([4, 2], [1, 2], [0])
%!error id=Octave:incorrect-size imref2d ([4, 2], [1, 2], [1, 2, 3])
%!error id=Octave:incorrect-size imref2d ([4, 2], [1, 2, 3], [1, 2])
%!error id=Octave:incorrect-size imref2d ([4, 2], [1; 2], [1, 2])
%!error id=Octave:incorrect-size imref2d ([4, 2], [1, 2], [1; 2])
%!error id=Octave:invalid-indexing imref2d().InvalidProperty
%!error id=Octave:expected-increasing imref2d ([100 200], [1.5 0.5], [2.5 3.5])
%!error id=Octave:expected-increasing imref2d ([100 200], [1.5 2.5], [2.5 1.5])

%!test
%! r = imref2d;
%! assert (r.XWorldLimits, [0.5, 2.5])
%! assert (r.YWorldLimits, [0.5, 2.5])
%! assert (r.ImageSize, [2, 2])
%! assert (r.PixelExtentInWorldX, 1)
%! assert (r.PixelExtentInWorldY, 1)
%! assert (r.ImageExtentInWorldX, 2)
%! assert (r.ImageExtentInWorldY, 2)
%! assert (r.XIntrinsicLimits, [0.5, 2.5])
%! assert (r.YIntrinsicLimits, [0.5, 2.5])

%!test
%! r = imref2d ([100, 200]);
%! assert (r.XWorldLimits, [0.5, 200.5])
%! assert (r.YWorldLimits, [0.5, 100.5])
%! assert (r.ImageSize, [100, 200])
%! assert (r.PixelExtentInWorldX, 1)
%! assert (r.PixelExtentInWorldY, 1)
%! assert (r.ImageExtentInWorldX, 200)
%! assert (r.ImageExtentInWorldY, 100)
%! assert (r.XIntrinsicLimits, [0.5, 200.5])
%! assert (r.YIntrinsicLimits, [0.5, 100.5])

%!test
%! xWorldLimits = [2, 5];
%! yWorldLimits = [3, 6];
%! r = imref2d ([291, 240], xWorldLimits, yWorldLimits);
%! assert (r.XWorldLimits, [2, 5])
%! assert (r.YWorldLimits, [3, 6])
%! assert (r.ImageSize, [291, 240])
%! assert (r.PixelExtentInWorldX, 0.0125)
%! assert (r.PixelExtentInWorldY, 0.0103, 1e-3)
%! assert (r.ImageExtentInWorldX, 3)
%! assert (r.ImageExtentInWorldY, 3)
%! assert (r.XIntrinsicLimits, [0.5, 240.5])
%! assert (r.YIntrinsicLimits, [0.5, 291.5])

%!test
%! pixelExtentInWorldX = 0.3125;
%! pixelExtentInWorldY = 0.3125;
%! r = imref2d ([512, 512], pixelExtentInWorldX, pixelExtentInWorldY);
%! assert (r.XWorldLimits, [0.15625, 160.1562], 1e-4)
%! assert (r.YWorldLimits, [0.15625, 160.1562], 1e-4)
%! assert (r.ImageSize, [512, 512])
%! assert (r.PixelExtentInWorldX, 0.3125)
%! assert (r.PixelExtentInWorldY, 0.3125)
%! assert (r.ImageExtentInWorldX, 160)
%! assert (r.ImageExtentInWorldY, 160)
%! assert (r.XIntrinsicLimits, [0.5, 512.5])
%! assert (r.YIntrinsicLimits, [0.5, 512.5])

%!test
%! pixelExtentInWorldX = 0.1;
%! pixelExtentInWorldY = 0.4;
%! r = imref2d ([100, 200], pixelExtentInWorldX, pixelExtentInWorldY);
%! assert (r.XWorldLimits, [0.05, 20.05], 1e-4)
%! assert (r.YWorldLimits, [0.2, 40.2], 1e-4)
%! assert (r.ImageSize, [100, 200])
%! assert (r.PixelExtentInWorldX, 0.1)
%! assert (r.PixelExtentInWorldY, 0.4)
%! assert (r.ImageExtentInWorldX, 20)
%! assert (r.ImageExtentInWorldY, 40)
%! assert (r.XIntrinsicLimits, [0.5, 200.5])
%! assert (r.YIntrinsicLimits, [0.5, 100.5])

## changing ImageSize
%!test
%! r = imref2d;
%! assert (r.XWorldLimits, [0.5, 2.5])
%! assert (r.YWorldLimits, [0.5, 2.5])
%! assert (r.ImageSize, [2, 2])
%! assert (r.PixelExtentInWorldX, 1)
%! assert (r.PixelExtentInWorldY, 1)
%! assert (r.ImageExtentInWorldX, 2)
%! assert (r.ImageExtentInWorldY, 2)
%! assert (r.XIntrinsicLimits, [0.5, 2.5])
%! assert (r.YIntrinsicLimits, [0.5, 2.5])
%! r.ImageSize = [800, 600];
%! assert (r.XWorldLimits, [0.5, 2.5])
%! assert (r.YWorldLimits, [0.5, 2.5])
%! assert (r.ImageSize, [800, 600])
%! assert (r.PixelExtentInWorldX, 0.003333, 1e-5)
%! assert (r.PixelExtentInWorldY, 0.0025)
%! assert (r.ImageExtentInWorldX, 2)
%! assert (r.ImageExtentInWorldY, 2)
%! assert (r.XIntrinsicLimits, [0.5, 600.5])
%! assert (r.YIntrinsicLimits, [0.5, 800.5])

## changing XWorldLimits and YWorldLimits
%!test
%! r = imref2d;
%! assert (r.XWorldLimits, [0.5, 2.5])
%! assert (r.YWorldLimits, [0.5, 2.5])
%! assert (r.ImageSize, [2, 2])
%! assert (r.PixelExtentInWorldX, 1)
%! assert (r.PixelExtentInWorldY, 1)
%! assert (r.ImageExtentInWorldX, 2)
%! assert (r.ImageExtentInWorldY, 2)
%! assert (r.XIntrinsicLimits, [0.5, 2.5])
%! assert (r.YIntrinsicLimits, [0.5, 2.5])
%! r.XWorldLimits = [-60, 13.33];
%! r.YWorldLimits = [-900.8, -560.26];
%! assert (r.XWorldLimits, [-60, 13.33])
%! assert (r.YWorldLimits, [-900.8, -560.26])
%! assert (r.PixelExtentInWorldX, 36.6650)
%! assert (r.PixelExtentInWorldY, 170.27, 1e-5)
%! assert (r.ImageExtentInWorldX, 73.33, 1e-5)
%! assert (r.ImageExtentInWorldY, 340.54, 1e-5)
%! assert (r.XIntrinsicLimits, [0.5, 2.5])
%! assert (r.YIntrinsicLimits, [0.5, 2.5])

%!test
%! r = imref2d;
%! fail ("r.XWorldLimits = []", "")
%! fail ("r.XWorldLimits = [1]", "")
%! fail ("r.XWorldLimits = [j]", "")
%! fail ("r.XWorldLimits = [1; 2]", "")
%! fail ("r.YWorldLimits = []", "")
%! fail ("r.YWorldLimits = [1]", "")
%! fail ("r.YWorldLimits = [j]", "")
%! fail ("r.YWorldLimits = [1; 2]", "")

%!assert (imref2d ([4, 2, 3]).ImageSize, [4, 2]);