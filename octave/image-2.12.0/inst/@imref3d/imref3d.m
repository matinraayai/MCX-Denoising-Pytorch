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
## @deftypefn {} {@var{r} =} imref3d
## @deftypefnx {} {@var{r} =} imref3d (@var{imageSize})
## @deftypefnx {} {@var{r} =} imref3d (@var{imageSize}, @var{pixelExtentInWorldX}, @var{pixelExtentInWorldY}, @var{pixelExtentInWorldZ})
## @deftypefnx {} {@var{r} =} imref3d (@var{imageSize}, @var{xWorldLimits}, @var{yWorldLimits}, @var{zWorldLimits})
## Reference 3-D image to world coordinates.
##
## Creates an imref3d object referencing a 3-D m-by-n-by-p image with the size
## @var{imageSize} to world coordinates.  The world extent is either given
## by @var{xWorldLimits}, @var{yWorldLimits} and @var{xWorldLimits} or computed
## from @var{pixelExtentInWorldX}, @var{pixelExtentInWorldY} and
## @var{pixelExtentInWorldZ}.  @var{imageSize} is [2, 2, 2] by default.
##
## Intrinsic coordinates are x = 1.0, y = 1.0, z = 1.0 in the center of the
## top left pixel in the first plane and x = n, y = m, z = p in the center
## of the bottom right pixel in the last plane.  Spatial resolution in each
## dimension can be different.
##
## imref3d object has the following properties:
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
## ZWorldLimits - limits of the image along the z-axis in world units
## specified as a two element real vector @code{[zMin, zMax]}.
##
## PixelExtentInWorldX - pixel extent along the x-axis in world units
## specified as a real scalar.
##
## PixelExtentInWorldY - pixel extent along the y-axis in world units
## specified as a real scalar.
##
## PixelExtentInWorldZ - pixel extent along the z-axis in world units
## specified as a real scalar.
##
## ImageExtentInWorldX - image extent along the x-axis in world units
## specified as a real scalar.
##
## ImageExtentInWorldY - image extent along the y-axis in world units
## specified as a real scalar.
##
## ImageExtentInWorldZ - image extent along the z-axis in world units
## specified as a real scalar.
##
## XIntrinsicLimits - limits of the image along the x-axis in intrinsic
## units, equals to @code{[n - 0.5, n + 0.5]}.
##
## YIntrinsicLimits - limits of the image along the y-axis in intrinsic
## units, equals to @code{[m - 0.5, m + 0.5]}.
##
## ZIntrinsicLimits - limits of the image along the z-axis in intrinsic
## units, equals to @code{[p - 0.5, p + 0.5]}.
##
## @seealso{imref2d}
## @end deftypefn

function r = imref3d (imageSize, varargin)
  if (nargin > 4)
    print_usage ();
  endif

  if (nargin == 0)
    imageSize = [2, 2, 2];
  elseif (nargin > 0)
    validateattributes (imageSize, {"numeric"}, ...
    {"positive", "integer", "vector", "size", [1, 3]}, "imref3d", "imageSize");
    
    imageSize = imageSize(1:3);
  endif
  
  m = imageSize(1);
  n = imageSize(2);
  p = imageSize(3);
  
  if (numel (varargin) == 0)
    xWorldLimits = [0.5, n + 0.5];
    yWorldLimits = [0.5, m + 0.5];
    zWorldLimits = [0.5, p + 0.5];
    r2 = @imref2d (imageSize);
  elseif (numel (varargin) == 3)
    if (isscalar (varargin{1}))
      validateattributes (varargin{1}, {"numeric"}, ...
      {"real", "positive", "scalar"}, "imref3d", "pixelExtentInWorldX");
      validateattributes (varargin{2}, {"numeric"}, ...
      {"real", "positive", "scalar"}, "imref3d", "pixelExtentInWorldY");
      validateattributes (varargin{3}, {"numeric"}, ...
      {"real", "positive", "scalar"}, "imref3d", "pixelExtentInWorldZ");
      pixelExtentInWorldX = varargin{1};
      pixelExtentInWorldY = varargin{2};
      pixelExtentInWorldZ = varargin{3};
    else
      validateattributes (varargin{1}, {"numeric"}, ...
      {"real", "increasing", "vector", "size", [1, 2]}, "imref3d", ...
      "xWorldLimits");
      validateattributes (varargin{2}, {"numeric"}, ...
      {"real", "increasing", "vector", "size", [1, 2]}, "imref3d", ...
      "yWorldLimits");
      validateattributes (varargin{3}, {"numeric"}, ...
      {"real", "increasing", "vector", "size", [1, 2]}, "imref3d", ...
      "zWorldLimits");
      xWorldLimits = varargin{1};
      yWorldLimits = varargin{2};
      zWorldLimits = varargin{3};
    endif
  endif

  if (exist ("pixelExtentInWorldX") && exist ("pixelExtentInWorldY") ...
    && exist ("pixelExtentInWorldZ"))
    imageExtentInWorldX = pixelExtentInWorldX * m;
    imageExtentInWorldY = pixelExtentInWorldY * n;
    imageExtentInWorldZ = pixelExtentInWorldZ * p;
    xWorldLimits = [pixelExtentInWorldX / 2, imageExtentInWorldX + ...
    pixelExtentInWorldX / 2];
    yWorldLimits = [pixelExtentInWorldY / 2, imageExtentInWorldY + ...
    pixelExtentInWorldY / 2];
    zWorldLimits = [pixelExtentInWorldZ / 2, imageExtentInWorldZ + ...
    pixelExtentInWorldZ / 2];
  elseif (exist ("xWorldLimits") && exist ("yWorldLimits") ...
    && exist ("zWorldLimits"))
    imageExtentInWorldX = xWorldLimits(2) - xWorldLimits(1);
    imageExtentInWorldY = yWorldLimits(2) - yWorldLimits(1);
    imageExtentInWorldZ = zWorldLimits(2) - zWorldLimits(1);
    pixelExtentInWorldX = imageExtentInWorldX / n;
    pixelExtentInWorldY = imageExtentInWorldY / m;
    pixelExtentInWorldZ = imageExtentInWorldZ / p;
  endif

  xIntrinsicLimits = [0.5, n + 0.5];
  yIntrinsicLimits = [0.5, m + 0.5];
  zIntrinsicLimits = [0.5, p + 0.5];

  r.ImageSize = imageSize;
  r.XWorldLimits = xWorldLimits;
  r.YWorldLimits = yWorldLimits;
  r.ZWorldLimits = zWorldLimits;
  r.PixelExtentInWorldX = pixelExtentInWorldX;
  r.PixelExtentInWorldY = pixelExtentInWorldY;
  r.PixelExtentInWorldZ = pixelExtentInWorldZ;
  r.ImageExtentInWorldX = imageExtentInWorldX;
  r.ImageExtentInWorldY = imageExtentInWorldY;
  r.ImageExtentInWorldZ = imageExtentInWorldZ;
  r.XIntrinsicLimits = xIntrinsicLimits;
  r.YIntrinsicLimits = yIntrinsicLimits;
  r.ZIntrinsicLimits = zIntrinsicLimits;
  
  ## in MATLAB imref3d isa imref2d
  r = class (r, "imref3d", @imref2d());
endfunction

%!error id=Octave:invalid-fun-call imref3d (1, 2, 3, 4, 5)
%!error id=Octave:incorrect-size imref3d (42)
%!error id=Octave:incorrect-size imref3d ([42])
%!error id=Octave:incorrect-size imref3d ([4, 2])
%!error id=Octave:incorrect-size imref3d ([4, 2, 3, 3])
%!error id=Octave:expected-integer imref3d ([4.2, 42])
%!error id=Octave:expected-positive imref3d ([0, 0])
%!error id=Octave:expected-positive imref3d ([-4, 2])
%!error id=Octave:expected-positive imref3d ([4, 2, 3], 0, 1, 2)
%!error id=Octave:expected-positive imref3d ([4, 2, 3], 1, 0, 2)
%!error id=Octave:expected-positive imref3d ([4, 2, 3], 1, 2, 0)
%!error id=Octave:expected-real imref3d ([4, 2, 3], j, 1, 2)
%!error id=Octave:expected-real imref3d ([4, 2, 3], 1, j, 2)
%!error id=Octave:expected-real imref3d ([4, 2, 3], 1, 2, j)
%!error id=Octave:expected-real imref3d ([4, 2, 3], [j, 2], [3, 4], [5, 6])
%!error id=Octave:expected-real imref3d ([4, 2, 3], [1, 2], [j, 4], [5, 6])
%!error id=Octave:expected-real imref3d ([4, 2, 3], [1, 2], [3, 4], [5, j])
%!error id=Octave:expected-vector imref3d ([4, 2, 3], [], [], [])
%!error id=Octave:expected-vector imref3d ([4, 2, 3], [], [1], [2])
%!error id=Octave:expected-scalar imref3d ([4, 2, 3], [1], [], [])
%!error id=Octave:incorrect-size imref3d ([4, 2, 3], [1, 2], [3, 4], [0])
%!error id=Octave:incorrect-size imref3d ([4, 2, 3], [1, 2], [3, 4, 5], [6, 7])
%!error id=Octave:incorrect-size imref3d ([4, 2, 3], [1, 2], [3, 4], [5, 6, 7])
%!error id=Octave:incorrect-size imref3d ([4, 2, 3], [1; 2], [3, 4], [5, 6])
%!error id=Octave:incorrect-size imref3d ([4, 2, 3], [1, 2], [3; 4], [5, 6])
%!error id=Octave:incorrect-size imref3d ([4, 2, 3], [1, 2], [3, 4], [5; 6])
%!error id=Octave:invalid-indexing imref3d().InvalidProperty
%!error id=Octave:expected-increasing imref3d ([100, 200, 3], [1.5 0.5], [2.5, 3.5], [0.5, 1.5])
%!error id=Octave:expected-increasing imref3d ([100, 200, 3], [1.5 2.5], [2.5, 1.5], [0.5, 1.5])
%!error id=Octave:expected-increasing imref3d ([100, 200, 3], [1.5 2.5], [2.5, 3.5], [1.5, 0.5])

%!assert (imref3d ([4, 2, 3]).ImageSize, [4, 2, 3])

%!test
%! r = imref3d;
%! assert (r.XWorldLimits, [0.5, 2.5])
%! assert (r.YWorldLimits, [0.5, 2.5])
%! assert (r.ZWorldLimits, [0.5, 2.5])
%! assert (r.ImageSize, [2, 2, 2])
%! assert (r.PixelExtentInWorldX, 1)
%! assert (r.PixelExtentInWorldY, 1)
%! assert (r.PixelExtentInWorldZ, 1)
%! assert (r.ImageExtentInWorldX, 2)
%! assert (r.ImageExtentInWorldY, 2)
%! assert (r.ImageExtentInWorldZ, 2)
%! assert (r.XIntrinsicLimits, [0.5, 2.5])
%! assert (r.YIntrinsicLimits, [0.5, 2.5])
%! assert (r.ZIntrinsicLimits, [0.5, 2.5])

%!test
%! r = imref3d ([128, 128, 27]);
%! assert (r.XWorldLimits, [0.5, 128.5])
%! assert (r.YWorldLimits, [0.5, 128.5])
%! assert (r.ZWorldLimits, [0.5, 27.5])
%! assert (r.ImageSize, [128, 128, 27])
%! assert (r.PixelExtentInWorldX, 1)
%! assert (r.PixelExtentInWorldY, 1)
%! assert (r.PixelExtentInWorldZ, 1)
%! assert (r.ImageExtentInWorldX, 128)
%! assert (r.ImageExtentInWorldY, 128)
%! assert (r.ImageExtentInWorldZ, 27)
%! assert (r.XIntrinsicLimits, [0.5, 128.5])
%! assert (r.YIntrinsicLimits, [0.5, 128.5])
%! assert (r.ZIntrinsicLimits, [0.5, 27.5])

%!test
%! r = imref3d ([128, 128, 27], 2, 2, 4);
%! assert (r.XWorldLimits, [1, 257])
%! assert (r.YWorldLimits, [1, 257])
%! assert (r.ZWorldLimits, [2, 110])
%! assert (r.ImageSize, [128, 128, 27])
%! assert (r.PixelExtentInWorldX, 2)
%! assert (r.PixelExtentInWorldY, 2)
%! assert (r.PixelExtentInWorldZ, 4)
%! assert (r.ImageExtentInWorldX, 256)
%! assert (r.ImageExtentInWorldY, 256)
%! assert (r.ImageExtentInWorldZ, 108)
%! assert (r.XIntrinsicLimits, [0.5, 128.5])
%! assert (r.YIntrinsicLimits, [0.5, 128.5])
%! assert (r.ZIntrinsicLimits, [0.5, 27.5])

## changing ImageSize
%!test
%! r = imref3d;
%! assert (r.XWorldLimits, [0.5, 2.5])
%! assert (r.YWorldLimits, [0.5, 2.5])
%! assert (r.ZWorldLimits, [0.5, 2.5])
%! assert (r.ImageSize, [2, 2, 2])
%! assert (r.PixelExtentInWorldX, 1)
%! assert (r.PixelExtentInWorldY, 1)
%! assert (r.PixelExtentInWorldZ, 1)
%! assert (r.ImageExtentInWorldX, 2)
%! assert (r.ImageExtentInWorldY, 2)
%! assert (r.ImageExtentInWorldZ, 2)
%! assert (r.XIntrinsicLimits, [0.5, 2.5])
%! assert (r.YIntrinsicLimits, [0.5, 2.5])
%! assert (r.ZIntrinsicLimits, [0.5, 2.5])
%! r.ImageSize = [128, 128, 27];
%! assert (r.XWorldLimits, [0.5, 2.5])
%! assert (r.YWorldLimits, [0.5, 2.5])
%! assert (r.ZWorldLimits, [0.5, 2.5])
%! assert (r.ImageSize, [128, 128, 27])
%! assert (r.PixelExtentInWorldX, 0.015625, 1e-6)
%! assert (r.PixelExtentInWorldY, 0.015625, 1e-6)
%! assert (r.PixelExtentInWorldZ, 0.074074, 1e-6)
%! assert (r.ImageExtentInWorldX, 2)
%! assert (r.ImageExtentInWorldY, 2)
%! assert (r.ImageExtentInWorldZ, 2)
%! assert (r.XIntrinsicLimits, [0.5, 128.5])
%! assert (r.YIntrinsicLimits, [0.5, 128.5])
%! assert (r.ZIntrinsicLimits, [0.5, 27.5])

## changing XWorldLimits, YWorldLimits and ZWorldLimits
%!test
%! r = imref3d;
%! assert (r.XWorldLimits, [0.5, 2.5])
%! assert (r.YWorldLimits, [0.5, 2.5])
%! assert (r.ZWorldLimits, [0.5, 2.5])
%! assert (r.ImageSize, [2, 2, 2])
%! assert (r.PixelExtentInWorldX, 1)
%! assert (r.PixelExtentInWorldY, 1)
%! assert (r.PixelExtentInWorldZ, 1)
%! assert (r.ImageExtentInWorldX, 2)
%! assert (r.ImageExtentInWorldY, 2)
%! assert (r.ImageExtentInWorldZ, 2)
%! assert (r.XIntrinsicLimits, [0.5, 2.5])
%! assert (r.YIntrinsicLimits, [0.5, 2.5])
%! assert (r.ZIntrinsicLimits, [0.5, 2.5])
%! r.XWorldLimits = [-60, 13.33];
%! r.YWorldLimits = [-900.8, -560.26];
%! r.ZWorldLimits = [-302.48, 1500.333];
%! assert (r.XWorldLimits, [-60, 13.33])
%! assert (r.YWorldLimits, [-900.8, -560.26])
%! assert (r.ZWorldLimits, [-302.48, 1500.333])
%! assert (r.ImageSize, [2, 2, 2])
%! assert (r.PixelExtentInWorldX, 36.6650)
%! assert (r.PixelExtentInWorldY, 170.27, 1e-5)
%! assert (r.PixelExtentInWorldZ, 901.4065)
%! assert (r.ImageExtentInWorldX, 73.33, 1e-5)
%! assert (r.ImageExtentInWorldY, 340.54, 1e-5)
%! assert (r.ImageExtentInWorldZ, 1802.813, 1e-5)
%! assert (r.XIntrinsicLimits, [0.5, 2.5])
%! assert (r.YIntrinsicLimits, [0.5, 2.5])
%! assert (r.ZIntrinsicLimits, [0.5, 2.5])

%!test
%! r = imref3d;
%! fail ("r.XWorldLimits = []", "")
%! fail ("r.XWorldLimits = [1]", "")
%! fail ("r.XWorldLimits = [j]", "")
%! fail ("r.XWorldLimits = [1; 2]", "")
%! fail ("r.YWorldLimits = []", "")
%! fail ("r.YWorldLimits = [1]", "")
%! fail ("r.YWorldLimits = [j]", "")
%! fail ("r.YWorldLimits = [1; 2]", "")
%! fail ("r.ZWorldLimits = []", "")
%! fail ("r.ZWorldLimits = [1]", "")
%! fail ("r.ZWorldLimits = [j]", "")
%! fail ("r.ZWorldLimits = [1; 2]", "")