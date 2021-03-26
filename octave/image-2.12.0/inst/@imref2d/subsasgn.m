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
## @deftypefn {} {@var{rout} =} subsasgn (@var{r}, @var{index}, @var{val})
##
## @seealso{}
## @end deftypefn

function rout = subsasgn (r, index, val)
  switch (index.type)
    case "."
      fld = index.subs;
      switch (fld)
        case "ImageSize"
          imageSize = val;

          if (length (imageSize) < 2)
            error ("Octave:invalid-input-arg", ...
            "ImageSize must have at least two elements");
          endif

          validateattributes (imageSize, {"numeric"}, ...
          {"positive", "integer", "vector"}, "imref2d", "imageSize");
          
          m = imageSize(1);
          n = imageSize(2);
          
          rout = r;
          rout.ImageSize = imageSize;
          rout.PixelExtentInWorldX = r.ImageExtentInWorldX / n;
          rout.PixelExtentInWorldY = r.ImageExtentInWorldY / m;
          rout.XIntrinsicLimits = [0.5, n + 0.5];
          rout.YIntrinsicLimits = [0.5, m + 0.5];
        case "XWorldLimits"
          xWorldLimits = val;
          
          validateattributes (xWorldLimits, {"numeric"}, ...
          {"increasing", "real", "vector", "size", [1, 2]}, ...
          "imref2d", "xWorldLimits");
          
          imageSize = r.ImageSize;
          imageExtentInWorldX = xWorldLimits(2) - xWorldLimits(1);
          
          rout = r;
          rout.XWorldLimits = val;
          rout.ImageExtentInWorldX = imageExtentInWorldX;
          rout.PixelExtentInWorldX = imageExtentInWorldX / imageSize(2);
        case "YWorldLimits"
          yWorldLimits = val;
          
          validateattributes (yWorldLimits, {"numeric"}, ...
          {"increasing", "real", "vector", "size", [1, 2]}, ...
          "imref2d", "yWorldLimits");
          
          imageSize = r.ImageSize;
          imageExtentInWorldY = yWorldLimits(2) - yWorldLimits(1);
          
          rout = r;
          rout.YWorldLimits = val;
          rout.ImageExtentInWorldY = imageExtentInWorldY;
          rout.PixelExtentInWorldY = imageExtentInWorldY / imageSize(1);
        otherwise
          error ("Octave:invalid-indexing", ...
          "@imref2d/subsasgn: invalid property '%s'", fld);
      endswitch

    otherwise
      error ("Octave:invalid-indexing", "@imref2d/subsasgn: invalid index type")
  endswitch
endfunction