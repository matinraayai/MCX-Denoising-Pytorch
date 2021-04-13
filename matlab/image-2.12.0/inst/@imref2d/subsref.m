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
## @deftypefn {} {@var{r} =} subref (@var{val}, @var{idx})
##
## @seealso{}
## @end deftypefn

function r = subsref (val, idx)
  if (strcmp (idx(1).type, "."))
    switch (idx(1).subs)
      case "ImageSize"
        r = val.ImageSize;
      case "XWorldLimits"
        r = val.XWorldLimits;
      case "YWorldLimits"
        r = val.YWorldLimits;
      case "PixelExtentInWorldX"
        r = val.PixelExtentInWorldX;
      case "PixelExtentInWorldY"
        r = val.PixelExtentInWorldY;
      case "ImageExtentInWorldX"
        r = val.ImageExtentInWorldX;
      case "ImageExtentInWorldY"
        r = val.ImageExtentInWorldY;
      case "XIntrinsicLimits"
        r = val.XIntrinsicLimits;
      case "YIntrinsicLimits"
        r = val.YIntrinsicLimits;
      otherwise
        error ("Octave:invalid-indexing", ...
        strcat ("unknown property '", idx(1).subs, "' for class imref2d"));
    endswitch
    if (length (idx) > 1)
      switch (idx(2).type)
        case "()"
          i = idx(2).subs;
          r = r(i{1});
        otherwise
          error ("Octave:invalid-indexing", ...
          strcat ("can't index '", idx(1).subs, "' with ", idx(2).type));
      endswitch
    endif
  endif
endfunction
