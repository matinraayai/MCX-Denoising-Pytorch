## Copyright (C) 2016 Carnë Draug <carandraug@octave.org>
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

## Internal function that does all the work of lab2uint8, lab2uin16,
## and lab2double functions.

function [lab] = lab2cls (lab, out_cls)

  nd = ndims (lab);
  sz = size (lab);

  ## We want to minimize operations on the input but we also want to
  ## have the same code for colormap and ND images.  So we reshape into
  ## a colormap with other images on the 3rd dimension.  The actual
  ## conversion is broadcasted, and then we reshape back at the end.
  was_image = false;
  if (nd == 2 && sz(2) == 3) # colormap shape
    ## Do nothing, we already have it shaped like we want.
  elseif (nd == 3 && sz(3) == 3) # MxNx3 image
    was_image = true;
    lab = reshape (lab, [sz(1)*sz(2) 3]);
  elseif (nd == 4 && sz(3) == 3) # MxNx3xK image
    was_image = true;
    lab = reshape (lab, [sz(1)*sz(2) 3 sz(4)]);
  else
    error ("lab2%s: LAB must be Mx3, MxNx3, or MxNx3xK size", out_cls);
  endif

  in_cls = class (lab);
  switch (out_cls)
    case {"double", "single"}
      lab = cast (lab, out_cls);
      switch (in_cls)
        case "uint8"
          lab(:, 1, :) .*= (100 / 255);
          lab(:, [2 3], :) .-= 128;
        case "uint16"
          lab(:, 1, :) .*= (100 / 65280);
          lab(:, [2 3], :) .*= (255 / 65280);
          lab(:, [2 3], :) .-= 128;
        case {"double", "single"}
          ## Do nothing, we already casted to the other type.
        otherwise
          error ("lab2%s: invalid class '%s' for LAB", out_cls, in_cls);
      endswitch

    case "uint8"
      switch (in_cls)
        case {"double", "single"}
          lab(:, 1, :) .*= (255 / 100);
          lab(:, [2 3], :) .+= 128;
          lab(isnan (lab)) = 255; # for Matlab compatibility
          lab = uint8 (lab);
        case "uint16"
          lab /= 256;
          lab = uint8 (lab);
        case "uint8"
          ## Do nothing.
        otherwise
          error ("lab2uint8: invalid class '%s' for LAB", in_cls);
      endswitch

    case "uint16"
      switch (in_cls)
        case {"double", "single"}
          lab(:, 1, :) .*= (65280 / 100);
          lab(:, [2 3], :) .+= 128;
          lab(:, [2 3], :) .*= (65280 / 255);
          lab(isnan (lab)) = 65535; # for Matlab compatibility
          lab = uint16 (lab);
        case "uint8"
          lab = uint16 (lab) * 256;
        case "uint16"
          ## Do nothing.
        otherwise
          error ("lab2uint16: invalid class '%s' for LAB", in_cls);
      endswitch

    otherwise
      error ("lab2%s: non-supported conversion (internal error)", out_cls);
  endswitch

  if (was_image)
    lab = reshape (lab, sz);
  endif

endfunction
