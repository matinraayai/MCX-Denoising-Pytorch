## Copyright (C) 2015 Carnë Draug <carandraug+dev@gmail.com>
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

## Private function for functions that convert between color spaces, i.e.,
## rgb2xyz, xyz2rgb, xyz2lab, lab2xyz, rgb2lab and lab2rgb.  All of these
## functions need to handle input in the same way.  The returned flags
## are meant to be handled by the complementary private function
## colorspace_conversion_revert()
## Setting the only_foats flag to 1 tells this function to only accept single
## and double input types, otherwise it will also accept uint8 and uint16
## input values.
##
## adapted by: Hartmut Gimpel, 2015

function [in_arg, cls, sz, is_im, is_nd, is_int] ...
            = colorspace_conversion_input_check (func, arg_name, in_arg, only_floats)

  cls = class (in_arg);
  sz = size (in_arg);

  ## If we have an image convert it into a color map.
  if (! iscolormap (in_arg))
    if (! any (strcmp (cls, {"uint8", "int8", "uint16", "single", "double"})))
      error ("%s: %s of invalid data type '%s'", func, arg_name, cls);
    elseif (only_floats && ! any (strcmp (cls, {"single", "double"})))
      error ("%s: %s of invalid data type '%s'", func, arg_name, cls);
    elseif (size (in_arg, 3) != 3 && !(size (in_arg) == [1, 3] || size (in_arg) == [3, 1]))
      error ("%s: %s must be a colormap or %s image", func, arg_name, arg_name);
    elseif (! isreal (in_arg) || ! isnumeric (in_arg))
      error ("%s: %s must be numeric and real", func, arg_name);
    endif
    is_im = true;

    ## Some floating point values (like R, G, B values) should be in the [0 1] range,
    ## otherwise they don't make any sense. We accept those values
    ## anyways because we must return something for Matlab compatibility.
    ## User case is when a function returns an RGB image just slightly outside
    ## the range due to floating point rounding errors.

    ## Allow for ND images, i.e., multiple images on the 4th dimension.
    nd = ndims (in_arg);
    if (nd == 2 || nd == 3)
      is_nd = false;
    elseif (nd == 4)
      is_nd = true;
      in_arg = permute (in_arg, [1 2 4 3]);
    elseif (nd > 4)
      error ("%s: invalid %s with more than 4 dimensions", func, arg_name);
    endif
    in_arg = reshape (in_arg, [numel(in_arg)/3 3]);
  else
    is_im = false;
    is_nd = false;
  endif

  ## Convert to floating point (remember to leave class single alone)
  if (isinteger (in_arg))
    in_arg = double (in_arg) / double (intmax (cls));
    is_int = true;
  else
    is_int = false;
  endif

endfunction
