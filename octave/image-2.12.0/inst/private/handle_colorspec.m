## Copyright (C) 2013 Carnë Draug <carandraug+dev@gmail.com>
##
## This program is free software; you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program; if not, see <http://www.gnu.org/licenses/>.

## There's 3 ways to specify colors:
##    RGB triplet values
##    short names
##    long names
##
## And probably more undocumented ways to do it.

function rgb = handle_colorspec (func, spec)
  if (iscolormap (spec))
    rgb = spec;
  elseif (ischar (spec))
    switch (tolower (spec))
      case {"b", "blue"   }, rgb = [0 0 1];
      case {"c", "cyan"   }, rgb = [0 1 1];
      case {"g", "green"  }, rgb = [0 1 0];
      case {"k", "black"  }, rgb = [0 0 0];
      case {"m", "magenta"}, rgb = [1 0 1];
      case {"r", "red"    }, rgb = [1 0 0];
      case {"w", "white"  }, rgb = [1 1 1];
      case {"y", "yellow" }, rgb = [1 1 0];
      otherwise
        error("%s: unknown color '%s'", func, spec);
    endswitch
  else
    error ("%s: invalid color specification");
  endif
endfunction
