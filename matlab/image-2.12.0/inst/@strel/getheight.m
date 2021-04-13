## Copyright (C) 2012 Roberto Metere <roberto@metere.it>
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
## @deftypefn {Function File} {@var{H} =} getheight (@var{SE})
## Return the heights of a non-flat structuring element.
## If SE is flat, then all heights are zeros.
##
## @seealso{getnhood, strel}
## @end deftypefn

function H = getheight (SE)

  if (SE.flat)
    H = zeros (size (SE.nhood));
  else
    H = SE.height;
  endif

endfunction
