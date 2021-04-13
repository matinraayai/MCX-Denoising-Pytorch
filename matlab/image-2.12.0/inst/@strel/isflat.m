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
## @deftypefn {Function File} {@var{TF} =} isflat (@var{SE})
## Say if a structuring element object is flat or not.
##
## Although in accordance with me TF should be a logical value, it is typed
## double due to compatibility with MATLAB
## (Perhaps there is a good reason which I can't figure out now)
##
## @seealso{strel}
## @end deftypefn

## TODO:  if SE is an array of strel, this function return results for any
##        strel included.
function TF = isflat (SE)

  TF = double (SE.flat);

endfunction
