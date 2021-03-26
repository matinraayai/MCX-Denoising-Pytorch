## Copyright (C) 2019 David Miguel Susano Pinto <carandraug@octave.org>
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
## along with this program.  If not, see <https://www.gnu.org/licenses/>.

## -*- texinfo -*-
## @deftypefn {Loadable Function} {@dots{}} nonmax_supress (@dots{})
## Incorrect spelling of nonmax_suppress function.
##
## This function was originally released with the incorrect spelling.
## This function with the incorrect name is kept for backwards
## compatibility reasons.
##
## @seealso{nonmax_suppress}
## @end deftypefn

function [varargout] = nonmax_supress (varargin)
  [varargout{1:nargout}] = nonmax_suppress (varargin{:});
endfunction
