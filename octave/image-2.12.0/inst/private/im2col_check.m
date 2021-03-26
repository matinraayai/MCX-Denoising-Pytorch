## Copyright (C) 2013 Carnë Draug <carandraug@octave.org>
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

## A private function originally written for im2col but that can also
## be used by colfilt and nlfilt, since the syntax is very similar.

function [p, block_size, padval] = im2col_check (func, nargin, A, varargin)

  if (nargin < 2)
    print_usage (func);
  elseif (! isnumeric (A) && ! islogical (A))
    error ("%s: A must be a numeric of logical matrix", func);
  endif
  p = 1;  # varargin param being processsed

  padval = 0;

  ## Check for 'indexed' presence
  if (ischar (varargin{p}) && strcmpi (varargin{p}, "indexed"))
    if (nargin < 3)
      print_usage (func);
    endif
    ## We pad with value of 0 for indexed images of uint8 or uint16 class.
    ## Indexed images of signed integer, or above uint16, are treated the
    ## same as floating point (a value of 1 is index 1 on the colormap)
    if (any (isa (A, {"uint8", "uint16"})))
      padval = 0;
    else
      padval = 1;
    endif
    p++;
  endif

  ## check [m,n]
  block_size = varargin{p++};
  if (! isnumeric (block_size) || ! isvector (block_size) ||
      any (block_size(:) < 1))
    error ("%s: BLOCK_SIZE must be a vector of positive elements.", func);
  endif
  block_size(end+1:ndims(A)) = 1; # expand singleton dimensions if required
  block_size = block_size(:).';   # make sure it's a row vector

endfunction
