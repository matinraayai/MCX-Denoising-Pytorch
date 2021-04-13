## Copyright (C) 2013 Carnë Draug <carandraug+dev@gmail.com>
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
## @deftypefn {Function File} {[@var{offsets}, @var{heights}] =} getneighbors (@var{se})
## Get neighbors relative position and height.
##
## For each of the neighbors in the strel object @var{se}, @var{offsets} are
## their positions relative to the origin (center).  It is a @var{P}x@var{N}
## matrix, with @var{P} as the number of neighbors, and @var{N} as the number
## of dimensions.
##
## @var{heights} is a vector with the height of each neighbor in @var{se}.
##
## @seealso{getnhood, getheight, strel}
## @end deftypefn

function [offsets, heights] = getneighbors (se)

  se_ndims  = ndims (se.nhood);
  se_size   = size (se.nhood);

  ## To calculate the offsets we get the subscript indexes of all elements
  ## in a cell array, one cell for each dimension and the that dimension
  ## indexes in a column. We then just subtract the center position of each
  ## dimension to the respective column.
  sub = cell (1, se_ndims);
  ## The (:) on find(...)(:) is because find() will return a row (instead
  ## of a column), if the input is just a row vector. We need to make sure
  ## that we always get a column for ind2sub().
  [sub{1:se_ndims}] = ind2sub (se_size, find (se.nhood)(:));
  offsets = cell2mat (sub) - floor ((se_size + 1) / 2);

  heights = getheight (se)(se.nhood);

endfunction
