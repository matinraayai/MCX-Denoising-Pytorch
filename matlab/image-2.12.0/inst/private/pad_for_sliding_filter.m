## Copyright (C) 2008 Søren Hauberg <soren@hauberg.org>
## Copyright (C) 2012 Carnë Draug <carandraug+dev@gmail.com>
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

## This private function is to be common to all functions that call
## __spatial_filtering__. It may be better to have __spatial_filtering__ do this
## part as well then

function im = pad_for_sliding_filter (im, window_size, padval)

  if (ndims (im) != numel (window_size))
    error ("image and domain must have the same number of dimensions")
  endif

  pad  = floor (window_size / 2);
  im   = padarray (im, pad, padval);
  even = ! mod (window_size, 2);

  ## if one of the domain dimensions is even, its origin is must be
  ##     floor ([size(domain)/2] + 1)
  ## so after padarray, we need to remove the 1st element (1st row or column for
  ## dimension 1 and 2) from the dimensions where the domain is even
  if (any (even))
    idx = cell (1, ndims (im));
    for k = 1:ndims(im)
      idx{k} = (even(k)+1):size(im, k);
    endfor
    im = im(idx{:});
  endif

endfunction
