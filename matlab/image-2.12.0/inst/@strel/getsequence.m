## Copyright (C) 2012 Roberto Metere <roberto@metere.it>
## Copyright (C) 2013 Carnë Draug <carandraug@octave.org>
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
## @deftypefn {Function File} {@var{seq} =} getsequence (@var{se})
## Decompose structuring element.
##
## Returns a strel object @var{se} that can be indexed with @code{()} to obtain
## the decomposed structuring elements that can be used to "rebuild" @var{se}.
##
## Decomposing a structuring element may lead to faster operations by
## replacing a single operation with a large @var{se} (large nhood), with
## multiple operations with smaller @var{se}.
##
## Most functions will automatically perform SE decomposition provided a
## strel object is used (instead of a logical array).  This also requires
## that specific shapes are specified instead of @qcode{"arbitrary"}, since
## it may prevent SE decomposition.
##
## @seealso{imdilate, imerode, strel}
## @end deftypefn

function se = getsequence (se)

  if (isempty (se.seq))
    ## guess a square/cube/rectangle/hyperrectangle shape even if shape
    ## is arbitrary. There's no need to decompose when it's very small
    ## hence the no bother if numel > 15
    if (se.flat && all (se.nhood(:)) && ! isvector (se.nhood) &&
        numel (se.nhood) > 15)
      nd = ndims (se.nhood);
      se.seq = cell (nd, 1);
      for idx = 1:nd
        vec_size      = ones (1, nd);
        vec_size(idx) = size (se.nhood, idx);
        se.seq{idx}   = strel ("arbitrary", true (vec_size));
      endfor
    elseif (strcmp (se.shape, "octagon") && se.opt.apothem > 0)
      persistent octagon_template = get_octagon_template ();
      se.seq = repmat (octagon_template, se.opt.apothem /3, 1);
    else
        se.seq{1,1} = se;
    endif
  endif

endfunction

function template = get_octagon_template ()
  template = repmat ({false(3)}, 4, 1);
  template{1}(2,:)     = true;
  template{2}(:,2)     = true;
  template{3}([1 5 9]) = true;
  template{4}([3 5 7]) = true;
  template = cellfun (@(x) strel ("arbitrary", x), template, "UniformOutput", false);
endfunction
