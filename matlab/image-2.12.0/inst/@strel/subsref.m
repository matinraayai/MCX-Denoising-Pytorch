## Copyright (C) 2012 Carnë Draug <carandraug+dev@gmail.com>
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

function req = subsref (se, idx)

  if (! isa (se, "strel"))
    error ("object must be of the strel class but '%s' was used", class (se));
  elseif (! strcmp (idx(1).type, "()") || numel (idx) > 1)
    error ("incorrect syntax to reference strel object. Use obj(idx) to access elements.");
  endif

  if (isempty (se.seq))
    ## for matlab compatibility, if we don't have the sequence, then this is
    ## still an array of strel objects, it just happens we have we give
    ## back the object as longsame as 1 element sequence
    ## so we give the object back. This means the following:
    ##     se(1)        <-- works fine
    ##     se(2)        <-- fails
    ##     se(1,1,1,1)  <-- works fine
    ##     se(1:3)      <-- fails
    ## Yes, this is ugly but we don't have classdef yet...
    if (any (cellfun (@(x) any (x != 1), idx(1).subs)))
      error ("A(I): index out of bounds. This is not a sequence of strel object");
    endif
    req = se;
  else
    req = se.seq{idx(1).subs{:}};
  endif

endfunction
