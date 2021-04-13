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

## This function will be common to imopen, imclose, imbothat, imtophat,
## and any other function that can take a strel object or a matrix of 0
## and 1. It checks the input and gets a strel object in the later case.

function se = prepare_strel (func, se)

  ## We could do a lot more of input checking but it'd be repeated again
  ## in imerode and imdilate. We just do the minimum for strel as well.
  ## Since imerode and imdilate will create a strel object out of the SE
  ## we pass them, we can create it now ourselves.
  if (! strcmpi (class (se), "strel"))
    if (! islogical (se) && (isnumeric (se) && any (se(:) != 1 & se(:) != 0)))
      error ("%s: SE must be a strel object or matrix of 0 and 1", func);
    endif
    se = strel ("arbitrary", se);
  endif

endfunction
