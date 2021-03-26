## Copyright (C) 2000 Kai Habel <kai.habel@gmx.de>
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
## @deftypefn {Function File} @var{r} = corr2 (@var{I},@var{J})
## Compute correlation coefficients of images.
##
## The two images @var{I} and @var{J} must be real type matrices or vectors of
## same size.
## @seealso{corr, cov, std2}
## @end deftypefn

function r = corr2 (I, J)

  if (nargin != 2)
    print_usage ();
  elseif (! isimage (I) || ! isimage (J))
    error ("corr2: I and J must be real matrices");
  elseif (! size_equal (I, J))
    error ("corr2: I and J must be of same size");
  endif
  r = corr (I(:), J(:));

endfunction
