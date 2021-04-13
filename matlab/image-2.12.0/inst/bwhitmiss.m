## Copyright (C) 2008 Søren Hauberg <soren@hauberg.org>
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
## @deftypefn  {Function File} {} bwhitmiss (@var{bw}, @var{se1}, @var{se2})
## @deftypefnx {Function File} {} bwhitmiss (@var{bw}, @var{interval})
## Perform binary hit-or-miss transform.
##
## This transform returns the set of positions, where the structuring
## element @var{se1} fits in the foregrond of @var{bw}, while the
## structuring element @var{se2} misses it completely.  It is equivalent
## to:
##
## @example
## imerode (@var{bw}, @var{se1}) & imerode (! @var{bw}, @var{se2})
## @end example
##
## For example, the following will remove every pixel with adjacent
## horizontal foreground pixels:
##
## @example
## >> bw = [ 0   1   0   1   1   0
##           0   1   0   1   1   0
##           0   1   0   1   1   0];
##
## >> bwhitmiss (bw, [1; 0; 1], [1 0 1])
##   @result{} ans =
##
##           0   1   0   0   0   0
##           0   1   0   0   0   0
##           0   1   0   0   0   0
## @end example
##
## Note that while @var{se1} and @var{se2} must have disjoint neighbourhoods
## for this transform to be meaningful, no error or warning is throw about
## it.
##
## Alternatively a single array @var{interval} can be defined, of values
## from @code{[1 0 -1]}.  In this case, the two structuring elements are
## extracted as:
##
## @example
## @var{se1} = (@var{interval} ==  1)
## @var{se2} = (@var{interval} == -1)
## @end example
##
## @seealso{bwmorph}
## @end deftypefn

function bw = bwhitmiss(im, varargin)
  ## Checkinput
  if (nargin != 2 && nargin != 3)
    print_usage();
  endif
  if (! isreal(im))
    error("bwhitmiss: first input argument must be a real matrix");
  endif

  ## Get structuring elements
  if (nargin == 2) # bwhitmiss (im, interval)
    interval = varargin{1};
    if (!isreal(interval))
      error("bwhitmiss: second input argument must be a real matrix");
    endif
    if (!all( (interval(:) == 1) | (interval(:) == 0) | (interval(:) == -1) ))
      error("bwhitmiss: second input argument can only contain the values -1, 0, and 1");
    endif
    se1 = (interval ==  1);
    se2 = (interval == -1);
  else # bwhitmiss (im, se1, se2)
    se1 = varargin{1};
    se2 = varargin{2};
    if (!all((se1(:) == 1) | (se1(:) == 0)) || !all((se2(:) == 1) | (se2(:) == 0)))
      error("bwhitmiss: structuring elements can only contain zeros and ones.");
    endif
  endif

  ## Perform filtering
  bw = imerode (im, se1) & imerode (! im, se2);

endfunction

%!test
%! bw1 = repmat ([0 1 0 1 1], [3 1]);
%! bw2 = repmat ([0 1 0 0 0], [3 1]);
%! assert (bwhitmiss (bw1, [1; 0; 1], [1 0 1]), logical (bw2))
%! assert (bwhitmiss (bw1, [0 1 0; -1 0 -1; 0 1 0]), logical (bw2))
