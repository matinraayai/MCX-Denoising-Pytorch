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
## @deftypefn {Function File} {} imopen (@var{img}, @var{SE})
## Perform morphological opening.
##
## The matrix @var{img} must be numeric while @var{SE} can be a:
## @itemize @bullet
## @item
## strel object;
## @item
## array of strel objects as returned by `@@strel/getsequence';
## @item
## matrix of 0's and 1's.
## @end itemize
##
## The opening corresponds to an erosion followed by a dilation of @var{img},
## using the same @var{SE}, i.e., it is equivalent to:
## @example
## imdilate (imerode (img, se), se);
## @end example
##
## @seealso{imdilate, imerode, imclose}
## @end deftypefn

function opened = imopen (img, se)

  if (nargin != 2)
    print_usage ();
  elseif (! isimage (img))
    error("imopen: IMG must be a numeric matrix");
  endif
  se = prepare_strel ("imopen", se);

  ## Perform filtering
  opened = imdilate (imerode (img, se), se);

endfunction

%!shared in, out
%! in =  [ 0   0   0   1   1   1   0   0   1   1
%!         0   1   0   1   1   1   0   0   0   1
%!         1   1   1   1   1   0   0   0   0   0
%!         0   1   1   1   1   0   0   0   0   0
%!         0   0   0   1   0   0   0   0   1   0
%!         0   0   0   0   0   0   0   1   1   1
%!         0   0   0   0   1   0   1   0   1   0
%!         0   0   0   1   1   1   1   1   0   0
%!         0   0   0   0   1   1   1   0   0   0
%!         0   0   0   1   1   1   0   0   0   0];
%!
%! out = [ 0   0   0   1   1   1   0   0   0   0
%!         0   0   0   1   1   1   0   0   0   0
%!         0   0   0   0   0   0   0   0   0   0
%!         0   0   0   0   0   0   0   0   0   0
%!         0   0   0   0   0   0   0   0   0   0
%!         0   0   0   0   0   0   0   0   0   0
%!         0   0   0   0   0   0   0   0   0   0
%!         0   0   0   0   0   0   0   0   0   0
%!         0   0   0   0   0   0   0   0   0   0
%!         0   0   0   0   0   0   0   0   0   0];
%!assert (imopen (logical (in), ones (3)), logical (out));
%!
%! out = [80   80    1    8   15   51   51   51   51   40
%!        80   80    7    8   15   54   55   55   55   40
%!         4    7    7    8   15   54   55   55   55   40
%!        17   17   17    7    3   54   55   55   55   28
%!        17   17   17    2    9   54   54   54   52   33
%!        17   17   17   29   29   29   29   26   33   33
%!         5    5   13   29   29   29   30   32   39   39
%!         6    6   13   29   29   29   30   32   39   39
%!        10   12   77   77   77   35   35   35   39   39
%!        10   12   77   77   77   35   35   35   27   27];
%!assert (imopen (magic (10), ones (3)), out);
%!assert (imopen (uint8 (magic (10)), strel ("square", 3)), uint8 (out));
%!
%! ## using a se that will be decomposed in 2 pieces
%! out =[ 1    1    1    8   15   40   40   40   40   40
%!        4    4    4    8   15   40   40   40   40   40
%!        4    4    4    8   15   40   40   40   40   40
%!        5    5    5    3    3   28   28   28   28   28
%!        5    5    5    2    9   28   28   28   28   28
%!        5    5   13   26   26   26   26   26   26   26
%!        5    5   13   29   29   29   29   29   27   27
%!        6    6   13   29   29   29   29   29   27   27
%!        6    6   13   29   29   29   29   29   27   27
%!        6    6   13   29   29   29   29   29   27   27];
%!assert (imopen (magic (10), ones(5)), out);
%!
%! ## using a weird non-symmetric and even-size se
%! out =[ 7    7    1    8   15   55   51   51   41   40
%!        7    7    7    8   16   55   55   55   51   41
%!        4    9    7    7   16   54   55   54   55   47
%!       25   25    9    9    3   52   54   52   54   28
%!       25   24   25    2    9   33   52   34   52   34
%!       17   24   29   31   29   30   33   26   33   34
%!       17    5   29   31   31   31   30   32   39   33
%!       10    6   13   35   35   29   31   32   45   39
%!       10   12   77   36   36   35   35   31   45   45
%!       11   12   77   77   77   36   36   35   27   45];
%!assert (imopen (magic (10), [1 0 0 0; 1 1 1 0; 0 1 0 1]), out);
