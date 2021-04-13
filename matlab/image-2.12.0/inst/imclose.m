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
## @deftypefn {Function File} {} imclose (@var{img}, @var{SE})
## Perform morphological closing.
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
## The closing corresponds to a dilation followed by an erosion of @var{img},
## using the same @var{SE}, i.e., it is equivalent to:
## @example
## imerode (imdilate (img, se), se);
## @end example
##
## @seealso{imdilate, imerode, imopen}
## @end deftypefn

function closed = imclose (img, se)

  if (nargin != 2)
    print_usage ();
  elseif (! isimage (img))
    error("imclose: IMG must be a numeric matrix");
  endif
  se = prepare_strel ("imclose", se);

  ## Perform filtering
  closed = imerode (imdilate (img, se), se);

endfunction

%!shared in, out
%! in =  [ 0   0   0   1   1   1   0   0   1   1
%!       0   1   0   1   1   1   0   0   0   1
%!       1   1   1   1   1   0   0   0   0   0
%!       0   1   1   1   1   0   0   0   0   0
%!       0   0   0   1   0   0   0   0   1   0
%!       0   0   0   0   0   0   0   1   1   1
%!       0   0   0   0   1   0   1   0   1   0
%!       0   0   0   1   1   1   1   1   0   0
%!       0   0   0   0   1   1   1   0   0   0
%!       0   0   0   1   1   1   0   0   0   0];
%!
%! out = [ 1   1   1   1   1   1   1   1   1   1
%!       1   1   1   1   1   1   0   0   0   1
%!       1   1   1   1   1   0   0   0   0   1
%!       1   1   1   1   1   0   0   0   0   1
%!       0   0   0   1   1   0   0   0   1   1
%!       0   0   0   1   1   1   1   1   1   1
%!       0   0   0   1   1   1   1   1   1   1
%!       0   0   0   1   1   1   1   1   0   0
%!       0   0   0   1   1   1   1   0   0   0
%!       0   0   0   1   1   1   1   0   0   0];
%!assert (imclose (logical (in), ones (3)), logical (out));
%!
%! out = [99   99   16   16   16   73   74   64   64   64
%!      98   88   16   16   16   73   71   64   64   64
%!      93   88   88   61   61   61   68   70   70   70
%!      93   88   88   61   61   61   68   71   71   71
%!      93   93   88   61   61   61   68   75   66   66
%!      79   79   82   90   90   49   49   49   49   66
%!      79   79   82   91   91   48   46   46   46   66
%!      79   79   82   95   97   48   46   46   46   72
%!      18   18   94   96   84   48   46   46   46   59
%!      18   18  100   96   84   50   50   50   50   59];
%!assert (imclose (magic (10), ones (3)), out);
%!assert (imclose (uint8 (magic (10)), strel ("square", 3)), uint8 (out));
%!
%! ## using a se that will be decomposed in 2 pieces
%! out =[ 99   99   88   74   74   74   74   70   70   70
%!        98   93   88   74   74   74   74   70   70   70
%!        93   93   88   74   74   74   74   70   70   70
%!        93   93   88   74   74   74   74   71   71   71
%!        93   93   88   75   75   75   75   75   75   75
%!        93   93   90   90   90   72   72   72   72   72
%!        93   93   91   91   91   72   72   72   72   72
%!        93   93   93   95   97   72   72   72   72   72
%!        94   94   94   96   97   72   72   72   72   72
%!       100  100  100   97   97   72   72   72   72   72];
%!assert (imclose (magic (10), ones(5)), out);
%!
%! ## using a weird non-symmetric and even-size se
%! out =[ 92   99   16   16   16   70   74   58   58   58
%!        98   88   60   73   16   73   69   70   64   58
%!        88   81   88   60   60   60   69   69   70   70
%!        87   87   61   68   61   60   68   69   71   69
%!        86   93   87   61   61   61   68   75   68   69
%!        23   82   89   89   90   45   68   45   68   66
%!        23   23   82   89   91   48   45   45   45   66
%!        79   23   82   95   97   46   48   46   45   72
%!        18   79   94   96   78   50   46   46   46   59
%!        18   18  100   94   94   78   50   50   46   59];
%!assert (imclose (magic (10), [1 0 0 0; 1 1 1 0; 0 1 0 1]), out);
