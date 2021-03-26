## Copyright (C) 2005 Carvalho-Mariel
## Copyright (C) 2010-2013 Carnë Draug <carandraug@octave.org>
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
## @deftypefn {Function File} {} imbothat (@var{img}, @var{SE})
## Perform morphological bottom hat filtering.
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
## A bottom hat transform corresponds to the difference between the closing
## of @var{img} and @var{img} itself, i.e., it is equivalent to:
## @example
## imclose (img, se) - img;
## @end example
##
## A bottom-hat transform is also known as 'black', or 'closing', top-hat
## transform.
##
## @seealso{imerode, imdilate, imopen, imclose, imtophat, mmgradm}
## @end deftypefn

function black = imbothat (img, se)

  if (nargin != 2)
    print_usage ();
  elseif (! isimage (img))
    error("imbothat: IMG must be a numeric matrix");
  endif
  se = prepare_strel ("imbothat", se);

  ## Perform filtering
  ## Note that in case that the transform is to applied to a logical image,
  ## subtraction must be handled in a different way (x & !y) instead of (x - y)
  ## or it will return a double precision matrix
  if (islogical (img))
    black = imclose (img, se) & ! img;
  else
    black = imclose (img, se) - img;
  endif

endfunction

%!assert (imbothat (ones (3), [1 1; 0 1]), zeros (3));
%!assert (imbothat (true (3), [1 1; 0 1]), false (3));

%!shared in, out, se
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
%! out = [ 1   1   1   0   0   0   1   1   0   0
%!         1   0   1   0   0   0   0   0   0   0
%!         0   0   0   0   0   0   0   0   0   1
%!         1   0   0   0   0   0   0   0   0   1
%!         0   0   0   0   1   0   0   0   0   1
%!         0   0   0   1   1   1   1   0   0   0
%!         0   0   0   1   0   1   0   1   0   1
%!         0   0   0   0   0   0   0   0   0   0
%!         0   0   0   1   0   0   0   0   0   0
%!         0   0   0   0   0   0   1   0   0   0];
%!assert (imbothat (logical (in), ones (3)), logical (out));
%!
%! out = [ 7    0   15    8    1    6    0   13    6   24
%!         0    8    9    2    0    0   16    7    0   23
%!        89    7    0   41   39    7   12    7    0   23
%!         8    1   69   40   58    1    6    2    0   43
%!         7    0   63   59   52    0    0    0   14   32
%!        62   55    6    7    0    7    0   23   16    1
%!        56   74    0    2    0    0   16   14    7    0
%!         0   73   69    0    0   19   15    8    1    0
%!         8    6    0    0    6   13    9    2    0    6
%!         7    0    0   19    0   14    7    0   23    0];
%!assert (imbothat (magic (10), ones (3)), out);
%!assert (imbothat (uint8 (magic (10)), strel ("square", 3)), uint8 (out));
%!
%! ## using a se that will be decomposed in 2 pieces
%! out =[ 7    0   87   66   59    7    0   19   12   30
%!        0   13   81   60   58    1   19   13    6   29
%!       89   12    0   54   52   20   18    7    0   23
%!        8    6   69   53   71   14   12    2    0   43
%!        7    0   63   73   66   14    7    0   23   41
%!       76   69   14    7    0   30   23   46   39    7
%!       70   88    9    2    0   24   42   40   33    6
%!       14   87   80    0    0   43   41   34   27    0
%!       84   82    0    0   19   37   35   28   26   19
%!       89   82    0   20   13   36   29   22   45   13];
%!assert (imbothat (magic (10), ones(5)), out);
%!
%! ## using a weird non-symmetric and even-size se
%! out =[ 0    0   15    8    1    3    0    7    0   18
%!        0    8   53   59    0    0   14   13    0   17
%!       84    0    0   40   38    6   13    6    0   23
%!        2    0   42   47   58    0    6    0    0   41
%!        0    0   62   59   52    0    0    0   16   35
%!        6   58   13    6    0    3   19   19   35    1
%!        0   18    0    0    0    0   15   13    6    0
%!        0   17   69    0    0   17   17    8    0    0
%!        8   67    0    0    0   15    9    2    0    6
%!        7    0    0   17   10   42    7    0   19    0];
%!assert (imbothat (magic (10), [1 0 0 0; 1 1 1 0; 0 1 0 1]), out);
%!
%! ## N dimensional and weird se
%! in = reshape (magic(16), [4 8 4 2]);
%! se = ones (3, 3, 3);
%! se(:,:,1) = [1 0 1; 0 1 1; 0 0 0];
%! se(:,:,3) = [1 0 1; 0 1 1; 0 0 1];
%! out = zeros (size (in));
%! out(:,:,1,1) = [
%!     0   17   81  145  237  146   64    0
%!   205  128   64    0    0   37   83  147
%!   175  111   47    0    0   64  117  181
%!     0   64  128  209  173  109   45    0];
%! out(:,:,2,1) = [
%!   235  142   78   18    0   23   69  133
%!     0   35  103  163  215  128   46    0
%!     0   64  128  195  183  123   48    0
%!   153   93   43    0   14   78  146  215];
%! out(:,:,3,1) = [
%!     0   25   89  153  229  142   64    0
%!   201  128   64    0    0   41   91  155
%!   167  103   57    0    0   64  125  189
%!     0   64  146  217  165  101   37    0];
%! out(:,:,4,1) = [
%!   227  142   78   14    0   31   77  141
%!     0   43  107  171  211  128   46    0
%!     0   64  128  203  179  115   48    0
%!   149   99   35    0   18   82  146  223];
%! out(:,:,1,2) = [
%!     0   33   97  161  221  146   64    0
%!   189  125   61    0    0   53   99  163
%!   159   95   31    0    0   64  128  197
%!     0   64  128  225  157   93   29    0];
%! out(:,:,2,2) = [
%!   219  142   78   18    0   39   85  149
%!     0   51  119  179  199  128   46    0
%!     0   64  128  211  167  107   43    0
%!   137   77   27    0   14   78  146  231];
%! out(:,:,3,2) = [
%!     0   41  105  169  213  142   64    0
%!   185  121   64    0    0   57  107  171
%!   151   87   41    0    0   64  128  205
%!     0   64  146  233  149   85   21    0];
%! out(:,:,4,2) = [
%!   211  142   78   14    0   47   93  157
%!     0   59  123  187  195  128   46    0
%!     0   64  128  219  163   99   35    0
%!   133   83   19    0   18   82  146  239];
%!assert (imbothat (in, se), out);
