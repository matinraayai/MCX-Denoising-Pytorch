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
## @deftypefn {Function File} {} imtophat (@var{img}, @var{SE})
## Perform morphological top hat filtering.
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
## A top hat transform corresponds to the difference between @var{img},
## and the opening of @var{img}, i.e., it is equivalent to:
## @example
## img - imopen (img, se);
## @end example
##
## Use @code{imbothat} to perform a 'black' or 'closing', top-hat transform
## (is is also known as bottom-hat transform).
##
## @seealso{imerode, imdilate, imopen, imclose, imbothat, mmgradm}
## @end deftypefn

function white = imtophat (img, se)

  if (nargin != 2)
    print_usage ();
  elseif (! isimage (img))
    error("imtophat: IMG must be a numeric matrix");
  endif
  se = prepare_strel ("imtophat", se);

  ## Perform filtering
  ## Note that in case that the transform is to applied to a logical image,
  ## subtraction must be handled in a different way (x & !y) instead of (x - y)
  ## or it will return a double precision matrix
  if (islogical (img))
    white = img & ! imopen (img, se);
  else
    white = img - imopen (img, se);
  endif

endfunction

%!assert (imtophat (ones (3), [1 1; 0 1]), zeros (3));
%!assert (imtophat (true (3), [1 1; 0 1]), false (3));

%!shared in, out, se
%! in =  [ 0   0   0   1   1   1   0   0   1   1
%!     0   1   0   1   1   1   0   0   0   1
%!     1   1   1   1   1   0   0   0   0   0
%!     0   1   1   1   1   0   0   0   0   0
%!     0   0   0   1   0   0   0   0   1   0
%!     0   0   0   0   0   0   0   1   1   1
%!     0   0   0   0   1   0   1   0   1   0
%!     0   0   0   1   1   1   1   1   0   0
%!     0   0   0   0   1   1   1   0   0   0
%!     0   0   0   1   1   1   0   0   0   0];
%!
%! out = [ 0   0   0   0   0   0   0   0   1   1
%!         0   1   0   0   0   0   0   0   0   1
%!         1   1   1   1   1   0   0   0   0   0
%!         0   1   1   1   1   0   0   0   0   0
%!         0   0   0   1   0   0   0   0   1   0
%!         0   0   0   0   0   0   0   1   1   1
%!         0   0   0   0   1   0   1   0   1   0
%!         0   0   0   1   1   1   1   1   0   0
%!         0   0   0   0   1   1   1   0   0   0
%!         0   0   0   1   1   1   0   0   0   0];
%!assert (imtophat (logical (in), ones (3)), logical (out));
%!
%! out = [12  19   0   0   0  16  23   0   7   0
%!        18   0   0   6   1  19   0   2   9   1
%!         0  74  81  12   7   0   1   8  15   7
%!        68  70   2  14   0   6   7  14  16   0
%!        69  76   8   0   0   7  14  21   0   1
%!         0   7  59  54  61  13  20   0   0  32
%!        18   0  69  60  62  19   0   0   0  27
%!        73   0   0  66  68   0   1   6   6  33
%!         0   0  17  19   1   0   2   9   7  14
%!         1   6  23   0   7   1   8  15   0  32];
%!assert (imtophat (magic (10), ones (3)), out);
%!assert (imtophat (uint8 (magic (10)), strel ("square", 3)), uint8 (out));
%!
%! ## using a se that will be decomposed in 2 pieces
%! out =[91  98   0   0   0  27  34  11  18   0
%!       94  76   3   6   1  33  15  17  24   1
%!        0  77  84  12   7  14  16  23  30   7
%!       80  82  14  18   0  32  34  41  43   0
%!       81  88  20   0   0  33  40  47  24   6
%!       12  19  63  57  64  16  23   0   7  39
%!       18   0  69  60  62  19   1   3  12  39
%!       73   0   0  66  68   0   2   9  18  45
%!        4   6  81  67  49   6   8  15  19  26
%!        5  12  87  48  55   7  14  21   0  32];
%!assert (imtophat (magic (10), ones(5)), out);
%!
%! ## using a weird non-symmetric and even-size se
%! out =[85  92   0   0   0  12  23   0  17   0
%!       91  73   0   6   0  18   0   2  13   0
%!        0  72  81  13   6   0   1   9  15   0
%!       60  62  10  12   0   8   8  17  17   0
%!       61  69   0   0   0  28  16  41   0   0
%!        0   0  47  52  61  12  16   0   0  31
%!        6   0  53  58  60  17   0   0   0  33
%!       69   0   0  60  62   0   0   6   0  33
%!        0   0  17  60  42   0   2  13   1   8
%!        0   6  23   0   7   0   7  15   0  14];
%!assert (imtophat (magic (10), [1 0 0 0; 1 1 1 0; 0 1 0 1]), out);
%!
%! ## N dimensional and weird se
%! in = reshape (magic(16), [4 8 4 2]);
%! se = ones (3, 3, 3);
%! se(:,:,1) = [1 0 1; 0 1 1; 0 0 0];
%! se(:,:,3) = [1 0 1; 0 1 1; 0 0 1];
%! out = zeros (size (in));
%! out(:,:,1,1) = [
%!   239  146   82   18    0   19   83  133
%!     0   35   99  163  219  128   64    0
%!     0   46  128  195  187  123   59    0
%!   157   93   47    0   14   78  142  211];
%! out(:,:,2,1) = [
%!     0   21   85  149  233  146   64    0
%!   205  128   64    0    0   41   87  151
%!   171  107   57    0    0   64  121  185
%!     0   64  142  213  169  105   41    0];
%! out(:,:,3,1) = [
%!   231  146   78   14    0   27   77  137
%!     0   43  107  167  211  128   64    0
%!     0   46  128  199  179  119   51    0
%!   149   85   39    0   18   78  142  219];
%! out(:,:,4,1) = [
%!     0   29   93  157  225  128   64    0
%!   197  128   64    0    0   31   95  159
%!   163   99   53    0    0   61  125  189
%!     0   64  146  221  161   97   33    0];
%! out(:,:,1,2) = [
%!   223  146   82   18    0   35   99  149
%!     0   48  115  179  203  128   64    0
%!     0   46  128  211  171  107   43    0
%!   141   77   31    0   14   78  142  227];
%! out(:,:,2,2) = [
%!     0   37  101  165  217  146   64    0
%!   189  125   64    0    0   57  103  167
%!   155   91   41    0    0   64  128  201
%!     0   64  142  229  153   89   25    0];
%! out(:,:,3,2) = [
%!   215  146   78   14    0   43   93  153
%!     0   48  123  183  195  128   64    0
%!     0   46  128  215  163  103   35    0
%!   133   69   23    0   18   78  142  235];
%! out(:,:,4,2) = [
%!     0   45  109  173  209  128   64    0
%!   181  117   64    0    0   47  111  175
%!   147   83   37    0    0   64  128  205
%!     0   64  146  237  145   81   17    0];
%!assert (imtophat (in, se), out);
