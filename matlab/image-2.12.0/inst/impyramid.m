## Copyright (C) 2015 Avinoam Kalma <a.kalma@gmail.com>
## Copyright (C) 2015 Carnë Draug <carandraug@octave.org>
##
## This program is free software; you can redistribute it and/or
## modify it under the terms of the GNU General Public License as
## published by the Free Software Foundation; either version 3 of the
## License, or (at your option) any later version.
##
## This program is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
## General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program; if not, see
## <http:##www.gnu.org/licenses/>.

## -*- texinfo -*-
## @deftypefn {Function File} {} impyramid (@var{im}, @var{direction})
## Compute gaussian pyramid expansion or reduction.
##
## Create image which is one level up or down in the Gaussian
## pyramid.  @var{direction} must be @qcode{"reduce"} or
## @qcode{"expand"}.  These operations are only done in the first
## two dimensions, so that even if @var{im} is a N dimensional
## array, only the number of rows and columns will change.
##
## The @qcode{"reduce"} stage is done by low-pass filtering and
## subsampling of 1:2 in each axis.  If the size of the original
## image is [M N], the size of the reduced image is
## @qcode{[ceil((M+1)/2) ceil((N+1)/2)]}.
##
## The @qcode{"expand"} stage is done by upsampling the image
## (2:1 in each axis), and then low-pass filtering.  If the size
## of the original image is [M N], the size of the expanded image
## is @code{[2M-1 2N-1]}.
##
## Note that image processing pyramids are upside down, so
## @qcode{"reduce"} is going one level @emph{down} in the pyramid,
## while @qcode{"expand"} is going one level @emph{up} in the pyramid.
##
## @example
## @group
## impyramid (im, "reduce");   # return reduced image (one level down)
## impyramid (im, "expand");   # return expanded image (one level up)
## @end group
## @end example
##
## The low-pass filter is defined according to Burt & Adelson [1]
## @code{W(i,j) = w(i)w(j)} where
## @code{w = [0.25-alpha/2 0.25 alpha 0.25 0.25-alpha/2]} with
## @code{alpha = 0.375}
##
## [1] Peter J. Burt and Edward H. Adelson (1983).  The Laplacian Pyramid
## as a Compact Image Code.  IEEE Transactions on Communications,
## vol. COM-31(4), 532-540.
##
## @seealso{imresize, imfilter}
## @end deftypefn

## Author: Avinoam Kalma <a.kalma@gmail.com>

function imp = impyramid (im, direction)

  if (nargin != 2)
    print_usage ();
  elseif (! isnumeric (im) && ! isbool (im))
    error ("impyramid: IM must be numeric or logical")
  endif

  ## low pass filters to be used
  alpha = 0.375;
  filt_horz = [(0.25-alpha/2)  0.25  alpha  0.25  (0.25-alpha/2)];
  filt_vert = filt_horz.';

  nd = ndims (im);
  sz = size (im);
  cl = class (im);
  switch (tolower (direction))
    case "reduce"
      ## vertical low pass filtering
      im = padarray (im, floor (size (filt_vert) /2), "replicate");
      im = convn (im, filt_vert, "valid");

      ## horizontal low pass filtering
      im = padarray (im, floor (size (filt_horz) /2), "replicate");
      im = convn (im, filt_horz, "valid");

      im = cast (im, cl);

      ## subsampling
      idx = repmat ({":"}, 1, nd);
      idx([1 2]) = {1:2:sz(1), 1:2:sz(2)};
      imp = im(idx{:});

    case "expand"
      ## Create image, twice the size (rows and columns only),
      ## with the original image on the odd pixels.
      imp_sz = sz .* postpad ([2 2], nd, 1);
      imp_sz([1 2]) -= 1;
      imp = zeros (imp_sz, cl);
      idx = repmat ({":"}, 1, nd);
      idx([1 2]) = {1:2:imp_sz(1), 1:2:imp_sz(2)};
      imp(idx{:}) = im;

      ## horizontal low pass filtering
      imp = padarray (imp, floor (size (filt_horz) /2));
      imp = convn (imp, filt_horz, "valid");
      imp *= 2;

      ## vertical low pass filtering
      imp = padarray (imp, floor (size (filt_vert) /2));
      imp = convn (imp, filt_vert, "valid");
      imp *= 2;

      imp = cast (imp, cl);

    otherwise
      error ("impyramid: DIRECTION must be 'reduce' or 'expand'")
  endswitch
endfunction

## Note that there are small differences, 1 and 2 gray levels, between
## the results here (the ones we get in Octave), and the ones we should
## have for Matlab compatibility.  This is specially true for elements
## in the border, and worse when expanding.

%!xtest
%! ## bug #51979 (results are not matlab compatible)
%! in = [116  227  153   69  146  194   59  130  139  106
%!         2   47  137  249   90   75   16   24  158   44
%!       155   68   46   84  166  156   69  204   32  152
%!        71  221  137  230  210  153  192  115   30  118
%!       107  143  108   52   51   73  101   21  175   90
%!        54  158  143   77   26  168  113  229  165  225
%!         9   47  133  135  130  207  236   43   19   73];
%!
%! reduced = [
%!     114  139  131  103  111
%!      97  122  141  111  100
%!     103  123  112  123  122
%!      47  107  134  153   94];
%!
%! expanded = [
%!   115  154  185  178  150  122  105  116  138  159  158  117   78   86  112  129  133  120  103
%!    69   98  128  141  146  152  152  139  125  127  121   87   55   58   81  113  131  112   84
%!    40   54   74  100  131  167  184  157  119  104   92   64   41   44   66  100  121  103   74
%!    76   69   65   75   97  130  153  148  131  122  108   80   61   79  103  105   98   97   98
%!   120  105   88   77   78   96  121  143  155  154  140  112   98  124  143  109   74   91  123
%!   117  129  134  119  107  125  153  173  180  172  156  143  138  146  140   96   60   83  122
%!    99  139  170  157  139  156  181  188  180  164  151  154  156  140  112   81   65   84  110
%!   101  136  163  153  133  132  138  136  130  122  120  130  133  108   82   86   99  104  104
%!   103  126  143  136  116   97   81   73   73   82   94  105  105   87   78  108  138  133  116
%!    90  116  139  139  122   96   69   52   53   80  109  114  111  116  128  148  163  164  160
%!    66   99  131  140  131  109   83   62   62  102  142  144  138  154  169  164  157  169  184
%!    41   68   99  121  130  122  107   92   95  133  173  182  172  156  135  114  105  121  142
%!    21   38   64   98  124  131  127  123  129  160  194  212  199  144   82   52   48   65   85];
%!
%! assert (impyramid (uint8 (in), "reduce"), uint8 (reduced))
%! assert (impyramid (uint8 (in), "expand"), uint8 (expanded))

## Test that that reduction and expansion are done in the
## first 2 dimensions only.
%!test
%! in = randi ([0 255], [40 39 3 5], "uint8");
%! red = impyramid (in, "reduce");
%! for p = 1:3
%!   for n = 1:5
%!     assert (red(:,:,p,n), impyramid (in(:,:,p,n), "reduce"))
%!   endfor
%! endfor
%!
%! exp = impyramid (in, "expand");
%! for p = 1:3
%!   for n = 1:5
%!     assert (exp(:,:,p,n), impyramid (in(:,:,p,n), "expand"))
%!   endfor
%! endfor

%!xtest
%! ## bug #51979 (results are not matlab compatible)
%! in = repmat (uint8 (255), [10 10]);
%! assert (impyramid (in, "reduce"), repmat (uint8 (255), [5 5]))
%! assert (impyramid (in, "expand"), repmat (uint8 (255), [19 19]))

%!xtest
%! ## bug #51979  (results are not matlab compatible)
%! in = logical ([
%!   1  0  1  1  0  0  1  1  0  0
%!   1  1  0  0  0  1  0  0  1  0
%!   0  1  1  0  1  1  1  1  1  1
%!   1  0  1  0  1  0  1  0  1  1
%!   1  1  1  0  0  0  1  1  1  1
%!   0  0  1  1  0  0  1  0  0  0
%!   0  0  1  1  0  1  1  0  1  1
%!   1  1  0  0  1  0  0  0  1  0
%!   1  1  1  1  1  1  0  1  0  0
%!   1  1  0  0  1  0  0  0  1  0]);
%!
%! reduced = logical ([
%!   1  1  0  1  0
%!   1  1  0  1  1
%!   1  1  0  1  1
%!   0  1  0  0  0
%!   1  1  1  0  0]);
%!
%! expanded = logical ([
%!   1  1  0  0  1  1  1  0  0  0  0  0  1  1  1  0  0  0  0
%!   1  1  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
%!   1  1  1  1  0  0  0  0  0  0  1  1  0  0  0  1  1  0  0
%!   1  1  1  1  0  0  0  0  0  1  1  1  1  0  1  1  1  1  1
%!   0  1  1  1  1  0  0  0  1  1  1  1  1  1  1  1  1  1  1
%!   0  0  1  1  1  0  0  0  1  1  1  1  1  1  1  1  1  1  1
%!   1  1  0  1  1  0  0  0  1  0  0  1  1  1  0  1  1  1  1
%!   1  1  1  1  1  0  0  0  0  0  0  0  1  1  1  1  1  1  1
%!   1  1  1  1  1  1  0  0  0  0  0  0  1  1  1  1  1  1  1
%!   0  0  1  1  1  1  0  0  0  0  0  0  1  1  1  0  0  0  0
%!   0  0  0  1  1  1  1  0  0  0  0  1  1  1  0  0  0  0  0
%!   0  0  0  0  1  1  1  0  0  0  0  1  1  0  0  0  0  0  0
%!   0  0  0  0  1  1  1  0  0  0  1  1  1  0  0  0  1  1  1
%!   0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  1  1
%!   1  1  1  1  0  0  0  1  1  1  0  0  0  0  0  0  1  0  0
%!   1  1  1  1  1  0  1  1  1  1  0  0  0  0  0  0  0  0  0
%!   1  1  1  1  1  1  1  1  1  1  1  0  0  0  1  0  0  0  0
%!   1  1  1  1  1  0  1  1  1  1  0  0  0  0  0  0  0  0  0
%!   1  1  1  1  0  0  0  1  1  1  0  0  0  0  0  0  1  0  0]);
%!
%! assert (impyramid (in, "reduce"), reduced)
%! assert (impyramid (in, "expand"), expanded)
