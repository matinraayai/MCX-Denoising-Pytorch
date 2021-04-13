## Copyright (C) 2005 Søren Hauberg <soren@hauberg.org>
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
## @deftypefn  {Function File} {} imresize (@var{im}, @var{scale})
## @deftypefnx {Function File} {} imresize (@var{im}, [@var{M} @var{N}])
## @deftypefnx {Function File} {} imresize (@dots{}, @var{method})
## Resize image with interpolation
##
## Scales the image @var{im} by a factor @var{scale} or into the size @var{M}
## rows by @var{N} columns.  For example:
##
## @example
## @group
## imresize (im, 1);    # return the same image as input
## imresize (im, 1.5);  # return image 1.5 times larger
## imresize (im, 0.5);  # return image with half the size
## imresize (im, 2);    # return image with the double size
## imresize (im, [512 610]); # return image of size 512x610
## @end group
## @end example
##
## If @var{M} or @var{N} is @code{NaN}, it will be determined automatically so
## as to preserve aspect ratio.
##
## The optional argument @var{method} defines the interpolation method to be
## used.  All methods supported by @code{interp2} can be used.  By default, the
## @code{cubic} method is used.
##
## For @sc{matlab} compatibility, the methods @code{bicubic} (same as
## @code{cubic}), @code{bilinear} and @code{triangle} (both the same as
## @code{linear}) are also supported.
##
## @table @asis
## @item bicubic (default)
## Same as @code{cubic}.
##
## @item bilinear
## Same as @code{linear}.
##
## @item triangle
## Same as @code{linear}.
## @end table
##
## @seealso{imremap, imrotate, interp2}
## @end deftypefn

function im = imresize (im, scale, method = "cubic")

  if (nargin < 2 || nargin > 3)
    print_usage ();
  elseif (! isimage (im))
    error ("imresize: IM must be an image")
  elseif (! isnumeric (scale) || any (scale <= 0) || all (isnan (scale)))
    error ("imresize: SCALE or [M N] must be numeric positive values")
  elseif (! ischar (method))
    error ("imresize: METHOD must be a string with interpolation method")
  endif
  method = interp_method (method);

  inRows = rows (im);
  inCols = columns (im);

  ## we may be able to use clever indexing instead of interpolation
  int_row_scale = false;
  int_col_scale = false;
  if (isscalar (scale))
    outRows = ceil (rows (im) * scale);
    outCols = ceil (columns (im) * scale);

    ## check if we can use clever indexing
    scale_rows = scale_cols = scale;
    int_row_scale = int_col_scale = is_for_integer (scale);
  elseif (numel (scale) == 2)
    outRows = scale(1);
    outCols = scale(2);
    ## maintain aspect ratio if requested
    if (isnan (outRows))
      outRows = inRows * (outCols / inCols);
    elseif (isnan (outCols))
      outCols = inCols * (outRows / inRows);
    endif
    outRows = ceil (outRows);
    outCols = ceil (outCols);

    ## we will need this to use clever indexing. In this case, we will also need
    ## to check that we are changing the rows and columns of the image in the
    ## same direction
    scale_rows = (outRows/inRows);
    scale_cols = (outCols/inCols);
    int_row_scale = is_for_integer (scale_rows);
    int_col_scale = is_for_integer (scale_cols);
  else
    error ("imresize: SCALE argument must be a scalar or a 2 element vector");
  end

  ## Perform the actual resizing
  if (inRows == outRows && inCols == outCols)
    ## no resizing to do
  elseif (strcmpi (method, "nearest") && all ([int_row_scale int_col_scale]))
    ## we are matlab incompatible here on purpose. We can the stuff here in 2
    ## ways. With interp2 or by clever indexing. Indexing is much much faster
    ## than interp2 but they return different results (the way we are doing it
    ## at least). Matlab does the same as we are doing if the both columns and
    ## rows go the same direction but if they increase one and decrease the
    ## other, then they return the same as if we were using interp2. We are
    ## smarter and use indexing even in that case but then the results differ
    if (int_row_scale == 1)
      row_idx = (1:rows (im))(ones (1, scale_rows), :);
    elseif (int_row_scale == -1)
      row_idx = ceil (linspace (floor (1/(scale_rows * 2)) + 1, inRows, outRows));
    endif
    if (int_col_scale == 1)
      col_idx = (1:columns (im))(ones (scale_cols, 1), :);
    elseif (int_col_scale == -1)
      col_idx = ceil (linspace (floor (1/(scale_cols * 2)) + 1, inCols, outCols));
    endif
    im = im(row_idx, col_idx);

  else
    [XI, YI] = meshgrid (linspace (1, inCols, outCols), linspace (1, inRows, outRows));
    im = imremap (im, XI, YI, method);
  endif
endfunction

function retval = is_for_integer (scale)
  retval = false;
  if (fix (scale) == scale)
    retval = 1;
  elseif (fix (1/scale) == (1/scale))
    ## if scale/n is an integer then we are resizing to one half, one third, etc
    ## and we can also use clever indexing
    retval = -1;
  endif
endfunction

%!test
%! in = [116  227  153   69  146  194   59  130  139  106
%!         2   47  137  249   90   75   16   24  158   44
%!       155   68   46   84  166  156   69  204   32  152
%!        71  221  137  230  210  153  192  115   30  118
%!       107  143  108   52   51   73  101   21  175   90
%!        54  158  143   77   26  168  113  229  165  225
%!         9   47  133  135  130  207  236   43   19   73];
%! assert (imresize (uint8 (in), 1, "nearest"), uint8 (in))
%! assert (imresize (uint8 (in), 1, "bicubic"), uint8 (in))
%!
%! out = [116  116  227  227  153  153   69   69  146  146  194  194   59   59  130  130  139  139  106  106
%!        116  116  227  227  153  153   69   69  146  146  194  194   59   59  130  130  139  139  106  106
%!          2    2   47   47  137  137  249  249   90   90   75   75   16   16   24   24  158  158   44   44
%!          2    2   47   47  137  137  249  249   90   90   75   75   16   16   24   24  158  158   44   44
%!        155  155   68   68   46   46   84   84  166  166  156  156   69   69  204  204   32   32  152  152
%!        155  155   68   68   46   46   84   84  166  166  156  156   69   69  204  204   32   32  152  152
%!         71   71  221  221  137  137  230  230  210  210  153  153  192  192  115  115   30   30  118  118
%!         71   71  221  221  137  137  230  230  210  210  153  153  192  192  115  115   30   30  118  118
%!        107  107  143  143  108  108   52   52   51   51   73   73  101  101   21   21  175  175   90   90
%!        107  107  143  143  108  108   52   52   51   51   73   73  101  101   21   21  175  175   90   90
%!         54   54  158  158  143  143   77   77   26   26  168  168  113  113  229  229  165  165  225  225
%!         54   54  158  158  143  143   77   77   26   26  168  168  113  113  229  229  165  165  225  225
%!          9    9   47   47  133  133  135  135  130  130  207  207  236  236   43   43   19   19   73   73
%!          9    9   47   47  133  133  135  135  130  130  207  207  236  236   43   43   19   19   73   73];
%! assert (imresize (uint8 (in), 2, "nearest"), uint8 (out))
%! assert (imresize (uint8 (in), 2, "neAreST"), uint8 (out))
%! assert (imresize (uint8 (in), [14 NaN], "nearest"), uint8 (out))
%! assert (imresize (uint8 (in), [NaN 20], "nearest"), uint8 (out))
%!
%! out = [116  116  227  227  153  153   69   69  146  146  194  194   59   59  130  130  139  139  106  106
%!          2    2   47   47  137  137  249  249   90   90   75   75   16   16   24   24  158  158   44   44
%!        155  155   68   68   46   46   84   84  166  166  156  156   69   69  204  204   32   32  152  152
%!         71   71  221  221  137  137  230  230  210  210  153  153  192  192  115  115   30   30  118  118
%!        107  107  143  143  108  108   52   52   51   51   73   73  101  101   21   21  175  175   90   90
%!         54   54  158  158  143  143   77   77   26   26  168  168  113  113  229  229  165  165  225  225
%!          9    9   47   47  133  133  135  135  130  130  207  207  236  236   43   43   19   19   73   73];
%! assert (imresize (uint8 (in), [7 20], "nearest"), uint8 (out))
%!
%! assert (imresize (uint8 (in), 1.5, "bicubic"), imresize (uint8 (in), 1.5, "cubic"))
%! assert (imresize (uint8 (in), [NaN, size(in,2)*1.5], "bicubic"), imresize (uint8 (in), 1.5, "cubic"))
%! assert (imresize (uint8 (in), [size(in,1)*1.5, NaN], "bicubic"), imresize (uint8 (in), 1.5, "cubic"))
%! assert (imresize (uint8 (in), 1.5, "linear"), imresize (uint8 (in), 1.5, "LIneAR"))
%! assert (imresize (uint8 (in), 1.5, "linear"), imresize (uint8 (in), 1.5, "triangle"))
%!
%! out = [ 47  249   75   24   44
%!        221  230  153  115  118
%!        158   77  168  229  225
%!        47   135  207   43   73];
%! assert (imresize (uint8 (in), 0.5, "nearest"), uint8 (out))

## Do not enforce floating point images to be in the [0 1] range (bug #43846)
%!assert (imresize (repmat (5, [3 3]), 2), repmat (5, [6 6]), eps*100)

## Similarly, do not enforce images to have specific dimensions and only
## expand on the first 2 dimensions.
%!assert (imresize (repmat (5, [3 3 2]), 2), repmat (5, [6 6 2]), eps*100)

## The following are the matlab results. We have slighlty different
## results but not by much. If there's would be any fixes, they
## would have to be on interp2 or maybe in imremap.

%!shared in
%! in = [116  227  153   69  146  194   59  130  139  106
%!         2   47  137  249   90   75   16   24  158   44
%!       155   68   46   84  166  156   69  204   32  152
%!        71  221  137  230  210  153  192  115   30  118
%!       107  143  108   52   51   73  101   21  175   90
%!        54  158  143   77   26  168  113  229  165  225
%!         9   47  133  135  130  207  236   43   19   73];

%!xtest
%! out = [116  185  235  171   96   64  134  189  186   74   90   141  140  124  108
%!         44   92  143  149  164  163  119  123  118   44   38    80  151  118   63
%!         14   21   47  107  195  228  115   81   70   24   19    56  137  105   49
%!        145   98   49   49   71  107  148  159  132   58  124   176   61   85  145
%!        118  139  144   92  116  168  201  188  159  140  167   158   27   69  152
%!         62  151  218  145  174  219  201  164  146  187  148    84   48   76  115
%!        102  132  151  119   90   72   72   72   83  114   60    31  144  130   81
%!         82  121  154  133   87   41   19   67  116   95  108   140  183  180  164
%!         40   96  152  149  117   74   34  108  179  131  175   215  153  177  219
%!         11   33   73  127  137  125  113  158  212  229  148    55   35   63   96
%!          4   17   53  121  141  138  133  171  220  253  141    16    7   36   67];
%! assert (imresize (uint8 (in), 1.5, "bicubic"), uint8 (out))

%!xtest
%! out = [116  172  215  165  111   82  133  170  171   81   95   132  138  123  106
%!         59   98  138  144  152  152  125  127  119   54   58    89  137  112   75
%!         27   39   62  110  172  202  123   96   78   36   40    68  123  100   62
%!        129   97   64   62   87  119  146  148  128   74  117   154   73   94  134
%!        113  129  136  101  125  162  183  172  151  135  146   139   53   83  135
%!         77  143  195  145  166  197  186  162  146  171  138    92   62   84  113
%!        101  129  149  120   98   81   78   82   91  111   77    56  132  123   95
%!         81  116  147  130   96   61   43   80  119  109  116   132  162  164  158
%!         46   93  139  141  114   80   50  109  168  141  166   189  151  171  200
%!         16   41   77  123  130  123  115  157  204  214  145    69   48   71   98
%!          9   28   61  119  134  134  131  169  212  231  140    39   23   46   73];
%! assert (imresize (uint8 (in), 1.5, "bilinear"), uint8 (out))

%!xtest
%! out = [108  136  125   89  107
%!        111  132  143  114   99
%!        106  110  106  127  136
%!         47  121  163  138   68];
%! assert (imresize (uint8 (in), 0.5, "bilinear"), uint8 (out))

%!xtest
%! out = [103  141  124   78  110
%!        111  134  153  114   91
%!        115  108   93  128  146
%!         38  124  175  143   54];
%! assert (imresize (uint8 (in), 0.5, "bicubic"), uint8 (out))

%!xtest
%! ## bug #55202
%! assert (imresize (zeros (1, 20), [1 30], "nearest"), zeros (1, 30))

%!test
%! ## a test for bug #55780
%! a = zeros (7, "uint8");
%! a (4,4) = 255;
%! c = [   4   13   24   31   31   24   13    4;
%!        13   44   79  103  103   79   44   13;
%!        24   79  140  182  182  140   79   24;
%!        31  103  182  238  238  182  103   31;
%!        31  103  182  238  238  182  103   31;
%!        24   79  140  182  182  140   79   24;
%!        13   44   79  103  103   79   44   13;
%!         4   13   24   31   31   24   13    4];
%! b = imresize (a, 4, "bicubic");
%! assert (b(11:18,11:18), c, eps)

