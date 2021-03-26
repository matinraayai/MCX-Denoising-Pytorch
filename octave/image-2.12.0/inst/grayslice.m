## Copyright (C) 2014-2018 Carnë Draug <carandraug@octave.org>
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
## @deftypefn  {} {} grayslice (@var{I})
## @deftypefnx {} {} grayslice (@var{I}, @var{n})
## @deftypefnx {} {} grayslice (@var{I}, @var{v})
## Create indexed image from intensity image using multilevel thresholding.
##
## The intensity image @var{I} is split into multiple threshold levels.
## For regularly spaced intervals, the number of levels can be specified as the
## numeric scalar @var{n} (defaults to 10), which will use the intervals:
##
## @tex
## \def\frac#1#2{{\begingroup#1\endgroup\over#2}}
## $$ \frac{1}{n}, \frac{2}{n}, \dots{}, \frac{n - 1}{n} $$
## @end tex
## @ifnottex
## @verbatim
## 1  2       n-1
## -, -, ..., ---
## n  n        n
## @end verbatim
## @end ifnottex
##
## For irregularly spaced intervals, a numeric vector @var{v} of
## threshold values can be used instead.
##
## The output image will be of class uint8 if the number of levels is
## less than 256, otherwise it will be double.
##
## @seealso{im2bw, gray2ind}
## @end deftypefn

function sliced = grayslice (I, n = 10)

  if (nargin < 1 || nargin > 2)
    print_usage ();
  elseif (! isnumeric (n))
    error ("Octave:invalid-invalid-input-arg",
           "grayslice: N and V must be numeric");
  endif

  if (isscalar (n) && n >= 1)
    ## For Matlab compatibility, don't check if N is an integer but
    ## don't allow n < 1 either.
    n = double (n);
    v = (1:(n-1)) ./ n;
    v = imcast (v, class (I));
  elseif ((isvector (n) && ! isscalar (n)) || (isscalar (n) && n > 0 && n <1))
    ## For Matlab compatibility, a 0>N>1 is handled like V.
    v = sort (n(:));
    n = numel (v) + 1;
    ## The range is [0 1] but if the image is floating point we may
    ## need to increase the range (but never decrease it).
    if (isfloat (I))
      imax = max (I(:));
      imin = min (I(:));
      v(v < imin) = imin;
      v(v > imax) = imax;
    endif
  else
    if (isscalar (n) && n <= 0)
      error ("Octave:invalid-invalid-input-arg",
             "grayslice: N must be a positive number");
      endif
    error ("Octave:invalid-invalid-input-arg",
           "grayslice: N and V must be a numeric scalar an vector");
  endif

  sliced_tmp = lookup (v, I);

  if (n < 256)
    sliced_tmp = uint8 (sliced_tmp);
  else
    ## Indexed images of class double have indices base 1
    sliced_tmp++;
  endif

  if (nargout < 1)
    imshow (sliced_tmp, jet (n));
  else
    sliced = sliced_tmp;
  endif
endfunction

%!test
%! expected = uint8 ([0 4 5 5 9]);
%! im = [0 0.45 0.5 0.55 1];
%! assert (grayslice (im), expected)
%! assert (grayslice (im, 10), expected)
%! assert (grayslice (im, uint8 (10)), expected)
%! assert (grayslice (im, [.1 .2 .3 .4 .5 .6 .7 .8 .9]), expected)

%!test
%! im = [0 0.45 0.5 0.55 1];
%! assert (grayslice (im, 2), uint8 ([0 0 1 1 1]))
%! assert (grayslice (im, 3), uint8 ([0 1 1 1 2]))
%! assert (grayslice (im, 4), uint8 ([0 1 2 2 3]))
%! assert (grayslice (im, [0 0.5 1]), uint8 ([1 1 2 2 3]))
%! assert (grayslice (im, [0.5 1]), uint8 ([0 0 1 1 2]))
%! assert (grayslice (im, [0.6 1]), uint8 ([0 0 0 0 2]))

%!test
%% ## non-integer values of N when N>1 are used anyway
%! im = [0 .55 1];
%! assert (grayslice (im, 9), uint8 ([0 4 8]))
%! assert (grayslice (im, 9.1), uint8 ([0 5 8]))
%! assert (grayslice (im, 10), uint8 ([0 5 9]))

## handle unsorted V
%!assert (grayslice ([0 .5 1], [0 1 .5]), uint8 ([1 2 3]))

%!test
%! ## 0 > N > 1 values are treated as if they are V and N=2
%! im = [0 .5 .55 .7 1];
%! assert (grayslice (im, .5), uint8 ([0 1 1 1 1]))
%! assert (grayslice (im, .51), uint8 ([0 0 1 1 1]))
%! assert (grayslice (im, .7), uint8 ([0 0 0 1 1]))
%! assert (grayslice (im, 1), uint8 ([0 0 0 0 0]))
%! assert (grayslice (im, 1.2), uint8 ([0 0 0 0 0]))

## V is outside the [0 1] and image range
%!assert (grayslice ([0 .5 .7 1], [0 .5 1 2]), uint8 ([1 2 2 4]))

## repeated values in V
%!assert (grayslice ([0 .45 .5 .65 .7 1], [.4 .5 .5 .7 .7 1]),
%!        uint8 ([0 1 3 3 5 6]))

## Image an V with values outside [0 1] range
%!assert (grayslice ([-.5 .1 .8 1.2], [-1 -.4 .05 .6 .9 1.1 2]),
%!        uint8 ([1 3 4 7]))
%!assert (grayslice ([0 .5 1], [-1 .5 1 2]), uint8 ([1 2 4]))
%!assert (grayslice ([-2 -1 .5 1], [-1 .5 1]), uint8 ([0 1 2 3]))

%!test
%! sliced = [
%!   repmat(0, [26 1])
%!   repmat(1, [25 1])
%!   repmat(2, [26 1])
%!   repmat(3, [25 1])
%!   repmat(4, [26 1])
%!   repmat(5, [25 1])
%!   repmat(6, [26 1])
%!   repmat(7, [25 1])
%!   repmat(8, [26 1])
%!   repmat(9, [26 1])
%! ];
%! sliced = uint8 (sliced(:).');
%! assert (grayslice (uint8 (0:255)), sliced)

%!assert (grayslice (uint8 (0:255), 255), uint8 ([0:254 254]))

## Returns class double if n >= 256 and not n > 256
%!assert (class (grayslice (uint8 (0:255), 256)), "double")

%!xtest
%! assert (grayslice (uint8 (0:255), 256), [1:256])
%!
%! ## While the above fails, this passes and should continue to do so
%! ## since it's the actual formula in the documentation.
%! assert (grayslice (uint8 (0:255), 256),
%!         grayslice (uint8 (0:255), (1:255)./256))

%!test
%! ## Use of threshold in the [0 1] range for images of integer type does
%! ## not really work despite the Matlab documentation.  It's Matlab
%! ## documentation that is wrong, see bug #55059
%!
%! assert (grayslice (uint8([0 100 200 255]), [.1 .4 .5]),
%!         uint8 ([0 3 3 3]))
%! assert (grayslice (uint8([0 100 200 255]), [100 199 200 210]),
%!         uint8 ([0 1 3 4]))
%!
%! ## P (penny) is a 2d image of class double in [1 255] range
%! q = warning ("query", "Octave:data-file-in-path");
%! warning ("off", "Octave:data-file-in-path");
%! load ("penny.mat");
%! warning (q.state, "Octave:data-file-in-path");
%! assert (grayslice (P), repmat (uint8 (9), size (P)))

%!function gs = test_grayslice_v (I, v)
%!  ## This is effectively what grayslice does but slower with a for
%!  ## loop internally.
%!  gs = zeros (size (I));
%!  for idx = 1:numel (v)
%!    gs(I >= v(idx)) = idx;
%!  endfor
%! if (numel (v) >= 256)
%!   gs = gs +1;
%! else
%!   gs = uint8 (gs);
%! endif
%!endfunction

%!test
%! q = warning ("query", "Octave:data-file-in-path");
%! warning ("off", "Octave:data-file-in-path");
%! load ("penny.mat");
%! warning (q.state, "Octave:data-file-in-path");
%!
%! ## The loaded P in penny.mat is of size 128x128, class double, and
%! ## with values in the [1 255] range
%! penny_uint8 = uint8 (P);
%! penny_double = im2double (penny_uint8); # rescales to [0 1] range]
%!
%! ## default of N = 10
%! expected = test_grayslice_v (penny_uint8,
%!                              [26 51 77 102 128 153 179 204 230]);
%! assert (grayslice (penny_uint8, 10), expected)
%! assert (grayslice (penny_uint8), expected)
%!
%! expected = test_grayslice_v (penny_double,
%!                              [.1 .2 .3 .4 .5 .6 .7 .8 .9]);
%! assert (grayslice (penny_double, 10), expected)
%! assert (grayslice (penny_double), expected)

%!test
%! ## For images with more than 2d
%! q = warning ("query", "Octave:data-file-in-path");
%! warning ("off", "Octave:data-file-in-path");
%! load ("penny.mat");
%! warning (q.state, "Octave:data-file-in-path");
%! penny_double = im2double (uint8 (P));
%! P_3d = repmat (penny_double, [1 1 3]);
%! P_5d = repmat (penny_double, [1 1 3 2 3]);
%!
%! v = [.3 .5 .7];
%! expected_2d = test_grayslice_v (penny_double, v);
%! assert (grayslice (P_3d, v), repmat (expected_2d, [1 1 3]))
%! assert (grayslice (P_5d, v), repmat (expected_2d, [1 1 3 2 3]))

%!test
%! q = warning ("query", "Octave:data-file-in-path");
%! warning ("off", "Octave:data-file-in-path");
%! load ("penny.mat");
%! warning (q.state, "Octave:data-file-in-path");
%! penny_double = uint8 (P);
%!
%! ## Test that change from uint8 to double happens at 256 exactly
%! assert (class (grayslice (penny_double, 255)), "uint8")
%! assert (class (grayslice (penny_double, 256)), "double")
%!
%! ## If returns in class double, it's +1.
%! v = [10 150 200];
%! v_long = [v 256:600];
%! assert (double (grayslice (penny_double, v)) +1,
%!         grayslice (penny_double, v_long))

%!test
%! ## If there's a vector for floating point and goes outside the
%! ## range, it uses the last index of the vector.
%! q = warning ("query", "Octave:data-file-in-path");
%! warning ("off", "Octave:data-file-in-path");
%! load ("penny.mat");
%! warning (q.state, "Octave:data-file-in-path");
%! penny_double = im2double (uint8 (P));
%! v = [.3 .5 .7 2:10];
%! idx_1 = find (penny_double == 1);
%! assert (grayslice (penny_double, v)(idx_1), uint8 ([12; 12]))

%!error <N must be a positive number> x = grayslice ([1 2; 3 4], 0)
%!error <N must be a positive number> x = grayslice ([1 2; 3 4], -1)
%!error <N and V must be numeric> x = grayslice ([1 2; 3 4], "foo")
