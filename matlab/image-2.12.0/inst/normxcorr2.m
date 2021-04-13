## Copyright (C) 2014 Benjamin Eltzner <b.eltzner@gmx.de>
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
## @deftypefn {Function File} {} normxcorr2 (@var{template}, @var{img})
## Compute normalized cross-correlation.
##
## Returns the cross-correlation coefficient of matrices @var{template}
## and @var{img}, a matrix of (roughly) the same size as @var{img} with
## values ranging between -1 and 1.
##
## Normalized correlation is mostly used for template matching, finding an
## object or pattern, @var{template}, withing an image @var{img}.  Higher
## values on the output show their locations, even in the presence of noise.
##
## @group
## @example
## img = randi (255, 600, 400);
## template = imnoise (img(100:150, 300:320), "gaussian");
## cc = normxcorr2 (template, img);
## [r, c] = find (cc == max (cc(:)))
## @result{r} 150
## @result{c} 320
## @end example
## @end group
##
## Despite the function name, this function will accept input with an arbitrary
## number of dimensions.
##
## Note that the size of the cross-correlation array is slightly bigger
## than @var{img} because the input image is padded during the calculation.
##
## @seealso{conv2, convn, corr2, xcorr, xcorr2}
## @end deftypefn

## Author: Benjamin Eltzner <b.eltzner@gmx.de>

function c = normxcorr2 (a, b)
  if (nargin != 2)
    print_usage ();
  endif

  ## If this happens, it is probably a mistake
  if (ndims (a) > ndims (b) || any (postpad (size (a), ndims (b)) > size (b)))
    warning ("normxcorr2: TEMPLATE larger than IMG. Arguments may be swapped.");
  endif

  a = double (a) - mean (a(:));
  b = double (b) - mean (b(:));

  a1 = ones (size (a));
  ar = reshape (a(end:-1:1), size (a));

  c = convn (b, conj (ar));
  b = convn (b.^2, a1) .- convn (b, a1).^2 ./ (prod (size (a)));

  ## remove small machine precision errors after substraction
  b(b < 0) = 0;

  a = sumsq (a(:));
  c = reshape (c ./ sqrt (b * a), size (c));

  c(isinf (c) | isnan (c)) = 0;
endfunction

%!function offsets = get_max_offsets (c)
%!  l = find (c == max (c(:)));
%!  offsets = nthargout (1:ndims (c), @ind2sub, size (c), l);
%!endfunction

## test basic usage
%!test
%! row_shift = 18;
%! col_shift = 20;
%! a = randi (255, 30, 30);
%! b = a(row_shift-10:row_shift, col_shift-7:col_shift);
%! c = normxcorr2 (b, a);
%! ## should return exact coordinates
%! assert (get_max_offsets (c), {row_shift col_shift});
%!
%! ## Even with some small noise, should return exact coordinates
%! b = imnoise (b, "gaussian");
%! c = normxcorr2 (b, a);
%! assert (get_max_offsets (c), {row_shift col_shift});

## The value for a "perfect" match should be 1. However, machine precision
## creeps in most of the times.
%!test
%! a = rand (10, 10);
%! c = normxcorr2 (a(5:7, 6:9), a);
%! assert (c(7, 9), 1, eps*100);

## coeff of autocorrelation must be same as negative of correlation
## by additive inverse
%!test
%! a = 10 * randn (100, 100);
%! auto = normxcorr2 (a, a);
%! add_in = normxcorr2 (a, -a);
%! assert (auto, -add_in);

## Normalized correlation should be independent of scaling and shifting
## up to rounding errors
%!test
%! a = 10 * randn (50, 50);
%! b = 10 * randn (100, 100);
%! do
%!   scale = 100 * rand ();
%! until (scale != 0)
%!
%! assert (max ((normxcorr2 (scale*a,b) .- normxcorr2 (a,b))(:)), 0, 1e-10);
%! assert (max ((normxcorr2 (a,scale*b) .- normxcorr2 (a,b))(:)), 0, 1e-10);
%!
%! a_shift1 = a .+ scale * ones (size (a));
%! b_shift1 = b .+ scale * ones (size (b));
%! a_shift2 = a .- scale * ones (size (a));
%! b_shift2 = b .- scale * ones (size (b));
%! assert (max ((normxcorr2 (a_shift1,b) .- normxcorr2 (a,b))(:)), 0, 1e-10);
%! assert (max ((normxcorr2 (a,b_shift1) .- normxcorr2 (a,b))(:)), 0, 1e-10);
%! assert (max ((normxcorr2 (a_shift2,b) .- normxcorr2 (a,b))(:)), 0, 1e-10);
%! assert (max ((normxcorr2 (a,b_shift2) .- normxcorr2 (a,b))(:)), 0, 1e-10);

## test n dimensional input
%!test
%! a = randi (100, 15, 15, 15);
%! c = normxcorr2 (a(5:10, 2:6, 3:7), a);
%! assert (get_max_offsets (c), {10 6 7});
%!
%! a = randi (100, 15, 15, 15);
%! c = normxcorr2 (a(5:10, 2:6, 1:1), a);
%! assert (get_max_offsets (c), {10 6 1});

%!warning <swapped> normxcorr2 (rand (20), rand (5));
%!error normxcorr2 (rand (5));
%!error normxcorr2 (rand (5), rand (20), 2);

## test for no small imaginary parts in result (bug #46160)
%!test
%! a =  [ 252  168   50    1   59;
%!        114    0    0    0    0] ./ 255;
%! b = [    1  171  255  255  255  255  240   71  131  254  255  255  255;
%!          0  109  254  255  255  233   59    0  131  254  255  255  255;
%!         76   13  195  253  194   34    0   19  217  255  255  255  255;
%!        110    0    0    0    0    0    3  181  255  255  255  255  255;
%!        153    0    0    0    0    2  154  254  255  255  255  255  255]./255;
%!  c = normxcorr2 (a, b);
%! assert (max (imag (c(:))), 0);

## test for inf results on image regions with constant value (bug #50122)
%!test
%! img = [1 1 1 0];
%! t = [1 1 0];
%! c = normxcorr2 (t, img);
%! assert (c(3), 0)
