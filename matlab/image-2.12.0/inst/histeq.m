## Copyright (C) 2000 Kai Habel <kai.habel@gmx.de>
## Copyright (C) 2008 Jonas Wagner <j.b.w@gmx.ch>
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
## @deftypefn {Function File} {@var{J} =} histeq (@var{I}, @var{n})
## Equalize histogram of grayscale image.
##
## The histogram contains
## @var{n} bins, which defaults to 64.
##
## @var{I}: Image in double format, with values from 0.0 to 1.0.
##
## @var{J}: Returned image, in double format as well.
##
## Note that the algorithm used for histogram equalization gives results
## qualitatively comparable but numerically different from @sc{matlab}
## implementation.
##
## @seealso{imhist, mat2gray, brighten}
## @end deftypefn

function J = histeq (I, n = 64)

  if (nargin < 1 || nargin > 3)
    print_usage ();
  endif

  if (isempty (I))
    J = [];
    return
  endif

  [r, c]   = size (I);
  I        = mat2gray (I);
  [X, map] = gray2ind (I, n);
  [nn, xx] = imhist (I, n);
  Icdf     = 1 / prod (size (I)) * cumsum (nn);
  J        = reshape (Icdf(X + 1), r, c);
endfunction

## FIXME: the method we are using is different from Matlab so our results
##        are slightly different. The following xtest show the Matlab
##        results that we should be aiming at.

%!assert (histeq ([]), []);

## One value
%!assert (histeq (0), 1);
%!assert (histeq (1), 1);
%!assert (histeq (1.5), 1);
%!assert (histeq (zeros (100, 200)), ones (100, 200));            # matrix

## Two values
%!xtest assert (histeq ([0    1]),  [0.4920634921  1],  10^-8);
%!xtest assert (histeq ([0    1]'), [0.4920634921  1]', 10^-8);   # column array
%!xtest assert (histeq ([0  255]),  [0.4920634921  1],  10^-8);
%!xtest assert (histeq (uint8  ([0      1])), [  125    190]);    # uint8
%!xtest assert (histeq (uint8  ([0    255])), [  125    255]);
%!xtest assert (histeq (uint16 ([0      1])), [65535  65535]);    # uint16
%!xtest assert (histeq (uint16 ([0    255])), [32247  48891]);
%!xtest assert (histeq (uint16 ([0    256])), [32247  48891]);
%!xtest assert (histeq (uint16 ([0  65535])), [32247  65535]);

## Three values
%!test assert (histeq  ([0 1 1] ),             [  1/3     1     1] ,  10^-8);
%!test assert (histeq  ([0 0 1]'),             [  2/3   2/3     1]',  10^-8);
%!xtest assert (histeq ([0 1 2] ),             [  1/3     1     1] ,  10^-8);
%!xtest assert (histeq (uint8  ([0   1   2])), [   85   125   215]);
%!xtest assert (histeq (uint16 ([0   1   2])), [65535 65535 65535]);
%!xtest assert (histeq (uint16 ([0 100 200])), [43690 43690 55133]);

## Many values
%!xtest
%! J = [20    32    57    81   105   125   150   174   198   223   247];
%! assert (histeq (uint8 (0:10:100)), J);

%!xtest
%! J = [0.0793650794
%!      0.1269841270
%!      0.2222222222
%!      0.3174603175
%!      0.4126984127
%!      0.4920634921
%!      0.5873015873
%!      0.6825396825
%!      0.7777777778
%!      0.8730158730
%!      1.0000000000];
%! assert (histeq (0:0.1:1), J', 10^-8);
