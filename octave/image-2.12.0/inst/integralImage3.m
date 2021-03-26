## Copyright (C) 2019 Avinoam Kalma <a.kalma@gmail.com>
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <https://www.gnu.org/licenses/>.

## -*- texinfo -*-
## @deftypefn  {Function File} {} integralImage (@var{img})
## Calculate the 3D integral image.
##
## @var{img} is the input image for 3D integral image calculation.
##
## The value of the 3D integral image is J = cumsum (cumsum (cumsum (I), 2), 3)
## with padding.
## The padding adds a zeros plane, zero rows and zero column, so size (J) = size (I) + 1.
##
## @seealso{integralImage, cumsum}
## @end deftypefn

function J = integralImage3 (I)

  if (nargin != 1)
    print_usage ();
endif

  if (! isimage (I))
    error ("integralImage3: I should be an image");
  endif

  if (ndims (I) > 3)
    error ("integralImage3: I should be a 3-dimensional image");
  endif

  if (! isa (I, "double"))
    I = double (I);
  endif

  J = cumsum (cumsum (cumsum (I), 2), 3);
  J = padarray (J, [1 1 1], "pre");

endfunction

%!test
%! assert (integralImage3 (zeros (4)), zeros (5, 5, 2));

%!test
%! J_res = zeros (2, 2, 2);
%! J_res(2, 2, 2) = 10;
%! assert (integralImage3 (10), J_res);

%!test
%! J = integralImage3 (10);
%! assert (class (J), "double");
%! J = integralImage3 (uint8 (10));
%! assert (class (J), "double");

%!test
%! I = [1, 2; 3, 4];
%! J = integralImage3 (I);
%! J_res = zeros (3, 3, 2);
%! J_res(2:3, 2:3, 2) = [1 3; 4 10];
%! assert (J, J_res)

%!test
%! I1 = [1, 2; 3, 4];
%! I2 = [5, 6; 7, 8];
%! I3 = [9, 10; 11, 12];
%! I = cat (3, I1, I2, I3);
%! J = integralImage3 (I);
%! J2 = [0 0 0; 0 1 3; 0 4 10];
%! J3 = [0 0 0; 0 6 14; 0 16 36];
%! J4 = [0 0 0; 0 15 33; 0 36 78];
%! J_res = cat (3, zeros (3), J2, J3, J4);
%! assert (J, J_res)

%!test
%! I = magic (5);
%! J = integralImage3 (I);
%! J_res = zeros (6, 6, 2);
%! J_res(:, :, 2) = [0   0   0  0    0   0;
%!                    0  17  41 42   50  65;
%!                    0  40  69 77   99 130;
%!                    0  44  79 100 142 195;
%!                    0  54 101 141 204 260;
%!                    0  65 130 195 260 325];
%! assert (J, J_res)

%!# test of 3d input image:

%!test
%! K = magic (8);
%! K = reshape (K, [4 4 4]);
%! L = integralImage3 (K);
%! L1_ML = zeros (5);
%! L2_ML = [0 0 0 0 0;
%!    0 64 96 98 132;
%!    0 73 146 203 260;
%!    0 90 212 316 388;
%!    0 130 260 390 520];
%! L3_ML = [0 0 0 0 0;
%!   0 67 134 197 260;
%!   0 130 260 390 520;
%!   0 193 386 583 780;
%!   0 260 520 780 1040];
%! L4_ML = [0 0 0 0 0;
%!   0 127 222 291 392;
%!   0 203 406 593 780;
%!   0 287 606 903 1168;
%!   0 390 780 1170 1560];
%! L5_ML = [0 0 0 0 0;
%!   0 134 268 394 520;
%!   0 260 520 780 1040;
%!   0 386 772 1166 1560;
%!   0 520 1040 1560 2080];
%! L_ML = cat (3, L1_ML, L2_ML, L3_ML, L4_ML, L5_ML);
%! assert (L, L_ML)

%!# test of 2d input image:
%!test
%! X = ones (3);
%! Y = integralImage3 (X);
%! Y_ML = zeros (4, 4, 2);
%! Y_ML(:, :, 2) = [0 0 0 0; 0 1 2 3; 0 2 4 6; 0 3 6 9];
%! assert(Y, Y_ML);

%!error <Invalid call to integralImage3>
%! integralImage3 ();

%!error <Invalid call to integralImage3>
%! integralImage3 (zeros (3), zeros (3));

%!error <integralImage3: I should be an image>
%! integralImage3 ("abcd");

%!error <integralImage3: I should be an image>
%! integralImage3 (1+i);

%!error <integralImage3: I should be a 3-dimensional image>
%! integralImage3 (reshape (1:81, 3, 3, 3, 3));
