// Copyright (C) 2004 Stefan van der Walt <stefan@sun.ac.za>
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     1 Redistributions of source code must retain the above copyright notice,
//       this list of conditions and the following disclaimer.
//     2 Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ''AS IS''
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <octave/oct.h>

DEFUN_DLD(hough_line, args, , "\
-*- texinfo -*-\n\
@deftypefn {Loadable Function} {[@var{H}, @var{R}] =} hough_line(@var{I}, @var{angles})\n\
Calculate the straight line Hough transform of a binary image @var{I}.\n\
\n\
The angles are given in radians and default to -pi/2:pi/2.\n\
\n\
@var{H} is the resulting Hough transform, and @var{R} is the radial distances.\n\
\n\
The algorithm is described in\n\
Digital Image Processing by Gonzales & Woods (2nd ed., p. 587)\n\
\n\
For a Matlab compatible Hough transform see hough.m\n\
@end deftypefn\n\
")
{
  octave_value_list retval;
  const int nargin = args.length ();
  const bool DEF_THETA = (nargin == 1);

  if (nargin < 1 || nargin > 2)
    print_usage ();

  const Matrix I = args (0).matrix_value ();
  const ColumnVector thetas = (DEF_THETA) ? ColumnVector (Range (-M_PI/2.0, M_PI/2.0, M_PI/180.0).matrix_value ()) 
                                          : ColumnVector (args (1).vector_value ());

  const int r = I.rows ();
  const int c = I.columns ();
  const int thetas_length = thetas.numel ();

  const double diag_length = sqrt ((r-1)*(r-1) + (c-1)*(c-1));
  const int nr_bins = 2 * (int)ceil (diag_length) + 1;
  RowVector bins = RowVector (Range(1, nr_bins).matrix_value ()) - ceil (nr_bins/2.0);
  const int bins_length = bins.numel ();

  Matrix J (bins_length, thetas_length, 0.0);

  for (int i = 0; i < thetas_length; i++)
    {
      const double theta = thetas (i);

      const double cT = cos (theta);
      const double sT = sin (theta);
      for (int x = 0; x < r; x++)
        {
          for (int y = 0; y < c; y++)
            {
              if (I(x, y) == 1)
                {
                  const int rho = (int)floor (cT*x + sT*y + 0.5);
                  const int bin = (int)(rho - bins (0));
                  if ((bin > 0) && (bin < bins_length))
                    J (bin, i)++;
                }
            }
        }
    }

  retval.append (J);
  retval.append (bins);
  return retval;
}

/*
%!test
%! I = zeros(100, 100);
%! I(1,1) = 1; I(100,100) = 1; I(1,100) = 1; I(100, 1) = 1; I(50,50) = 1;
%! [J, R] = houghtf(I); J = J / max(J(:));
%! assert(size(J) == [length(R) 181]);
%!

%!demo
%! I = zeros(100, 150);
%! I(30,:) = 1; I(:, 65) = 1; I(35:45, 35:50) = 1;
%! for i = 1:90, I(i,i) = 1;endfor
%! I = imnoise(I, 'salt & pepper');
%! imshow(I);
%! J = houghtf(I); J = J / max(J(:));
%! imshow(J, bone(128), 'truesize');

*/
