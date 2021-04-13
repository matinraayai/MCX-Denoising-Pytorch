// Copyright (C) 2008 Søren Hauberg <soren@hauberg.org>
//
// This program is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation; either version 3 of the License, or (at your option) any later
// version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
// details.
//
// You should have received a copy of the GNU General Public License along with
// this program; if not, see <http://www.gnu.org/licenses/>.

#include <octave/oct.h>

template <class MT> 
MT
custom_gaussian_smoothing (const MT &I, const Matrix &lambda1, const Matrix &lambda2,
                           const Matrix &theta)
{
  const octave_idx_type rows = I.rows ();
  const octave_idx_type cols = I.columns ();

  // Allocate output
  MT J (I.dims ());

  // Iterate over every element of 'I'
  for (octave_idx_type row = 0; row < rows; row++)
    {
      for (octave_idx_type col = 0; col < cols; col++)
        {
          // Extract parameters
          const double v1 = lambda1 (row, col);
          const double v2 = lambda2 (row, col);
          const double t  = theta (row, col);
          
          // Should we perform any filtering?
          if (std::min (v1, v2) > 0)
            {
              // Compute inverse covariance matrix, C^-1 = [a, b; b, c]
              const double iv1 = 1.0/v1;
              const double iv2 = 1.0/v2;
              const double ct = cos (t);
              const double st = sin (t);
              const double ct2 = ct*ct;
              const double st2 = st*st;
              const double ctst = ct*st;
              const double a = ct2*iv2 + st2*iv1;
              const double b = (iv2-iv1)*ctst;
              const double c = st2*iv2 + ct2*iv1;
              
              // Compute bounding box of the filter
              const double k = 3.0; // The maximally allowed Mahalanobis' distance
              const double sqrtv1 = sqrt (v1);
              const double sqrtv2 = sqrt (v2);
              const octave_idx_type rur = abs (k*(ct*sqrtv2 - st*sqrtv1)); // 'rur' means 'row-upper-right'
              const octave_idx_type cur = abs (k*(st*sqrtv2 + ct*sqrtv1));
              const octave_idx_type rlr = abs (k*(ct*sqrtv2 + st*sqrtv1));
              const octave_idx_type clr = abs (k*(st*sqrtv2 - ct*sqrtv1));
              const octave_idx_type rul = abs (k*(-ct*sqrtv2 - st*sqrtv1));
              const octave_idx_type cul = abs (k*(-st*sqrtv2 + ct*sqrtv1));
              const octave_idx_type rll = abs (k*(-ct*sqrtv2 + st*sqrtv1));
              const octave_idx_type cll = abs (k*(-st*sqrtv2 - ct*sqrtv1));
              const octave_idx_type r_delta = std::max (std::max (rur, rlr), std::max (rul, rll));
              const octave_idx_type c_delta = std::max (std::max (cur, clr), std::max (cul, cll));;
              // The bounding box is now (row-r_delta):(row+r_delta)x(col-c_delta):(col+c_delta).
              // We, however, represent the bounding box in a local coordinate system around (row, col).
              const octave_idx_type r1 = std::max (row-r_delta, octave_idx_type (0)) - row;
              const octave_idx_type r2 = std::min (row+r_delta, rows-1) - row;
              const octave_idx_type c1 = std::max (col-c_delta, octave_idx_type (0)) - col;
              const octave_idx_type c2 = std::min (col+c_delta, cols-1) - col;
              
              // Perform the actual filtering
              double sum = 0;
              double wsum = 0; // for normalisation
              for (octave_idx_type rl = r1; rl <= r2; rl++)
                {
                  for (octave_idx_type cl = c1; cl <= c2; cl++)
                    {
                      // Compute Mahalanobis' distance
                      const double dsquare = rl*(a*rl + b*cl) + cl*(b*rl + c*cl);
                      
                      // We only do the filtering in an elliptical window
                      if (dsquare > k*k) continue;
                      
                      // Update filter values
                      const double w = exp (-0.5*dsquare);
                      wsum += w;
                      sum += w*(double)I.elem (row + rl, col + cl);
                    } // End: cl
                } // End: rl
              
              // Compute final result
              J (row, col) = sum/wsum;
            }
          else // No filtering is performed
            {
              J.elem (row, col) = I.elem (row, col);
            }
        } // End: column iteration
    } // End: row iteration
    
  // Return
  return J;
}

DEFUN_DLD (__custom_gaussian_smoothing__, args, ,"\
-*- texinfo -*-\n\
@deftypefn {Loadable Function} {@var{J} =} __custom_gaussian_smooting__ (@var{I}, @var{lambda1}, @var{lambda2}, @var{theta})\n\
Performs Gaussian smoothing on the image @var{I}. In pixel @math{(r,c)} the \n\
Eigenvalues of the Gaussian is @var{lambda1}@math{(r,c)} and @var{lambda2}@math{(r,c)}.\n\
The Gaussian is rotated with the angle given in @var{theta}@math{(r,c)}.\n\
\n\
@strong{Warning:} this function should @i{never} be called directly! The user\n\
interface to this function is available in @code{imsmooth}.\n\
@seealso{imsmooth}\n\
@end deftypefn\n\
")
{
  // Handle Input
  octave_value_list retval;
  const int nargin = args.length ();

  if (nargin != 4)
    print_usage ();

  const Matrix lambda1 = args (1).matrix_value ();
  const Matrix lambda2 = args (2).matrix_value ();
  const Matrix theta   = args (3).matrix_value ();

  const octave_idx_type rows = args (0).rows();
  const octave_idx_type cols = args (0).columns();
  if (lambda1.rows () != rows || lambda1.columns () != cols
      || lambda2.rows () != rows || lambda2.columns () != cols
      || theta.rows () != rows || theta.columns () != cols)
    error ("__custom_gaussian_smoothing__: size mismatch");

  // Take action depending on input type
  //octave_value J;
  if (args(0).is_real_matrix())
    {
      const Matrix I = args(0).matrix_value();
      retval.append (custom_gaussian_smoothing<Matrix>(I, lambda1, lambda2, theta));
    } 
  else if (args(0).is_int8_type())
    {
      const int8NDArray I = args(0).int8_array_value();
      retval.append (custom_gaussian_smoothing<int8NDArray>(I, lambda1, lambda2, theta));
    } 
  else if (args(0).is_int16_type())
    {
      const int16NDArray I = args(0).int16_array_value();
      retval.append (custom_gaussian_smoothing<int16NDArray>(I, lambda1, lambda2, theta));
    } 
  else if (args(0).is_int32_type())
    {
      const int32NDArray I = args(0).int32_array_value();
      retval.append (custom_gaussian_smoothing<int32NDArray>(I, lambda1, lambda2, theta));
    } 
  else if (args(0).is_int64_type())
    {
      const int64NDArray I = args(0).int64_array_value();
      retval.append (custom_gaussian_smoothing<int64NDArray>(I, lambda1, lambda2, theta));
    } 
  else if (args(0).is_uint8_type())
    {
      const uint8NDArray I = args(0).uint8_array_value();
      retval.append (custom_gaussian_smoothing<uint8NDArray>(I, lambda1, lambda2, theta));
    } 
  else if (args(0).is_uint16_type())
    {
      const uint16NDArray I = args(0).uint16_array_value();
      retval.append (custom_gaussian_smoothing<uint16NDArray>(I, lambda1, lambda2, theta));
    } 
  else if (args(0).is_uint32_type())
    {
      const uint32NDArray I = args(0).uint32_array_value();
      retval.append (custom_gaussian_smoothing<uint32NDArray>(I, lambda1, lambda2, theta));
    } 
  else if (args(0).is_uint64_type())
    {
      const uint64NDArray I = args(0).uint64_array_value();
      retval.append (custom_gaussian_smoothing<uint64NDArray>(I, lambda1, lambda2, theta));
    } 
  else
    error("__custom_gaussian_smoothing__: first input should be a real or integer array");

  return retval;
}
