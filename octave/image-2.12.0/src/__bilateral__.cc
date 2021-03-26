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

#include <vector>

#include <octave/oct.h>

inline
double gauss (const std::vector<double> x, const std::vector<double> mu,
              const double sigma)
{
  double s = 0;
  for (size_t i = 0; i < x.size (); i++)
    {
      const double d = x[i] - mu[i];
      s += d*d;
    }
  return exp (-0.5*s/(sigma*sigma));
}

template <class MatrixType> 
octave_value
bilateral (const MatrixType &im, const double sigma_d, const double sigma_r)
{
  // Get sizes
  const octave_idx_type ndims = im.ndims ();
  const dim_vector size = im.dims ();
  const octave_idx_type num_planes = (ndims == 2) ? 1 : size (2);
  
  // Build spatial kernel
  const int s = std::max ((int)std::round (3*sigma_d), 1);
  Matrix kernel (2*s+1, 2*s+1);
  for (octave_idx_type r = 0; r < 2*s+1; r++)
    {
      for (octave_idx_type c = 0; c < 2*s+1; c++)
        {
          const int dr = r-s;
          const int dc = c-s;
          kernel (r,c) = exp (-0.5 * (dr*dr + dc*dc)/(sigma_d*sigma_d));
        }
    }
  
  // Allocate output
  dim_vector out_size (size);
  out_size (0) = std::max (size (0) - 2*s, (octave_idx_type)0);
  out_size (1) = std::max (size (1) - 2*s, (octave_idx_type)0);
  MatrixType out = MatrixType (out_size);

  // Iterate over every element of 'out'.
  for (octave_idx_type r = 0; r < out_size (0); r++)
    {
      for (octave_idx_type c = 0; c < out_size (1); c++)
        {
          OCTAVE_QUIT;

          // For each neighbour
          std::vector<double> val (num_planes);
          std::vector<double> sum (num_planes);
          double k = 0;
          for (octave_idx_type i = 0; i < num_planes; i++)
            {
              val[i] = im (r+s,c+s,i);
              sum[i] = 0;
            }
          for (octave_idx_type kr = 0; kr < 2*s+1; kr++)
            {
              for (octave_idx_type kc = 0; kc < 2*s+1; kc++)
                {
                  std::vector<double> lval (num_planes);
                  for (octave_idx_type i = 0; i < num_planes; i++)
                    lval[i] = im (r+kr, c+kc, i);
                  const double w = kernel (kr, kc) * gauss (val, lval, sigma_r);
                  for (octave_idx_type i = 0; i < num_planes; i++)
                    sum[i] += w * lval[i];
                  k += w;
                }
            }
          for (octave_idx_type i = 0; i < num_planes; i++)
            out (r, c, i) = sum[i]/k;
        }
    }
    
  return octave_value (out);
}

DEFUN_DLD (__bilateral__, args, , "\
-*- texinfo -*-\n\
@deftypefn {Loadable Function} __bilateral__(@var{im}, @var{sigma_d}, @var{sigma_r})\n\
Performs Gaussian bilateral filtering in the image @var{im}. @var{sigma_d} is the\n\
spread of the Gaussian used as closenes function, and @var{sigma_r} is the spread\n\
of Gaussian used as similarity function. This function is internal and should NOT\n\
be called directly. Instead use @code{imsmooth}.\n\
@end deftypefn\n\
")
{
  octave_value_list retval;
  if (args.length () != 3)
    print_usage ();

  const octave_idx_type ndims = args (0).ndims ();
  if (ndims != 2 && ndims != 3)
    error ("__bilateral__: only 2 and 3 dimensional is supported");

  const double sigma_d = args (1).scalar_value ();
  const double sigma_r = args (2).scalar_value ();

  // Take action depending on input type
  if (args (0).is_real_matrix ())
    {
      const NDArray im = args(0).array_value ();
      retval = bilateral<NDArray> (im, sigma_d, sigma_r);
    } 
  else if (args (0).is_int8_type ())
    {
      const int8NDArray im = args (0).int8_array_value ();
      retval = bilateral<int8NDArray> (im, sigma_d, sigma_r);
    } 
  else if (args (0).is_int16_type ())
    {
      const int16NDArray im = args (0).int16_array_value ();
      retval = bilateral<int16NDArray> (im, sigma_d, sigma_r);
    } 
  else if (args (0).is_int32_type ())
    {
      const int32NDArray im = args (0).int32_array_value ();
      retval = bilateral<int32NDArray> (im, sigma_d, sigma_r);
    } 
  else if (args (0).is_int64_type ())
    {
      const int64NDArray im = args (0).int64_array_value ();
      retval = bilateral<int64NDArray> (im, sigma_d, sigma_r);
    } 
  else if (args (0).is_uint8_type ())
    {
      const uint8NDArray im = args (0).uint8_array_value ();
      retval = bilateral<uint8NDArray> (im, sigma_d, sigma_r);
    } 
  else if (args(0).is_uint16_type())
    {
      const uint16NDArray im = args (0).uint16_array_value ();
      retval = bilateral<uint16NDArray> (im, sigma_d, sigma_r);
    } 
  else if (args (0).is_uint32_type ())
    {
      const uint32NDArray im = args (0).uint32_array_value ();
      retval = bilateral<uint32NDArray> (im, sigma_d, sigma_r);
    } 
  else if (args (0).is_uint64_type ())
    {
      const uint64NDArray im = args (0).uint64_array_value ();
      retval = bilateral<uint64NDArray> (im, sigma_d, sigma_r);
    } 
  else
    error ("__bilateral__: first input should be a real or integer array");

  return retval;
}
