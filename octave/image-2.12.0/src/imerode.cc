// Copyright (C) 2013 Carnë Draug <carandraug@octave.org>
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

#include <string>
#include <typeinfo> // for an optimization using logical matrices

#include <octave/Array.h>
#include <octave/Array-util.h>  // to get ind2sub
#include <octave/boolNDArray.h>
#include <octave/dim-vector.h>
#include <octave/lo-ieee.h>  // octave_Inf

#include <octave/defun-dld.h>
#include <octave/defun-int.h>
#include <octave/error.h>
#include <octave/ovl.h>

#include "strel.h"

#define WANTS_OCTAVE_IMAGE_VALUE 1
#include "octave-wrappers.h"

using namespace octave::image;

// How this works:
//
// Erosion and dilation are simply minimum and maximum filters (in case of
// dilation, the filter needs to be reflected). We have a binary matrix as
// Structuring Element (SE) which slides through the image, and using the
// maximum or minimum value of those elements for the output matrix. The
// border of the matrix is considered to be +Inf or -Inf for the minimum
// (erosion) and maximum (dilation) respectively.
//
// We start by padding the input matrix accordingly to the requested shape
// (maybe this step could be avoided by doing the filtering in some different
// method around the borders of the matrix0.
//
// For performance (so we can use a pointer to access the data) while
// supporting ND matrices, we calculate the offset for all the points in the
// input matrix that affects a single point in the output. Note that as we
// slide through this offset values will always be the same.
//
// We could implement something more close to convn which is quite efficient
// but that requires to go through every element of the SE which would be a
// waste because not all elements in the SE will be true. Anyway, at least for
// binary images (and convn can only be used to do erosion and dilation of
// binary images), we already perform faster.

// Pads the matrix MT with PADVAL, so it has the correct size to perform a
// spatial filtering with SE, for the requested SHAPE. The SHAPE argument
// is the same as in convn().
// FIXME: apparently it is not the same as convn(). Matlab seems to have
//        changed how this is done and will trim the SE, effectively changing
//        what its origin is. For example, requesting full erosion with the
//        following SE's will return the same
//
//              0 0 0
//              0 0 1         0 1
//              0 1 1         1 1
//
//        because in the first case, the first column and row are ignored. This
//        means that the size of output for full erosion will differ depending
//        on the SE.
template <class T>
static T
pad_matrix (const T& mt, const strel& se,
            const double& padval, const std::string& shape)
{
  // If the shape is valid, we can return the input matrix.
  if (shape == "valid")
    return mt;

  const octave_idx_type ndims = mt.ndims ();
  const Array<octave_idx_type> pre_pad  = se.pre_pad  (ndims, shape);
  const Array<octave_idx_type> post_pad = se.post_pad (ndims, shape);

  dim_vector padded_size (mt.dims ());
  for (octave_idx_type dim = 0; dim < ndims; dim++)
    padded_size(dim) += pre_pad(dim) + post_pad(dim);
  T padded (padded_size, padval);

  // Ammount of pre_pad is also how much the original must be shifted
  // when inserting into the new padded matrix.
  padded.insert (mt, pre_pad);

  return padded;
}

// The general idea about the following is to look at each point for the
// output, one at a time, and evaluate all the points from the input. This
// at least allows us to skip many points in the case of binary images. For
// each output point we consider the one with same index in the input as
// "under" the SE element with index 0, and shift from that point to all the
// others. Then we move to the next point of output.
//
// SE:
//    0 1 1
//
// Input in:
//  0 1 0 0 1 0 0 1 0 1 1 0 0 1 1 1 1 1 0 0 1 0
//
// Input out:
//    1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
//
// Output:
//    0 0 0 0 0 0 0 1 0 0 0 1 1 1 1 0 0 0 0 0
//
// Note that the output is shorter in size since we have already padded the
// input as appropriate for the requested shape. When we slide the SE over
// the input, its center (origin) shows what value will be on the output. But
// we won't actually use the center of the SE, only the first element and the
// distance from it. This means that in this example, the NNZ elements of the
// SE will have an offset of 1 and 2.
// We match the first element of output with the first from output, and shift
// their offset values to move to all other points in the input and assign
// them to the output.
//
// To deal with N dimensional images we have the cumulative dimensions of the
// input matrix, e.g. for 10x20x4x5 matrix, this would be the array
// [10 200 800 4000]. This is how much we must shift the pointer in the input
// matrix to get to the next value in a specific dimension. For example, to get
// to the second column we would add 10*(2-1) to the input matrix. To get to
// element 4 of the 3rd dimension we would add 200*(4-3). So we use this with
// recursion, and adding to the pointer of the input matrix until we only
// have a column to erode).

// The values of erosion and isflat come as template since they are checked
// at the deepest of the loop. By using a template instead of function
// argument, it's all done at compile time so we get better performance.
// If erosion is false, we perform dilation instead
template<class P, bool erosion, bool flat>
inline static void
erode_line (const P* in, P* out, const octave_idx_type* offsets, const P* height,
            const octave_idx_type& nnz, const octave_idx_type& line_length)
{
  for (octave_idx_type line_idx = 0; line_idx < line_length; line_idx++)
    {
      for (octave_idx_type nnz_idx = 0; nnz_idx < nnz; nnz_idx++)
        {
          if (flat)
            {
              if (erosion)
                {
                  if (in[offsets[nnz_idx]] < out[line_idx])
                    out[line_idx] = in[offsets[nnz_idx]];
                }
              else
                {
                  if (in[offsets[nnz_idx]] > out[line_idx])
                    out[line_idx] = in[offsets[nnz_idx]];
                }
            }
          else
            {
              // If non-flat, there is no need to check if typeid is boolean
              // since non-flat makes no sense for binary images.
              if (erosion)
                {
                  P val = in[offsets[nnz_idx]] - height[nnz_idx];
                  if (val < out[line_idx])
                    out[line_idx] = val;
                }
              else
                {
                  P val = in[offsets[nnz_idx]] + height[nnz_idx];
                  if (val > out[line_idx])
                    out[line_idx] = val;
                }
            }
        }
      in++;
    }
}

// For the specific case of boolean dilation/erosion, we may be able to
// break from the loop sooner.  Also, there is non-flat binary erosion
// and dilation.
template<class P, bool erosion, bool flat>
inline static void
erode_line (const bool* in, bool* out, const octave_idx_type* offsets, const bool* height,
            const octave_idx_type& nnz, const octave_idx_type& line_length)
{
  for (octave_idx_type line_idx = 0; line_idx < line_length; line_idx++)
    {
      for (octave_idx_type nnz_idx = 0; nnz_idx < nnz; nnz_idx++)
        {
          if (erosion)
            {
              if (! in[offsets[nnz_idx]])
                {
                  out[line_idx] = false;
                  break;
                }
            }
          else
            {
              if (in[offsets[nnz_idx]])
                {
                  out[line_idx] = true;
                  break;
                }
            }
        }
      in++;
    }
}

template<class P, bool erosion, bool flat>
static void
erode_nd (const P* in, const dim_vector& in_cd,
          P* out, const dim_vector& out_cd, const dim_vector& out_d,
          const octave_idx_type* offsets, const P* height,
          const octave_idx_type& nnz, const octave_idx_type& dim)
{
  if (dim == 0)
    erode_line<P, erosion, flat> (in, out, offsets, height, nnz, out_d(0));
  else
    for (octave_idx_type elem = 0; elem < out_d(dim); elem++)
      erode_nd<P, erosion, flat> (in  + in_cd(dim-1) * elem, in_cd,
                                  out + out_cd(dim-1)* elem, out_cd, out_d,
                                  offsets, height, nnz, dim -1);
  OCTAVE_QUIT;
}

template<class T>
static octave_value
erode (const T& im, const strel& se, const std::string& shape, const bool& erosion)
{
  typedef typename T::element_type P;

  // If image is empty, return empty of the same class.
  if (octave_image::value (im).isempty ())
    return octave_value (im);

  // In the case of floating point, complex and integers numbers, both
  // octave_Inf and -octave_Inf actually become that type max and min value.
  // However, for boolean, both of them are converted to "true" so for
  // dilation, where we want false, we check the type.
  T padded;
  if (erosion)
    padded = pad_matrix<T> (im, se, octave_Inf, shape);
  else
    {
      if (typeid (P) == typeid (bool))
        padded = pad_matrix<T> (im, se, false, shape);
      else
        padded = pad_matrix<T> (im, se, -octave_Inf, shape);
    }

  const boolNDArray nhood = se.get_nhood ();

  const octave_idx_type ndims   = padded.ndims ();
  const dim_vector nhood_size   = nhood.dims ().redim (ndims);
  const dim_vector padded_size  = padded.dims ();

  const dim_vector cum_size = padded_size.cumulative ();
  const Array<octave_idx_type> offsets = se.offsets (cum_size);

  const Array<P> heights = se.true_heights<P> ();

  const bool flat = se.flat ();

  if (typeid (P) == typeid (bool) && ! flat)
    error ("only non flat structuring elements for binary images");

  dim_vector out_size (padded_size);
  for (octave_idx_type i = 0; i < ndims; i++)
    out_size(i) -= nhood_size(i) - 1;

  T out;

  // When there's only a single neighbor on the SE, then we will only shift
  // the matrix by its distance to the origin of the SE.
  if (se.get_nnz () == 1)
    {
      octave_idx_type ind = nhood.find (1)(0);
      Array<idx_vector> sub = ind2sub (nhood_size, idx_vector (ind));

      Array<idx_vector> ranges (dim_vector (ndims, 1));
      for (octave_idx_type dim = 0; dim < ndims; dim++)
        {
          octave_idx_type start (sub(dim)(0));
          octave_idx_type limit (start + out_size(dim));
          ranges(dim) = idx_vector (start, limit);
        }
      out = padded.index (ranges);
    }
  else
    {
      if (erosion)
        out = T (out_size, octave_Inf);
      else
        if (typeid (P) == typeid (bool))
          out = T (out_size, false);
        else
          out = T (out_size, -octave_Inf);

      if (flat)
        if (erosion)
          erode_nd<P, true, true> (padded.data (), cum_size, out.fortran_vec (),
            out_size.cumulative (), out_size, offsets.data (), heights.data (),
            offsets.numel (), ndims -1);
        else
          erode_nd<P, false, true> (padded.data (), cum_size, out.fortran_vec (),
            out_size.cumulative (), out_size, offsets.data (), heights.data (),
            offsets.numel (), ndims -1);

      else
        if (erosion)
          erode_nd<P, true, false> (padded.data (), cum_size, out.fortran_vec (),
            out_size.cumulative (), out_size, offsets.data (), heights.data (),
            offsets.numel (), ndims -1);
        else
          erode_nd<P, false, false> (padded.data (), cum_size, out.fortran_vec (),
            out_size.cumulative (), out_size, offsets.data (), heights.data (),
            offsets.numel (), ndims -1);
    }
  return octave_value (out);
}

static octave_value
base_action (const std::string& func, const bool& erosion, const octave_value_list& args)
{
  octave_value retval;
  const octave_idx_type nargin = args.length ();
  if (nargin < 2 || nargin > 4)
    print_usage (func);

  // Default shape is "same"
  const std::string shape = nargin > 2? args(2).string_value () : "same";

  strel se (args(1));
  if (! erosion) // must be dilation, then get the se reflection
    se = se.reflect ();

  octave_image::value im (args(0));
  for (octave_idx_type idx = 0; idx < se.numel (); idx++)
    {
      const strel se_elem = se(idx);
      if (im.islogical ())
        im = erode<boolNDArray> (im.bool_array_value (), se_elem, shape, erosion);
      else if (im.is_int8_type ())
        im = erode<int8NDArray> (im.int8_array_value (), se_elem, shape, erosion);
      else if (im.is_int16_type ())
        im = erode<int16NDArray> (im.int16_array_value (), se_elem, shape, erosion);
      else if (im.is_int32_type ())
        im = erode<int32NDArray> (im.int32_array_value (), se_elem, shape, erosion);
      else if (im.is_int64_type ())
        im = erode<int64NDArray> (im.int64_array_value (), se_elem, shape, erosion);
      else if (im.is_uint8_type ())
        im = erode<uint8NDArray> (im.uint8_array_value (), se_elem, shape, erosion);
      else if (im.is_uint16_type ())
        im = erode<uint16NDArray> (im.uint16_array_value (), se_elem, shape, erosion);
      else if (im.is_uint32_type ())
        im = erode<uint32NDArray> (im.uint32_array_value (), se_elem, shape, erosion);
      else if (im.is_uint64_type ())
        im = erode<uint64NDArray> (im.uint64_array_value (), se_elem, shape, erosion);
      else if (im.isreal ())
        if (im.is_single_type ())
          im = erode<FloatNDArray> (im.float_array_value (), se_elem, shape, erosion);
        else // must be double
          im = erode<NDArray> (im.array_value (), se_elem, shape, erosion);
      else if (im.iscomplex ())
        if (im.is_single_type ())
          im = erode<FloatComplexNDArray> (im.float_complex_array_value (), se_elem, shape, erosion);
        else // must be double
          im = erode<ComplexNDArray> (im.complex_array_value (), se_elem, shape, erosion);
      else
        im = octave_value ();
    }

  return im;
}

DEFUN_DLD(imerode, args, , "\
-*- texinfo -*-\n\
@deftypefn  {Loadable Function} {} imerode (@var{im}, @var{SE})\n\
@deftypefnx {Loadable Function} {} imerode (@var{im}, @var{SE}, @var{shape})\n\
Perform morphological erosion.\n\
\n\
The image @var{im} must be a numeric matrix with any number of dimensions.\n\
The erosion is performed with the structuring element @var{se} which can\n\
be a:\n\
\n\
@itemize @bullet\n\
@item strel object;\n\
@item array of strel objects as returned by @code{@@strel/getsequence};\n\
@item matrix of 0's and 1's.\n\
@end itemize\n\
\n\
To perform a non-flat erosion, @var{SE} must be a strel object.\n\
\n\
The size of the result is determined by the optional @var{shape} argument\n\
which takes the following values:\n\
\n\
@table @asis\n\
@item @qcode{\"same\"} (default)\n\
Return image of the same size as input @var{im}.\n\
\n\
@item @qcode{\"full\"}\n\
Return the full erosion (image is padded to accommodate @var{se} near the\n\
borders).\n\
\n\
@item @qcode{\"valid\"}\n\
Return only the parts which do not include the padded edges.\n\
@end table\n\
\n\
In case of a @var{SE} with a size of even length, the center is considered\n\
at indices @code{floor ([size(@var{SE})/2] + 1)}.\n\
\n\
@seealso{imdilate, imopen, imclose, strel}\n\
@end deftypefn")
{
  return base_action ("imerode", true, args);
}

/*
## using [1] as mask returns the same value
%!assert (imerode (eye (3), [1]), eye (3));
## and an empty SE returns all Inf
%!assert (imerode (eye (3), []), Inf (3, 3));

## test normal usage with non-symmetric SE
%!test
%! im = [0 1 0
%!       1 1 1
%!       0 1 0];
%! se = [1 0 0
%!       0 1 0
%!       0 1 1];
%! assert (imerode (im,          se),          [0 1 0; 0 0 0; 0 1 0]);
%! assert (imerode (logical(im), se), logical ([0 1 0; 0 0 0; 0 1 0]));
%! assert (imerode (im, se, "full"),
%!                 [  0    0    0    0  Inf
%!                    1    0    1    0  Inf
%!                    0    0    0    0    0
%!                  Inf    0    1    0    1
%!                  Inf  Inf    0    1    0]);
%! assert (imerode (logical(im), se, "full"),
%!                 logical([0     0     0     0     1
%!                          1     0     1     0     1
%!                          0     0     0     0     0
%!                          1     0     1     0     1
%!                          1     1     0     1     0]));

%!test
%! a = rand ([10 40 15 6 8 5]) > 0.2;
%! se = ones ([5 3 7]);
%!
%! ## the image is not really indexed but this way it is padded with 1s
%! assert (imerode (a, se), colfilt (a, "indexed", size (se), "sliding", @all))
%!
%! assert (imerode (a, se, "valid"), convn (a, se, "valid") == nnz (se))
%! ## again, we need to pad it ourselves because convn pads with zeros
%! b = true (size (a) + [4 2 6 0 0 0]);
%! b(3:12, 2:41, 4:18,:,:,:) = a;
%! assert (imdilate (b, se, "same"), convn (b, se, "same") > 0)
%! b = true (size (a) + [8 4 12 0 0 0]);
%! b(5:14, 3:42, 7:21,:,:,:) = a;
%! assert (imdilate (b, se, "full"), convn (b, se, "full") > 0)

%!test
%! im = [0 0 0 0 0 0 0
%!       0 0 1 0 1 0 0
%!       0 0 1 1 0 1 0
%!       0 0 1 1 1 0 0
%!       0 0 0 0 0 0 0];
%! se = [0 0 0
%!       0 1 0
%!       0 1 1];
%! out = [0 0 0 0 0 0 0
%!        0 0 1 0 0 0 0
%!        0 0 1 1 0 0 0
%!        0 0 0 0 0 0 0
%!        0 0 0 0 0 0 0];
%! assert (imerode (im, se), out);
%! assert (imerode (logical (im), se), logical (out));
%! assert (imerode (im, logical (se)), out);
%! assert (imerode (logical (im), logical (se)), logical (out));
%!
%! # with an even-size SE
%! se =  [0 0 0 1
%!        0 1 0 0
%!        0 1 1 1];
%! out = [0 0 0 0 0 0 0
%!        0 0 0 0 0 0 0
%!        0 0 1 0 0 0 0
%!        0 0 0 0 0 0 0
%!        0 0 0 0 0 0 0];
%! assert (imerode (im, se), out);
%! out = [ 0 0 0 0 1 0 1
%!        0 0 1 0 1 1 0
%!        0 0 1 1 1 1 1
%!        0 0 1 1 1 1 1
%!        0 0 1 1 1 1 1];
%! assert (imdilate (im, se), out);

## normal usage for grayscale images
%!test
%! a = [ 82    2   97   43   79   43   41   65   51   11
%!       60   65   21   56   94   77   36   38   75   39
%!       32   68   78    1   16   75   76   90   81   56
%!       43   90   82   41   36    1   87   19   18   63
%!       63   64    2   48   18   43   38   25   22   99
%!       12   46   90   79    3   92   39   79   10   22
%!       38   98   11   10   40   90   88   38    4   76
%!       54   37    9    4   33   98   36   47   53   57
%!       38   76   82   50   14   74   64   99    7   33
%!       88   96   41   62   84   89   97   23   41    3];
%!
%! domain = ones (3);
%! out = [  2    1    1    1   16   36   36   11
%!         21    1    1    1    1    1   18   18
%!          2    1    1    1    1    1   18   18
%!          2    2    2    1    1    1   10   10
%!          2    2    2    3    3   25    4    4
%!          9    4    3    3    3   36    4    4
%!          9    4    4    4   14   36    4    4
%!          9    4    4    4   14   23    7    3];
%! assert (imerode (a, domain, "valid"), out);
%! assert (imerode (uint8 (a), domain, "valid"), uint8 (out));
%! assert (imerode (uint8 (a), strel ("arbitrary", domain), "valid"), uint8 (out));
%! assert (imerode (uint8 (a), strel ("square", 3), "valid"), uint8 (out));
%!
%!## Test for non-flat strel
%! assert (imerode (a, strel ("arbitrary", domain, ones (3)), "valid"), out -1);
%!
%! out = [ 97   97   97   94   94   90   90   90
%!         90   90   94   94   94   90   90   90
%!         90   90   82   75   87   90   90   99
%!         90   90   90   92   92   92   87   99
%!         98   98   90   92   92   92   88   99
%!         98   98   90   98   98   98   88   79
%!         98   98   82   98   98   99   99   99
%!         96   96   84   98   98   99   99   99];
%! assert (imdilate (a, domain, "valid"), out);
%! assert (imdilate (uint8 (a), domain, "valid"), uint8 (out));
%!
%!## Test for non-flat strel
%! assert (imdilate (a, strel ("arbitrary", domain, ones (3)), "valid"), out +1);
%!
%! ## test while using SE that can be decomposed and an actual sequence
%! domain = ones (5);
%! out = [   2   1   1   1   1   1  16  11  11  11
%!           2   1   1   1   1   1   1   1  11  11
%!           2   1   1   1   1   1   1   1  11  11
%!           2   1   1   1   1   1   1   1  10  10
%!           2   1   1   1   1   1   1   1   4   4
%!           2   2   2   1   1   1   1   1   4   4
%!           2   2   2   2   2   3   3   4   4   4
%!           9   4   3   3   3   3   3   3   3   3
%!           9   4   4   4   4   4   4   3   3   3
%!           9   4   4   4   4   4   7   3   3   3];
%! assert (imerode (a, domain), out);
%! assert (imerode (a, strel ("square", 5)), out);
%! assert (imerode (a, getsequence (strel ("square", 5))), out);
%!
%! ## using a non-symmetric SE
%! domain = [ 1 1 0
%!            0 1 1
%!            0 1 0];
%!
%! out = [  2    2    1   16   36   36   38   39
%!         60    1    1   16    1   36   19   18
%!         32    2    1    1    1   19   18   18
%!          2    2   18    3    1    1   19   10
%!         46    2    2    3   18   38   10    4
%!         11    9    4    3    3   36    4    4
%!          9    4    4   10   36   36   38    4
%!         37    9    4    4   33   36    7    7];
%! assert (imerode (a, domain, "valid"), out);
%! assert (imerode (a, strel ("arbitrary", domain, ones (3)), "valid"), out -1);
%!
%! out = [ 78   97   56   94   94   90   90   81
%!         90   82   78   94   87   87   90   90
%!         90   90   82   43   75   87   90   99
%!         90   90   79   92   92   87   79   25
%!         98   90   90   90   92   92   79   79
%!         98   98   79   98   98   90   88   57
%!         98   82   50   74   98   99   99   53
%!         96   82   84   89   98   97   99   99];
%! assert (imdilate (a, domain, "valid"), out);
%! assert (imdilate (a, strel ("arbitrary", domain, ones (3)), "valid"), out +1);

// Tests for N-dimensions
%!test
%! im = reshape (magic(16), [4 8 4 2]);
%! se = true (3, 3, 3);
%! out = zeros (4, 8, 4, 2);
%! out(:,:,1,1) = [
%!     3   3  46   2   2   2  47  47
%!     3   3  30   2   2   2  31  31
%!    17  17  16  16  16  20  13  13
%!    33  33  16  16  16  36  13  13];
%! out(:,:,2,1) = [
%!     3   3  46   2   2   2  43  43
%!     3   3  30   2   2   2  27  27
%!    17  17  12  12  12  20  13  13
%!    33  33  12  12  12  36  13  13];
%! out(:,:,3,1) = [
%!     3   3  42   6   6   6  43  43
%!     3   3  26   6   6   6  27  27
%!    21  21  12  12  12  20   9   9
%!    37  37  12  12  12  36   9   9];
%! out(:,:,4,1) = [
%!     7   7  42   6   6   6  43  43
%!     7   7  26   6   6   6  27  27
%!    21  21  12  12  12  24   9   9
%!    37  37  12  12  12  40   9   9];
%! out(:,:,1,2) = [
%!    11  11  38  10  10  10  39  39
%!    11  11  22  10  10  10  23  23
%!    25  25   8   8   8  28   5   5
%!    41  41   8   8   8  44   5   5];
%! out(:,:,2,2) = [
%!    11  11  38  10  10  10  35  35
%!    11  11  22  10  10  10  19  19
%!    25  25   4   4   4  28   5   5
%!    41  41   4   4   4  44   5   5];
%! out(:,:,3,2) = [
%!    11  11  34  14  14  14  35  35
%!    11  11  18  14  14  14  19  19
%!    29  29   4   4   4  28   1   1
%!    45  45   4   4   4  44   1   1];
%! out(:,:,4,2) = [
%!    15  15  34  14  14  14  35  35
%!    15  15  18  14  14  14  19  19
%!    29  29   4   4   4  32   1   1
%!    45  45   4   4   4  48   1   1];
%! assert (imerode (im, se), out);
%! assert (imerode (uint16 (im), se), uint16 (out));
%!
%! ## trying a more weird SE
%! se(:,:,1) = [1 0 1; 0 1 1; 0 0 0];
%! se(:,:,3) = [1 0 1; 0 1 1; 0 0 1];
%! out(:,:,1,1) = [
%!    3  17  46   2   2   2  47  47
%!   17   3  30   2   2   2  31  31
%!   17  17  16  16  16  20  13  31
%!   33  33  16  16  16  36  13  13];
%! out(:,:,2,1) = [
%!    3   3  46   2   2  20  43  61
%!    3   3  30   2  20   2  27  43
%!   33  17  12  20  20  20  13  13
%!   51  33  12  12  30  36  13  13];
%! out(:,:,3,1) = [
%!    3  21  42   6   6   6  43  43
%!   21   3  26   6   6   6  27  27
%!   21  21  12  12  12  20   9  27
%!   37  37  12  12  12  36   9   9];
%! out(:,:,4,1) = [
%!    7   7  42   6   6  24  57  57
%!    7   7  26   6  24   6  43  43
%!   37  21  26  24  24  24   9   9
%!   55  37  12  12  26  40   9   9];
%! out(:,:,1,2) = [
%!   11  25  38  10  10  10  39  39
%!   25  11  22  10  10  10  23  23
%!   25  25   8   8   8  28   5  23
%!   41  41   8   8   8  44   5   5];
%! out(:,:,2,2) = [
%!   11  11  38  10  10  28  35  53
%!   11  11  22  10  22  10  19  35
%!   41  25   4  22  22  28   5   5
%!   59  41   4   4  22  44   5   5];
%! out(:,:,3,2) = [
%!   11  29  34  14  14  14  35  35
%!   29  11  18  14  14  14  19  19
%!   29  29   4   4   4  28   1  19
%!   45  45   4   4   4  44   1   1];
%! out(:,:,4,2) = [
%!   15  15  34  14  14  32  49  49
%!   15  15  18  14  18  14  35  35
%!   45  29  18  18  18  32   1   1
%!   63  45   4   4  18  48   1   1];
%! assert (imerode (im, se), out);
%! assert (imerode (uint16 (im), se), uint16 (out));

## Test input check
%!error imerode (ones (10), 45)
%!error imerode (ones (10), "some text")
%!error imerode (ones (10), {23, 45})

## No binary erosion for non-flat strel
%!error imerode (rand (10) > 10 , strel ("arbitrary", true (3), ones (3)))
*/

// PKG_ADD: autoload ("imdilate", which ("imerode"));
// PKG_DEL: autoload ("imdilate", which ("imerode"), "remove");
DEFUN_DLD(imdilate, args, , "\
-*- texinfo -*-\n\
@deftypefn  {Loadable Function} {} imdilate (@var{im}, @var{SE})\n\
@deftypefnx {Loadable Function} {} imdilate (@var{im}, @var{SE}, @var{shape})\n\
Perform morphological dilation.\n\
\n\
The image @var{im} must be a numeric matrix with any number of dimensions.\n\
The dilation is performed with the structuring element @var{se} which can\n\
be a:\n\
\n\
@itemize @bullet\n\
@item strel object;\n\
@item array of strel objects as returned by @code{@@strel/getsequence};\n\
@item matrix of 0's and 1's.\n\
@end itemize\n\
\n\
To perform a non-flat dilation, @var{SE} must be a strel object.\n\
\n\
The size of the result is determined by the optional @var{shape} argument\n\
which takes the following values:\n\
\n\
@table @asis\n\
@item @qcode{\"same\"} (default)\n\
Return image of the same size as input @var{im}.\n\
\n\
@item @qcode{\"full\"}\n\
Return the full dilation (matrix is padded to accommodate @var{se} near the\n\
borders).\n\
\n\
@item @qcode{\"valid\"}\n\
Return only the parts which do not include the padded edges.\n\
@end table\n\
\n\
In case of a @var{SE} with a size of even length, the center is considered\n\
at indices @code{floor ([size(@var{SE})/2] + 1)}.\n\
\n\
@seealso{imerode, imopen, imclose}\n\
@end deftypefn")
{
  return base_action ("imdilate", false, args);
}

/*
// Tests for N-dimensions

%!test
%! a = rand ([10 40 15 6 8 5]) > 0.8;
%! se = ones ([5 3 7]);
%! assert (imdilate (a, se), convn (a, se, "same") > 0)
%! assert (imdilate (a, se, "full"), convn (a, se, "full") > 0)
%! assert (imdilate (a, se, "valid"), convn (a, se, "valid") > 0)
%! assert (imdilate (a, se), colfilt (a, size (se), "sliding", @any))

%!test
%! im = reshape (magic(16), [4 8 4 2]);
%! se = true (3, 3, 3);
%! out = zeros (4, 8, 4, 2);
%!
%! out(:,:,1,1) = [
%!   256   256   209   253   253   253   212   212
%!   256   256   225   253   253   253   228   228
%!   238   238   243   243   243   239   242   242
%!   222   222   243   243   243   223   242   242];
%! out(:,:,2,1) = [
%!   256   256   213   253   253   253   212   212
%!   256   256   229   253   253   253   228   228
%!   238   238   243   243   243   239   246   246
%!   222   222   243   243   243   223   246   246];
%! out(:,:,3,1) = [
%!   252   252   213   253   253   253   216   216
%!   252   252   229   253   253   253   232   232
%!   238   238   247   247   247   235   246   246
%!   222   222   247   247   247   219   246   246];
%! out(:,:,4,1) = [
%!   252   252   213   249   249   249   216   216
%!   252   252   229   249   249   249   232   232
%!   234   234   247   247   247   235   246   246
%!   218   218   247   247   247   219   246   246];
%! out(:,:,1,2) = [
%!   248   248   217   245   245   245   220   220
%!   248   248   233   245   245   245   236   236
%!   230   230   251   251   251   231   250   250
%!   214   214   251   251   251   215   250   250];
%! out(:,:,2,2) = [
%!   248   248   221   245   245   245   220   220
%!   248   248   237   245   245   245   236   236
%!   230   230   251   251   251   231   254   254
%!   214   214   251   251   251   215   254   254];
%! out(:,:,3,2) = [
%!   244   244   221   245   245   245   224   224
%!   244   244   237   245   245   245   240   240
%!   230   230   255   255   255   227   254   254
%!   214   214   255   255   255   211   254   254];
%! out(:,:,4,2) = [
%!   244   244   221   241   241   241   224   224
%!   244   244   237   241   241   241   240   240
%!   226   226   255   255   255   227   254   254
%!   210   210   255   255   255   211   254   254];
%! assert (imdilate (im, se), out);
%! assert (imdilate (uint16 (im), se), uint16 (out));
%!
%! ## trying a more weird SE
%! se(:,:,1) = [1 0 1; 0 1 1; 0 0 0];
%! se(:,:,3) = [1 0 1; 0 1 1; 0 0 1];
%! out(:,:,1,1) = [
%!  256   256   209   239   253   253   212   194
%!  256   256   225   239   239   239   228   212
%!  222   222   243   239   243   239   242   242
%!  208   208   225   243   243   223   242   242];
%! out(:,:,2,1) = [
%!  256   256   213   253   253   253   212   212
%!  238   256   229   253   253   253   228   228
%!  238   238   243   243   243   239   246   228
%!  222   222   243   243   243   223   228   246];
%! out(:,:,3,1) = [
%!  252   252   213   235   253   253   216   198
%!  252   252   229   235   235   253   232   216
%!  222   238   247   235   247   235   246   246
%!  204   222   229   247   247   219   246   246];
%! out(:,:,4,1) = [
%!  252   252   213   249   249   249   216   216
%!  234   252   229   249   249   249   232   232
%!  234   234   247   247   247   235   246   232
%!  218   218   247   247   247   219   232   246];
%! out(:,:,1,2) = [
%!  248   248   217   231   245   245   220   202
%!  248   248   233   233   233   231   236   220
%!  214   214   251   233   251   231   250   250
%!  200   200   233   251   251   215   250   250];
%! out(:,:,2,2) = [
%!  248   248   221   245   245   245   220   220
%!  230   248   237   245   245   245   236   236
%!  230   230   251   251   251   231   254   236
%!  214   214   251   251   251   215   236   254];
%! out(:,:,3,2) = [
%!  244   244   221   227   245   245   224   206
%!  244   244   237   237   237   245   240   224
%!  214   230   255   237   255   227   254   254
%!  196   214   237   255   255   211   254   254];
%! out(:,:,4,2) = [
%!  244   244   221   241   241   241   224   224
%!  226   244   237   241   241   241   240   240
%!  226   226   255   255   255   227   254   240
%!  210   210   255   255   255   211   240   254];
%! assert (imdilate (im, se), out);
%! assert (imdilate (uint16 (im), se), uint16 (out));
*/

/*
## bug #47879 (invalid but mathematically interesting corner-case)
## This is all about empty sets, either by using a blank/empty/zeros SE
## or by picking a SE that "picks" elements only from the borders.  These
## are two completely different issues that may look the same.  See a more
## detailed explanation at http://stackoverflow.com/a/37117842/1609556

%!test    # scalar blank SE
%! se = 0;
%! assert (imerode (5, se), Inf)
%! assert (imerode (true, se), true)
%! assert (imerode (false, se), true)
%! assert (imerode (uint8 (3), se), uint8 (255))
%!
%! assert (imdilate (5, se), -Inf)
%! assert (imdilate (true, se), false)
%! assert (imdilate (false, se), false)
%! assert (imdilate (uint8 (3), se), uint8 (0))

%!test    # empty SE
%! se = [];
%! assert (imerode (5, se), Inf)
%! assert (imerode (true, se), true)
%! assert (imerode (false, se), true)
%! assert (imerode (uint8 (3), se), uint8 (255))
%!
%! assert (imdilate (5, se), -Inf)
%! assert (imdilate (true, se), false)
%! assert (imdilate (false, se), false)
%! assert (imdilate (uint8 (3), se), uint8 (0))

%!test    # non-scalar blank SE
%! se = zeros (3, 3);
%! assert (imerode (5, se), Inf)
%! assert (imerode (true, se), true)
%! assert (imerode (false, se), true)
%! assert (imerode (uint8 (3), se), uint8 (255))
%!
%! assert (imdilate (5, se), -Inf)
%! assert (imdilate(true, se), false)
%! assert (imdilate (false, se), false)
%! assert (imdilate (uint8 (3), se), uint8 (0))

%!test    # erode only with out-of-border elements
%! se = [1 1 1; 1 0 1; 1 1 1];
%! assert (imerode (5, se), Inf)
%! assert (imerode (true, se), true)
%!
%! assert (imdilate (5, se), -Inf)
%! assert (imdilate (true, se), false)

%!test    # only true elements of SE are out-of-border
%! se = [0 0 0; 1 0 0; 1 1 0];
%! assert (imerode (zeros (3), se), [0 0 0; 0 0 0; Inf 0 0])
%! assert (imerode (false (3), se), logical ([0 0 0; 0 0 0; 1 0 0]))
%! assert (imdilate (zeros (3), se), [0 0 -Inf; 0 0 0; 0 0 0])
%! assert (imdilate (false (3), se), false (3, 3))
%!
%! se = [0 0 0; 0 0 0; 1 1 1];
%! assert (imerode (zeros (3, 3), se), [0 0 0; 0 0 0; Inf Inf Inf])
%! assert (imerode (false (3, 3), se), logical ([0 0 0; 0 0 0; 1 1 1]))
%! assert (imdilate (zeros (3, 3), se), [-Inf -Inf -Inf; 0 0 0; 0 0 0])
%! assert (imdilate (false (3, 3), se), false (3, 3))

%!test  # only true elements of even-sized SE are out-of-border
%! se = logical ([0 1; 1 1]);
%! assert (imerode (false (3, 3), se), logical ([0 0 0; 0 0 0; 0 0 1]))
%! assert (imerode (zeros (3, 3), se), [0 0 0; 0 0 0; 0 0 Inf])
%!
%! assert (imdilate (false (3, 3), se), false (3, 3))
%! assert (imdilate (zeros (3, 3), se), [-Inf 0 0; 0 0 0; 0 0 0])
*/
