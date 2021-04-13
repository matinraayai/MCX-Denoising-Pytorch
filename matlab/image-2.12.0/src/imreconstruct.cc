// Copyright (C) 2014 Carnë Draug <carandraug@octave.org>
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 3 of the
// License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, see
// <http://www.gnu.org/licenses/>.

// This file implements imreconstruct as described on "Vincent, L. (1993).
// Morphological grayscale reconstruction in image analysis: applications
// and efficient algorithms. Image Processing, IEEE Transactions on, 2(2),
// 176-201."
//
// Our strategy to handle elements in the border is to simply pad it with
// the lowest value for the type, which will be ignored on the comparisons.
// This should still be more efficient than using subscript indices to find
// when we are on the border.

#include <functional>
#include <queue>

#include <octave/lo-mappers.h>

#include <octave/defun-dld.h>
#include <octave/defun-int.h>
#include <octave/error.h>
#include <octave/ovl.h>

#include "connectivity.h"

using namespace octave::image;

#define WANTS_OCTAVE_IMAGE_VALUE 1
#include "octave-wrappers.h"

/*
## A dirty implementation of the fast hybrid reconstruction as m file
## for testing purposes.

function marker = fast_hybrid_reconstruction (marker, mask)
  ## works for 10x10 matrices, padded to 12x12 with zeros, when
  ## connectivity is ones (3, 3)

  offsets = [-13 -12 -11 -1 1 11 12 13];
  pos_offsets = [0 1 11 12 13]; # don't forget the zero

  neg_offsets = - [pos_offsets];

  ## the raster scan
  for c = 2:(columns(marker) -1)
    for r = 2:(rows(marker) -1)
      i = sub2ind (size (marker), r, c);
      marker(r,c) = min (mask(r,c), max (marker(i + [neg_offsets])));
    endfor
  endfor

  ## the antiraster scan
  fifo = [];
  for c = (columns(marker) -1):-1:2
    for r = (rows(marker) -1):-1:2
      i = sub2ind (size (marker), r, c);
      offs = marker(i + [pos_offsets]);
      marker(r,c) = min (mask(r,c), max (offs));

      offs(1) = []; #remove itself
      picks = offs < marker(i) & offs < mask(i + [pos_offsets(2:end)]);
      if (any (picks))
        fifo(end+1) = i;
      endif
    endfor
  endfor

  ## the propagation step
  while (numel (fifo) != 0)
    p = fifo(1);
    fifo(1) = [];

    for i = offsets;
      if (marker(p +i) < marker(p) && mask(p+i) != marker(p+i))
        marker(p +i) = min (marker(p), mask(p+i));
        fifo(end+1) = p+i;
      endif
    endfor
  endwhile

endfunction
*/

template<class T>
static void
scan_raster_order (T& padded_marker, const T& padded_mask,
                   const dim_vector& original_size,
                   const Array<octave_idx_type>& padding_lengths,
                   const Array<octave_idx_type>& raster_neighbours)
{
  typedef typename T::element_type P;

  P* J = padded_marker.fortran_vec ();
  const P* I = padded_mask.fortran_vec ();
  const octave_idx_type* pads = padding_lengths.fortran_vec ();
  const octave_idx_type* neighbours = raster_neighbours.fortran_vec ();

  const octave_idx_type n_neighbours = raster_neighbours.numel ();

  // We probably should not be using this but converting to Array
  // just to have fortran_vec seems a bit too much.
  const octave_idx_type* s = original_size.to_jit ();

  std::function<void(const octave_idx_type)> scan;
  scan = [&] (const octave_idx_type dim) -> void
  {
    J += pads[dim];
    I += pads[dim];

    if (dim == 0)
      {
        for (octave_idx_type k = 0; k < s[0]; k++, J++, I++)
          {
            for (octave_idx_type i = 0; i < n_neighbours; i++)
              if (*J < J[neighbours[i]])
                *J = J[neighbours[i]];

            if (*J > *I)
              *J = *I;
          }
      }
    else
      for (octave_idx_type i = 0; i < s[dim]; i++)
        scan (dim-1);

    J += pads[dim];
    I += pads[dim];
    return;
  };
  scan (original_size.length () -1);
  return;
}

template<class T>
static std::queue<octave_idx_type>
scan_antiraster_order (T& padded_marker, const T& padded_mask,
                       const dim_vector& original_size,
                       const Array<octave_idx_type>& padding_lengths,
                       const Array<octave_idx_type>& antiraster_neighbours)
{
  typedef typename T::element_type P;
  std::queue<octave_idx_type> unfinished;

  P* J = padded_marker.fortran_vec ();
  const P* I = padded_mask.fortran_vec ();
  const octave_idx_type* pads = padding_lengths.fortran_vec ();
  const octave_idx_type* neighbours = antiraster_neighbours.fortran_vec ();

  const octave_idx_type n_neighbours = antiraster_neighbours.numel ();

  // We probably should not be using this but converting to Array
  // just to have fortran_vec seems a bit too much.
  const octave_idx_type* s = original_size.to_jit ();

  J += padded_marker.numel () -1;
  I += padded_marker.numel () -1;

  octave_idx_type ind = padded_marker.numel () -1;
  std::function<void(const octave_idx_type)> scan;
  scan = [&] (const octave_idx_type dim) -> void
  {
    J   -= pads[dim];
    I   -= pads[dim];
    ind -= pads[dim];

    if (dim == 0)
      {
        for (octave_idx_type k = 0; k < s[0]; k++, J--, I--, ind--)
          {
            for (octave_idx_type i = 0; i < n_neighbours; i++)
              if (*J < J[neighbours[i]])
                *J = J[neighbours[i]];

            if (*J > *I)
              *J = *I;

            for (octave_idx_type i = 0; i < n_neighbours; i++)
              if (J[neighbours[i]] < *J && J[neighbours[i]] < I[neighbours[i]])
                unfinished.push (ind);
          }
      }
    else
      for (octave_idx_type i = 0; i < s[dim]; i++)
        scan (dim-1);

    J   -= pads[dim];
    I   -= pads[dim];
    ind -= pads[dim];

    return;
  };
  scan (original_size.length () -1);
  return unfinished;
}

template<class T>
static void
propagation_step (T& padded_marker, const T& padded_mask,
                  std::queue<octave_idx_type>& unfinished,
                  const Array<octave_idx_type>& deleted_neighbours)
{
  typedef typename T::element_type P;

  P* J = padded_marker.fortran_vec ();
  const P* I = padded_mask.fortran_vec ();
  const octave_idx_type* neighbours = deleted_neighbours.fortran_vec ();

  const octave_idx_type n_neighbours = deleted_neighbours.numel ();

  while (! unfinished.empty ())
    {
      octave_idx_type p = unfinished.front ();
      unfinished.pop ();
      for (octave_idx_type k = 0; k < n_neighbours; k++)
        {
          octave_idx_type q = p + neighbours[k];
          if (J[q] < J[p] && I[q] != J[q])
            {
              J[q] = octave::math::min (J[p], I[q]);
              unfinished.push (q);
            }
        }
      OCTAVE_QUIT;
    }
  return;
}

template<class T>
static T
fast_hybrid_reconstruction (const T& marker, const T& mask,
                            const connectivity& conn)
{
  typedef typename T::element_type P;

  const dim_vector original_size = marker.dims ();

  T padded_marker = conn.create_padded (marker, connectivity::min_value<P> ());
  const T padded_mask = conn.create_padded (mask, connectivity::min_value<P> ());

  const dim_vector padded_size = padded_marker.dims ();
  const Array<octave_idx_type> padding_lengths
    = connectivity::padding_lengths (original_size, padded_size);

  scan_raster_order (padded_marker, padded_mask, original_size,
                     padding_lengths,
                     conn.negative_neighbourhood (padded_size));

  OCTAVE_QUIT;

  std::queue<octave_idx_type> unfinished =
    scan_antiraster_order (padded_marker, padded_mask, original_size,
                           padding_lengths,
                           conn.positive_neighbourhood (padded_size));

  OCTAVE_QUIT;

  propagation_step (padded_marker, padded_mask, unfinished,
                    conn.deleted_neighbourhood (padded_size));

  conn.unpad (padded_marker);
  return padded_marker;
}

template<class T>
static T
reconstruct (const T& marker, const T& mask, const connectivity& conn)
{
  return fast_hybrid_reconstruction (marker, mask, conn);
}

// TODO implement the following by reusing the code in bwlabeln
//static boolNDArray
//reconstruct (const boolNDArray& marker, const boolNDArray& mask,
//             const connectivity& conn)
//{
//  /*
//    1. Label the connected components of the mask image, i.e., each of these
//       components is assigned a unique number. Note that this step can itself
//       be implemented very efficiently by using algorithms based on chain an
//       loops [16] or queues of pixels [23, 26].
//    2. Determine the labels of the connected components which contain at
//       least a pixel of the marker image.
//    3. Remove all the connected components whose label is not one of the
//       previous ones.
//  */
//  return boolNDArray ();
//}

DEFUN_DLD(imreconstruct, args, , "\
-*- texinfo -*-\n\
@deftypefn  {Loadable Function} {} imreconstruct (@var{marker}, @var{mask})\n\
@deftypefnx {Loadable Function} {} imreconstruct (@var{marker}, @var{mask}, @var{conn})\n\
\n\
@seealso{imclearborder, imdilate, imerode}\n\
@end deftypefn")
{
  const octave_idx_type nargin = args.length ();

  if (nargin < 2 || nargin > 3)
    print_usage ();
  if (args(0).class_name () != args(1).class_name ())
    error ("imreconstruct: MARKER and MASK must be of same class");

  connectivity conn;
  if (nargin > 2)
    conn = conndef (args(2));
  else
    {
      try
        {
          conn = connectivity (args(0).ndims (), "maximal");
        }
      catch (invalid_connectivity& e)
        {
          error ("imreconstruct: unable to create connectivity (%s)", e.what ());
        }
    }
  octave_image::value marker (args(0));

#define RECONSTRUCT(TYPE) \
  ret = reconstruct (marker.TYPE ## _array_value (), \
                     args(1).TYPE ## _array_value (), conn);

#define IF_TYPE(TYPE) \
if (marker.is_ ## TYPE ## _type ()) \
  RECONSTRUCT (TYPE)

#define INT_BRANCH(TYPE) \
IF_TYPE(u ## TYPE) \
else IF_TYPE(TYPE)

#define FLOAT_BRANCH(CR) \
if (marker.is_single_type ()) \
  ret = reconstruct (marker.float_ ## CR ## array_value (), \
                    args(1).float_ ## CR ## array_value (), conn); \
else \
  ret = reconstruct (marker.CR ## array_value (), \
                     args(1).CR ## array_value (), conn);

  octave_value ret;
  if (marker.islogical ())
    RECONSTRUCT(bool)
  else INT_BRANCH (int8)
  else INT_BRANCH (int16)
  else INT_BRANCH (int32)
  else INT_BRANCH (int64)
  else if (marker.isreal ())
    {
      FLOAT_BRANCH()
    }
  else if (marker.iscomplex ())
    {
      FLOAT_BRANCH(complex_)
    }
  else
    error ("imreconstruct: unsupported class %s for MARKER",
           marker.class_name ().c_str ());

#undef IF_TYPE
#undef INT_BRANCH
#undef FLOAT_BRANCH

  return ret;
}

/*
## When using the fast hybrid reconstruction (and specially with random
## images), and if the images are small, it is often finished after the
## antiraster scan and before the propagation step.  Using larger images
## makes sure we get in the propagation step and that we catch bugs in there.

## This function does exactly what imreconstruct is meant to but is, in
## the words of Luc Vicent 1993, and I can attest to it, "[...] not suited
## to conventional computers, where its execution time is often of several
## minutes."
%!function recon = parallel_reconstruction (marker, mask,
%!                                          conn = conndef (ndims (marker), "maximal"))
%!  do
%!    previous = marker;
%!    marker = imdilate (marker, conn);
%!    ## FIXME https://savannah.gnu.org/bugs/index.php?43712
%!    if (strcmp (class (marker), "logical"))
%!      marker = marker & mask;
%!    else
%!      marker = min (marker, mask);
%!    endif
%!  until (all ((marker == previous)(:)))
%!  recon = marker;
%!endfunction

%!test
%! for cl = {"int8", "uint8", "int16", "uint16", "int32", "uint32"}
%!   cl = cl{1};
%!   a = randi ([intmin(cl) intmax(cl)-30], 100, 100, cl);
%!   b = a + randi (20, 100, 100, cl);
%!   assert (imreconstruct (a, b), parallel_reconstruction (a, b))
%! endfor
%! for cl = {"double", "single"}
%!   cl = cl{1};
%!   a = (rand (100, 100, cl) - 0.5) .* 1000;
%!   b = a + rand (100, 100, cl) * 100;
%!   assert (imreconstruct (a, b), parallel_reconstruction (a, b))
%! endfor

%!test
%! for cl = {"int8", "uint8", "int16", "uint16", "int32", "uint32"}
%!   cl = cl{1};
%!   a = randi ([intmin(cl) intmax(cl)-30], 100, 100, cl);
%!   b = a + randi (20, 100, 100, cl);
%!   c = [0 1 0; 1 1 1; 0 1 0];
%!   assert (imreconstruct (a, b, c), parallel_reconstruction (a, b, c))
%! endfor

%!test
%! a = randi (210, 100, 100);
%! b = a + randi (20, 100, 100);
%! c = ones (3, 1);
%! assert (imreconstruct (a, b, c), parallel_reconstruction (a, b, c))

%!test
%! a = randi (210, 500, 500, 10, 4);
%! b = a + randi (20, 500, 500, 10, 4);
%! c = ones (3, 3, 3);
%! assert (imreconstruct (a, b, c), parallel_reconstruction (a, b, c))

%!test
%! a = randi (210, 500, 500, 10, 4);
%! b = a + randi (20, 500, 500, 10, 4);
%! c = conndef (4, "minimal");
%! assert (imreconstruct (a, b, c), parallel_reconstruction (a, b, c))

%!test
%! a = [   0   0   0   0   0   0   0   1   0   0
%!         0   0   0   0   0   0   0   1   0   0
%!         1   0   0   0   0   0   0   0   0   0
%!         0   0   0   0   0   0   0   0   0   0
%!         0   0   0   0   0   0   0   1   0   0
%!         0   0   0   0   0   0   1   0   0   0
%!         0   0   0   0   0   0   0   0   0   0
%!         0   0   0   0   0   0   0   0   0   0
%!         0   0   0   0   1   0   0   0   0   0
%!         0   0   0   0   0   0   0   1   0   0];
%!
%! b = [   0   1   0   0   0   0   0   1   1   0
%!         1   1   0   0   0   1   0   1   1   0
%!         1   1   0   0   1   0   0   0   0   0
%!         1   1   0   0   0   1   1   0   0   0
%!         1   0   0   0   0   0   1   1   0   0
%!         0   1   0   0   0   0   1   1   0   0
%!         0   0   0   1   0   0   0   0   0   0
%!         0   0   0   0   1   1   0   0   0   0
%!         0   0   0   1   1   0   0   0   0   0
%!         1   0   0   0   1   0   0   1   0   1];
%!
%! c = [   0   1   0   0   0   0   0   1   1   0
%!         1   1   0   0   0   1   0   1   1   0
%!         1   1   0   0   1   0   0   0   0   0
%!         1   1   0   0   0   1   1   0   0   0
%!         1   0   0   0   0   0   1   1   0   0
%!         0   1   0   0   0   0   1   1   0   0
%!         0   0   0   1   0   0   0   0   0   0
%!         0   0   0   0   1   1   0   0   0   0
%!         0   0   0   1   1   0   0   0   0   0
%!         0   0   0   0   1   0   0   1   0   0];
%! assert (imreconstruct (logical (a), logical (b)), logical (c));
%!
%! c = [   0   1   0   0   0   0   0   1   1   0
%!         1   1   0   0   0   0   0   1   1   0
%!         1   1   0   0   0   0   0   0   0   0
%!         1   1   0   0   0   1   1   0   0   0
%!         1   0   0   0   0   0   1   1   0   0
%!         0   0   0   0   0   0   1   1   0   0
%!         0   0   0   0   0   0   0   0   0   0
%!         0   0   0   0   1   1   0   0   0   0
%!         0   0   0   1   1   0   0   0   0   0
%!         0   0   0   0   1   0   0   1   0   0];
%! assert (imreconstruct (logical (a), logical (b), [0 1 0; 1 1 1; 0 1 0]),
%!         logical (c));

%!test
%! do
%!   b = rand (100, 100, 100) > 0.98;
%! until (nnz (b) > 4)
%! b = imdilate (b, ones (5, 5, 5));
%! a = false (size (b));
%! f = find (b);
%! a(f(randi (numel (f), 6, 1))) = true;
%! assert (imreconstruct (a, b), parallel_reconstruction (a, b))

## we try to be smart about the padding so make sure this works.  There
## was a nasty bug during development which this test brings up.
%!test
%! a = randi (200, 100,100, 10, 10);
%! b = a + randi (20, 100,100, 10, 10);
%! c1 = ones (3, 3, 3);
%! c2 = zeros (3, 3, 3, 3);
%! c2(:,:,:,2) = c1;
%! assert (imreconstruct (a, b, c1), imreconstruct (a, b, c2))

%!test
%! ## Values in MARKER above MASK should be clipped (bug #48794)
%! ## (well, treated internally as if they were clipped)
%! mask = logical ([1 1 1; 1 0 1; 1 1 1]);
%! assert (imreconstruct (true (3, 3), mask), mask)
%!
%! mask = ones (5, 5);
%! mask(2:4,2:4) = 0;
%! assert (imreconstruct (ones (5, 5), mask), mask)
%!
%! mask = ones (5, 5);
%! mask(2:4,2:4) = 0;
%! assert (imreconstruct (repmat (2, [5, 5]), mask), mask)
%!
%! mask = ones (5, 5);
%! mask(2:4,2:4) = 0;
%! assert (imreconstruct (repmat (2, [5, 5]), mask), mask)
%!
%! marker = ones (3, 3, 3, 3);
%! mask = marker;
%! mask(2, 2, 2, 2) = 0;
%! assert (imreconstruct (marker, mask), mask)
%!
%! marker = randi (210, 100, 100);
%! assert (imreconstruct (marker +1, marker), marker)
%! assert (imreconstruct (marker +1, marker), imreconstruct (marker, marker))
*/
