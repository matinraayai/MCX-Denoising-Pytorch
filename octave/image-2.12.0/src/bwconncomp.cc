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

// Implements connected components but using flood-fill algorithm instead
// of union find (like bwlabeln) so it uses a lot less memory.

// TODO: functions that could be here
//      * bwareafilt
//      * imfill / bwfill
//      * bwselect
//      * labelmatrix

#include <string>
#include <vector>

#include <octave/oct.h>
#include <octave/oct-map.h>

#include "connectivity.h"
using namespace octave::image;

static std::vector<std::vector<octave_idx_type>>
connected_components (const boolNDArray& BW, const connectivity& conn)
{
  boolNDArray BW_pad = conn.create_padded (BW, false);
  bool* BW_vec = BW_pad.fortran_vec ();

  const Array<octave_idx_type> offsets
    = conn.deleted_neighbourhood (BW_pad.dims ());
  const octave_idx_type n_offsets = offsets.numel ();
  const octave_idx_type* off_v = offsets.fortran_vec ();

  std::vector<std::vector<octave_idx_type>> all_components;
  const octave_idx_type numel = BW_pad.numel ();
  for (octave_idx_type i = 0; i < numel; BW_vec++, i++)
    {
      if (! *BW_vec)
        continue;

      // We want a queue but we will mimic one with a vector because in the
      // end all elements that go in the queue go in the vector anyway.
      std::vector<octave_idx_type> conn_comp {0};
      *BW_vec = false;

      std::vector<octave_idx_type>::size_type front = 0;
      while (front < conn_comp.size ())
        {
          octave_idx_type base_offset = conn_comp[front++];

          for (octave_idx_type j = 0; j < n_offsets; j++)
            {
              const octave_idx_type this_offset = base_offset + off_v[j];
              if (BW_vec[this_offset])
                {
                  BW_vec[this_offset] = false;
                  conn_comp.push_back (this_offset);
                }
            }
        }

      for (octave_idx_type& offset : conn_comp)
        offset += i;

      all_components.push_back (conn_comp);
    }

  // The collected indices are for the padded image so they need fixing

  const dim_vector original_size = BW.dims ();
  const dim_vector padded_size = BW_pad.dims ();
  const octave_idx_type* o = original_size.to_jit ();
  const octave_idx_type* p = padded_size.to_jit ();
  const octave_idx_type ndims_m1 = BW_pad.ndims () -1;

  std::vector<bool> dim_padded (BW_pad.ndims (), true);
  for (octave_idx_type i = 0; i < BW_pad.ndims (); i++)
    if (p[i] == o[i])
      dim_padded[i] = false;

  for (std::vector<octave_idx_type>& conn_comp : all_components)
    {
      for (octave_idx_type& offset : conn_comp)
        {
          octave_idx_type mult = 1;
          octave_idx_type ind = 0;
          for (octave_idx_type d = 0; d < ndims_m1; d++)
            {
              if (dim_padded[d])
                {
                  ind  += mult * (offset % p[d] - 1);
                  mult *= p[d] - 2;
                  offset /= p[d];
                }
              else
                {
                  ind  += mult * (offset % p[d]);
                  mult *= p[d];
                  offset /= p[d];
                }
            }
          if (dim_padded[ndims_m1])
            ind  += mult * (offset % p[ndims_m1] - 1);
          else
            ind  += mult * (offset % p[ndims_m1]);

          offset = ind;
        }
    }

  return all_components;
}

static Array<octave_idx_type>
dim_vector_2_array (const dim_vector& dims)
{
  RowVector size (dim_vector (1, dims.length ()));
  for (octave_idx_type i = 0; i < dims.length (); i++)
    size(i) = dims(i);
  return size;
}

// We should just return the connectivity used as input, args(1), but for
// Matlab compatibility, we must return 4, 8, etc if it matches
static octave_value
conn_to_octave_value (const connectivity& conn)
{
  const octave_idx_type n = conn.mask.numel ();
  const octave_idx_type nnz = conn.mask.nnz ();
  const bool* b_v = conn.mask.fortran_vec ();

  octave_idx_type nr;
  if ((n == 9 || n == 27) && n == nnz)
    nr = nnz -1;
  else if (nnz == 5 && b_v[1] && b_v[3] && b_v[4] && b_v[5] && b_v[7])
    nr = 4;
  else if (nnz == 7 && b_v[4] && b_v[10] && b_v[12] && b_v[13] && b_v[14]
           && b_v[16] && b_v[22])
    nr = 6;
  else if (nnz == 19 && ! b_v[0] && ! b_v[2] && ! b_v[6] && ! b_v[8]
           && ! b_v[18] && ! b_v[20] && ! b_v[24] && ! b_v[26])
    nr = 18;
  else
    return octave_value (conn.mask);

  return octave_value (nr);
}


DEFUN_DLD(bwconncomp, args, , "\
-*- texinfo -*-\n\
@deftypefn  {Function File} {@var{cc} =} bwconncomp (@var{bw})\n\
@deftypefnx {Function File} {@var{cc} =} bwconncomp (@var{bw}, @var{conn})\n\
Find connected objects.\n\
\n\
Elements from the matrix @var{bw}, belong to an object if they have a\n\
non-zero value.  The output @var{cc} is a structure with information about\n\
each object;\n\
\n\
@table @asis\n\
@item @qcode{\"Connectivity\"}\n\
The connectivity used in the boundary tracing. This may be different from\n\
the input argument, e.g., if @var{conn} is defined as a matrix of 1s and\n\
size 3x3, the @qcode{\"Connectivity\"} value will still be 8.\n\
\n\
@item @qcode{\"ImageSize\"}\n\
The size of the matrix @var{bw}.\n\
\n\
@item @qcode{\"NumObjects\"}\n\
The number of objects in the image @var{bw}.\n\
\n\
@item @qcode{\"PixelIdxList\"}\n\
A cell array with linear indices for each element of each object in @var{bw}\n\
A cell array containing where each element corresponds to an object in @var{BW}.\n\
Each element is represented as a vector of linear indices of the boundary of\n\
the given object.\n\
\n\
@end table\n\
\n\
Element connectivity @var{conn}, to define the size of objects, can be\n\
specified with a numeric scalar (number of elements in the neighborhood):\n\
\n\
@table @samp\n\
@item 4 or 8\n\
for 2 dimensional matrices;\n\
@item 6, 18 or 26\n\
for 3 dimensional matrices;\n\
@end table\n\
\n\
or with a binary matrix representing a connectivity array.  Defaults to\n\
@code{conndef (ndims (@var{bw}), \"maximal\")} which is equivalent to\n\
@var{conn} of 8 and 26 for 2 and 3 dimensional matrices respectively.\n\
\n\
@seealso{bwlabel, bwlabeln, bwboundaries, ind2sub, regionprops}\n\
@end deftypefn")
{
  const octave_idx_type nargin = args.length ();
  if (nargin < 1 || nargin > 2)
    print_usage ();

  const boolNDArray BW = args(0).bool_array_value ();

  connectivity conn;
  if (nargin > 1)
    conn = conndef (args(1));
  else
    {
      try
        {
          conn = connectivity (BW.ndims (), "maximal");
        }
      catch (invalid_connectivity& e)
        {
          error ("bwconncomp: failed to create MASK (%s)", e.what ());
        }
    }

  const std::vector<std::vector<octave_idx_type>> all_cc
    = connected_components (BW, conn);

  static const char *fields[] =
  {
    "Connectivity",
    "ImageSize",
    "NumObjects",
    "PixelIdxList",
    0
  };

  octave_scalar_map cc = octave_scalar_map (string_vector (fields));
  cc.assign ("Connectivity", conn_to_octave_value (conn));
  cc.assign ("NumObjects", octave_value (all_cc.size ()));
  cc.assign ("ImageSize", octave_value (dim_vector_2_array (BW.dims ())));

  Cell idx_cell (dim_vector (1, all_cc.size ()), ColumnVector ());

  octave_idx_type i_out = 0;
  for (auto it = all_cc.begin (); it != all_cc.end (); it++, i_out++)
    {
      ColumnVector idx (dim_vector (it->size (), 1));
      double* idx_v = idx.fortran_vec ();

      octave_idx_type i_in = 0;
      for (auto it_it = it->begin (); it_it != it->end (); it_it++, i_in++)
        idx_v[i_in] = *it_it +1; // +1 (we fix it for Octave indexing)

      idx_cell(i_out) = idx;
    }
  cc.setfield ("PixelIdxList", idx_cell);

  return octave_value (cc);
}

/*
%!test
%! a = rand (10) > 0.5;
%! cc = bwconncomp (a, 4);
%! assert (cc.Connectivity, 4)
%! assert (cc.ImageSize, [10 10])
%!
%! b = false (10);
%! for i = 1:numel (cc.PixelIdxList)
%!   b(cc.PixelIdxList{i}) = true;
%! endfor
%! assert (a, b)

%!test
%! a = rand (10, 13) > 0.5;
%! cc = bwconncomp (a, 4);
%! assert (cc.ImageSize, [10 13])
%!
%! b = false (10, 13);
%! for i = 1:numel (cc.PixelIdxList)
%!   b(cc.PixelIdxList{i}) = true;
%! endfor
%! assert (a, b)

%!test
%! a = rand (15) > 0.5;
%! conn_8 = bwconncomp (a, 8);
%! assert (conn_8, bwconncomp (a))
%! assert (conn_8, bwconncomp (a, ones (3)))
%! assert (conn_8.Connectivity, 8)
%! assert (bwconncomp (a, ones (3)).Connectivity, 8)
%! assert (bwconncomp (a, [0 1 0; 1 1 1; 0 1 0]).Connectivity, 4)

%!test
%! bw = logical ([
%!   1  0  0  1  0  1  0
%!   1  0  0  1  0  1  0
%!   0  0  0  0  0  1  0
%!   0  0  0  0  1  0  0
%!   1  1  0  1  1  0  0
%!   0  1  0  0  0  0  0
%!   1  1  0  0  0  0  0
%! ]);
%! cc = bwconncomp (bw);
%! cc = struct ();
%! cc.Connectivity = 8;
%! cc.ImageSize = [7 7];
%! cc.NumObjects = 4;
%! ## The commented line has the results from Matlab.  We return the
%! ## same result but in a slightly different order.  Since the order
%! ## is not defined, it is not required for compatibility.
%! #cc.PixelIdxList = {[1;2], [5;7;12;13;14], [22;23], [26;32;33;36;37;38]};
%! cc.PixelIdxList = {[1;2], [5;12;13;7;14], [22;23], [26;32;33;38;37;36]};
%! assert (bwconncomp (bw), cc)

%!test
%! ## test that PixelIdxList is a row vector
%! a = rand (40, 40) > 0.2;
%! cc = bwconncomp (a, 4);
%! assert (rows (cc.PixelIdxList), 1)
%! assert (columns (cc.PixelIdxList), cc.NumObjects)

## PixelIdxList is a row vector, even when there's zero objects
%!assert (bwconncomp (false (5)), struct ("ImageSize", [5 5], "NumObjects", 0,
%!                                        "PixelIdxList", {cell(1, 0)},
%!                                        "Connectivity", 8))
*/

// PKG_ADD: autoload ("bwareaopen", which ("bwconncomp"));
// PKG_DEL: autoload ("bwareaopen", which ("bwconncomp"), "remove");
DEFUN_DLD(bwareaopen, args, , "\
-*- texinfo -*-\n\
@deftypefn  {Function File} {} bwareaopen (@var{bw}, @var{lim})\n\
@deftypefnx {Function File} {} bwareaopen (@var{bw}, @var{lim}, @var{conn})\n\
Perform area opening.\n\
\n\
Remove objects with less than @var{lim} elements from a binary image\n\
@var{bw}.\n\
\n\
Element connectivity @var{conn}, to define the size of objects, can be\n\
specified with a numeric scalar (number of elements in the neighborhood):\n\
\n\
@table @samp\n\
@item 4 or 8\n\
for 2 dimensional matrices;\n\
@item 6, 18 or 26\n\
for 3 dimensional matrices;\n\
@end table\n\
\n\
or with a binary matrix representing a connectivity array.  Defaults to\n\
@code{conndef (ndims (@var{bw}), \"maximal\")} which is equivalent to\n\
@var{conn} of 8 and 26 for 2 and 3 dimensional matrices respectively.\n\
\n\
@seealso{bwconncomp, conndef, bwboundaries}\n\
@end deftypefn")
{
  const octave_idx_type nargin = args.length ();
  if (nargin < 2 || nargin > 3)
    print_usage ();

  boolNDArray BW = args(0).bool_array_value ();

  const std::vector<octave_idx_type>::size_type lim = args(1).idx_type_value ();
  if (lim < 0)
    error ("bwareaopen: LIM must be a non-negative scalar integer");

  connectivity conn;
  if (nargin > 2)
    conn = conndef (args(2));
  else
    {
      try
        {
          conn = connectivity (BW.ndims (), "maximal");
        }
      catch (invalid_connectivity& e)
        {
          error ("bwareaopen: failed to create MASK (%s)", e.what ());
        }
    }

  const std::vector<std::vector<octave_idx_type>> all_cc
    = connected_components (BW, conn);

  bool* BW_v = BW.fortran_vec ();
  for (std::vector<octave_idx_type> cc : all_cc)
    {
      if (cc.size () < lim)
        for (octave_idx_type ind : cc)
          BW_v[ind] = false;
    }

  return octave_value (BW);
}

/*
%!test
%! in = [ 0   0   1   0   0   1   0   1   0   0
%!        0   0   1   0   0   0   0   0   1   1
%!        1   0   0   0   0   1   1   0   0   0
%!        1   0   0   0   1   0   0   0   0   0
%!        1   1   1   1   0   0   0   0   0   1
%!        0   1   0   1   1   0   0   1   0   0
%!        1   0   0   0   1   0   0   0   0   0
%!        0   0   0   1   1   0   0   1   0   0
%!        0   1   0   1   1   0   0   1   1   0
%!        0   1   0   1   1   1   0   0   1   0];
%! assert (bwareaopen (in, 1, 4), logical (in))
%!
%! out = [0   0   0   0   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        1   0   0   0   0   0   0   0   0   0
%!        1   0   0   0   0   0   0   0   0   0
%!        1   1   1   1   0   0   0   0   0   0
%!        0   1   0   1   1   0   0   0   0   0
%!        0   0   0   0   1   0   0   0   0   0
%!        0   0   0   1   1   0   0   0   0   0
%!        0   0   0   1   1   0   0   0   0   0
%!        0   0   0   1   1   1   0   0   0   0];
%! assert (bwareaopen (logical (in), 10, 4), logical (out))
%! assert (bwareaopen (in, 10, 4), logical (out))
%! assert (bwareaopen (in, 10, [0 1 0; 1 1 1; 0 1 0]), logical (out))
%!
%! out = [0   0   0   0   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        1   0   0   0   0   1   1   0   0   0
%!        1   0   0   0   1   0   0   0   0   0
%!        1   1   1   1   0   0   0   0   0   0
%!        0   1   0   1   1   0   0   0   0   0
%!        1   0   0   0   1   0   0   0   0   0
%!        0   0   0   1   1   0   0   0   0   0
%!        0   0   0   1   1   0   0   0   0   0
%!        0   0   0   1   1   1   0   0   0   0];
%! assert (bwareaopen (in, 10, 8), logical (out))
%! assert (bwareaopen (in, 10, ones (3)), logical (out))
%! assert (bwareaopen (in, 10), logical (out))
%!
%! out = [0   0   0   0   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        1   0   0   0   0   0   0   0   0   0
%!        1   0   0   0   0   0   0   0   0   0
%!        1   1   1   1   0   0   0   0   0   0
%!        0   1   0   1   1   0   0   0   0   0
%!        0   0   0   0   1   0   0   0   0   0
%!        0   0   0   1   1   0   0   1   0   0
%!        0   0   0   1   1   0   0   1   1   0
%!        0   0   0   1   1   1   0   0   1   0];
%! assert (bwareaopen (in, 4, [1 1 0; 1 1 1; 0 1 1]), logical (out))

%!error bwareaopen ("not an image", 78, 8)
%!error bwareaopen (rand (10) > 0.5, 10, 100)
%!error bwareaopen (rand (10) > 0.5, 10, "maximal")
%!error bwareaopen (rand (10) > 0.5, 10, [1 1 1; 0 1 1; 0 1 0])
*/
