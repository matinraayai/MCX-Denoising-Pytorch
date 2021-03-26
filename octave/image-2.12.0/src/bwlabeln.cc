// Copyright (C) 2002 Jeffrey E. Boyd <boyd@cpsc.ucalgary.ca>
// Copyright (C) 2011-2012 Jordi Gutiérrez Hermoso <jordigh@octave.org>
// Copyright (C) 2013 Carnë Draug <carandraug@octave.org>
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

// Copyright
// Jeffrey E. Boyd and Carnë Draug for bwlabel_2d
// Jordi Gutiérrez Hermoso for bwlabel_nd

#include <vector>
#include <algorithm>
#include <unordered_map>

#include <octave/oct.h>

#include "union-find.h"
#include "connectivity.h"

#define WANTS_OCTAVE_IMAGE_VALUE 1
#include "octave-wrappers.h"

using namespace octave::image;

static union_find
pre_label (NDArray& L, const connectivity& conn)
{
  double* L_vec = L.fortran_vec ();
  const octave_idx_type numel = L.numel ();

  const Array<octave_idx_type> neighbours
    = conn.negative_neighbourhood (L.dims ());
  const octave_idx_type* nbr = neighbours.fortran_vec ();
  const octave_idx_type nbr_numel = neighbours.numel ();

  union_find u_f (numel);
  for (octave_idx_type Lidx = 0; Lidx < numel; Lidx++)
    {
      // The boundary is always zero, so we'll always skip it, so
      // we're never considering the neighbours of the boundary. Thus,
      // there is no possibility of out-of-bounds error below.
      if (L_vec[Lidx])
        {
          //Insert this one into its group
          u_f.add (Lidx);

          for (octave_idx_type i = 0; i < nbr_numel; i++)
            {
              octave_idx_type n = *nbr++ + Lidx;
              if (L_vec[n])
                u_f.unite (n, Lidx);
            }
          nbr -= nbr_numel;
        }
    }
  return u_f;
}

static octave_idx_type
paint_labels (NDArray& L, union_find& u_f)
{
  double* L_vec = L.fortran_vec ();

  std::unordered_map<octave_idx_type, octave_idx_type> ids_to_label;
  octave_idx_type next_label = 1;

  std::vector<octave_idx_type> idxs = u_f.get_ids (L);
  for (auto idx = idxs.begin (); idx != idxs.end (); idx++)
    {
      octave_idx_type label;
      octave_idx_type id = u_f.find (*idx);
      auto try_label = ids_to_label.find (id);
      if (try_label == ids_to_label.end ())
        {
          label = next_label++;
          ids_to_label[id] = label;
        }
      else
        label = try_label->second;

      L_vec[*idx] = label;
    }
  return ids_to_label.size ();
}

static octave_value_list
bwlabel_nd (const boolNDArray& BW, const connectivity& conn)
{
  boolNDArray conn_mask = conn.mask;

  const dim_vector size_vec = BW.dims ();

  // Use temporary array with borders padded with zeros. Labels will
  // also go in here eventually.
  NDArray L = conn.create_padded (BW, 0);

  union_find u_f = pre_label (L, conn);
  octave_idx_type n_labels = paint_labels (L, u_f);

  // Remove the zero padding...
  conn.unpad (L);

  octave_value_list rval;
  rval(0) = L;
  rval(1) = n_labels;
  return rval;
}

static octave_idx_type
find (std::vector<octave_idx_type>& lset, octave_idx_type x)
{
  // Follow lset until we find a value that points to itself
  while (lset[x] != x)
    x = lset[x];
  return x;
}

static octave_value_list
bwlabel_2d (const boolMatrix& BW, const octave_idx_type& n)
{
  // This algorithm was derived from  BKP Horn, Robot Vision, MIT Press,
  // 1986, p 65 - 89 by Jeffrey E. Boyd in 2002. Some smaller changes
  // were then introduced by Carnë Draug in 2013 to speed up by iterating
  // down a column, and what values to use when connecting two labels
  // to increase chance of getting them in the right order in the end.

  const octave_idx_type nr = BW.rows ();
  const octave_idx_type nc = BW.columns ();

  // The labelled image
  Matrix L (nr, nc);

  std::vector<octave_idx_type> lset (nc*nr);    // label table/tree

  octave_idx_type ntable = 0; // number of elements in the component table/tree
  octave_idx_type ind    = 0; // linear index

  bool n4, n6, n8;
  n4 = n6 = n8 = false;
  if (n == 4)
    n4 = true;
  else if (n == 6)
    n6 = true;
  else if (n == 8)
    n8 = true;

  const bool* BW_vec = BW.data ();
  double* L_vec = L.fortran_vec ();

  for (octave_idx_type c = 0; c < nc; c++)
    {
      for (octave_idx_type r = 0; r < nr; r++, ind++)
        {
          if (BW_vec[ind]) // if A is an object
            {
              octave_idx_type stride = ind - nr;
              // Get the neighboring pixels B, C, D, and E
              //
              //  D  B
              //  C  A  <-- ind is linear index to A
              //  E
              //
              // C and B will always be needed so we get them here, but
              // D is only needed when n is 6 or 8, and E when n is 8.

              octave_idx_type B, C;
              if (c == 0)
                C = 0;
              else
                C = find (lset, L_vec[stride]);

              if (r == 0)
                B = 0;
              else
                B = find (lset, L_vec[ind -1]);

              if (n4)
                {
                  // apply 4 connectedness
                  if (B && C) // B and C are labeled
                    {
                      if (B != C)
                        lset[B] = C;

                      L_vec[ind] = C;
                    }
                  else if (B) // B is object but C is not
                    L_vec[ind] = B;
                  else if (C) // C is object but B is not
                    L_vec[ind] = C;
                  else // B, C not object - new object
                    {
                      // label and put into table
                      ntable++;
                      L_vec[ind] = lset[ntable] = ntable;
                    }
                }
              else if (n6)
                {
                  // Apply 6 connectedness. Seem there's more than one
                  // possible way to do this for 2D images but for some
                  // reason, the most common seems to be the top left pixel
                  // and the bottom right
                  // See http://en.wikipedia.org/wiki/Pixel_connectivity

                  octave_idx_type D;
                  // D is only required for n6 and n8
                  if (r == 0 || c == 0)
                    D = 0;
                  else
                    D = find (lset, L_vec[stride -1]);

                  if (D) // D object, copy label and move on
                    L_vec[ind] = D;
                  else if (B && C) // B and C are labeled
                    {
                      if (B == C)
                        L_vec[ind] = B;
                      else
                        {
                          octave_idx_type tlabel = std::min (B, C);
                          lset[B] = tlabel;
                          lset[C] = tlabel;
                          L_vec[ind] = tlabel;
                        }
                    }
                  else if (B) // B is object but C is not
                    L_vec[ind] = B;
                  else if (C) // C is object but B is not
                    L_vec[ind] = C;
                  else // B, C, D not object - new object
                    {
                      // label and put into table
                      ntable++;
                      L_vec[ind] = lset[ntable] = ntable;
                    }
                }
              else if (n8)
                {
                  octave_idx_type D, E;
                  // D is only required for n6 and n8
                  if (r == 0 || c == 0)
                    D = 0;
                  else
                    D = find (lset, L_vec[stride -1]);

                  // E is only required for n8
                  if (c == 0 || r == nr -1)
                    E = 0;
                  else
                    E = find (lset, L_vec[stride +1]);

                  // apply 8 connectedness
                  if (B || C || D || E)
                    {
                      octave_idx_type tlabel = D;
                      if (D)
                        ; // do nothing (tlabel is already D)
                      else if (C)
                        tlabel = C;
                      else if (E)
                        tlabel = E;
                      else if (B)
                        tlabel = B;

                      L_vec[ind] = tlabel;

                      if (B && B != tlabel)
                        lset[B] = tlabel;
                      if (C && C != tlabel)
                        lset[C] = tlabel;
                      if (D)
                        // we don't check if B != tlabel since if B
                        // is true, tlabel == B
                        lset[D] = tlabel;
                      if (E && E != tlabel)
                        lset[E] = tlabel;
                    }
                  else
                    {
                      // label and put into table
                      ntable++;  // run image through the look-up table
                      L_vec[ind] = lset[ntable] = ntable;
                    }
                }
            }
          else
            L_vec[ind] = 0; // A is not an object so leave it
        }
    }

  const octave_idx_type numel = BW.numel ();

  // consolidate component table
  for (octave_idx_type i = 0; i <= ntable; i++)
    lset[i] = find (lset, i);

  // run image through the look-up table
  for (octave_idx_type ind = 0; ind < numel; ind++)
    L_vec[ind] = lset[L_vec[ind]];

  // count up the objects in the image
  for (octave_idx_type i = 0; i <= ntable; i++)
    lset[i] = 0;

  for (octave_idx_type ind = 0; ind < numel; ind++)
    lset[L_vec[ind]]++;

  // number the objects from 1 through n objects
  octave_idx_type nobj = 0;
  lset[0] = 0;
  for (octave_idx_type i = 1; i <= ntable; i++)
    if (lset[i] > 0)
      lset[i] = ++nobj;

  // Run through the look-up table again, so that their numbers
  // match the number of labels
  for (octave_idx_type ind = 0; ind < numel; ind++)
    L_vec[ind] = lset[L_vec[ind]];

  octave_value_list rval;
  rval(0) = L;
  rval(1) = double (nobj);
  return rval;
}

DEFUN_DLD(bwlabeln, args, , "\
-*- texinfo -*-\n\
@deftypefn  {Loadable Function} {[@var{l}, @var{num}] =} bwlabeln (@var{bw})\n\
@deftypefnx {Loadable Function} {[@var{l}, @var{num}] =} bwlabeln (@var{bw}, @var{n})\n\
Label foreground objects in the n-dimensional binary image @var{bw}.\n\
\n\
The optional argument @var{n} sets the connectivity and defaults 26,\n\
for 26-connectivity in 3-D images. Other possible values are 18 and 6\n\
for 3-D images, 4 and 8 for 2-D images, or an arbitrary N-dimensional\n\
binary connectivity mask where each dimension is of size 3.\n\
\n\
The output @var{l} is an Nd-array where 0 indicates a background\n\
pixel, 1 indicates that the pixel belongs to object number 1, 2 that\n\
the pixel belongs to object number 2, etc. The total number of objects\n\
is @var{num}.\n\
\n\
The algorithm used is a disjoint-set data structure, a.k.a. union-find.\n\
See, for example, http://en.wikipedia.org/wiki/Union-find\n\
\n\
@seealso{bwconncomp, bwlabel, regionprops}\n\
@end deftypefn\n\
")
{
  octave_value_list rval;

  const octave_idx_type nargin = args.length ();
  if (nargin < 1 || nargin > 2)
    print_usage ();

  octave_image::value bw_value (args(0));
  if (! bw_value.isnumeric () && ! bw_value.islogical ())
    error ("bwlabeln: BW must be a numeric or logical matrix");

  boolNDArray BW = bw_value.bool_array_value ();
  dim_vector size_vec = BW.dims ();

  connectivity conn;
  if (nargin == 2)
    conn = conndef (args(1));
  else
    {
      try
        {
          conn = connectivity (BW.ndims (), "maximal");
        }
      catch (invalid_connectivity& e)
        {
          error ("bwlabeln: faild to create MASK (%s)", e.what ());
        }
    }

  // The implementation in bwlabel_2d is faster so use it if we can
  const octave_idx_type ndims = BW.ndims ();
  if (ndims == 2 && boolMatrix (conn.mask) == connectivity (4).mask)
    rval = bwlabel_2d (BW, 4);
  else if (ndims == 2 && boolMatrix (conn.mask) == connectivity (8).mask)
    rval = bwlabel_2d (BW, 8);
  else
    rval = bwlabel_nd (BW, conn);

  return rval;
}

/*
%!shared a2d, a3d
%! a2d = [1   0   0   0   0   0   1   0   0   1
%!        1   0   0   1   0   1   0   1   0   1
%!        1   0   1   0   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   1   0   0   0   0   0   0   0   0
%!        1   1   0   1   1   1   0   0   0   0
%!        1   1   0   1   0   0   0   1   0   0
%!        1   1   0   0   0   0   1   0   1   0
%!        1   1   0   0   0   0   0   0   0   0
%!        1   1   0   0   0   1   1   0   0   1];
%!
%! a3d = a2d;
%! a3d(:,:,2) = [
%!        0   0   0   0   0   0   0   0   0   0
%!        1   0   0   1   1   0   0   1   0   0
%!        0   0   0   1   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   1   0   0   0   0   0   0   0   0
%!        1   1   0   0   1   1   0   0   0   0
%!        1   1   0   1   0   0   0   0   0   0
%!        1   0   0   0   0   0   1   0   0   0
%!        0   1   0   0   0   0   0   0   0   1
%!        1   1   0   0   0   0   1   0   0   0];
%!
%! a3d(:,:,3) = [
%!        1   0   0   0   0   0   0   0   0   0
%!        0   1   0   1   1   0   0   1   0   0
%!        0   0   0   1   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   0   0   1   1   1   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        1   0   0   0   0   0   0   0   0   0
%!        1   1   0   0   0   0   0   0   0   1
%!        1   1   0   0   0   0   0   0   0   0];

%!test
%! label2dc4 = [
%!        1   0   0   0   0   0   8   0   0  13
%!        1   0   0   4   0   6   0  10   0  13
%!        1   0   3   0   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   2   0   0   0   0   0   0   0   0
%!        2   2   0   5   5   5   0   0   0   0
%!        2   2   0   5   0   0   0  11   0   0
%!        2   2   0   0   0   0   9   0  12   0
%!        2   2   0   0   0   0   0   0   0   0
%!        2   2   0   0   0   7   7   0   0  14];
%! assert (bwlabeln (a2d, 4), label2dc4)
%! assert (bwlabeln (a2d, [0 1 0; 1 1 1; 0 1 0]), label2dc4)
%! assert (bwlabeln (a2d, conndef (2, "minimal")), label2dc4)
%! assert (bwlabeln (a2d, conndef (3, "minimal")), label2dc4)

%!test
%! label2dc8 = [
%!        1   0   0   0   0   0   5   0   0   8
%!        1   0   0   3   0   5   0   5   0   8
%!        1   0   3   0   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   2   0   0   0   0   0   0   0   0
%!        2   2   0   4   4   4   0   0   0   0
%!        2   2   0   4   0   0   0   7   0   0
%!        2   2   0   0   0   0   7   0   7   0
%!        2   2   0   0   0   0   0   0   0   0
%!        2   2   0   0   0   6   6   0   0   9];
%! assert (bwlabeln (a2d, 8), label2dc8)
%! assert (bwlabeln (a2d, ones (3)), label2dc8)
%! assert (bwlabeln (a2d, conndef (2, "maximal")), label2dc8)
%! assert (bwlabeln (a2d, conndef (3, "maximal")), label2dc8)

%!test
%! label3dc8 = [
%!        1   0   0   0   0   0   5   0   0   8
%!        1   0   0   3   0   5   0   5   0   8
%!        1   0   3   0   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   2   0   0   0   0   0   0   0   0
%!        2   2   0   4   4   4   0   0   0   0
%!        2   2   0   4   0   0   0   7   0   0
%!        2   2   0   0   0   0   7   0   7   0
%!        2   2   0   0   0   0   0   0   0   0
%!        2   2   0   0   0   6   6   0   0   9];
%! label3dc8(:,:,2) = [
%!        0   0   0   0   0   0   0   0   0   0
%!       10   0   0  12  12   0   0  16   0   0
%!        0   0   0  12   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0  11   0   0   0   0   0   0   0   0
%!       11  11   0   0  13  13   0   0   0   0
%!       11  11   0  13   0   0   0   0   0   0
%!       11   0   0   0   0   0  14   0   0   0
%!        0  11   0   0   0   0   0   0   0  17
%!       11  11   0   0   0   0  15   0   0   0];
%! label3dc8(:,:,3) = [
%!       18   0   0   0   0   0   0   0   0   0
%!        0  18   0  20  20   0   0  22   0   0
%!        0   0   0  20   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   0   0  21  21  21   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!       19   0   0   0   0   0   0   0   0   0
%!       19  19   0   0   0   0   0   0   0  23
%!       19  19   0   0   0   0   0   0   0   0];
%! assert (bwlabeln (a3d, 8), label3dc8)
%! assert (bwlabeln (a3d, ones (3, 3)), label3dc8)
%! assert (bwlabeln (a3d, conndef (2, "maximal")), label3dc8)

%!test
%! label3dc26 = [
%!        1   0   0   0   0   0   3   0   0   7
%!        1   0   0   3   0   3   0   3   0   7
%!        1   0   3   0   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   2   0   0   0   0   0   0   0   0
%!        2   2   0   4   4   4   0   0   0   0
%!        2   2   0   4   0   0   0   6   0   0
%!        2   2   0   0   0   0   6   0   6   0
%!        2   2   0   0   0   0   0   0   0   0
%!        2   2   0   0   0   5   5   0   0   6];
%! label3dc26(:,:,2) = [
%!        0   0   0   0   0   0   0   0   0   0
%!        1   0   0   3   3   0   0   3   0   0
%!        0   0   0   3   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   2   0   0   0   0   0   0   0   0
%!        2   2   0   0   4   4   0   0   0   0
%!        2   2   0   4   0   0   0   0   0   0
%!        2   0   0   0   0   0   6   0   0   0
%!        0   2   0   0   0   0   0   0   0   6
%!        2   2   0   0   0   0   5   0   0   0];
%! label3dc26(:,:,3) = [
%!        1   0   0   0   0   0   0   0   0   0
%!        0   1   0   3   3   0   0   3   0   0
%!        0   0   0   3   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   0   0   4   4   4   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        2   0   0   0   0   0   0   0   0   0
%!        2   2   0   0   0   0   0   0   0   6
%!        2   2   0   0   0   0   0   0   0   0];
%! assert (bwlabeln (a3d, 26), label3dc26)
%! assert (bwlabeln (a3d, ones (3, 3, 3)), label3dc26)
%! assert (bwlabeln (a3d, conndef (3, "maximal")), label3dc26)

%!test
%! label3dc18 = [
%!        1   0   0   0   0   0   3   0   0   7
%!        1   0   0   3   0   3   0   3   0   7
%!        1   0   3   0   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   2   0   0   0   0   0   0   0   0
%!        2   2   0   4   4   4   0   0   0   0
%!        2   2   0   4   0   0   0   6   0   0
%!        2   2   0   0   0   0   6   0   6   0
%!        2   2   0   0   0   0   0   0   0   0
%!        2   2   0   0   0   5   5   0   0   8];
%! label3dc18(:,:,2) = [
%!        0   0   0   0   0   0   0   0   0   0
%!        1   0   0   3   3   0   0   3   0   0
%!        0   0   0   3   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   2   0   0   0   0   0   0   0   0
%!        2   2   0   0   4   4   0   0   0   0
%!        2   2   0   4   0   0   0   0   0   0
%!        2   0   0   0   0   0   6   0   0   0
%!        0   2   0   0   0   0   0   0   0   8
%!        2   2   0   0   0   0   5   0   0   0];
%! label3dc18(:,:,3) = [
%!        1   0   0   0   0   0   0   0   0   0
%!        0   1   0   3   3   0   0   3   0   0
%!        0   0   0   3   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   0   0   4   4   4   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        2   0   0   0   0   0   0   0   0   0
%!        2   2   0   0   0   0   0   0   0   8
%!        2   2   0   0   0   0   0   0   0   0];
%! assert (bwlabeln (a3d, 18), label3dc18)

%!test
%! label2dc3 = [
%!        1   0   0   0   0   0  11   0   0  17
%!        1   0   0   5   0   8   0  14   0  17
%!        1   0   4   0   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   3   0   0   0   0   0   0   0   0
%!        2   3   0   6   7   9   0   0   0   0
%!        2   3   0   6   0   0   0  15   0   0
%!        2   3   0   0   0   0  12   0  16   0
%!        2   3   0   0   0   0   0   0   0   0
%!        2   3   0   0   0  10  13   0   0  18];
%! assert (bwlabeln (a2d, [1 1 1]'), label2dc3)
%!
%! label3dc3 = label2dc3;
%! label3dc3(:,:,2) = [
%!        0   0   0   0   0   0   0   0   0   0
%!       19   0   0  24  26   0   0  31   0   0
%!        0   0   0  24   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0  22   0   0   0   0   0   0   0   0
%!       20  22   0   0  27  28   0   0   0   0
%!       20  22   0  25   0   0   0   0   0   0
%!       20   0   0   0   0   0  29   0   0   0
%!        0  23   0   0   0   0   0   0   0  32
%!       21  23   0   0   0   0  30   0   0   0];
%! label3dc3(:,:,3) = [
%!       33   0   0   0   0   0   0   0   0   0
%!        0  35   0  37  39   0   0  42   0   0
%!        0   0   0  37   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!        0   0   0  38  40  41   0   0   0   0
%!        0   0   0   0   0   0   0   0   0   0
%!       34   0   0   0   0   0   0   0   0   0
%!       34  36   0   0   0   0   0   0   0  43
%!       34  36   0   0   0   0   0   0   0   0];
%! assert (bwlabeln (a3d, [1 1 1]'), label3dc3)

%!test
%! label2dc1 = zeros (size (a2d));
%! label2dc1(a2d != 0) = 1:nnz (a2d);
%! assert (bwlabeln (a2d, [1]), label2dc1);
%! assert (bwlabeln (a2d, [0 1 0]'), label2dc1);
%!
%! label3dc1 = zeros (size (a3d));
%! label3dc1(a3d != 0) = 1:nnz (a3d);
%! assert (bwlabeln (a3d, [1]), label3dc1);
%! assert (bwlabeln (a3d, [0 1 0]'), label3dc1);
*/

// PKG_ADD: autoload ("bwlabel", which ("bwlabeln"));
// PKG_DEL: autoload ("bwlabel", which ("bwlabeln"), "remove");
DEFUN_DLD(bwlabel, args, , "\
-*- texinfo -*-\n\
@deftypefn  {Loadable Function} {[@var{l}, @var{num}] =} bwlabel(@var{BW})\n\
@deftypefnx {Loadable Function} {[@var{l}, @var{num}] =} bwlabel(@var{BW}, @var{n})\n\
Label binary 2 dimensional image.\n\
\n\
Labels foreground objects in the binary image @var{bw}.\n\
The output @var{l} is a matrix where 0 indicates a background pixel,\n\
1 indicates that the pixel belongs to object number 1, 2 that the pixel\n\
belongs to object number 2, etc.\n\
The total number of objects is @var{num}.\n\
\n\
Two pixels belong to the same object if they are neighbors. By default\n\
the algorithm uses 8-connectivity to define a neighborhood, but this\n\
can be changed through the argument @var{n}, which can be either 4, 6, or 8.\n\
\n\
@seealso{bwconncomp, bwlabeln, regionprops}\n\
@end deftypefn\n\
")
{
  octave_value_list rval;

  const octave_idx_type nargin = args.length ();
  if (nargin < 1 || nargin > 2)
    print_usage ();

  // We do not check error state after conversion to boolMatrix
  // because what we want is to actually get a boolean matrix
  // with all non-zero elements as true (Matlab compatibility).
  octave_image::value bw_value (args(0));
  if ((! bw_value.isnumeric () && ! bw_value.islogical ()) ||
      bw_value.ndims () != 2)
    error ("bwlabel: BW must be a 2D matrix");

  // For some reason, we can't use bool_matrix_value() to get a
  // a boolMatrix since it will error if there's values other
  // than 0 and 1 (whatever bool_array_value() does, bool_matrix_value()
  // does not).
  const boolMatrix BW = bw_value.bool_array_value ();

  // N-hood connectivity
  const octave_idx_type n = nargin < 2 ? 8 : args(1).idx_type_value ();
  if (n != 4 && n!= 6 && n != 8)
    error ("bwlabel: BW must be a 2 dimensional matrix");

  return bwlabel_2d (BW, n);
}

/*
%!shared in
%! in = rand (10) > 0.8;
%!assert (bwlabel (in, 4), bwlabeln (in, 4));
%!assert (bwlabel (in, 4), bwlabeln (in, [0 1 0; 1 1 1; 0 1 0]));
%!assert (bwlabel (in, 8), bwlabeln (in, 8));
%!assert (bwlabel (in, 8), bwlabeln (in, [1 1 1; 1 1 1; 1 1 1]));

%!assert (bwlabel (logical ([0 1 0; 0 0 0; 1 0 1])), [0 2 0; 0 0 0; 1 0 3]);
%!assert (bwlabel ([0 1 0; 0 0 0; 1 0 1]), [0 2 0; 0 0 0; 1 0 3]);

## Support any type of real non-zero value
%!assert (bwlabel ([0 -1 0; 0 0 0; 5 0 0.2]), [0 2 0; 0 0 0; 1 0 3]);

%!shared in, out
%!
%! in = [  0   1   1   0   0   1   0   0   0   0
%!         0   0   0   1   0   0   0   0   0   1
%!         0   1   1   0   0   0   0   0   1   1
%!         1   0   0   0   0   0   0   1   0   0
%!         0   0   0   0   0   1   1   0   0   0
%!         0   0   0   0   0   0   0   0   0   0
%!         0   0   0   1   0   0   0   0   0   0
%!         0   0   0   0   1   1   0   1   0   0
%!         0   0   0   1   0   1   0   1   0   1
%!         1   1   0   0   0   0   0   1   1   0];
%!
%! out = [ 0   3   3   0   0   9   0   0   0   0
%!         0   0   0   5   0   0   0   0   0  13
%!         0   4   4   0   0   0   0   0  13  13
%!         1   0   0   0   0   0   0  11   0   0
%!         0   0   0   0   0  10  10   0   0   0
%!         0   0   0   0   0   0   0   0   0   0
%!         0   0   0   6   0   0   0   0   0   0
%!         0   0   0   0   8   8   0  12   0   0
%!         0   0   0   7   0   8   0  12   0  14
%!         2   2   0   0   0   0   0  12  12   0];
%!assert (nthargout ([1 2], @bwlabel, in, 4), {out, 14});
%!assert (nthargout ([1 2], @bwlabel, logical (in), 4), {out, 14});
%!
%! out = [ 0   3   3   0   0   7   0   0   0   0
%!         0   0   0   3   0   0   0   0   0  11
%!         0   4   4   0   0   0   0   0  11  11
%!         1   0   0   0   0   0   0   9   0   0
%!         0   0   0   0   0   8   8   0   0   0
%!         0   0   0   0   0   0   0   0   0   0
%!         0   0   0   5   0   0   0   0   0   0
%!         0   0   0   0   5   5   0  10   0   0
%!         0   0   0   6   0   5   0  10   0  12
%!         2   2   0   0   0   0   0  10  10   0];
%!assert (nthargout ([1 2], @bwlabel, in, 6), {out, 12});
%!assert (nthargout ([1 2], @bwlabel, logical (in), 6), {out, 12});
%!
%! ## The labeled image is not the same as Matlab, but they are
%! ## labeled correctly. Do we really need to get them properly
%! ## ordered? (the algorithm in bwlabeln does it)
%! mout = [0   1   1   0   0   4   0   0   0   0
%!         0   0   0   1   0   0   0   0   0   5
%!         0   1   1   0   0   0   0   0   5   5
%!         1   0   0   0   0   0   0   5   0   0
%!         0   0   0   0   0   5   5   0   0   0
%!         0   0   0   0   0   0   0   0   0   0
%!         0   0   0   3   0   0   0   0   0   0
%!         0   0   0   0   3   3   0   6   0   0
%!         0   0   0   3   0   3   0   6   0   6
%!         2   2   0   0   0   0   0   6   6   0];
%!
%! out = [ 0   2   2   0   0   4   0   0   0   0
%!         0   0   0   2   0   0   0   0   0   5
%!         0   2   2   0   0   0   0   0   5   5
%!         2   0   0   0   0   0   0   5   0   0
%!         0   0   0   0   0   5   5   0   0   0
%!         0   0   0   0   0   0   0   0   0   0
%!         0   0   0   3   0   0   0   0   0   0
%!         0   0   0   0   3   3   0   6   0   0
%!         0   0   0   3   0   3   0   6   0   6
%!         1   1   0   0   0   0   0   6   6   0];
%!assert (nthargout ([1 2], @bwlabel, in, 8), {out, 6});
%!assert (nthargout ([1 2], @bwlabel, logical (in), 8), {out, 6});
%!
%!error bwlabel (rand (10, 10, 10) > 0.8, 4)
%!error bwlabel (rand (10) > 0.8, "text")
%!error bwlabel ("text", 6)
*/
