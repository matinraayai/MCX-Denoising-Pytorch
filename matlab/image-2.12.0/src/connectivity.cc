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

#include "connectivity.h"

#include <string>
#include <vector>
#include <typeinfo>

#include <octave/oct.h>

#define WANTS_OCTAVE_IMAGE_VALUE 1
#include "octave-wrappers.h"

using namespace octave::image;


connectivity::connectivity (const boolNDArray& mask_arg)
{
  mask = mask_arg;

  // Must be 1x1, 3x1, or 3x3x3x...x3
  const octave_idx_type numel = mask.numel ();
  const dim_vector dims = mask.dims ();
  const octave_idx_type ndims = connectivity::ndims (dims);

  for (octave_idx_type i = 0; i < ndims; i++)
    if (dims(i) != 3)
      throw invalid_connectivity ("is not 1x1, 3x1, 3x3, or 3x3x...x3");

  // Center must be true
  const octave_idx_type center = floor (numel /2);
  if (! mask(center))
    throw invalid_connectivity ("center is not true");

  // Must be symmetric relative to its center
  const bool* start = mask.fortran_vec ();
  const bool* end   = mask.fortran_vec () + (numel -1);
  for (octave_idx_type i = 0; i < center; i++)
    if (start[i] != end[-i])
      throw invalid_connectivity ("is not symmetric relative to its center");

  return;
}

connectivity::connectivity (const unsigned int conn)
{
  if (conn == 4)
    {
      mask = boolNDArray (dim_vector (3, 3), true);
      bool* md = mask.fortran_vec ();
      md[ 0] = false;
      md[ 2] = false;
      md[ 6] = false;
      md[ 8] = false;
    }
  else if (conn == 6)
    {
      mask = boolNDArray (dim_vector (3, 3, 3), false);
      bool* md = mask.fortran_vec ();
      md[ 4] = true;
      md[10] = true;
      md[12] = true;
      md[13] = true;
      md[14] = true;
      md[16] = true;
      md[22] = true;
    }
  else if (conn == 8)
    mask = boolNDArray (dim_vector (3, 3), true);
  else if (conn == 18)
    {
      mask = boolNDArray (dim_vector (3, 3, 3), true);
      bool* md = mask.fortran_vec ();
      md[ 0] = false;
      md[ 2] = false;
      md[ 6] = false;
      md[ 8] = false;
      md[18] = false;
      md[20] = false;
      md[24] = false;
      md[26] = false;
    }
  else if (conn == 26)
    mask = boolNDArray (dim_vector (3, 3, 3), true);
  else
    throw invalid_connectivity ("must be in the set [4 6 8 18 26]"
                                " (was " + std::to_string (conn) + ")");

  return;
}


connectivity::connectivity (const octave_idx_type& ndims,
                            const std::string& type)
{
  dim_vector size;
  if (ndims == 1)
    size = dim_vector (3, 1);
  else
    {
      size = dim_vector (3, 3);
      size.resize (ndims, 3);
    }

  if (type == "maximal")
    {
      mask = boolNDArray (size, true);
    }
  else if (type == "minimal")
    {
      mask = boolNDArray (size, false);
      bool* md = mask.fortran_vec ();

      md += int (floor (pow (3, ndims) /2));  // move to center
      md[0] = true;
      for (octave_idx_type dim = 0; dim < ndims; dim++)
        {
          const octave_idx_type stride = pow (3, dim);
          md[ stride] = true;
          md[-stride] = true;
        }
    }
  else
    throw invalid_connectivity ("must be \"maximal\" or \"minimal\"");

  return;
}


// A couple of things:
//  * it is handy that the offsets come sorted since they will be used to
//    access the elements and we want to jump around as little as possible.
//  * the number of dimensions used may be different than the mask.
Array<octave_idx_type>
connectivity::neighbourhood (const dim_vector& size) const
{
  const octave_idx_type ndims = connectivity::ndims (mask);
  const octave_idx_type numel = mask.numel ();

  // offset to adjacent element on correspoding dimension
  Array<octave_idx_type> strides (dim_vector (ndims, 1));
  strides(0) = 1;
  for (octave_idx_type dim = 1; dim < ndims; dim++)
    strides(dim) = strides(dim-1) * size(dim-1);

  Array<octave_idx_type> pow3 (dim_vector (ndims, 1));
  pow3(0) = 1;
  for (octave_idx_type dim = 1; dim < ndims; dim++)
    pow3(dim) = pow3(dim-1) * 3;

  // We calculate this for all elements. We could do it only for the "true"
  // elements but that's slightly more complex and in most cases we will
  // already want most, if not all, elements anyway.
  Array<octave_idx_type> all_offsets (dim_vector (numel, 1), 0);
  for (octave_idx_type dim = 0; dim < ndims; dim++)
    {
      octave_idx_type i (0);

      for (int x = 0; x < pow3(ndims -1 -dim); x++)
        {
          for (octave_idx_type k = 0; k < pow3(dim); k++)
            all_offsets(i++) -= strides(dim);
          i += pow3(dim);
          for (octave_idx_type k = 0; k < pow3(dim); k++)
            all_offsets(i++) += strides(dim);
        }
    }

  octave_idx_type start_idx = 0;
  for (octave_idx_type dim = ndims; dim > connectivity::ndims (size); dim--)
    start_idx += pow3(dim -1);

  const bool* m = mask.fortran_vec ();
  const octave_idx_type* ao = all_offsets.fortran_vec ();

  octave_idx_type nnz = 0;
  for (octave_idx_type i = start_idx; i < (numel - start_idx); i++)
    if (m[i])
      nnz++;

  Array<octave_idx_type> offsets (dim_vector (nnz, 1));
  octave_idx_type* o = offsets.fortran_vec ();
  for (octave_idx_type i = start_idx, j = 0; i < (numel - start_idx); i++)
    if (m[i])
      o[j++] = ao[i];

  return offsets;
}

Array<octave_idx_type>
connectivity::deleted_neighbourhood (const dim_vector& size) const
{
  Array<octave_idx_type> offsets = neighbourhood (size);
  for (octave_idx_type i = 0; i < offsets.numel (); i++)
    if (offsets(i) == 0)
      offsets.delete_elements (idx_vector (i));
  return offsets;
}

Array<octave_idx_type>
connectivity::positive_neighbourhood (const dim_vector& size) const
{
  Array<octave_idx_type> offsets = neighbourhood (size);
  std::vector<octave_idx_type> to_keep;

  for (octave_idx_type i = 0; i < offsets.numel (); i++)
    if (offsets(i) > 0)
      to_keep.push_back (offsets(i));

  octave_idx_type numel = to_keep.size ();
  Array<octave_idx_type> neg (dim_vector (numel, 1));
  for (octave_idx_type i = 0; i < numel; i++)
    neg(i) = to_keep[i];

  return neg;
}

Array<octave_idx_type>
connectivity::negative_neighbourhood (const dim_vector& size) const
{
  Array<octave_idx_type> offsets = neighbourhood (size);
  std::vector<octave_idx_type> to_keep;

  for (octave_idx_type i = 0; i < offsets.numel (); i++)
    if (offsets(i) < 0)
      to_keep.push_back (offsets(i));

  octave_idx_type numel = to_keep.size ();
  Array<octave_idx_type> neg (dim_vector (numel, 1));
  for (octave_idx_type i = 0; i < numel; i++)
    neg(i) = to_keep[i];

  return neg;
}


octave_idx_type
connectivity::ndims (const dim_vector& dims)
{
  // We do not bother with 1x3 arrays since those are not valid
  // connectivity masks anyway.
  if (dims(1) == 1)
    {
      if (dims(0) == 1)
        return 0;
      else
        return 1;
    }
  else
    return dims.length ();
}

template<class T>
octave_idx_type
connectivity::ndims (const Array<T>& a)
{
  return connectivity::ndims (a.dims ());
}


Array<octave_idx_type>
connectivity::padding_lengths (const dim_vector& size,
                               const dim_vector& padded_size)
{
  const octave_idx_type ndims = size.length ();
  Array<octave_idx_type> lengths (dim_vector (ndims, 1), 0);
  lengths(0) = 1;
  for (octave_idx_type i = 1; i < ndims; i++)
    if (size(i) < padded_size(i))
      lengths(i) = lengths(i -1) * padded_size(i-1);
  return lengths;
}

boolNDArray
connectivity::padding_mask (const dim_vector& size,
                            const dim_vector& padded_size)
{
  boolNDArray mask (padded_size, false);
  set_padding (size, padded_size, mask, true);
  return mask;
}


connectivity
octave::image::conndef (const octave_value& _val)
{
  octave_image::value val (_val);

  // A mask may not not be of type logical/bool, it can be of any
  // numeric type as long all values are zeros and ones (usually
  // manually typed masks which by default are of type double.
  if (val.islogical ()
      || (val.isnumeric() && ! val.array_value ().any_element_not_one_or_zero ()))
    {
      try
        {
          return connectivity (val.bool_array_value ());
        }
      catch (invalid_connectivity& e)
        {
          error ("conndef: CONN %s", e.what ());
        }
    }
  else if (val.isnumeric () && val.is_scalar_type ())
    {
      if (val.double_value () < 1)
        error ("conndef: if CONN is a scalar it must be a positive number");
      try
        {
          return connectivity (val.uint_value ());
        }
      catch (invalid_connectivity& e)
        {
          error ("conndef: CONN %s", e.what ());
        }
    }
  else
    error ("conndef: CONN must either be a logical array or a numeric scalar");
}
