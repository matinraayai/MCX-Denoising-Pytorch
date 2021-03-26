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

// This is wrapper class for the @strel class so that it can be used by
// the rest of the image package using SE's. It's not a perfect wrapper
// on purpose. For example, the reflect method behaves kinda weird for
// matlab compatibility. In here we try to make a bit more sense.

#include "strel.h"

#include <vector>

#include <octave/oct.h>

#include <octave/parse.h>  // gives us feval so we can use @strel
#include <octave/oct-map.h>

#define WANTS_FEVAL 1
#define WANTS_OCTAVE_IMAGE_VALUE 1
#include "octave-wrappers.h"

// Constructors

// Expects a @strel object, or a boolean matrix (or something
// that can be converted into one with bool_matrix_value()
octave::image::strel::strel (const octave_value& arg)
{
  octave_value se = arg;
  // We are only creating a strel object here so that we can use
  // getsequence which tries to guess a shape to decompose. In
  // the cases where we don't get a @strel object we could also:
  //
  // 1) don't do any automatic decomposition, use the matrix as
  //    it is.
  // 2) implement the guessing in C++ and make a oct file for it.

  // If we don't have a strel object, then make one.
  if (se.class_name () != "strel")
    {
      octave_value_list strel_args (2);
      strel_args(0) = "arbitrary";
      strel_args(1) = se;
      // We are leaving the input check up to @strel
      se = octave_image::feval ("strel", strel_args)(0);
    }

  nhood   = octave_image::feval ("getnhood",  se)(0).bool_array_value ();
  height  = octave_image::feval ("getheight", se)(0).array_value ();

  // Maybe we should handle this better but how?  This makes imerode
  // and imdilate work with a [] strel
  if (octave_image::value (nhood).isempty ())
    {
      nhood = boolNDArray (dim_vector (1, 1), false);
      height = NDArray (dim_vector (1, 1), 0);
    }

  ini_ctor ();
  origin  = default_origin ();

  const octave_value se_seq = octave_image::feval ("getsequence", se)(0);
  const octave_idx_type seq_numel
    = octave_image::feval ("numel", se_seq)(0).idx_type_value ();

  // This is to emulate the strel_obj(idx) syntax in function form
  static const char *fields[] = {"type", "subs", 0};
  octave_scalar_map ref = octave_scalar_map (string_vector (fields));
  ref.setfield ("type", octave_value ("()"));
  octave_value_list subsref_args (2);
  subsref_args(0) = se_seq;

  if (seq_numel > 1)
    {
      for (octave_idx_type subs = 0; subs < seq_numel; subs++)
        {
          // subs+1 because octave is index base 1
          ref.setfield ("subs", Cell (octave_value (subs+1)));
          subsref_args(1) = ref;
          // Equivalent to "selem = strel_obj(subs)"
          const octave_value_list elem
            = octave_image::feval ("subsref", subsref_args)(0);

          const boolNDArray elem_nhood
            = octave_image::feval ("getnhood",  elem)(0).bool_array_value ();
          const NDArray elem_height
            = octave_image::feval ("getheight", elem)(0).array_value ();

          decomposition.push_back (strel (elem_nhood, elem_height));
        }
    }

  end_ctor ();
  return;
}

octave::image::strel::strel (const boolNDArray& nhood, const NDArray& height)
  : nhood (nhood), height (height)
{
  ini_ctor ();
  origin = default_origin ();
  end_ctor ();
  return;
}

octave::image::strel::strel (const boolNDArray& nhood, const NDArray& height,
                             const Array<octave_idx_type>& origin)
  : nhood (nhood), height (height), origin (origin)
{
  ini_ctor ();
  validate_origin ();
  end_ctor ();
  return;
}

boolNDArray
octave::image::strel::get_nhood (void) const
{
  return nhood;
}

octave_idx_type
octave::image::strel::get_nnz (void) const
{
  return nnz;
}

Array<octave_idx_type>
octave::image::strel::get_origin (void) const
{
  return origin;
}

octave::image::strel
octave::image::strel::operator () (const octave_idx_type& i) const
{
  assert (i >= 0 && i < octave_idx_type (decomposition.size ()));
  return decomposition[i];
}

// Number of strel elements after decomposition
octave_idx_type
octave::image::strel::numel (void) const
{
  return octave_idx_type (decomposition.size ());
}

// Pretty much rotates the matrix by 180 degrees in all dimensions. The
// only tricky thing that we doo it purpose is that the origin is also
// rotated. For example, if the origin is set to the bottom right point,
// after refleection it will be in the top left. If the origin is at the
// center of the matrix, there will be no change.
// The reason for this is so that we can keep the origin in matrices with
// sides with an even length.
octave::image::strel
octave::image::strel::reflect (void) const
{
  boolNDArray ref_nhood   (size);
  NDArray     ref_height  (size);
  const octave_idx_type numel = nhood.numel ();
  for (octave_idx_type ind = 0; ind < numel; ind++)
    {
      ref_nhood(ind)  = nhood(numel - ind -1);
      ref_height(ind) = height(numel - ind -1);
    }
  Array<octave_idx_type> ref_origin (origin);
  for (octave_idx_type dim = 0; dim < ndims; dim++)
    ref_origin(dim) = size(dim) - origin(dim) -1;

  return octave::image::strel (ref_nhood, ref_height, ref_origin);
}

void
octave::image::strel::set_origin (const Array<octave_idx_type>& sub)
{
  origin = sub;
  validate_origin ();
  return;
}

bool
octave::image::strel::flat (void) const
{
  bool flat = true;
  if (! height.all_elements_are_zero ())
    {
      const octave_idx_type numel = height.numel ();
      for (octave_idx_type ind = 0; ind < numel; ind++)
        {
          if (height(ind))
            {
              flat = false;
              break;
            }
        }
    }
  return flat;
}

// For any given point in the input matrix, calculates the memory offset for
// all the others that will have an effect on the erosion and dilation with it.
// How much we need to shift the input matrix to cover the nnz of the SE.
// That is, how many elements away is each nnz of the SE, for any element
// of the input matrix. Given a 10x10 input matrix (cumulative dimensions
// of [10 100]), and a SE with:
//   [1 0 0
//    1 1 1
//    0 0 1]
// linear shift is [0 1 11 21 22]
// The second element is the matching height for each.
Array<octave_idx_type>
octave::image::strel::offsets (const dim_vector& cum_size) const
{
  Array<octave_idx_type> sub (dim_vector (ndims, 1), 0);
  Array<octave_idx_type> offsets (dim_vector (nnz, 1));

  for (octave_idx_type found = 0; found < nnz; boolNDArray::increment_index (sub, size))
    {
      if (nhood(sub))
        {
          offsets(found) = sub(0);
          for (octave_idx_type dim = 1; dim < ndims; dim++)
            offsets(found) += cum_size(dim-1) * sub(dim);
          found++;
        }
    }
  return offsets;
}

// The final size of the output matrix will be the size of the input
// matrix, plus the size of the SE, less its center. Consider a square SE
// at the corner of the input matrix. The origin (center) of the SE will be
// at coordinates (0,0) of the input matrix and we need enough padding for
// it. If the shape is "full", then we add the double.
Array<octave_idx_type>
octave::image::strel::pre_pad (const octave_idx_type& mt_ndims,
                               const std::string& shape) const
{
  Array<octave_idx_type> pad (dim_vector (mt_ndims, 1), 0);

  octave_idx_type pad_times;
  if (shape == "valid")
    return pad;
  else if (shape == "same")
    pad_times = 1;
  else if (shape == "full")
    pad_times = 2;
  else
    error ("invalid SHAPE");

  Array<octave_idx_type> resized_origin (origin);
  dim_vector             resized_size (size);
  if (ndims < mt_ndims)
    {
      resized_origin.resize (dim_vector (mt_ndims, 1), 0);
      resized_size.resize (mt_ndims, 1);
    }

  for (octave_idx_type dim = 0; dim < mt_ndims; dim++)
    pad(dim) = resized_origin(dim) * pad_times;

  return pad;
}

Array<octave_idx_type>
octave::image::strel::post_pad (const octave_idx_type& mt_ndims,
                                const std::string& shape) const
{
  Array<octave_idx_type> pad (dim_vector (mt_ndims, 1), 0);

  octave_idx_type pad_times;
  if (shape == "valid")
    return pad;
  else if (shape == "same")
    pad_times = 1;
  else if (shape == "full")
    pad_times = 2;
  else
    error ("invalid SHAPE");

  Array<octave_idx_type> resized_origin (origin);
  dim_vector             resized_size (size);
  if (ndims < mt_ndims)
    {
      resized_origin.resize (dim_vector (mt_ndims, 1), 0);
      resized_size.resize (mt_ndims, 1);
    }

  for (octave_idx_type dim = 0; dim < mt_ndims; dim++)
    pad(dim) = (resized_size(dim) - resized_origin(dim) -1) * pad_times;

  return pad;
}

void
octave::image::strel::ini_ctor ()
{
  size    = nhood.dims ();
  ndims   = nhood.ndims ();
  nnz     = nhood.nnz ();
  return;
}

Array<octave_idx_type>
octave::image::strel::default_origin ()
{
  Array<octave_idx_type> origin (dim_vector (ndims, 1));
  for (octave_idx_type dim = 0; dim < ndims; dim++)
    origin(dim) = floor ((size(dim) +1) /2) -1; // -1 for zero based indexing

  return origin;
}

void
octave::image::strel::end_ctor (void)
{
  if (decomposition.empty ())
    decomposition.push_back (*this);
}

void
octave::image::strel::validate_origin (void)
{
  assert (ndims == origin.numel ());
  for (octave_idx_type dim = 0; dim < ndims; dim++)
    assert (origin(dim) >= 0 && origin(dim) < size(dim));
  return;
}
