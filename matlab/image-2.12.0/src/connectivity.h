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

#ifndef OCTAVE_IMAGE_CONNDEF
#define OCTAVE_IMAGE_CONNDEF

#include <string>
#include <stdexcept>
#include <functional>

#include <octave/Array.h>
#include <octave/idx-vector.h>
#include <octave/dim-vector.h>
#include <octave/boolNDArray.h>
#include <octave/lo-ieee.h>  // octave_Inf

#include <octave/ov.h>

namespace octave
{
  namespace image
  {
    class connectivity
    {
      public:
        connectivity () = default;

        //! Will throw if val is bad
        explicit connectivity (const boolNDArray& mask_arg);
        explicit connectivity (const unsigned int conn);
        connectivity (const octave_idx_type& ndims, const std::string& type);

        boolNDArray mask;

        // For a matrix of size `size', what are the offsets for all of its
        // connected elements (will have negative and positive values).
        Array<octave_idx_type> neighbourhood (const dim_vector& size) const;
        Array<octave_idx_type> deleted_neighbourhood (const dim_vector& size) const;
        Array<octave_idx_type> positive_neighbourhood (const dim_vector& size) const;
        Array<octave_idx_type> negative_neighbourhood (const dim_vector& size) const;

        template<class T, class P>
        T create_padded (const T& image, const P& val) const;

        template<class T>
        void unpad (T& image) const;

        //! Return a logical mask of elements that are part of the padding.
        static boolNDArray padding_mask (const dim_vector& size,
                                         const dim_vector& padded_size);

        //! Set the padding elements to a specific value.
        template<class T, class P>
        static void set_padding (const dim_vector& size,
                                 const dim_vector& padded_size,
                                 T& im, const P& val);

        template<class P>
        static P min_value (void);

        static Array<octave_idx_type> padding_lengths (const dim_vector& size,
                                                       const dim_vector& padded_size);

      private:
        //! Like Array::ndims() but will return 1 dimension for ColumnVector
        static octave_idx_type ndims (const dim_vector& d);
        template<class T>
        static octave_idx_type ndims (const Array<T>& a);
    };

    class invalid_connectivity : public std::invalid_argument
    {
      public:
        invalid_connectivity (const std::string& what_arg)
          : std::invalid_argument (what_arg) { }
    };

    connectivity conndef (const octave_value& val);
  }
}

// Templated methods

template<class T, class P>
T
octave::image::connectivity::create_padded (const T& image, const P& val) const
{
  const octave_idx_type pad_ndims = std::min (mask.ndims (), image.ndims ());

  Array<octave_idx_type> idx (dim_vector (image.ndims (), 1), 0);
  dim_vector padded_size = image.dims ();
  for (octave_idx_type i = 0; i < pad_ndims; i++)
    {
      padded_size(i) += 2;
      idx(i) = 1;
    }

  T padded (padded_size, val);

  // padded(2:end-1, 2:end-1, ..., 2:end-1) = BW
  padded.insert (image, idx);
  return padded;
}

template<class T>
void
octave::image::connectivity::unpad (T& image) const
{
  const octave_idx_type pad_ndims = std::min (mask.ndims (), image.ndims ());
  const dim_vector padded_size = image.dims ();

  Array<idx_vector> inner_slice (dim_vector (image.ndims (), 1));
  for (octave_idx_type i = 0; i < pad_ndims ; i++)
    inner_slice(i) = idx_vector (1, padded_size(i) - 1);
  for (octave_idx_type i = pad_ndims; i < image.ndims (); i++)
    inner_slice(i) = idx_vector (0, padded_size(i));

  image = image.index (inner_slice);
  return;
}

template<class P>
P
octave::image::connectivity::min_value (void)
{
  if (typeid (P) == typeid (bool))
    return false;
  else
    return P(-octave_Inf);
}

template<class T, class P>
void
octave::image::connectivity::set_padding (const dim_vector& size,
                                          const dim_vector& padded_size,
                                          T& im, const P& val)
{
  P* im_v = im.fortran_vec ();

  const Array<octave_idx_type> lengths = padding_lengths (size, padded_size);
  const octave_idx_type* lengths_v = lengths.fortran_vec ();

  const octave_idx_type* strides_v = size.to_jit ();
  const octave_idx_type row_stride = strides_v[0];

  std::function<void(const octave_idx_type)> fill;
  fill = [&] (const octave_idx_type dim) -> void
  {
    for (octave_idx_type i = 0; i < lengths_v[dim]; i++, im_v++)
      *im_v = val;

    if (dim == 0)
      im_v += row_stride;
    else
      for (octave_idx_type i = 0; i < strides_v[dim]; i++)
        fill (dim -1);

    for (octave_idx_type i = 0; i < lengths_v[dim]; i++, im_v++)
      *im_v = val;
  };
  fill (im.ndims () -1);
}

#endif
