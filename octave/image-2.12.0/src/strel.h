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

// An important thing about this class is how the origin (defaults to
// center coordinates which can have different interpretations when
// sides are of even length) moves when it's reflected. Consider the
// following (x is the origin)
//
//  o o o                   o o o
//  o x o  -- reflect -->   o x o
//  o o o                   o o o
//
//  o o o o                   o o o o
//  o o x o  -- reflect -->   o x o o
//  o o o o                   o o o o


#ifndef OCTAVE_IMAGE_STREL
#define OCTAVE_IMAGE_STREL

#include <vector>
#include <string>

#include <octave/Array.h>
#include <octave/dNDArray.h>
#include <octave/dim-vector.h>
#include <octave/boolNDArray.h>

#include <octave/ov.h>

namespace octave
{
  namespace image
  {
    class strel
    {
      public:
        explicit strel (const octave_value& arg);
        strel (const boolNDArray& nhood, const NDArray& height);
        strel (const boolNDArray& nhood, const NDArray& height,
               const Array<octave_idx_type>& origin);

        boolNDArray             get_nhood (void) const;
        octave_idx_type         get_nnz (void) const;
        Array<octave_idx_type>  get_origin (void) const;
        strel                   operator () (const octave_idx_type& i) const;

        // Number of strel objects after decomposition, NOT numel of nhood
        octave_idx_type         numel (void) const;

        // flat SE?
        bool flat (void) const;

        // set origin of the SE to specific coordinates
        void set_origin (const Array<octave_idx_type>& sub);

        // reflect the SE (rotates is 180 degrees in all dimensions).
        // Note that when rotating it, the origin coordinates move with it
        // (this is by design see ratinoal on top of this file).
        strel reflect (void) const;

        // given a matrix with a specific cumulative size, what's the offset
        // for each true element of the nhood from the first element (true or
        // false) of the nhood.
        Array<octave_idx_type> offsets (const dim_vector& cum_size) const;

        // array with height of each true element in the nhood (same order
        // as the one returned by offsets).
        template<class P>
        Array<P> true_heights (void) const;

        Array<octave_idx_type> pre_pad  (const octave_idx_type& mt_ndims,
                                         const std::string& shape) const;
        Array<octave_idx_type> post_pad (const octave_idx_type& mt_ndims,
                                         const std::string& shape) const;

      private:
        boolNDArray             nhood;
        NDArray                 height;
        octave_idx_type         nnz;
        Array<octave_idx_type>  origin;
        dim_vector              size;
        octave_idx_type         ndims;
        std::vector<strel>      decomposition;

        void ini_ctor (void);

        Array<octave_idx_type> default_origin (void);

        void end_ctor (void);

        void validate_origin (void);

    };
  }
}

// Define it on header or we we need to instantiate it for all
// possible classes in strel.cc
template<class P>
Array<P>
octave::image::strel::true_heights (void) const
{
  Array<P> true_heights (dim_vector (nnz, 1));
  for (octave_idx_type ind = 0, found = 0; found < nnz; ind++)
    if (nhood(ind))
      true_heights(found++) = height(ind);
  return true_heights;
}

#endif
