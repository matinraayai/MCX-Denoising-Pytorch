// Copyright (C) 2015 Carnë Draug <carandraug@octave.org>
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <queue>

#include <octave/oct.h>
#include <octave/error.h>
#include <octave/parse.h>
#include <octave/Cell.h>
#include <octave/oct-map.h>

#define WANTS_FEVAL 1
#define WANTS_OCTAVE_IMAGE_VALUE 1
#include "octave-wrappers.h"

#include "connectivity.h"
using namespace octave::image;

template<class T>
static boolNDArray
imregionalmin (const T& im, const connectivity& conn)
{
  octave_value_list args (2);
  args(0) = im;
  args(1) = conn.mask;
  const octave_value regional_min = octave_image::feval ("imregionalmin",
                                                         args)(0);
  return regional_min.bool_array_value ();
}

static NDArray
bwlabeln (const boolNDArray& bw, const connectivity& conn)
{
  octave_value_list args (2);
  args(0) = bw;
  args(1) = conn.mask;
  const octave_value label = octave_image::feval ("bwlabeln", args)(0);
  return label.array_value ();
}

// Implements watershed in a quite naïve way.  From the wikipedia, named
// "Meyer's flooding algorithm" (but I could not find the actual paper
// that reports it).  There are faster (and also nicer results) algorithms,
// but this is the only one I found that matches Matlab results.
//
//  1.  A set of markers, pixels where the flooding shall start, are chosen.
//      Each is given a different label.
//  2.  The neighboring pixels of each marked area are inserted into a
//      priority queue with a priority level corresponding to the gray level
//      of the pixel.
//  3.  The pixel with the lowest priority level is extracted from the
//      priority queue. If the neighbors of the extracted pixel that have
//      already been labeled all have the same label, then the pixel is
//      labeled with their label. All non-marked neighbors that are not yet
//      in the priority queue are put into the priority queue.
//  4.  Redo step 3 until the priority queue is empty.
//
// There is a detail missing on the description above.  On step 3, if the
// labeled neighbours do *not* have the same label, should the non-labeled
// neighbours be added to the queue?  Apparently not.

template<class P>
class
Voxel
{
  public:
    P val;
    octave_idx_type idx;
    // We need this to sort elements with the same priority.  We need them
    // to come out in the same order they went in.
    octave_idx_type pos;

    Voxel (const P val, const octave_idx_type idx, const octave_idx_type pos)
      : val (val), idx (idx), pos (pos)
    { }

    inline bool
    operator>(const Voxel& rhs) const
    {
      if (val == rhs.val)
        return pos > rhs.pos;
      else
        return val > rhs.val;
    }
};

// As part of this algorithm, we will check the neighbourhood for existing
// labels.  We don't know in advance the number of labeled neighbours, or
// where the first label will be.  But we do know the length of the
// neighbourhood.
template<class T>
class
Collection
{
  public:
    explicit Collection (const octave_idx_type n) : data (new T[n])
    { }

    ~Collection (void)
    { delete [] data; }

    inline octave_idx_type
    numel (void) const
    { return count; }

    inline void
    push_back (const T val)
    { data[count++] = val; }

    inline void
    reset (void)
    { count = 0; }

  protected:
    T* data = NULL;
    octave_idx_type count = 0;

  private:
    // Disable default and copy constructor and assignment
    Collection (void);
    Collection (Collection const& other);
    Collection& operator = (Collection const& other);
};


class
LabelsCollection : public Collection<double>
{
  public:
    using Collection<double>::Collection;

    inline double
    label (void) const
    { return *data; }

    inline bool
    all_equal (void) const
    {
      for (octave_idx_type i = 0; i < count; i++)
        if (data[0] != data[i])
          return false;
      return true;
    }
};

class
IdxCollection : public Collection<octave_idx_type>
{
  public:
    using Collection<octave_idx_type>::Collection;

    inline octave_idx_type
    operator [] (octave_idx_type i) const
    { return data[i]; }
};


template<class T>
NDArray
watershed (const T& im, const connectivity& conn)
{
  typedef typename T::element_type P;

//  1.  A set of markers, pixels where the flooding shall start, are chosen.
//      Each is given a different label.
  const boolNDArray markers = imregionalmin (im, conn);
  boolNDArray padded_markers = conn.create_padded (markers, false);
  NDArray label_array = bwlabeln (padded_markers, conn);
  double* label = label_array.fortran_vec ();

  const T  padded_im_array = conn.create_padded (im, 0);
  const P* padded_im = padded_im_array.fortran_vec ();

  const Array<octave_idx_type> neighbours_array
    = conn.deleted_neighbourhood (padded_im_array.dims ());
  const octave_idx_type* neighbours = neighbours_array.fortran_vec ();
  const octave_idx_type n_neighbours = neighbours_array.numel ();

  // We need two flags per voxel for this implementation:
  //  1. Whether a voxel has been labelled or not.  (TODO profile this later,
  //     maybe it's enough to do label > 0)
  //  2. Whether a voxel can go into the queue.  Reasons to not go into
  //    the queue are: it's a padding voxel, it's already in the queue,
  //    it's already been labelled.

  bool* label_flag = padded_markers.fortran_vec ();

  boolNDArray queue_flag_array (padded_markers);
  connectivity::set_padding (markers.dims (), padded_markers.dims (),
                             queue_flag_array, true);
  bool* queue_flag = queue_flag_array.fortran_vec ();

  const octave_idx_type n = padded_im_array.numel ();
  octave_idx_type pos = 0;

//  2.  The neighboring pixels of each marked area are inserted into a
//      priority queue with a priority level corresponding to the gray level
//      of the pixel.
  std::priority_queue<Voxel<P>, std::vector<Voxel<P>>, std::greater<Voxel<P>>> q;
  for (octave_idx_type i = 0; i < n; i++)
    if (label_flag[i])
      for (octave_idx_type j = 0; j < n_neighbours; j++)
        {
          const octave_idx_type ij = i + neighbours[j];
          if (! queue_flag[ij])
            {
              queue_flag[ij] = true;
              q.push (Voxel<P> (padded_im[ij], ij, pos++));
            }
        }

//  3.  The pixel with the lowest priority level is extracted from the
//      priority queue. If the neighbors of the extracted pixel that have
//      already been labeled all have the same label, then the pixel is
//      labeled with their label. All non-marked neighbors that are not yet
//      in the priority queue are put into the priority queue.
//  4.  Redo step 3 until the priority queue is empty.
//
// There is a detail missing on the description above.  On step 3, if the
// labeled neighbours do *not* have the same label, should the non-labeled
// neighbours be added to the queue?  Apparently not.
  LabelsCollection lc (n_neighbours);
  IdxCollection ic (n_neighbours);
  while (! q.empty ())
    {
      Voxel<P> v = q.top ();
      q.pop ();

      lc.reset ();
      ic.reset ();
      for (octave_idx_type j = 0; j < n_neighbours; j++)
        {
          const octave_idx_type ij = v.idx + neighbours[j];
          if (label_flag[ij])
            lc.push_back(label[ij]);
          else if (! queue_flag[ij])
            ic.push_back(ij);
        }
      if (lc.numel () > 0 && lc.all_equal ())
        {
          label[v.idx] = lc.label ();
          label_flag[v.idx] = true;
          for (octave_idx_type i = 0; i < ic.numel (); i++)
            {
              const octave_idx_type ij = ic[i];
              queue_flag[ij] = true;
              q.push (Voxel<P> (padded_im[ij], ij, pos++));
            }
        }
    }

  conn.unpad (label_array);
  return label_array;
}


DEFUN_DLD(watershed, args, , "\
-*- texinfo -*-\n\
@deftypefn  {Function File} {} watershed (@var{im})\n\
@deftypefnx {Function File} {} watershed (@var{im}, @var{conn})\n\
Compute watershed transform.\n\
\n\
Computes by immersion\n\
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
@seealso{bwdist, bwlabeln, regionprops}\n\
@end deftypefn")
{
  const octave_idx_type nargin = args.length ();
  if (nargin < 1 || nargin > 2)
    print_usage ();

  connectivity conn;
  if (nargin > 1)
    conn = octave::image::conndef (args(1));
  else
    {
      try
        {
          conn = connectivity (args(0).ndims (), "maximal");
        }
      catch (invalid_connectivity& e)
        {
          error ("bwconncomp: failed to create MASK (%s)", e.what ());
        }
    }

  const octave_image::value im (args(0));
#define IF_TYPE(IS_TYPE, VALUE_TYPE) \
  if (im.is ## IS_TYPE ()) \
    return octave_value (watershed (im. VALUE_TYPE ## array_value (), \
                                    conn)); \

  // My guess is that uint8, uint16, and double will be the most common types.
  IF_TYPE(_uint8_type, uint8_)
  else IF_TYPE(_uint16_type, uint16_)
  else if (im.isfloat ())
    {
      if (im.iscomplex ())
        {
          IF_TYPE(_double_type, complex_)
          else IF_TYPE(_single_type, float_complex_)
        }
      else
        {
          IF_TYPE(_double_type, )
          else IF_TYPE(_single_type, float_)
        }
    }
  else IF_TYPE(_uint32_type, uint32_)
  else IF_TYPE(_uint64_type, uint64_)
  else IF_TYPE(_int8_type, int8_)
  else IF_TYPE(_int16_type, int16_)
  else IF_TYPE(_int32_type, int32_)
  else IF_TYPE(_int64_type, int64_)
  else IF_TYPE(_uint8_type, uint8_)
  else IF_TYPE(logical, bool_)

  // default case if all other above fail.
  error ("watershed: IM of unsupported class `%s'",
         im.class_name ().c_str ());

#undef IF_TYPE
}

/*
## Some simple tests that will check the multiple ways to measure
## distances (comes to light on plateus)
%!test
%! ex = tril (ones (50), -1) + triu (repmat (2, [50 50]), 2);
%! ex(1, 1) = 1;
%! ex(end, end) = 1;
%!
%! in = ones (50);
%! in(end,1) = 0;
%! in(1,end) = 0;
%! assert (watershed (in), ex)

%!test
%! ex = tril (ones (49), -1) + triu (repmat (2, [49 49]), 2);
%! ex(1, 1) = 1;
%! ex(end, end) = 1;
%!
%! in = ones (49);
%! in(end,1) = 0;
%! in(1,end) = 0;
%! assert (watershed (in), ex)
%!
%! c = (fspecial ('disk', 5) > 0) + 1;
%! in(20:30,20:30) = c;
%! c = (fspecial ('disk', 4) > 0) + 2;
%! in(21:29,21:29) = c;
%! assert (watershed (in), ex)

%!test
%! ex = tril (ones (49), -1) + triu (repmat (2, [49 49]), 2);
%! ex(1:28,1:28) = (tril (ones (28) ,7) + triu (repmat (2, [28 28]), 10));
%! ex(1,9) = 1;
%! ex(end,end) = 1;
%! ex(20:29, 29) = 0;
%!
%! in = ones (49);
%! in(end,1) = 0;
%! in(1,end) = 0;
%! c = (fspecial ("disk", 5) > 0) + 1;
%! in(1:11,38:48) = c;
%!
%! assert (watershed (in), ex)

## See http://perso.esiee.fr/~info/tw/index.html for a page on topological
## watershed.  The following test cases were taken from a powerpoint
## presentation there http://perso.esiee.fr/~info/tw/isis03b.ppt
## "A topological approach to watersheds". Presentation made by Gilles Bertrand
## at the ISIS Workshop on Mathematical Morphology in Paris, France, 2003.
##
## From that presentation, the algorithm we must implement for Matlab
## compatibility is named "Meyer".

%!test
%! im = [
%!     3     4     5     6     0
%!     2     3     4     5     6
%!     1     2     3     4     5
%!     0     1     2     3     4
%!     1     0     1     2     3];
%!
%! labeled8 = [
%!     1     1     1     0     2
%!     1     1     1     0     0
%!     1     1     1     1     1
%!     1     1     1     1     1
%!     1     1     1     1     1];
%! labeled4 = [
%!     1     1     1     0     3
%!     1     1     1     0     0
%!     1     1     0     2     2
%!     1     0     2     2     2
%!     0     2     2     2     2];
%! labeled_weird = [
%!     1     1     1     0     2
%!     1     1     1     1     0
%!     1     1     1     1     1
%!     1     1     1     1     1
%!     1     1     1     1     1];
%!
%! assert (watershed (im), labeled8);
%! assert (watershed (im, 8), labeled8);
%! assert (watershed (im, 4), labeled4);
%! assert (watershed (im, [1 1 0; 1 1 1; 0 1 1]), labeled_weird);

%!test
%! im = [
%!     2     3    30     2
%!     3    30     3    30
%!   255    31    30     4
%!     2   255    31    30
%!     1     2   255     5];
%!
%! labeled4 = [
%!     1     1     0     4
%!     1     0     3     0
%!     0     2     0     5
%!     2     2     2     0
%!     2     2     0     6];
%! labeled_weird = [
%!     1     1     0     3
%!     1     1     1     0
%!     0     1     1     1
%!     2     0     0     0
%!     2     2     0     4];
%!
%! assert (watershed (im, 4), labeled4);
%! assert (watershed (im, [1 1 0; 1 1 1; 0 1 1]), labeled_weird);

%!xtest
%! ## The following test is required for Matlab compatibility.  There must be
%! ## something specific about their implementation that causes it to return
%! ## this value.  Even when solving it on paper, we get different results.
%! im = [
%!     2     3    30     2
%!     3    30     3    30
%!   255    31    30     4
%!     2   255    31    30
%!     1     2   255     5];
%!
%! labeled8 = [
%!     1     1     0     3
%!     1     1     0     3
%!     0     0     0     0
%!     2     2     0     4
%!     2     2     0     4];
%! assert (watershed (im), labeled8);
%! assert (watershed (im, 8), labeled8);

%!test
%! im = [
%!    2    2    2    2    2    2    2
%!    2    2   30   30   30    2    2
%!    2   30   20   20   20   30    2
%!   40   40   20   20   20   40   40
%!    1   40   20   20   20   40    0
%!    1    1   40   20   40    0    0
%!    1    1    1   20    0    0    0];
%!
%! labeled8 = [
%!    1    1    1    1    1    1    1
%!    1    1    1    1    1    1    1
%!    1    1    1    1    1    1    1
%!    0    0    0    0    0    0    0
%!    2    2    2    0    3    3    3
%!    2    2    2    0    3    3    3
%!    2    2    2    0    3    3    3];
%! labeled4 = [
%!    1    1    1    1    1    1    1
%!    1    1    1    1    1    1    1
%!    1    1    1    1    1    1    1
%!    0    1    1    1    1    1    0
%!    2    0    1    1    1    0    3
%!    2    2    0    1    0    3    3
%!    2    2    2    0    3    3    3];
%! labeled_weird = [
%!    1    1    1    1    1    1    1
%!    1    1    1    1    1    1    1
%!    1    1    1    1    1    1    1
%!    0    1    1    0    0    0    0
%!    2    0    0    0    3    3    3
%!    2    2    0    3    3    3    3
%!    2    2    2    0    3    3    3];
%!
%! assert (watershed (im), labeled8);
%! assert (watershed (im, 8), labeled8);
%! assert (watershed (im, 4), labeled4);
%! assert (watershed (im, [1 1 0; 1 1 1; 0 1 1]), labeled_weird);

%!test
%! im = [
%!   40   40   40   40   40   40   40   40   40   40   40   40   40
%!   40    3    3    5    5    5   10   10   10   10   15   20   40
%!   40    3    3    5    5   30   30   30   10   15   15   20   40
%!   40    3    3    5   30   20   20   20   30   15   15   20   40
%!   40   40   40   40   40   20   20   20   40   40   40   40   40
%!   40   10   10   10   40   20   20   20   40   10   10   10   40
%!   40    5    5    5   10   40   20   40   10   10    5    5   40
%!   40    1    3    5   10   15   20   15   10    5    1    0   40
%!   40    1    3    5   10   15   20   15   10    5    1    0   40
%!   40   40   40   40   40   40   40   40   40   40   40   40   40];
%!
%! labeled8 = [
%!    1    1    1    1    1    1    1    1    1    1    1    1    1
%!    1    1    1    1    1    1    1    1    1    1    1    1    1
%!    1    1    1    1    1    1    1    1    1    1    1    1    1
%!    1    1    1    1    1    1    1    1    1    1    1    1    1
%!    0    0    0    0    0    0    0    0    0    0    0    0    0
%!    2    2    2    2    2    2    0    3    3    3    3    3    3
%!    2    2    2    2    2    2    0    3    3    3    3    3    3
%!    2    2    2    2    2    2    0    3    3    3    3    3    3
%!    2    2    2    2    2    2    0    3    3    3    3    3    3
%!    2    2    2    2    2    2    0    3    3    3    3    3    3];
%! labeled4 = [
%!    1    1    1    1    1    1    1    1    1    1    1    1    1
%!    1    1    1    1    1    1    1    1    1    1    1    1    1
%!    1    1    1    1    1    1    1    1    1    1    1    1    1
%!    1    1    1    1    1    1    1    1    1    1    1    1    1
%!    0    0    0    0    1    1    1    1    1    0    0    0    0
%!    2    2    2    2    0    1    1    1    0    3    3    3    3
%!    2    2    2    2    2    0    1    0    3    3    3    3    3
%!    2    2    2    2    2    2    0    3    3    3    3    3    3
%!    2    2    2    2    2    2    0    3    3    3    3    3    3
%!    2    2    2    2    2    2    0    3    3    3    3    3    3];
%! labeled_weird = [
%!    1    1    1    1    1    1    1    1    1    1    1    1    1
%!    1    1    1    1    1    1    1    1    1    1    1    1    1
%!    1    1    1    1    1    1    1    1    1    1    1    1    1
%!    1    1    1    1    1    1    1    1    1    1    1    1    1
%!    0    0    0    0    1    1    0    0    0    0    0    0    0
%!    2    2    2    2    0    0    0    3    3    3    3    3    3
%!    2    2    2    2    2    0    3    3    3    3    3    3    3
%!    2    2    2    2    2    2    0    3    3    3    3    3    3
%!    2    2    2    2    2    2    0    3    3    3    3    3    3
%!    2    2    2    2    2    2    0    3    3    3    3    3    3];
%!
%! assert (watershed (im), labeled8);
%! assert (watershed (im, 8), labeled8);
%! assert (watershed (im, 4), labeled4);
%! assert (watershed (im, [1 1 0; 1 1 1; 0 1 1]), labeled_weird);

%!xtest
%! ## This test is failing for Matlab compatibility
%! im_full = [
%!   1   2  10   3   8   7   5
%!   3   2   5  10   8   1   4
%!   1   8   2   3   8   3   6];
%!
%! matlab_result_full = [
%!   1   1   0   3   0   4   4
%!   0   0   0   0   0   4   4
%!   2   2   2   0   4   4   4];
%!
%! assert (watershed (im_full), matlab_result_full);
%!
%! im_crop = [
%!       2  10   3   8   7   5
%!       2   5  10   8   1   4
%!       8   2   3   8   3   6];
%!
%! matlab_result_crop = [
%!       1   0   2   0   3   3
%!       1   0   0   0   3   3
%!       1   1   1   0   3   3];
%!
%! assert (watershed (im_crop), matlab_result_crop);
*/
