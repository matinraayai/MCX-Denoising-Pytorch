// Copyright (C) 2010 Andrew Kelly, IPS Radio & Space Services
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

/**
 *  Oct-file to trace the boundary of an object in a binary image.
 *
 *      b = boundary(region, conn=8)
 */
#include <octave/oct.h>

using namespace std;

DEFUN_DLD(__boundary__, args, nargout,
"-*- texinfo -*-\n\
@deftypefn  {Loadable Function} {} boundary(@var{region})\n\
@deftypefnx {Loadable Function} {} boundary(@var{region}, @var{conn})\n\
Trace the boundary of an object in a binary image.\n\
\n\
@code{boundary} computes the exterior clockwise boundary of the single \
@var{conn}-connected object represented by the non-zero pixels \
of @var{region}. It uses an algorithm based on Moore-neighbour tracing.\n\
\n\
@var{conn} can be either 8 (the default) or 4.\n\
\n\
@var{b} is an N-by-2 matrix containing the row/column coordinates of points \
on the boundary. The first boundary point is the first non-zero \
pixel of @var{region}, as determined by @code{find}. The last boundary \
point is the same as the first.\n\
@seealso{boundaries, bwlabel, find}\n\
@end deftypefn")
{
  octave_value_list retval;
  
  enum { ROW, COL };

  // check number of arguments
  const int nargin = args.length ();
  if (nargin > 2 || nargout != 1)
    error ("__boundary__: wrong number of input arguments");

  // extract arguments
  const boolMatrix unpadded = args (0).bool_matrix_value ();
  const int conn = (nargin > 1)
      ? (int) args (1).scalar_value ()
      : 8;

  // pad to avoid boundary issues
  int rows = unpadded.rows ();
  int cols = unpadded.columns ();
  boolMatrix region (rows + 2, cols + 2, false);
  for (int r = 0; r < rows; r++)
    for (int c = 0; c < cols; c++)
      region.elem (r+1, c+1) = unpadded (r, c);

  // the padded size
  rows += 2;
  cols += 2;

  // find the (first two) true pixels, if any
  std::vector <int> pixels;
  for (int i = 0; pixels.size () < 2 && i < region.numel (); ++i)
    if (region.elem (i))
      pixels.push_back (i);

  if (pixels.empty ())
    return retval;

  // the starting boundary point
  const int start = pixels [0];
  std::vector <int> bound;
  bound.push_back (start);

  // is this the only point?
  if (pixels.size () == 1)
    bound.push_back (start);

  // otherwise, find the boundary by tracing the Moore neighbourhood of its pixels
  //
  //      8-connected:    7 0 1      4-connected:      0
  //                      6 . 2                      3 . 1
  //                      5 4 3                        2
  else
    {
      // relative row/column positions
      static const int row8 [] = {-1, -1,  0,  1,  1,  1,  0, -1};
      static const int col8 [] = { 0,  1,  1,  1,  0, -1, -1, -1};
      static const int row4 [] = {-1,      0,      1,      0    };
      static const int col4 [] = { 0,      1,      0,     -1    };
      const int* mr = (conn == 4) ? row4 : row8;
      const int* mc = (conn == 4) ? col4 : col8;

      // next after backing-up
      static const int back8 [] = {7, 7, 1, 1, 3, 3, 5, 5};
      static const int back4 [] = {3, 0, 1, 2};
      const int* mBack = (conn == 4) ? back4 : back8;

      // relative indexes into the region for the Moore neighbourhood pixels
      std::vector<int> mi (conn);
      for (int i = 0; i < conn; ++i)
        mi[i] = mr[i] + (rows * mc [i]);

      // next neighbourhood pixel
      static const int next8 [] = {1, 2, 3, 4, 5, 6, 7, 0};
      static const int next4 [] = {1, 2, 3, 0};
      const int* mNext = (conn == 4) ? next4 : next8;

      // the final boundary point to be visited
      int finish = 0;
      for (int i = 0; i < conn; ++i)
        if (region.elem(start + mi[i]))
          finish = start + mi[i];

      // look for the next boundary point, starting at the next neighbour
      int bp = start;
      int mCurrent = mNext [0];
      bool done = false;
      while (!done)
        {
          // next neighbour
          int cp = bp + mi[mCurrent];

          // if this pixel is false, try the next one
          if (!region.elem (cp))
            {
              mCurrent = mNext [mCurrent];
            }
          // otherwise, we have another boundary point
          else
            {
              bound.push_back (cp);

              // either we're back at the start for the last time
              if (bp == finish && cp == start)
                {
                  done = true;
                }
              // or we step back to where we came in from, and continue
              else
                {
                  bp = cp;
                  mCurrent = mBack [mCurrent];
                }
            }
        }
    }

  // convert boundary points to row/column coordinates
  Matrix b (bound.size (), 2);
  for (unsigned int i = 0; i < bound.size (); i++)
    {
      const int point = bound [i];
      b (i, ROW) = point % rows;
      b (i, COL) = point / rows;
    }
  
  retval.append (b);
  return retval;
}
