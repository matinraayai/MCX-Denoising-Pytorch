// Copyright (C) 1999 Andy Adler <adler@sce.carleton.ca>
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

#include <octave/oct.h>

#define   ptUP     (-1)
#define   ptDN     (+1)
#define   ptRT     (+ioM)
#define   ptLF     (-ioM)

/*
 * check if the point needs to be filled, if so
 * fill it and change the appropriate variables
 */
void checkpoint (int pt, unsigned char *imo, int *ptstack, int *npoints)
{
// printf("filling %d np=%d fill=%d\n",pt,*npoints, *(imo+pt)==0 );
  if (*(imo+pt) != 0) return;

  *(imo+pt) = 2;
  *(ptstack + (*npoints))= pt;
  (*npoints)++;
}

DEFUN_DLD (bwfill, args, ,"\
-*- texinfo -*-\n\
@deftypefn {Loadable Function} {[@var{bw2}, @var{idx}] =} bwfill(@var{bw1}, @var{c}, @var{r}, @var{n})\n\
Perform a flood-fill operation on the binary image @var{bw1}.\n\
\n\
The flood-filling starts in the pixel (@var{r}, @var{c}). If @var{r} and @var{c}\n\
are vectors of the same length, each pixel pair (@var{r}(i), @var{c}(i)) will\n\
be a starting point for a flood-fill operation.\n\
The argument @var{n} changes the neighborhood connectivity (of the holes) for the flood-fill\n\
operation. @var{n} can be either 4 or 8, and has a default value of 8.\n\
\n\
Note that  @var{n} is the connectivity of the foreground, and not of the background,\n\
even though the function acts on the background.\n\
\n\
The output is the processed image @var{bw2} and the indexes of the filled\n\
pixels @var{idx}\n\
\n\
@end deftypefn\n\
@deftypefn {Loadable Function} {[@var{bw2}, @var{idx}] =} bwfill(@var{bw1}, \"holes\", @var{n})\n\
If the string \"holes\" is given instead of starting points for the flood-fill\n\
operation, the function finds interior holes in @var{bw1} and fills them.\n\
\n\
Note: bwfill is not recommended. Please use \"imfill\" instead.\n\
@seealso{imfill}\n\
@end deftypefn\n\
")
{
  octave_value_list retval;
  octave_value tmp;
  ColumnVector xseed, yseed ;
  const int nargin = args.length ();

  if (nargin < 2 || nargin > 4)
    print_usage ();

  const Matrix im = args (0).matrix_value ();
  const int imM = im.rows ();
  const int imN = im.columns ();

  if (imM == 1 || imN == 1) // check for vector inputs.
    {
      retval (0) = im;
      retval (1) = ColumnVector (0);
      return retval;
    }

  int nb = 8;
  int npoints = 0;
  bool fillmode = false;
  if (args (1).is_string () && args (1).string_value () == "holes")
    {
      // usage: bwfill (A, "holes", [N])
      if (nargin > 3)
        print_usage ();
      fillmode = true;

      npoints = 2 * (imM + imN - 4); // don't start fill from corners

      xseed = ColumnVector (npoints);
      yseed = ColumnVector (npoints);
      int idx = 0;
      for (int j = 2; j <= imN-1; j++)
        {
          xseed (idx)   = j;
          yseed (idx++) = 1;
          xseed (idx)   = j;
          yseed (idx++) = imM;
        }

      for (int i = 2; i <= imM-1; i++)
        {
          yseed (idx)   = i;
          xseed (idx++) = 1;
          yseed (idx)   = i;
          xseed (idx++) = imN;
        }

      if (nargin >= 3)
        nb = (int)args (2).double_value ();
    }
  else
    {
      // usage: bwfill (A, C, R, [N])
      if (nargin < 3)
        print_usage ();

      {
        ColumnVector tmp (args (1).vector_value ());
        xseed = tmp;
      }
      {
        ColumnVector tmp (args (2).vector_value ());
        yseed = tmp;
      }
      npoints= xseed.numel ();
      if (nargin >= 4)
        nb = (int)args (3).double_value ();
    }

  if (nb != 4 && nb != 8)
    error ("bwfill: connectivity must be 4 or 8");

/*
 * put a one pixel thick boundary around the image
 *  so that we can be more efficient in the main loop
 */
  int ioM = imM + 2;
  std::vector<unsigned char> imo ((imM+2) * (imN+2));

  for (int i = 0; i < imM; i++)
    for (int j = 0; j < imN; j++)
      imo[(i+1) + ioM*(j+1)] = (im (i, j) > 0);

  for (int i = 0; i < ioM; i++)
    imo[i]= imo[i + ioM*(imN+1)] = 3;

  for (int j = 1; j < imN+1; j++)
    imo[ioM*j]= imo[imM+1 + ioM*j] = 3;

  // This is obviously big enough for the point stack, but I'm
  // sure it can be smaller.
  std::vector<int> ptstack (ioM*imN);

  int seedidx = npoints;
  npoints = 0;
  while ((--seedidx) >= 0)
    {
      // no need to add 1 to convert indexing style because we're adding a boundary
      const int x = xseed (seedidx);
      const int y = yseed (seedidx);
      if (x < 1 || y < 1 || x > imN || y > imM)
        {
          warning ("bwfill: (%d, %d) out of bounds", x, y);
          continue;
        }
      const int pt = x * ioM + y;
      checkpoint (pt , imo.data (), ptstack.data (), &npoints);
    }

  while (npoints > 0)
    {
      npoints--;
      int pt = ptstack[npoints];

      checkpoint (pt + ptLF, imo.data (), ptstack.data (), &npoints);
      checkpoint (pt + ptRT, imo.data (), ptstack.data (), &npoints);
      checkpoint (pt + ptUP, imo.data (), ptstack.data (), &npoints);
      checkpoint (pt + ptDN, imo.data (), ptstack.data (), &npoints);

      if (nb==4)
        {
          checkpoint (pt + ptLF + ptUP, imo.data (), ptstack.data (), &npoints);
          checkpoint (pt + ptRT + ptUP, imo.data (), ptstack.data (), &npoints);
          checkpoint (pt + ptLF + ptDN, imo.data (), ptstack.data (), &npoints);
          checkpoint (pt + ptRT + ptDN, imo.data (), ptstack.data (), &npoints);
        }
    } // while ( npoints > 0)

  boolNDArray imout (dim_vector (imM, imN));
  ColumnVector idxout (imM*imN);
  int idx = 0;

  int notvalidpt = 0;
  int idxpoint = 2;
  if (fillmode)
    {
      notvalidpt = 2;
      idxpoint   = 0;
    }

  for (int i = 0; i < imM; i++)
    for (int j = 0; j < imN; j++)
      {
        imout (i, j) = imo[(i+1) + ioM*(j+1)] != notvalidpt;
        if (imo[(i+1) + ioM*(j+1)] == idxpoint)
          idxout (idx++) = (double) (i + j*imM + 1);
      }

  /*
  Matrix imout( imM+2, imN+2 );
  for (int i=0; i<imM+2; i++)
    for (int j=0; j<imN+2; j++)
      imout(i,j) = (double) imo[i + ioM*j];
  */

  retval (0) = imout;
  // we need to do this to be able to return a proper empty vector
  if (idx > 0)
    retval (1) = idxout.extract (0, idx-1);
  else
    retval (1) = ColumnVector (0);
  return retval;
}

/*
%!test
%! A = [0 1 0 0 1; 1 0 1 0 0; 1 0 1 1 0; 1 1 1 0 0; 1 0 0 1 0];
%! R4 = logical(ones(5));
%! R8 = logical([1 1 0 0 1; 1 0 1 0 0; 1 0 1 1 0; 1 1 1 0 0; 1 0 0 1 0]);
%! assert (bwfill (A,1,1,4), R4)
%! assert (bwfill (A,1,1,8), R8)
%! assert (bwfill (A,1,1), R8)
%! B = logical([0 1 0 0 1; 1 0 1 0 0; 1 0 1 1 0; 1 1 1 0 0; 1 0 0 1 0]);
%! assert (bwfill (A,3,3,4), B)
%! assert (bwfill (A,3,3,8), B)
%! assert (bwfill (A,3,3), B)
%! C = logical ([0 1 1 1 1; 1 0 1 1 1; 1 0 1 1 1; 1 1 1 1 1; 1 0 0 1 1]);
%! assert (bwfill (A,3,1,8), C)
%! assert (bwfill (A,3,1,4), R4)
%! assert (bwfill (A, [3 1], [1 3], 4), R4);
%! D = logical([0 1 1 1 1; 1 0 1 1 1; 1 0 1 1 1; 1 1 1 1 1; 1 0 0 1 1]);
%! assert (bwfill (A, [3 1], [1 3], 8), D);
%! assert (bwfill (A, [3 1], [1 3]), D);
%! E = logical ([0 1 0 0 1; 1 0 1 0 0; 1 0 1 1 0; 1 1 1 0 0; 1 0 0 1 0]);
%! assert (bwfill (A, "holes", 4), E);
%! F = logical ([1 1 0 0 1; 1 1 1 0 0; 1 1 1 1 0; 1 1 1 0 0; 1 0 0 1 0]);
%! assert (bwfill (A, "holes", 8), F);
%! assert (bwfill (A, "holes"), F);

%!error id=Octave:invalid-fun-call bwfill ()
%!error id=Octave:invalid-fun-call bwfill ("aaa")
%!error id=Octave:invalid-fun-call bwfill (rand (5) > 0.5)
%!error id=Octave:invalid-fun-call bwfill (rand (5) > 0.5, 2)
%!error <bwfill: connectivity must be 4 or 8> bwfill (rand (5) > 0.5, "holes", 1)
%!error <bwfill: connectivity must be 4 or 8> bwfill (rand (5) > 0.5, 2, 2, 5)
%!error id=Octave:invalid-fun-call bwfill (rand (5) > 0.5, "xxx")
%!error id=Octave:invalid-fun-call bwfill (rand (5) > 0.5, 2, 2, 4, 5)
%!error id=Octave:invalid-fun-call bwfill (rand (5) > 0.5, "holes", 4, 2)
*/
