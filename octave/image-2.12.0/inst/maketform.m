## Copyright (C) 2012 Pantxo Diribarne
##
## This program is free software; you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with Octave; see the file COPYING.  If not, see
## <http://www.gnu.org/licenses/>.

## -*- texinfo -*-
## @deftypefn  {Function File} {@var{T} =} maketform (@var{ttype}, @var{tmat})
## @deftypefnx {Function File} {@var{T} =} maketform (@var{ttype}, @var{inc}, @var{outc})
## @deftypefnx {Function File} {@var{T} =} maketform ("custom", @var{ndims_in}, @var{ndims_out}, @var{forward_fcn}, @var{inverse_fcn}, @var{tdata})
## Create a transform structure @var{T} to be used for spatial
## transformations between an input space and an output space.
##
## The fields of the transform structure are:
##
## @table @asis
## @item @qcode{"ndims_in"}, @qcode{"ndims_out"}: the number of
## dimensions of the input and output space.
##
## @item @qcode{"forward_fcn"}, @qcode{"inverse_fcn"}: the callback
## functions that are called for forward (input to output) and inverse
## transform.
##
## @item @qcode{"tdata"}: an inverse transform matrix or a structure
## containing forward and inverse transform matrices. 
## @end table
##
## The content of each field depends on the requested transform type
## @var{ttype}:
##
## @table @asis
## @item "projective"
## A ndims_in = ndims_out = N projective transform structure
## is returned.
## If a second input argument @var{tmat} is provided, it must be a
## (N+1)-by-(N+1) inverse transformation matrix.  The (N+1)th column must
## contain projection coefficients.  As an example a two 
## dimensional inverse transform from [x y] coordinates to [u v] coordinates
## is represented by an inverse transformation matrix defined so that:
##
## @example
## [xx yy zz] = [u v 1] * [a d g;
##                         b e h;
##                         c f i]
## [x y] =  [xx./zz yy./zz];
## @end example
## 
## Alternatively the transform can be specified using the coordinates
## of a quadrilateral (typically the 4 corners of the
## image) in the input space (@var{inc}, 4-by-ndims_in matrix) and in
## the output space (@var{outc}, 4-by-ndims_out matrix).  This is
## equivalent to building the transform using
## @code{T = cp2tform (@var{inc}, @var{outc}, "projective")}.
##
## @item "affine"
## Affine is a subset of projective transform (see above).  A
## ndims_in = ndims_out = N affine transformation structure is
## returned.
## If a second input argument @var{tmat} is provided, it must be a
## (N+1)-by-(N+1) or (N+1)-by-(N) transformation matrix. If present, the
## (N+1)th column  must contain [zeros(N,1); 1] so that projection is
## suppressed.
##
## Alternatively the transform can be specified using the coordinates
## of a triangle (typically 3 corners of the
## image)  in the input space (@var{inc}, 3-by-ndims_in matrix) and in
## the  output space (@var{outc}, 3-by-ndims_out matrix). This is
## equivalent to building the transform using 
## @code{T = cp2tform (@var{inc}, @var{outc}, "affine")}.
## 
## @item "custom"
## For user defined transforms every field of the transform structure
## must be supplied. The prototype of the transform functions,
## @var{forward_fcn} and @var{inverse_fcn}, should be X' =
## transform_fcn (X, T). X and X' are respectively p-by-ndims_in and
## p-by-ndims_out arrays for forward_fcn and reversed for inverse_fcn.
## The argument T is the transformation structure which will contain
## the user supplied transformation matrix @var{tdata}. 
## @end table
##
## @seealso{tformfwd, tforminv, cp2tform}
## @end deftypefn

## Author: Pantxo Diribarne <pantxo@dibona>

function T = maketform (ttype, varargin)

  if (nargin < 2 || ! any (strcmpi (ttype, {"affine", "projective", "custom"})))
    print_usage ();
  endif

  if (numel (varargin) == 1)
    tmat = varargin {1};
    ndin = rows (tmat) - 1;
    ndout = columns (tmat) - 1;
    if (ndin < 2);
      error ("maketform: expect at least 3-by-2 transform matrix")
    elseif ((ndin-ndout) > 1 || (ndout > ndin))
      print_usage ();
    endif

    switch (tolower (ttype))
      case "affine"
        if ((ndin - ndout) == 1)
          tmat = [tmat [zeros(ndin, 1); 1]];
          ndout += 1;
        elseif (!all (tmat(:,end) == [zeros(ndin, 1); 1]))
          error ("maketform: \"%s\" expect [zeros(N,1); 1] as (N+1)th column", ttype);
        endif
        forward_fcn = @fwd_affine; 
        inverse_fcn = @inv_affine;
      case "projective"
        if ((ndin - ndout) == 1)
          print_usage ();
        endif
        forward_fcn = @fwd_projective;
        inverse_fcn = @inv_projective;
    endswitch
    T.ndims_in = ndin;
    T.ndims_out = ndout;
    T.forward_fcn = forward_fcn;
    T.inverse_fcn = inverse_fcn;
    T.tdata.T = tmat;
    T.tdata.Tinv = inv (tmat);

  elseif (numel (varargin) == 2)
    inc = varargin{1};
    outc = varargin{2};
    if (strcmp (ttype, "affine"))
      if (all (size (inc) == size (outc)) &&
          all (size (inc) == [3 2]))
        T = cp2tform (inc, outc, ttype);
      else
        error ("maketform: expect INC and OUTC to be 3-by-2 vectors.");
      endif
    elseif (strcmp (ttype, "projective"))
      if (all (size (inc) == size (outc)) &&
          all (size (inc) == [4 2]))
        T = cp2tform (inc, outc, ttype);
      else
        error ("maketform: expect INC and OUTC to be 4-by-2 vectors.");
      endif
    endif

  elseif (numel (varargin) == 5 && strcmpi (ttype, "custom"))
    if (isscalar (varargin{1}) && isscalar (varargin{2})
        && varargin{1} > 0 && varargin{2} > 0)
      T.ndims_in = varargin{1};
      T.ndims_out = varargin{2};
    else
      error ("maketform: expect positive scalars as ndims.")
    endif
    if (is_function_handle (varargin{3}) || isempty (varargin{3}))
      T.forward_fcn = varargin{3};
    else
      error ("maketform: expect function handle as forward_fcn.")
    endif
    if (is_function_handle (varargin{4}) || isempty (varargin{4}))
      T.inverse_fcn = varargin{4};
    else
      error ("maketform: expect function handle as inverse_fcn.")
    endif
    
    T.tdata = varargin{5};

  else
    print_usage ();
  endif
endfunction

function X = fwd_affine (U, T)
  U = [U, ones(rows(U), 1)];
  X = U * T.tdata.T(:,1:end-1);
endfunction

function U = inv_affine (X, T)
  X = [X, ones(rows(X), 1)];
  U = X * T.tdata.Tinv(:,1:end-1);
endfunction

function X = fwd_projective (U, T)
  U = [U, ones(rows(U), 1)];
  XX = U * T.tdata.T;
  X = [XX(:,1)./XX(:,3) XX(:,2)./XX(:,3)];
endfunction

function U = inv_projective (X, T)
  X = [X, ones(rows(X), 1)];
  UU = X * T.tdata.Tinv;
  U = [UU(:,1)./UU(:,3) UU(:,2)./UU(:,3)];
endfunction
