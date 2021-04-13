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
## @deftypefn  {Function File} {@var{outbnd} =} findbounds (@var{T}, @var{inbnd})
## Estimate bounds for spatial transformation.
##
## Given a transformation structure @var{T} (see e.g. maketform) 
## and bounds @var{inbnd} (2-by-ndims_in) in an input space, returns
## an estimation of the bounds in the output space @var{outbnd} 
## (2-by-ndims_out). For instance two dimensionnal bounds could 
## be represented as : [xmin ymin; xmax ymax]. If @var{T} does not
## define a forward transform (i.e. for 'polynomial'), the output
## bounds are infered using fsolve and the inverse transform.
##
## @seealso{maketform, cp2tform, tformfwd}
## @end deftypefn

## Author: Pantxo Diribarne <pantxo@dibona>

function [outbnd] = findbounds (T, inbnd)
  if (nargin != 2)
    print_usage ();
  elseif (! istform (T))
    error ("imtransform: T must be a transformation structure (see `maketform')");
  elseif (! all (size (inbnd) == [2 T.ndims_in]))
    error ("imtransform: INBDN must have same size as T.ndims_in");
  endif

  ## Control points grid
  if (columns (inbnd) == 2)
    [xcp, ycp] = meshgrid (linspace (inbnd(1,1), inbnd(2,1), 3),
                           linspace (inbnd(1,2), inbnd(2,2), 3));
    xcp = reshape (xcp, numel (xcp), 1);
    ycp = reshape (ycp, numel (ycp), 1);
    xycp = [xcp, ycp];
  else
    error ("findbounds: support only 2D inputbounds.");
  endif

  ## Output bounds
  if (!is_function_handle (T.forward_fcn))
    outbnd = zeros (size (xycp));
    for ii = 1:rows (xycp)
      fun = @(x) tforminv (T, x) - xycp(ii,:);
      outbnd(ii,:) = fsolve (fun, xycp(ii,:));
    endfor
  else
    outbnd = tformfwd (T, xycp);
  endif
  outbnd = [min(outbnd); max(outbnd)];
endfunction

%!test
%! im = checkerboard ();
%! theta = pi/6;
%! T = maketform ('affine', [cos(theta) -sin(theta); ...
%!                           sin(theta) cos(theta); 0 0]);
%! inbnd = [0 0; 1 1];
%! outbnd = findbounds (T, inbnd);
%! diag = 2^.5;
%! ang = pi/4;
%! assert (diff (outbnd(:,1)), diag * abs (cos (theta - ang)), eps)
%! assert (diff (outbnd(:,2)), diag * abs (cos (theta - ang)), eps)
