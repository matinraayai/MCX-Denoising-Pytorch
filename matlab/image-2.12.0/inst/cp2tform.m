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
## @deftypefn  {Function File} {@var{T} =} cp2tform (@var{in_cp}, @var{out_cp}, @var{ttype})
## @deftypefnx {Function File} {@var{T} =} cp2tform (@dots{}, @var{opt})
## Return a transformation structure @var{T} (see "help maketform"
## for the form of the structure) that can be further used to
## transform coordinates between an input space and an ouput space.
##
## The transform is inferred from two n-by-2 arrays, @var{in_cp} and
## @var{out_cp}, which contain the coordinates of n control points in
## the two 2D spaces.
## Transform coefficients are stored in @var{T}.tdata. Interpretation of
## transform coefficients depends on the requested transform type @var{ttype}:
##
## @table @asis
## @item "affine"
## Return both forward (input space to output space) and inverse transform
## coefficients @var{T}.tdata.T and @var{T}.tdata.Tinv. Transform 
## coefficients are 3x2 matrices which can be used as follows:
##
## @example
## @group
## @var{in_cp} = [@var{out_cp} ones(rows (out_cp) ,1)] * T.tdata.Tinv
## @var{out_cp} = [@var{in_cp} ones(rows (in_cp) ,1)] * T.tdata.T
## @end group
## @end example
## This transformation is well suited when parallel lines in one space
## are still parallel in the other space (e.g. shear, translation, @dots{}).
## 
## @item "nonreflective similarity"
## Same as "affine" except that the transform matrices T and Tinv have
## the form
## @example
## @group
## Tcoefs = [a -b;
##           b  a;
##           c  d]
## @end group
## @end example
## This transformation may represent rotation, scaling and
## translation. Reflection is not included.
## 
## @item "similarity"
## Same as "nonreflective similarity" except that the transform matrices T and Tinv may also have
## the form
## @example
## @group
## Tcoefs = [a  b;
##           b -a;
##           c  d]
## @end group
## @end example
## This transformation may represent reflection, rotation, scaling and
## translation. Generates a warning if the nonreflective similarity is 
## better suited.
## 
## @item "projective"
## Return both forward (input space to output space) and inverse transform
## coefficients @var{T}.tdata.T and @var{T}.tdata.Tinv. Transform
## coefficients are 3x3 matrices which  can
## be used as follows:
##
## @example
## @group
## [u v w] = [@var{out_cp} ones(rows (out_cp) ,1)] * T.tdata.Tinv
## @var{in_cp} = [u./w, v./w];
## [x y z] = [@var{in_cp} ones(rows (in_cp) ,1)] * T.tdata.T
## @var{out_cp} = [x./z y./z];
## @end group
## @end example
## This transformation is well suited when parallel lines in one space
## all converge toward a vanishing point in the other space.
##
## @item "polynomial"
## Here the @var{opt} input argument is the order of the polynomial
## fit. @var{opt} must be 2, 3 or 4 and input control points number must
## be respectively at least 6, 10 or 15. Only the inverse transform
## (output space to input space) is included in  the structure @var{T}.
## Denoting x and y the output space coordinate vector and u, v the
## the input space coordinates. Inverse transform coefficients are
## stored in a (6,10 or 15)-by-2 matrix which can be used as follows:
##
## @example
## @group
## Second order:  
## [u v] = [1 x y x*y x^2 y^2] * T.tdata.Tinv
## @end group
## @group
## Third order:   
## [u v] = [1 x y x*y x^2 y^2 y*x^2 x*y^2 x^3 y^3] * T.tdata.Tinv
## @end group
## @group
## Fourth order:  
## [u v] = [1 x y x*y x^2 y^2 y*x^2 x*y^2 x^3 y^3 x^3*y x^2*y^2 x*y^3 x^4 y^4] * T.tdata.Tinv
## @end group
## @end example
## This transform is well suited when lines in one space become curves
## in the other space. 
## @end table
## @seealso{tformfwd, tforminv, maketform}
## @end deftypefn

function trans = cp2tform (crw, cap, ttype, opt)
  if (nargin < 3)
    print_usage ();
  endif

  if (! all (size (crw) == size (cap)) || columns (crw) != 2)
    error ("cp2tform: expect the 2 first input arguments to be (m x 2) matrices")
  elseif (! ischar (ttype))
    error ("cp2tform: expect a string as third input argument")
  endif

  ttype = lower (ttype);
  switch ttype
    case {'nonreflective similarity', 'similarity', 'affine', 'projective'}
      trans = gettrans (ttype, cap, crw);
    case 'polynomial'
      if (nargin < 4)
        error ("cp2tform: expect a fourth input argument for 'polynomial'")
      elseif (! isscalar (opt))
        error ("cp2tform: expect a scalar as fourth argument")
      endif
      trans = gettrans (ttype, cap, crw, round (opt));
    otherwise
      error ("cp2tform: expect 'nonreflective similarity', 'similarity', 'affine' or 'polynomial' as third input argument")
  endswitch  
endfunction

function trans = gettrans (ttype, cap, crw, ord = 0)
  switch ttype
    case "nonreflective similarity"
      x = cap(:,1);
      y = cap(:,2);
      u = crw(:,1);
      v = crw(:,2);
      tmp0 = zeros(size(u));
      tmp1 = ones(size(u));
      
      A = [u v tmp1 tmp0 ; v -u tmp0 tmp1];
      B = [x; y];
      tmat = A\B;
      tmat = [tmat(1) -tmat(2);
              tmat(2)  tmat(1);
              tmat(3)  tmat(4)];
      trans = maketform ("affine", tmat);

    case "similarity"
      x = cap(:,1);
      y = cap(:,2);
      u = crw(:,1);
      v = crw(:,2);
      tmp0 = zeros(size(u));
      tmp1 = ones(size(u));

      #without reflection
      A = [u v tmp1 tmp0 ; v -u tmp0 tmp1];
      B = [x; y];
      tmat1 = A\B;
      resid = norm (A*tmat1 - B);
      #with reflection
      A = [u v tmp1 tmp0 ; -v u tmp0 tmp1];
      tmat2 = A\B;
      if (norm (A*tmat2 - B) < resid)
        tmat = [tmat2(1)  tmat2(2); 
                tmat2(2) -tmat2(1); 
                tmat2(3)  tmat2(4)];
      else
        tmat = [tmat1(1) -tmat1(2); 
                tmat1(2)  tmat1(1); 
                tmat1(3)  tmat1(4)];
        warning ("cp2tform: reflection not included.")
      endif
      trans = maketform ("affine", tmat);

    case "affine"
      tmat  = [crw ones(rows(crw), 1)]\cap;
      trans = maketform ("affine", tmat);

    case "projective"
      x = cap(:,1);
      y = cap(:,2);
      u = crw(:,1);
      v = crw(:,2);
      tmp0 = zeros(size(u));
      tmp1 = ones(size(u));
      A = [-u -v -tmp1 tmp0 tmp0 tmp0 x.*u x.*v x;
           tmp0 tmp0 tmp0 -u -v -tmp1 y.*u y.*v y];
      B = - A(:,end);
      A(:,end) = [];
      tmat = A\B;
      tmat(9) = 1;
      tmat = reshape (tmat, 3, 3);
      trans = maketform ("projective", tmat);

    case "polynomial"
      x = cap(:,1);
      y = cap(:,2);
      u = crw(:,1);
      v = crw(:,2);
      tmp1 = ones(size(x));
      
      ndims_in = 2;
      ndims_out = 2;
      forward_fcn = [];
      inverse_fcn = @inv_polynomial;
      A = [tmp1, x, y, x.*y, x.^2, y.^2]; 
      B = [u v];
      switch ord
        case 2
        case 3
          A = [A, y.*x.^2 x.*y.^2 x.^3 y.^3]; 
        case 4
          A = [A, y.*x.^2 x.*y.^2 x.^3 y.^3];
          A = [A, x.^3.*y x.^2.*y.^2 x.*y.^3 x.^4 y.^4];
        otherwise
          error ("cp2tform: supported polynomial orders are 2, 3 and 4.")
      endswitch
      tmat = A\B;
      trans = maketform ("custom", ndims_in, ndims_out, ...
                         forward_fcn, inverse_fcn, tmat);
    otherwise
      error ("cp2tform: invalid TTYPE %s.", ttype);
    endswitch
endfunction

function out = inv_polynomial (x, pst)
  out = [];
  for ii = 1:2
    p = pst.tdata(:,ii);
    
    if (rows (p) == 6)
      ## 2nd order
      out(:,ii) = p(1) + p(2)*x(:,1) + p(3)*x(:,2) + p(4)*x(:,1).*x(:,2) + ...
                  p(5)*x(:,1).^2 + p(6)*x(:,2).^2;
    elseif (rows (p) == 10)
      ## 3rd order
      out(:,ii) = p(1) + p(2)*x(:,1) + p(3)*x(:,2) + p(4)*x(:,1).*x(:,2) + ...
                  p(5)*x(:,1).^2 + p(6)*x(:,2).^2 + p(7)*x(:,2).*x(:,1).^2 + ...
                  p(8)*x(:,1).*x(:,2).^2 + p(9)*x(:,1).^3 + p(10)*x(:,2).^3;
    elseif (rows (p) == 15)
      ## 4th order
      out(:,ii) = p(1) + p(2)*x(:,1) + p(3)*x(:,2) + p(4)*x(:,1).*x(:,2) + ...
                  p(5)*x(:,1).^2 + p(6)*x(:,2).^2 + p(7)*x(:,2).*x(:,1).^2 + ...
                  p(8)*x(:,1).*x(:,2).^2 + p(9)*x(:,1).^3 + p(10)*x(:,2).^3 + ...
                  p(11)*x(:,2).*x(:,1).^3 + p(12)*x(:,2).^2.*x(:,1).^2+ ...
                  p(13)*x(:,1).*x(:,2).^3 + p(14)*x(:,1).^4 + p(15)*x(:,2).^4;
    endif
  endfor
endfunction

%!function [crw, cap] = coords (npt = 1000, scale = 2, dtheta = pi/3,
%!                              dx = 2, dy = -6, sig2noise = 1e32)
%!  theta = (rand(npt, 1)*2-1)*2*pi;
%!  R = rand(npt,1);
%!  y = R.*sin(theta);
%!  x = R.*cos(theta);
%!  crw = [y x];
%!
%!  thetap = theta + dtheta;
%!  Rap = R * scale;
%!
%!  yap = Rap.*sin(thetap);
%!  yap = yap + dy;
%!  yap = yap + rand (size (yap)) * norm (yap) / sig2noise;
%!
%!  xap = Rap.*cos(thetap);
%!  xap = xap + dx;
%!  xap = xap + rand (size (xap)) * norm (xap) / sig2noise;
%!  cap = [yap xap];
%!endfunction


%!test
%! npt = 100000;
%! [crw, cap] = coords (npt);
%! ttype = 'projective';
%! T = cp2tform (crw, cap, ttype);
%! crw2 = tforminv (T, cap);
%! finalerr = norm (crw - crw2)/npt;
%! assert (finalerr < 2*eps, "norm = %3.2e ( > 2*eps)", finalerr)

%!test
%! npt = 100000;
%! [crw, cap] = coords (npt);
%! ttype = 'affine';
%! T = cp2tform (crw, cap, ttype);
%! crw2 = tforminv (T, cap);
%! finalerr = norm (crw - crw2)/npt;
%! assert (finalerr < eps, "norm = %3.2e ( > eps)", finalerr)

%!test
%! npt = 100000;
%! [crw, cap] = coords (npt);
%! ttype = 'nonreflective similarity';
%! T = cp2tform (crw, cap, ttype);
%! crw2 = tforminv (T, cap);
%! finalerr = norm (crw - crw2)/npt;
%! assert (finalerr < 2*eps, "norm = %3.2e ( > 2*eps)", finalerr)

%!test
%! npt = 100000;
%! [crw, cap] = coords (npt);
%! cap(:,2) *= -1; % reflection around y axis
%! ttype = 'similarity';
%! T = cp2tform (crw, cap, ttype);
%! crw2 = tforminv (T, cap);
%! finalerr = norm (crw - crw2)/npt;
%! assert (finalerr < 3*eps, "norm = %3.2e ( > 3*eps)", finalerr)

%!xtest
%! npt = 100000;
%! [crw, cap] = coords (npt);
%! ttype = 'polynomial';
%! ord = 2;
%! T = cp2tform (crw, cap, ttype, ord);
%! crw2 = tforminv (T, cap);
%! finalerr = norm (crw - crw2)/npt;
%! assert (finalerr < eps, "norm = %3.2e ( > eps)", finalerr)

%!xtest
%! npt = 100000;
%! [crw, cap] = coords (npt);
%! ttype = 'polynomial';
%! ord = 3;
%! T = cp2tform (crw, cap, ttype, ord);
%! crw2 = tforminv (T, cap);
%! finalerr = norm (crw - crw2)/npt;
%! assert (finalerr < eps, "norm = %3.2e ( > eps)", finalerr)

%!xtest
%! npt = 100000;
%! [crw, cap] = coords (npt);
%! ttype = 'polynomial';
%! ord = 4;
%! T = cp2tform (crw, cap, ttype, ord);
%! crw2 = tforminv (T, cap);
%! finalerr = norm (crw - crw2)/npt;
%! assert (finalerr < 6*eps, "norm = %3.2e ( > 6*eps)", finalerr)
