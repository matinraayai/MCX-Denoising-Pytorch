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
## @deftypefn  {Function File} {@var{IM_OUT} =} imtransform (@var{IM_IN}, @var{T})
## @deftypefnx {Function File} {@var{IM_OUT} =} imtransform (@var{IM_IN}, @var{T}, @var{interp})
## @deftypefnx {Function File} {@var{IM_OUT} =} imtransform (@dots{}, @var{prop}, @var{val})
## @deftypefnx {Function File} {[@var{IM_OUT}, @var{xdata}, @var{ydata}] =} imtransform (@dots{})
## Transform image.
##
## Given an image @var{IM_IN}, return an image @var{IM_OUT}
## resulting from the forward transform defined in the transformation
## structure @var{T}.  An additional input argument @var{interp},
## 'bicubic', 'bilinear' (default) or 'nearest',
## specifies the interpolation method to be used.  Finally, the
## transform can be tuned using  @var{prop}/@var{val} pairs.  The
## following properties @var{prop} are supported:
## 
## @table @asis
## @item "udata"
## Specifies the input space horizontal limits. @var{val} must be a two
## elements vector [minval maxval]. Default: [1 columns(@var{IM_IN})]
##
## @item "vdata"
## Specifies the input space vertical limits. @var{val} must be a two
## elements vector [minval maxval]. Default: [1 rows(@var{IM_IN})]
##
## @item "xdata"
## Specifies the required output space horizontal limits. @var{val} must
## be a two elements vector [minval maxval]. Default: estimated using
## udata, vdata and findbounds function.
##
## @item "ydata"
## Specifies the required output space vertical limits. @var{val} must
## be a two elements vector [minval maxval]. Default: estimated using
## udata, vdata and findbounds function.
##
## @item "xyscale"
## Specifies the size of pixels in the output space. If a scalar
## is provided, both vertical and horizontal dimensions are scaled the
## same way. If @var{val} is a two element vector, it must indicate
## consecutively width and height of the output pixels. The default is
## to use the width and height of input pixels provided
## that it does not lead to a too large output image.
##
## @item "size"
## Size of the output image (1-by-2 vector). Overrides the effect of
## "xyscale" property. 
##
## @item "fillvalues"
## Color of the areas where no interpolation is possible, e.g. when
## coordinates of points in the output space are out of the limits of
## the input space. @var{val} must be coherent with the input image
## format: for grayscale and indexed images (2D) @var{val} must be
## scalar, for RGB (n-by-m-by-3) @var{val} must be a 3 element vector.
## 
## @end table
##
## The actual output limits, @var{xdata} and @var{ydata} vectors,
## are returned respectively as second  and third output variables.
## @seealso{maketform, cp2tform, tforminv, tformfwd, findbounds}
## @end deftypefn

## Author: Pantxo Diribarne <pantxo@dibona>

function [varargout] = imtransform (im, T, varargin)

  if (nargin < 2)
    print_usage ();
  elseif (! istform (T))
    error ("imtransform: T must be a transformation structure (see `maketform')");
  endif

  ## Parameters
  interp = "linear";
  imdepth = size (im, 3);
  maximsize = [20000 20000];

  udata = [1; columns(im)];
  vdata = [1; rows(im)];
  xdata = ydata = [];
  xyscale = [];
  imsize = [];
  fillvalues = zeros (1, imdepth);

  if (isempty (varargin))
    xydata = findbounds (T, [udata vdata]);
    xdata = xydata(:,1);
    ydata = xydata(:,2);
  else
    ## interp
    if (floor (numel (varargin)/2) != numel (varargin)/2)
      allowed = {"bicubic", "bilinear", "nearest"};
      tst = strcmp (varargin{1}, allowed);
      if (!any (tst))
        error ("imtransform: expect one of %s as interp method", disp (allowed));
      else
        interp = {"pchip", "linear", "nearest"}{find (tst)};
      endif
      varargin = varargin(2:end);
    endif

    ## options
    allowed = {"udata", "vdata", "xdata", "ydata", ...
               "xyscale", "size", "fillvalues"};
    props = varargin(1:2:end);
    vals = varargin(2:2:end);
    np = numel (props);
    if (!all (cellfun (@ischar, props)))
      error ("imtransform: expect property/value pairs.");
    endif

    props = tolower (props);
    tst = cellfun (@(x) any (strcmp (x, allowed)), props);
    if (!all (tst))
      error ("imtransform: unknown property %s", disp (props{!tst}));
    endif

    ## u(vxy)data
    iolims = allowed(1:4);
    for ii = 1:numel (iolims)
      tst = cellfun (@(x) any (strcmp (x, iolims{ii})), props);
      if (any (tst))
        prop = props{find (tst)(1)};
        val = vals{find (tst)(1)};
        if (isnumeric (val) && numel (val) == 2)
          if (isrow (val))
            val = val';
          endif
          eval (sprintf ("%s = val;", prop),
                "error (\"imtransform: %s\n\", lasterr ());");
        else
          error ("imtransform: expect 2 elements real vector for %s", prop)
        endif
      endif
    endfor
    if (isempty (xdata) && isempty (ydata))
      xydata = findbounds (T, [udata vdata]);
      xdata = xydata(:,1);
      ydata = xydata(:,2);
    elseif (isempty (xdata))
      xydata = findbounds (T, [udata vdata]);
      xdata = xydata(:,1);
    elseif (isempty (ydata))
      xydata = findbounds (T, [udata vdata]);
      ydata = xydata(:,2);
    endif

    ## size and xyscale
    tst = strcmp ("size", props);
    if (any (tst))
      val = vals{find (tst)(1)};
      if (isnumeric (val) && numel (val) == 2 &&
          all (val > 0))
        imsize = val;
      else
        error ("imtransform: expect 2 elements real vector for size");
      endif
    elseif (any (tst = strcmp ("xyscale", props)))
      val = vals{find (tst)(1)};
      if (isnumeric (val) && all (val > 0))
        if (numel (val) == 1)
          xyscale(1:2) = val;
        elseif (numel (val) == 2)
          xyscale = val;
        else
          error ("imtransform: expect 1 or 2 element(s) real vector for xyscale");
        endif
      else
        error ("imtransform: expect 1 or 2 elements real vector for xyscale");
      endif
    endif
    ## Fillvalues
    tst = strcmp ("fillvalues", props);
    if (any (tst))
      val = vals{find (tst)(1)};
      if (isnumeric (val) && numel (val) == 1)
        fillvalues(1:end) = val;
      elseif (isnumeric (val) && numel (val) == 3)
        fillvalues = val;
      else
        error ("imtransform: expect 1 or 3 elements real vector for `fillvalues'");
      endif
    endif
  endif

  ## Output/Input pixels
  if (isempty (imsize))
    if (isempty (xyscale))
      nx = columns (im);
      ny = rows (im);
      xyscale = [(diff (udata) / (nx - 1)), ...
                 (diff (vdata) / (ny - 1))];
    endif
    xscale = xyscale(1);
    yscale = xyscale(2);
    xsize = ceil (diff (xdata) / xscale) + 1;
    ysize = ceil (diff (ydata) / yscale) + 1;
    ## xyscale takes precedence: recompute x/ydata considering roundoff errors
    xdata(2) = xdata(1) + (xsize - 1) * xscale;
    ydata(2) = ydata(1) + (ysize - 1) * yscale;
    if (xsize > maximsize(2) || ysize > maximsize(1))
      if (xsize >= ysize)
        scalefactor = (diff (xdata) / maximsize(2)) / xscale;
      else
        scalefactor = (diff (ydata) / maximsize(1)) / yscale;
      endif
      xscale *= scalefactor;
      yscale *= scalefactor;
      xsize = floor (diff (xdata) / xscale);
      ysize = floor (diff (ydata) / yscale);
      warning ("imtransform: output image two large, adjusting the largest dimension to %d", maximsize);
    endif
    imsize = [ysize xsize];
  endif
  [xx yy] = meshgrid (linspace (xdata(1), xdata(2), imsize(2)),
                      linspace (ydata(1), ydata(2), imsize(1)));

  [uu vv] = meshgrid (linspace (udata(1), udata(2), size(im)(2)),
                      linspace (vdata(1), vdata(2), size(im)(1)));

  ## Input coordinates
  [uui, vvi] = tforminv (T, reshape (xx, numel (xx), 1),
                         reshape (yy, numel (yy), 1));
  uui = reshape (uui, size (xx));
  vvi = reshape (vvi, size (yy));
  ## Interpolation
  for layer = 1:imdepth
    imout(:,:,layer) = interp2 (uu, vv, im(:,:,layer), ...
                                uui, vvi, interp, fillvalues(layer));
  endfor
  if (nargout != 0)
    varargout = {imout, xdata(:).', ydata(:).'};
  endif
endfunction

%!demo
%! ## Various linear transforms
%! figure (); 
%! im = [checkerboard(20, 2, 4); checkerboard(40, 1, 2)];
%! %input space corners
%! incp = [1 1; 160 1; 160 160; 1 160];
%! udata = [min(incp(:,1)) max(incp(:,1))];
%! vdata = [min(incp(:,2)) max(incp(:,2))];
%! subplot (2,3,1); 
%! imshow (im)
%! hold on
%! plot (incp(:,1), incp(:,2), 'ob')
%! axis on
%! xlabel ('Original')
%! 
%! % Translation and scaling
%! outcp = incp * 2;
%! outcp(:,1) += 200; 
%! outcp(:,2) += 500;
%! T = maketform ('affine', incp(1:3,:), outcp(1:3,:));
%! subplot (2,3,2); 
%! [im2 xdata ydata] = imtransform (im, T, 'udata', udata,
%!                                  'vdata', vdata, 'fillvalues', 1);
%! imh = imshow (im2); set (imh, 'xdata', xdata, 'ydata', ydata) 
%! set (gca, 'xlim', xdata, 'ylim', ydata)
%! axis on, hold on, xlabel ('Translation / Scaling');
%! plot (outcp(:,1), outcp(:,2), 'or')
%! 
%! % Shear
%! outcp = [1 1; 160 1; 140 160; -19 160]; % affine only needs 3 control points
%! T = maketform ('affine', incp(1:3,:), outcp(1:3,:));
%! subplot (2,3,3); 
%! [im2 xdata ydata] = imtransform (im, T, 'udata', udata,
%!                                  'vdata', vdata, 'fillvalues', 1);
%! imh = imshow (im2); set (imh, 'xdata', xdata, 'ydata', ydata) 
%! set (gca, 'xlim', xdata, 'ylim', ydata)
%! axis on, hold on, xlabel ('Shear');
%! plot (outcp(:,1), outcp(:,2), 'or')
%! 
%! % Rotation 
%! theta = pi/4;
%! T = maketform ('affine', [cos(theta) -sin(theta); ...
%!                           sin(theta) cos(theta); 0 0]);
%! outcp = tformfwd (T, incp);
%! subplot (2,3,4); 
%! [im2 xdata ydata] = imtransform (im, T, 'udata', udata,
%!                                  'vdata', vdata, 'fillvalues', 1 );
%! imh = imshow (im2); set (imh, 'xdata', xdata, 'ydata', ydata) 
%! set (gca, 'xlim', xdata, 'ylim', ydata)
%! axis on, hold on, xlabel ('Rotation');
%! plot (outcp(:,1), outcp(:,2), 'or')
%!
%! % Reflection around x axis
%! outcp = incp;
%! outcp(:,2) *= -1;
%! T = cp2tform (incp, outcp, 'similarity');
%! subplot (2,3,5); 
%! [im2 xdata ydata] = imtransform (im, T, 'udata', udata,
%!                                  'vdata', vdata, 'fillvalues', 1 );
%! imh = imshow (im2); set (imh, 'xdata', xdata, 'ydata', ydata) 
%! set (gca, 'xlim', xdata, 'ylim', ydata)
%! axis on, hold on, xlabel ('Reflection');
%! plot (outcp(:,1), outcp(:,2), 'or')
%!
%! % Projection
%! outcp = [1 1; 160 -40; 220 220; 12 140];
%! T = maketform ('projective', incp, outcp);
%! subplot (2,3,6); 
%! [im2 xdata ydata] = imtransform (im, T, 'udata', udata,
%!                                  'vdata', vdata, 'fillvalues', 1 );
%! imh = imshow (im2); set (imh, 'xdata', xdata, 'ydata', ydata) 
%! set (gca, 'xlim', xdata, 'ylim', ydata)
%! axis on, hold on, xlabel ('Projection');
%! plot (outcp(:,1), outcp(:,2), 'or')

%!demo
%! ## Streched image
%! rad = 2; % minimum value: 4/pi
%! [uu vv] = meshgrid ((-2:2)/rad, (-2:2)/rad);
%! rescfactor = sin ((uu.^2 + vv.^2).^.5);
%! inpts = [(reshape (uu, numel (uu), 1)), (reshape (vv, numel (uu), 1))]; 
%! xx = rescfactor .* sign(uu);
%! yy = rescfactor .* sign(vv);
%! outpts = [reshape(xx, numel (xx), 1) reshape(yy, numel (yy), 1)];
%! 
%! T = cp2tform (inpts, outpts, "polynomial", 4);
%! figure;
%! subplot (1,2,1)
%! im = zeros (800, 800, 3);
%! im(:,:,1) = checkerboard (100) > 0.2;
%! im(:,:,3) = checkerboard (100) < 0.2;
%! [im2 xdata ydata] = imtransform (im, T, 'udata', [-2 2],
%!                                  'vdata', [-2 2], 'fillvalues',
%!                                  [0 1 0]);
%! imh = imshow (im2);
%! set (imh, 'xdata', xdata, 'ydata', ydata)
%! set (gca, 'xlim', xdata, 'ylim', ydata)
%! [im cmap] = imread ('default.img');
%! subplot (1,2,2)
%! [im2 xdata ydata] = imtransform (im, T, 'udata', [-1 1],
%!                                  'vdata', [-1 1], 'fillvalues',
%!                                  round (length (cmap) / 2));
%! imh = imshow (im2, cmap);

%!test
%! im = checkerboard ();
%! incp = [0 0; 0 1; 1 1];
%! scl = 10;
%! outcp = scl * incp;
%! T = maketform ('affine', incp, outcp);
%! [im2 xdata ydata] = imtransform (im, T, 'udata', [0 1],
%!                                  'vdata', [0 1], 'size', [500 500]);
%! assert (xdata, scl * ([0 1]))
%! assert (ydata, scl * ([0 1]))
%! assert (size (im2), [500 500])

%!test
%! im = checkerboard ();
%! incp = [0 0; 0 1; 1 1];
%! scl = 10;
%! outcp = scl * incp;
%! xyscale = scl;
%! T = maketform ('affine', incp, outcp);
%! [im2 xdata ydata] = imtransform (im, T, 'xyscale', xyscale);
%! assert (size (im2), size (im), 1)

%!test
%! im = checkerboard (100, 10, 4);
%! theta = 2 * pi;
%! T = maketform ("affine", [cos(theta) -sin(theta); ...
%!                           sin(theta) cos(theta); 0 0]);
%! im2 = imtransform (im, T, "nearest", "xdata", [1 800], "ydata", [1 2000]);
%! im = im(2:end-1, 2:end-1); %avoid boundaries 
%! im2 = im2(2:end-1, 2:end-1);
%! assert (im, im2)

%!test
%! im = checkerboard (20, 10, 4);
%! theta = pi/6;
%! T = maketform ('affine', [cos(theta) -sin(theta); ...
%!                           sin(theta) cos(theta); 0 0]);
%! [im2, xdata] = imtransform (im, T);
%! nu = columns(im);
%! nv = rows(im);
%! nx = xdata(2);
%! diag = sqrt (nu^2 + nv^2);
%! ang = atan (nv / nu);
%! assert (nx, diag * abs (cos (theta - ang)),
%!         diag * 1 / size (im2, 2))

## Test default fillvalues
%!test
%! im = rand (2);
%! tmat = [eye(2); 0 0];
%! T = maketform ("affine", tmat);
%! im2 = imtransform (im, T, "xdata", [1 3]);
%! assert (im2(:,3), zeros (2,1))

## Test image size when forcing x/ydata
%!test
%! im = rand (2);
%! tmat = [eye(2); 0 0];
%! T = maketform ('affine', tmat);
%! im2 = imtransform (im, T, "xdata", [1 3]);
%! assert (size (im2), [2 3])

## Test image size when forcing xyscale
%!test
%! im = rand (2);
%! tmat = [eye(2); 0 0];
%! T = maketform ('affine', tmat);
%! im2 = imtransform (im, T, "xyscale", [0.5 0.5]);
%! assert (size (im2), [3 3])
