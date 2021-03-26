## Copyright (C) 2010 Alex Opie <lx_op@orcon.net.nz>
## Copyright (C) 2013 Carnë Draug <carandraug@octave.org>
##
## This program is free software; you can redistribute it and/or modify it under
## the terms of the GNU General Public License as published by the Free Software
## Foundation; either version 3 of the License, or (at your option) any later
## version.
##
## This program is distributed in the hope that it will be useful, but WITHOUT
## ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
## FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
## details.
##
## You should have received a copy of the GNU General Public License along with
## this program; if not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*-
## @deftypefn  {Function File} {@var{P}} = phantom ()
## @deftypefnx {Function File} {@var{P}} = phantom (@var{model})
## @deftypefnx {Function File} {@var{P}} = phantom (@var{E})
## @deftypefnx {Function File} {@var{P}} = phantom (@dots{}, @var{n})
## @deftypefnx {Function File} {[@var{P}, @var{E}]} = phantom (@dots{})
## Create computational phantom head.
##
## A phantom is a known object (either real or purely mathematical) that is
## used for testing image reconstruction algorithms.  The Shepp-Logan phantom
## is a popular mathematical model of a cranial slice, made up of a set of
## overlaying ellipses.  This allows rigorous testing of computed tomography
## (CT) algorithms as it can be analytically transformed with the radon
## transform (see the functions @code{radon} and @code{iradon}).
##
## The phantom @var{P}, is created by overlaying ellipses as defined by the
## matrix @var{E} or one of the standard @var{model}s, in a square of size
## @var{n} by @var{n} (defaults to 256).
##
## The available standard @var{model}s (use the output argument @var{E} to
## inspect the details of the different ellipses) are:
##
## @table @asis
## @item @qcode{"Shepp-Logan"}
## This is the original Shepp-Logan model with 10 ellipses as described in
## Table 1 of @cite{Shepp, Lawrence A., and Benjamin F. Logan. "The Fourier
## reconstruction of a head section." Nuclear Science, IEEE Transactions on
## 21, no. 3 (1974): 21-43.}
##
## @item @qcode{"Modified Shepp-Logan"} (default)
## A modification of the original Shepp-Logan model to give a better contrast,
## as described in Table B.3 of @cite{Toft, Peter Aundal. "The radon
## transform-theory and implementation." PhD diss., Department of Mathematical
## Modelling, Technical University of Denmark, 1996.}
##
## @end table
##
## A 6 column matrix @var{E} can be used to generate a custom image by
## superimposing arbitrary ellipses.  Each row defines a single ellipse, with
## each column for the values of @{I, a, b, x0, y0, phi@}:
##
## @table @abbr
## @item I
##   is the additive intensity of the ellipse
##
## @item a
##    is the length of the major axis
##
## @item b
##    is the length of the minor axis
##
## @item x0
##    is the horizontal offset of the centre of the ellipse
##
## @item y0
##    is the vertical offset of the centre of the ellipse
##
## @item phi
##    is the counterclockwise rotation of the ellipse in degrees,
##    measured as the angle between the x axis and the ellipse major axis.
##
## @end table
##
## The image bounding box in the algorithm is @{[-1, -1], [1, 1]@}, so the
## values of a, b, x0, y0 should all be specified with this in mind.
##
## Example:
##
## @example
## @group
## P = phantom (512);
## imshow (P);
## @end group
## @end example
##
## @seealso{iradon, radon}
## @end deftypefn

function [head, ellipses] = phantom (varargin)

  if (nargin > 2)
    print_usage ()
  endif

  ## Would be really cool if we implemented a 3D phantom as already described
  ## in Cheng Guan Koay, Joelle E. Sarlls, and Evren Ozarslan (2007).
  ## "Three-Dimensional Analytical Magnetic Resonance Imaging Phantom in the
  ## Fourier Domain". Magnetic Resonance in Medicine 58:430 - 436.
  ## The Table 1 on their paper to generate the 3D model, would take 8 columns,
  ## an extra value for z axis coordinates, and extra axis length.
  ## They mention other phantom heads as more canonical 3D head phantoms (read
  ## the introduction)

  ## Defaults
  ellipses  = mod_shepp_logan ();
  n         = 256;

  if (nargin)
    ## Check validity of N
    chk_n = @(x) isnumeric (x) && isscalar (x) && ceil (x) == x;

    in = varargin{1};
    if (ischar (in))
      switch (tolower (in))
        case "shepp-logan",           ellipses = shepp_logan ();
        case "modified shepp-logan",  ellipses = mod_shepp_logan ();
        otherwise
          error ("phantom: unknown MODEL `%s'", in);
      endswitch
    elseif (isnumeric (in) && ndims (in) == 2 && columns (in) == 6)
      ellipses = in;
    elseif (chk_n (in))
      n = in;
      ## If N is the first argument, we can't have more
      if (nargin > 1)
        print_usage ();
      endif
    else
      error ("phantom: first argument must either be MODEL, E, or N");
    endif

    ## If there is a second input argument, must be N
    if (nargin > 1)
      if (chk_n (varargin{2}))
        n = varargin{2};
      else
        error ("phantom: N must be numeric scalar");
      endif
    endif
  endif

  ## Initialize blank image
  head = zeros (n);

  # Create the pixel grid
  xvals = (-1 : 2 / (n - 1) : 1);
  xgrid = repmat (xvals, n, 1);

  for i = 1:rows (ellipses)
    I   = ellipses (i, 1);
    a2  = ellipses (i, 2)^2;
    b2  = ellipses (i, 3)^2;
    x0  = ellipses (i, 4);
    y0  = ellipses (i, 5);
    phi = ellipses (i, 6) * pi / 180;  # Rotation angle in radians

    ## Create the offset x and y values for the grid
    x = xgrid - x0;
    y = rot90 (xgrid) - y0;

    cos_p = cos (phi);
    sin_p = sin (phi);

    ## Find the pixels within the ellipse
    locs = find (((x .* cos_p + y .* sin_p).^2) ./ a2 ...
      + ((y .* cos_p - x .* sin_p).^2) ./ b2 <= 1);

    ## Add the ellipse intensity to those pixels
    head(locs) += I;
  endfor
endfunction

function ellipses = shepp_logan ()
  ## Standard head phantom, taken from Shepp & Logan
  ##
  ## Note that the first element of this matrix, the gray value for the first
  ## ellipse (human skull), has a value of 1.0 even though the paper gives it a
  ## a value of 2.0 (see Table 1 on page 32 and Figure 1 on page 34). This
  ## change is so that the **head** intensity values appear in the range [0 1]
  ## rather than the range [1 2].
  ##
  ##    **The problem with this**
  ##
  ## The background still need an intensity value which is going to be 0. This
  ## means that we can't distinguish between the background and the ventricles
  ## (ellipse "c" and "d" whose intensities are a + b + c and a + b + d, see
  ## Figure 1) since they will have an intensity value of 0 (actually, because
  ## of machine precision the ventricules will be almost 0). But if we didn't
  ## made this change, the ** image** range would be [0 2] with all of the head
  ## details compressed in half of the display range. Also, Matlab seems to be
  ## doing the same.
  ellipses = [ 1     0.69    0.92     0      0         0
              -0.98  0.6624  0.874    0     -0.0184    0
              -0.02  0.11    0.31     0.22   0       -18
              -0.02  0.16    0.41    -0.22   0        18
               0.01  0.21    0.25     0      0.35      0
               0.01  0.046   0.046    0      0.1       0
               0.01  0.046   0.046    0     -0.1       0
               0.01  0.046   0.023   -0.08  -0.605     0
               0.01  0.023   0.023    0     -0.606     0
               0.01  0.023   0.046    0.06  -0.605     0];
endfunction

function ellipses = mod_shepp_logan ()
  ## Modified version of Shepp & Logan's head phantom, adjusted to improve
  ## contrast.  Taken from Peter Toft PhD thesis, Table B.3
  ellipses = [ 1.0   0.69    0.92     0.0    0.0       0
              -0.8   0.6624  0.874    0.0   -0.0184    0
              -0.2   0.11    0.31     0.22   0.0     -18
              -0.2   0.16    0.41    -0.22   0.0      18
               0.1   0.21    0.25     0.0    0.35      0
               0.1   0.046   0.046    0.0    0.1       0
               0.1   0.046   0.046    0.0   -0.1       0
               0.1   0.046   0.023   -0.08  -0.605     0
               0.1   0.023   0.023    0.0   -0.606     0
               0.1   0.023   0.046    0.06  -0.605     0];
endfunction

%!demo
%! P = phantom (512);
%! imshow (P);
