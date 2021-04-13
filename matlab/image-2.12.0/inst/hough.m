## Copyright (C) 2017 Hartmut Gimpel <hg_code@gmx.de>
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
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*-
## @deftypefn  {Function File} {[@var{H}, @var{theta}, @var{rho}] =} @
##   hough (@var{BW})
## @deftypefnx {Function File} {[@var{H}, @var{theta}, @var{rho}] =} @
##   hough (@var{BW}, @var{property}, @var{value}, @dots{})
## Compute the Hough transform to find lines in a binary image.
##
## The resulting Hough transform matrix @var{H} (accumulator array) is
## 2D.  Its rows correspond to the distance values @var{rho} and its
## columns to the angle values @var{theta}.  Points of high value in
## @var{H} correspond to present lines in the given image.
##
## The distance @var{rho} is measured with respect to the image
## origin.  The angle @var{theta} is measured clockwise to the
## vertical axis.
##
## The following @var{property} values are possible:
##
## @table @asis
## @item @qcode{"Theta"}
## A vector of angle values.  The Hough transform will be calculated
## at those Theta angle values.  The angles are given in degrees.
## Defaults to [-90:89].
##
## @item @qcode{"ThetaResolution"}
## A scalar value to specify the Theta angles for the Hough transform
## in a different way.  This will result in Theta =
## [-90:ThetaResolution:90], with the +90° angle excluded.
##
## @end table
##
## @seealso{hough_line, hough_circle, immaximas}
## @end deftypefn

## TODO: add option RhoResolution
##
## @item @qcode{"RhoResolution"}
## A scalar value to specify the Resolution of the @var{rho} distance
## values. Defaults to 1.

function [H, theta, rho] = hough (bw, varargin)

  if (nargin < 1)
    print_usage ();
  endif

  validateattributes (bw, {"logical", "numeric"}, {"2d"}, "hough", "BW");
  bw = logical (bw);

  ## set default parameters:
  theta = [-90:1:89];
  theta_res = 1;

  ## process property/value pairs
  if (rem (numel (varargin), 2) != 0)
    error ("hough: PROPERTY/VALUE arguments must occur in pairs");
  endif

  for idx = 1:2:(numel (varargin))
    switch (tolower (varargin{idx}))

      case "rhoresolution"
        rho_res = varargin{idx+1};
        ## This option is not yet implemented.  The default will be 1
        ## so error out if the user tries to set it to anything else.
        if (rho_res != 1)
          error ("hough: option RHORESOLUTION is not implemented");
        endif

      case "thetaresolution"
        theta_res = varargin{idx+1};
        if (! (isreal (theta_res) && isscalar (theta_res)
               && (theta_res > 0) && (theta_res < 180)))
          error ("hough: value THETARESOLUTION must be between 0 and 180");
        endif

      case "theta"
        theta = varargin{idx+1};
        if (! (isreal (theta) && isvector (theta)))
          error ("hough: values THETA must be a vector of real numbers");
        endif

      otherwise
        error ("hough: unknown property `%s'", varargin{idx});

    endswitch
  endfor

   if (theta_res != 1)
     theta = [-90:theta_res:90];
     theta = theta(theta != 90);    # exclude +90 degrees
   endif

   ## Matlab's hough.m function measures the angle theta clockwise to the
   ## vertical axis, but Octave's hough_line.cc measure this angle (as usual)
   ## counter-clockwise the the horizontal axis. Octave's hough_line also
   ## need radians, Matlab uses degrees. So we translate this.
   theta_oct = (-theta+90) * (pi/180);

   ## eventually call hough_line.cc to do the real work
   [H, rho] = hough_line (bw, theta_oct);

endfunction

%!shared BW0, BW1, BW2, BWx, BWy
%!
%! BW0 = false (5);
%! BW0(2,2) = true;
%!
%! BW1 = zeros (100, 100);
%! BW1(1,1) = 1;
%! BW1(100,100) = 1;
%! BW1(1,100) = 1;
%! BW1(100, 1) = 1;
%! BW1(50,50) = 1;
%!
%! n = 100;
%! BW2 = false (n);
%! a = 50;    # line starts at left side at row a
%! b = 3;      # slope of line is 1:b
%! for column = 1:n
%!   if (rem (column, b) == 0)
%!     row = a - column/b;
%!     BW2(row, column) = true;
%!   endif
%! endfor
%!
%! BWx = false (10);
%! BWx(:,5) = true;
%!
%! BWy = false (10);
%! BWy(5,:) = true;

%!test
%! [H, T, R] = hough (BW1);
%! assert (size (H), [283 180]);

%!test
%! [H, T, R] = hough (BW1, "Theta", [-90 0 45 79]);
%! assert (size (H), [283 4]);

%!test
%! [H, T, R] = hough (BW1, "ThetaResolution", 0.5);
%! assert (size (H), [283 360]);

%!error <BW must be of class> hough ("foo")

## Non binary data, just gets cast to logical, not even rounding.
%!test
%! I = [0 0 1 0; 1 1 1 1; 0 0 1 1; 0 0 1 0];
%! I2 = I;
%! for v = [0.7 0.2 5]
%!   I2(1,3) = v;
%!   assert (hough (I2), hough (I))
%! endfor

%!error <must occur in pairs>
%! [H, T, R] = hough (BW0, "Theta");

%!error <must be a vector of real numbers>
%! [H, T, R] = hough (BW0, "Theta", ones (10));

%!error <must be a vector of real numbers>
%! [H, T, R] = hough (BW0, "Theta", [5 -i 7]);

%!error <not implemented>
%! [H, T, R] = hough (BW0, "RhoResolution", 0.5);

## RhoResolution defaults to 1
%!test
%! [Hd, Td, Rd] = hough (BW0);
%! [H1, T1, R1] = hough (BW0, "RhoResolution", 1);
%! assert (Hd, H1)
%! assert (Td, T1)
%! assert (Rd, R1)

%!test
%! [H, theta, rho] = hough (BW2);
%! H_max = max (H(:));
%! H_size = size (H);
%! [~, max_idx_lin] = max (H(:));
%! [max_row, max_column] = ind2sub (size (H), max_idx_lin);
%! theta_max = theta(max_column);
%! rho_max = rho(max_row);
%! assert (H_max , 33);
%! assert (H_size, [283 180]);
%! assert (max_row, 188);
%! assert (max_column, 163);
%! assert (theta_max, 72);
%! assert (rho_max, 46);

%!test
%! [H, theta, rho] = hough (BW2, "Theta", [65:1:75]);
%! H_max = max (H(:));
%! H_size = size (H);
%! [~, max_idx_lin] = max (H(:));
%! [max_row, max_column] = ind2sub (size (H), max_idx_lin);
%! theta_max = theta(max_column);
%! rho_max = rho(max_row);
%! assert (H_max , 33);
%! assert (H_size, [283 11]);
%! assert (max_row, 188);
%! assert (max_column, 8);
%! assert (theta_max, 72);
%! assert (rho_max, 46);

%!test
%! [H, theta, rho] = hough (BW2, "Theta", [-90:0.5:89.5]);
%! H_max = max (H(:));
%! H_size = size (H);
%! [~, max_idx_lin] = max (H(:));
%! [max_row, max_column] = ind2sub (size (H), max_idx_lin);
%! theta_max = theta(max_column);
%! rho_max = rho(max_row);
%! assert (H_max , 33);
%! assert (H_size, [283 360]);
%! assert (max_row, 188);
%! assert (max_column, 324);
%! assert (theta_max, 71.5);
%! assert (rho_max, 46);

%!test
%! [H, theta, rho] = hough (BW2, "ThetaResolution", 0.5);
%! H_max = max (H(:));
%! H_size = size (H);
%! [~, max_idx_lin] = max (H(:));
%! [max_row, max_column] = ind2sub (size (H), max_idx_lin);
%! theta_max = theta(max_column);
%! rho_max = rho(max_row);
%! assert (H_max , 33);
%! assert (H_size, [283 360]);
%! assert (max_row, 188);
%! assert (max_column, 324);
%! assert (theta_max, 71.5);
%! assert (rho_max, 46);

%!test
%! [H, theta, rho] = hough (BWx);
%! H_max = max (H(:));
%! [~, max_idx_lin] = max (H(:));
%! [max_row, max_column] = ind2sub (size (H), max_idx_lin);
%! theta_max = theta(max_column);
%! rho_max = rho(max_row);
%! assert (H_max , 10);
%! assert (max_column, 88);
%! assert (theta_max, -3);
%! assert (rho_max, 4);

%!test
%! [H, theta, rho] = hough (BWx);
%! H_size = size (H);
%! [~, max_idx_lin] = max (H(:));
%! [max_row, max_column] = ind2sub (size (H), max_idx_lin);
%! assert (H_size, [27 180]);
%! assert (max_row, 18);

%!test
%! [H, theta, rho] = hough (BWy);
%! H_max = max (H(:));
%! [~, max_idx_lin] = max (H(:));
%! [max_row, max_column] = ind2sub (size (H), max_idx_lin);
%! theta_max = theta(max_column);
%! rho_max = rho(max_row);
%! assert (H_max , 10);
%! assert (max_column, 1);
%! assert (theta_max, -90);
%! assert (rho_max, -4);

%!test
%! [H, theta, rho] = hough (BWy);
%! H_size = size (H);
%! [~, max_idx_lin] = max (H(:));
%! [max_row, max_column] = ind2sub (size (H), max_idx_lin);
%! assert (H_size, [27 180]);
%! assert (max_row, 10);

%!demo
%! BW = zeros (100, 150);
%! BW(30,:) = 1;
%! BW(:, 65) = 1;
%! BW(35:45, 35:50) = 1;
%! for i = 1:90
%!   BW(i,i) = 1;
%! endfor
%! BW = imnoise (BW, "salt & pepper");
%! figure ();
%! imshow (BW);
%! title ("BW");
%! [H, theta, rho] = hough (BW);
%! H /= max (H(:));
%! figure ();
%! imshow (H, "XData", theta, "YData", rho);
%! title ("hough transform of BW");
%! axis on;
%! xlabel ("angle \\theta [degrees]");
%! ylabel ("distance \\rho to origin [pixels]");
