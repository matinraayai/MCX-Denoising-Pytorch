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
## @deftypefn   {Function File} {@var{lines} =} @ houghlines (@var{BW}, @var{theta}, @var{rho}, @var{peaks})
## @deftypefnx  {Function File} {@var{lines} =} @ houghlines (@dots{}, @var{property}, @var{value}, @dots{})
## Extract line segments from a Hough transform.
##
## This function takes as inputs the binary 2d image @var{BW} and the vectors @var{theta} and @var{rho} with the coodinates
## of the Hough transform (as returned by the @code{hough} function). Its @var{peaks} input is an n-by-2 array where each row
## contains the coodinates of a peak of interest in the Hough transform. (Those peaks in the Hough transform can be
## found with the @code{houghpeaks} function.)
##
## The result @var{lines} of this function contains information about all the line segments
## in the image @var{BW} that correspond to the given @var{peak} positions of the Hough transform.
## The @var{lines} output is a struct array where each of the elements has the following four
## components to describe a single line segment: @code{point1} has the xy-coordinates of the first pixel, 
##@code{point2} the xy-coordinates of the last pixel, @code{theta} its angle to the vertical axis and @code{rho} its
## distance to the image origin. (output coordinate convention: [x, y] = [column, row])
##
## Additionally the following optional property-value-pairs can be used:
## @table @asis
## @item @var{FillGap}
## Gaps between line segments that are shorter or equal than @var{FillGap} will be ignored and both sides will still be
## considered as part of the same line segment.
## This value defaults to 20.
##
## @item @var{MinLength}
## Line segments that are shorter than @var{MinLength} will be suppressed in the output.
## This value defaults to 40.
## @end table
## 
## @seealso{hough, houghpeaks}
## @end deftypefn

## Algorithm:
## The Matlab help page does not cite any reference
## for the algorithm of this function.
##
## For this Octave implementation the information
## on Matlab's help page, as well as the information
## from this book was used:
##    "Digital Image Processing using Matlab"
##    by R.C. Gonzalez, R. E. Woods and S. L. Eddins
##    McGrawHill, 2nd edition 2010.
##    (Chapter 10.2.2. "Toolbox Hough Functions")
##
## The result is the following straight forward (brute force?)
## implementation. The individual steps are commented
## in the code below.

function lines = houghlines (BW, theta, rho, peaks, varargin)

  ## retrieve the input parameters:
  fillgap = [];
  minlength = [];
  
  if ((nargin < 4) || (nargin > 8) || any (nargin == [5, 7]))
    print_usage ();
  endif
    
  for n = 5:2:(nargin-1)         # process parameter-values pairs
    if (strcmpi (varargin{n-4}, "fillgap"))
      fillgap = varargin{n-4+1};
    elseif (strcmpi (varargin{n-4}, "minlength"))
      minlength = varargin{n-4+1};
    else
      error ("houghlines: invalid PROPERTY given")
    endif
  endfor
  
  ## set default parameters:
  if (isempty (fillgap))
    fillgap = 20;
  endif
  
  if (isempty (minlength))
    minlength = 40;
  endif
  
  ## check input parameters:
  if (! isimage (BW) || ndims (BW) != 2)
    error ("houghlines: BW must be a logical or numeric 2d array");
  endif

  if (! isimage (theta) || ! isnumeric (theta) || ! isvector (theta))
    error ("houghlines: THETA must be a numeric vector");
  endif  
  
  if (! isimage (rho) || ! isnumeric (rho) || ! isvector (rho))
    error ("houghlines: RHO must be a numeric vector");
  endif  
  
  if (! isimage (peaks) || ! isnumeric (peaks) || ndims  (peaks) > 2 || size (peaks, 2) != 2)
    error ("houghlines: PEAKS must be a n-by-2 numeric array");
  endif    
  
  if (! isnumeric (fillgap) || ! isreal (fillgap) || fillgap <= 0 || ! isscalar (fillgap))
    error ("houghlines: FILLGAP must be a positive scalar number");
  endif
  
  if (! isnumeric (minlength) || ! isreal (minlength) || minlength <= 0 || ! isscalar (minlength))
    error ("houghlines: MINLENGTH must be a positive scalar number");
  endif
      
  ## start the calculation:
  lines = struct ([]);
  numpeaks = size (peaks, 1);
  numlines = 0;
  
  ## find all foreground pixels and transform their 
  ## coordinates to conventions of Hough transform
  ## xy and (1,1) based
  [allpixels_r, allpixels_c] = find (BW);
  origin = [1 1];
  allpixels_x =  allpixels_c - origin(1);
  allpixels_y =  allpixels_r - origin(2);
  
  ## process each given Hough peak individually
  for n = 1:numpeaks
    rho_p_idx = peaks(n, 1);
    theta_p_idx = peaks(n, 2);
    rho_p = rho(rho_p_idx);           # distance from "origin" pixel at (1,1)
    theta_p = theta(theta_p_idx);  # measured clockwise to the vertical axis, in degrees

    ## Find all the image pixels that belong to this
    ## Hough accumulator cell with theta_p and rho_p:
    ## (What rho would those pixels have, if they really had theta_p?)
    ## (rho2idx_factor is a precaution for when hough.m will be able 
    ##  to deal with the RhoResolution parameter.)
    rho_all =  allpixels_x .* cosd (theta_p) +allpixels_y .* sind (theta_p);
    rho2idx_factor =  (length (rho) -1) ./ (rho(end) - rho(1)); 
    rho_all_idx = round(( rho_all - rho(1) ) .*  rho2idx_factor) + 1;     
    peak_pixels_idx = find (rho_all_idx == rho_p_idx);
    
    ## transform coordinates to output convention: xy and (0,0) based
    peak_pixels_x = allpixels_x(peak_pixels_idx) + origin(1);
    peak_pixels_y = allpixels_y(peak_pixels_idx) + origin(2);
       
    if (length (peak_pixels_x) == 0)
      continue     # avoid special cases for empty peak_pixel vectors
    endif
    
    ## order those image pixels, the "faster" axis first:
    ## (to avoid excessive index jumps in "wide" lines)
    x_span = max (peak_pixels_x) - min (peak_pixels_x);
    y_span =  max (peak_pixels_y) - min (peak_pixels_y);
    if (x_span > y_span)
      peak_pixels_yx = sortrows ([peak_pixels_y, peak_pixels_x], [1 2]);
    else
      peak_pixels_yx = sortrows ([peak_pixels_y, peak_pixels_x], [2 1]);
    endif
    peak_pixels = [peak_pixels_yx(:,2), peak_pixels_yx(:,1)]; # weired re-ordering needed for compatibility
    
    ## calculate the euclidean distance between adjacent (ordered) pixels:
     dist = sqrt (diff (peak_pixels(:,1)).^2 + diff (peak_pixels(:,2)).^2); 
    
    ## split line into segments, which are separated by more than fillgap:
    ## (always use very first and very last pixel in peak_pixels)
    endpoint_idx = find (dist > fillgap);
    num_peak_pixels = size (peak_pixels, 1);
    endpoint_idx = [0; endpoint_idx; num_peak_pixels];
    for m = 2 : length (endpoint_idx)
      first_pixel = peak_pixels(endpoint_idx(m-1)+1, :);  # point after last endpoint
      last_pixel = peak_pixels(endpoint_idx(m), :);          # this endpoint
      length_segment = sqrt (sum((last_pixel - first_pixel).^2));
      
      ## save this segment if it is long enough:
      if (length_segment < minlength)
        continue; 
      else
        numlines += 1;
        lines(numlines).point1 = first_pixel;
        lines(numlines).point2 = last_pixel;
        lines(numlines).theta = theta_p;
        lines(numlines).rho = rho_p;
      endif
      
    endfor # line segments
  endfor # peaks

endfunction

%!shared BW0, theta0, rho0, peaks0_1, peaks0_2, lines0_1, lines0_2, BW1, theta1, rho1, peaks1, lines1
%! BW0 = logical([0 0 0 0 1; 0 0 0 1 0; 1 0 1 0 0; 0 1 0 0 0; 1 1 1 1 1]);
%! theta0 = [-90:89];
%! rho0 = [-7:7];
%! peaks0_1 = [11 130];
%! peaks0_2 = [11 130; 4 1];
%! lines0_1 = struct ("point1", {[1,5]}, "point2", {[5,1]}, "theta", {39}, "rho", {3});
%! lines0_2 = struct ("point1", {[1,5], [1,5]}, "point2", {[5,1],[5,5]}, "theta", {39,-90}, "rho", {3, -4});
%! BW1 = diag(ones(50,1));
%! theta1 = [-90:89];
%! rho1 = -70:70;
%! peaks1 = [71 46];
%! lines1 = struct ("point1", {[1 1]}, "point2", {[50 50]}, "theta", {-45}, "rho", {0});

## test input syntax:
%!error houghlines ()
%!error houghlines (BW1)
%!error houghlines (BW1, theta1)
%!error houghlines (BW1, theta1, rho1)
%!assert (houghlines (BW1, theta1, rho1, peaks1), lines1)
%!error (houghlines (BW1, theta1, rho1, peaks1, [1 2 3]))
%!assert (houghlines (BW1, theta1, rho1, peaks1, "FillGap", 5), lines1)
%!assert (houghlines (BW1, theta1, rho1, peaks1, "MinLength", 2), lines1)
%!assert (houghlines (BW1, theta1, rho1, peaks1, "FillGap", 5, "MinLength", 2), lines1)
%!assert (houghlines (BW1, theta1, rho1, peaks1, "MinLength", 2, "FillGap", 5), lines1)
%!error houghlines (BW1, theta1, rho1, peaks1, "MinLength", 2, [1 2 3])
%!error houghlines (BW1, theta1, rho1, peaks1, "MinLength", 2, "FillGap", 5, [1 2 3])
%!assert (houghlines (double (BW1), theta1, rho1, peaks1), lines1)
%!error houghlines (ones(5, 5, 5), theta1, rho1, peaks1)
%!error houghlines ("nonsense", theta1, rho1, peaks1)
%!error houghlines (BW1, ones(5), rho1, peaks1)
%!error houghlines (BW1, "nonsense", rho1, peaks1)
%!error houghlines (BW1, theta1, ones(5), peaks1)
%!error houghlines (BW1, theta1, "nonsense", peaks1)
%!error houghlines (BW1, theta1, rho1, ones(5))
%!error houghlines (BW1, theta1, rho1, ones(2,2,2))
%!error houghlines (BW1, theta1, rho1, "nonsense")
%!error houghlines (BW1, theta1, rho1, peaks1, "nonsense", 5)
%!error houghlines (BW1, theta1, rho1, peaks1, "MinLength", -5)
%!error houghlines (BW1, theta1, rho1, peaks1, "MinLength", [3 4])
%!error houghlines (BW1, theta1, rho1, peaks1, "MinLength", "nonsense")
%!error houghlines (BW1, theta1, rho1, peaks1, "FillGap", -5)
%!error houghlines (BW1, theta1, rho1, peaks1, "FillGap", [3 4])
%!error houghlines (BW1, theta1, rho1, peaks1, "FillGap", "nonsense")

## output class and structure:
%!test
%! out =  houghlines(BW0, theta0, rho0, peaks0_2, "MinLength", 1);
%! assert (out, lines0_2) # includes class = struct, size = [1,2]

%!test   # for empty output
%! n = 100;
%! BW = false (n);
%! a = 50;    % line starts at left side at row a
%! b = 3;      % slope of line is 1:b
%! for column = 1:n
%!   if (rem (column, b) == 0)
%!     row = a - column/b;
%!     BW(row, column) = true;
%!     BW(row, column+1) = true;
%!   end
%! end
%! theta = [-90: 89];
%! rho = [-141:141];
%! peaks = [188, 163];
%! out = houghlines(BW, theta, rho, peaks, 'FillGap', 1, 'MinLength', 5);
%! assert (out, struct([]))

## test calculation results:
%!test
%! out0_1 = houghlines(BW0, theta0, rho0, peaks0_1, 'MinLength', 1);
%! out0_2 = houghlines(BW0, theta0, rho0, peaks0_2, 'MinLength', 1);
%! assert (out0_1, lines0_1);
%! assert (out0_2, lines0_2);

%!test
%! out = houghlines(BW1, theta1, rho1, peaks1);
%! assert (out, lines1);

%!test
%! n = 100;
%! BW = false (n);
%! a = 50;    % line starts at left side at row a
%! b = 3;      % slope of line is 1:b
%! for column = 1:n
%!   if (rem (column, b) == 0)
%!     row = a - column/b;
%!     BW(row, column) = true;
%!     BW(row, column+1) = true;
%!   end
%! end
%! theta = [-90:89];
%! rho = [-141:141];
%! peaks = [188, 163];
%! lines_1 = struct ("point1", {[99 17]}, "point2", {[3 49]}, "theta", {72}, "rho", {46});
%! out_1 = houghlines(BW, theta, rho, peaks);
%! out_n = houghlines(BW, theta, rho, peaks, 'FillGap', 1, 'MinLength', 1);
%! assert  (out_1, lines_1)
%! assert (size (out_n), [1, 29])

## show instructive demo:
%!demo
%! I = checkerboard (30, 1, 1);
%! I = imnoise(I, "salt & pepper", 0.2);
%! figure, imshow (I); 
%! title ("noisy image with some lines");
%! BW = edge (I, "canny");
%! figure, imshow(BW);
%! title ("edge image");
%! [H, theta, rho] = hough (BW);
%! figure, imshow (mat2gray (H), [], "XData", theta, "YData", rho);
%! title ("Hough transform of edge image \n 2 peaks marked");
%! axis on; xlabel("theta [degrees]"); ylabel("rho [pixels]");
%! peaks = houghpeaks (H, 2);
%! peaks_rho = rho(peaks(:,1));
%! peaks_theta = theta(peaks(:,2));
%! hold on; plot (peaks_theta, peaks_rho, "sr"); hold off;
%! lines = houghlines (BW, theta, rho, peaks);
%! figure, imshow (I), hold on;
%! for n = 1:length (lines)
%!    points = [lines(n).point1; lines(n).point2];
%!    plot (points(:,1), points(:,2), "r");
%! endfor
%! title ("the two strongest lines (edges) in the image"), hold off;
