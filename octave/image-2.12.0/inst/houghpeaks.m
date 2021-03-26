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
## @deftypefn   {Function File} {@var{peaks} =} @ houghpeaks (@var{H})
## @deftypefnx   {Function File} {@var{peaks} =} @ houghpeaks (@var{H}, @var{numpeaks})
## @deftypefnx  {Function File} {@var{peaks} =} @ houghpeaks (@var{H}, @dots{}, @var{property}, @var{value}, @dots{})
## Find peaks in a Hough transform.
##
## The houghpeaks function finds positions of maximum value ("peaks")
## in the 2d matrix @var{H} of a Hough (line) transform.
## This hough transform @var{H} will normally come from the 
## function hough. Its rows are distance coordinates (rho) of lines in an image,
## and its columns are angle coordinates (theta).
##
## The output @var{peaks} of the houghpeaks functions is a n-by-2 matrix where 
## each line is an x (rho) and y (theta) coordinate of a found peak, sorted
## in descending order of the corresponding peak value. 
##
## The number of returned peak coordinates can be limited with
## the @var{numpeaks} parameter, it defaults to 1.
##
## Additionally the following optional property-value-pairs can be used:
## @table @asis
## @item @var{Threshold}
## Maximum positions in @var{H} with a value below this threshold 
## will not counted as peaks.
## This defaults to 50% of the maximum value in @var{H}.
##
## @item @var{NHoodSize}
## After finding one peak, a neighborhood of this size will
## be excluded from the search for further peaks.
## This parameter must be given as a two-element row vector. 
## The first entry is the full width of the
## x (rho) neighborhood, the second entry of y (theta). 
## Both numbers need to be odd integers.
## This defaults to the smallest odd number equal or bigger than size(H)/50.
## @end table
## 
## @seealso{hough}
## @end deftypefn

## The algorithm is taken from the book 
## "Digital Image Processing using MATLAB"
## by R. C. Gonzalez, R. E. Woods and S. L. Eddins, 
## McGrawHill, 2nd edition 2010.

function peaks = houghpeaks (H, varargin)

  ## retrieve the input parameters:
  numpeaks = [];
  threshold = [];
  nhoodsize = [];  
  
  if ((nargin < 1) || (nargin > 6))
    print_usage ();
  endif
  
  if (nargin/2 == round(nargin/2))  # even number of inputs (2, 4 or 6)
   ## houghpeaks (H, numpeaks)
   ## houghpeaks (H, numpeaks, property1, value1)
   ## houghpeaks (H, numpeaks, property1, value1, property2, value2)
    numpeaks = varargin{1};
    n_start = 2;
  else                                               # odd number of inputs (1, 3 or 5)
    ## houghpeaks (H)
    ## houghpeaks (H, property1, value1)
    ## houghpeaks (H, property1, value1, property2, value2)
    n_start = 1;
  endif
    
  for n = n_start:2:(nargin-1)         # process parameter-values pairs
    if (strcmpi (varargin{n}, "threshold"))
      threshold = varargin{n+1};
    elseif (strcmpi (varargin{n}, "nhoodsize"))
      nhoodsize = varargin{n+1};
    else
      error ("houghpeaks: invalid PROPERTY given")
    endif
  endfor
  
  ## set default parameters:
  if (isempty (numpeaks))
    numpeaks = 1;
  endif
  
  if (isempty (threshold))
    threshold = 0.5 * max (H(:));
  endif
  
  if (isempty (nhoodsize))
    nhoodsize = size (H)/50;
    nhoodsize += 1;                                       # for Matlab compatibilty (against their documentation)
    nhoodsize = 2*ceil ((nhoodsize-1)/2)+1; # odd number (equal or bigger) as in Matlab documentation
    nhoodsize = max (nhoodsize, 3);             # for (undocumented) Matlab compatibility
  endif
  
  ## check input parameters:
  if (! isimage (H) || ndims(H) != 2)
    error ("houghpeaks: H must be a numeric 2d array");
  endif
  
  if ((! isscalar (numpeaks)) || (numpeaks <= 0) ||
       (numpeaks != round(numpeaks) ))
    error ("houghpeaks: NUMPEAKS must be a positive integer scalar.")
  endif
   
  if ((! isscalar (threshold) ) || (! isnumeric (threshold)) ||
        (threshold < 0))
    error ("houghpeaks: THRESHOLD must be a non-negative numeric scalar.")
  endif
    
  if ( (ndims (nhoodsize) != 2) || (any (size (nhoodsize) != [1 2])) || 
        (! isnumeric (nhoodsize)) || (any (nhoodsize <= 0)) ||
         (any (round ((nhoodsize-1)/2) * 2 + 1 != nhoodsize) ) )
    error ("houghpeaks: NHOODSIZE must be a 2-element vector of positive odd integers")
  endif
      
  ## do the calculation
  ##
  ## The algorithm is taken from the above cited book,
  ## chapter 10.2.2 "Toolbox Hough Functions",
  ## section "Function houghpeaks". It says
  ## "The basic idea behind this procedure is to 
  ## clean-up the peaks by setting to zero the Hough
  ## transform cells in the immediate neighborhood
  ## in which a peak was found."
  ##
  ## properties of the Hough transform data H:
  ## * rows (x) of H are distance (rho)
  ##   and columns (y) of H are angle (theta)
  ## * H is anti-symmetric in theta-direction
  nhood = (nhoodsize-1)/2;
  nhoodx = nhood(1);
  nhoody = nhood(2);
  sizex = size (H, 1);
  sizey = size (H, 2);
  
  peaks = [];
  for n = 1:numpeaks
    ## find the next peak
    [maxval, maxind] = max (H(:));
    [x0, y0] = ind2sub (size (H), maxind);
    
    ## if peak value is too low, stop the search
    if (maxval < threshold)
      break;
    endif
    peaks(n,:) = [x0, y0];
    
    ## limit the size of the deleted neighborhood to H
    xmin = max (x0 - nhoodx, 1);
    xmax = min (x0 + nhoodx, sizex);
    ymin = max (y0 - nhoody, 1);
    ymax = min (y0 + nhoody, sizey);
    H(xmin:xmax, ymin:ymax) = 0;
    
    ## use anti-symmetry in theta direction
    ## to also delete points on "the other side"
    if ((y0 + nhoody > sizey) || (y0 - nhoody < 1))
      xmin2 = sizex - xmax + 1;
      xmax2 = sizex - xmin + 1;
      if (y0 + nhoody > sizey)
        ymin2 = 1;
        ymax2 = y0 + nhoody - sizey;
      else   # (y0 - nhoody < 1)
        ymin2 = y0 - nhoody + sizey;
        ymax2 = sizey;
      endif
      H(xmin2:xmax2, ymin2:ymax2) = 0;
    endif
    
  endfor

endfunction

%!shared im1
%! im1 = magic (5);

## test input syntax:
%!error houghpeaks ()
%!error houghpeaks (1, 2, 3, 4, 5, 6, 7)
%!assert (houghpeaks (im1))
%!assert (houghpeaks (im1, 2))
%!assert (houghpeaks (im1, "Threshold", 10))
%!assert (houghpeaks (im1, 2, "Threshold", 10))
%!assert (houghpeaks (im1, "NHoodSize", [3 3]))
%!assert (houghpeaks (im1, 2, "NHoodSize", [3 3]))
%!assert (houghpeaks (im1, "Threshold", 10, "NHoodSize", [3 3]))
%!assert (houghpeaks (im1, "NHoodSize", [3 3], "Threshold", 10))
%!assert (houghpeaks (im1, 2, "Threshold", 10, "NHoodSize", [3 3]))
%!assert (houghpeaks (im1, 2, "NHoodSize", [3 3], "Threshold", 10))
%!error houghpeaks (ones (5, 5, 5))
%!error houghpeaks ("hello")
%!error houghpeaks (im1, 1.5)
%!error houghpeaks (im1, -2)
%!error houghpeaks (im1, [1 1])
%!error houghpeaks (im1, "Threshold", "hello")
%!error houghpeaks (im1, "Threshold", -2)
%!error houghpeaks (im1, "Threshold", [1 1])
%!error houghpeaks (im1, "NHoodSize", [3 3 3])
%!error houghpeaks (im1, "NHoodSize", "hello")
%!error houghpeaks (im1, "NHoodSize", [-3 -3])
%!error houghpeaks (im1, "NHoodSize", [4 4])

## test dimensions and classes:
%!test
%! out = houghpeaks (im1);
%! assert (size (out), [1 2])
%! assert (class (out), "double")

%!test
%! out = houghpeaks (im1, 3);
%! assert (size (out), [3 2])

## test calculation results:
%!test      
%! expected = [5 3; 1 2; 3 5; 1 5];
%! assert (houghpeaks (im1, 4), expected) # this checks for undocumented nhood >=3
%! assert (houghpeaks (im1, 4, "nhoodsize", [3,3]), expected)
%! assert (houghpeaks (im1, 4, "threshold", 10), expected)
%! assert (houghpeaks (im1, 4, "threshold", 24), expected(1:2,:))

%!test
%! im2 = magic (7);
%! expected_a = [7 4; 1 3; 3 1; 5 6];
%! expected_b = [7 4; 1 3; 4 7; 1 7];
%! assert (houghpeaks (im2, 4), expected_a)
%! assert (houghpeaks (im2, 4, "nhoodsize", [5,5]), expected_b)
%! assert (houghpeaks (im2, 4, "threshold", 24), expected_a)
%! assert (houghpeaks (im2, 4, "threshold", 47), expected_a(1:2,:))

%!test
%! im3 = magic (99);
%! expected_a = [99 50; 1 49; 3 47; 5 45; 7 43; 9 41; 11 39];
%! expected_b = [99 50; 1 49; 7 43; 13 37; 19 31; 25 25; 31 19];
%! expected_c = [99 50; 1 49; 2 48; 3 47; 4 46; 5 45; 6 44];
%! assert (houghpeaks (im3, 7), expected_a)
%! assert (houghpeaks (im3, 7, "nhoodsize", [11 11]), expected_b)
%! assert (houghpeaks (im3, 7, "nhoodsize", [11 1]), expected_c)
%! assert (houghpeaks (im3, 7, "nhoodsize", [11 1]), expected_c)

%!test
%! im4 = double (im2uint16 (peaks ()));
%! expected_a = [37 15; 39 15; 41 15; 15 16; 17 16];
%! expected_b = [37 15; 15 16; 26 21; 37 26; 20 32];
%! expected_c = [37 15; 15 16; 35 16; 15 17; 35 17];
%! expected_d = [37 15; 38 15; 39 15; 40 15; 41 15];
%! assert (houghpeaks (im4, 5), expected_a)
%! assert (houghpeaks (im4, 5, "nhoodsize", [21 21]), expected_b)
%! assert (houghpeaks (im4, 5, "nhoodsize", [21 1]), expected_c)
%! assert (houghpeaks (im4, 5, "nhoodsize", [1 21]), expected_d)

%!test                   # tests use of anti-symmetry in H
%! im5 = zeros (6,4); im5(2,1) = 1; im5(5,4) = 2;
%! expected = [5 4; 2 1];
%! assert (houghpeaks (im5, 2, "nhoodsize", [1 1]), expected);
%! assert (houghpeaks (im5, 2, "nhoodsize", [3 3]), expected(1,:));

%!test          #test use of anti-symmetry in the other direction
%! im6 = magic (100);
%! expected_a = [1 1; 100 99; 1 4; 100 95; 1 8; 100 91; 1 12];
%! expected_b = [1 1; 100 95; 1 8; 100 87; 1 16; 100 79; 1 24];
%! expected_c = [1 1; 100 99; 100 98; 1 4; 1 5; 100 95; 100 94];
%! expected_d = expected_b;
%! assert (houghpeaks (im6, 7), expected_a)
%! assert (houghpeaks (im6, 7, "nhoodsize", [11 11]), expected_b)
%! assert (houghpeaks (im6, 7, "nhoodsize", [11 1]), expected_c)
%! assert (houghpeaks (im6, 7, "nhoodsize", [1 11]), expected_d)

%!test          # test undocumented Matlab default value for nhoodsize
%! im = zeros (723, 180);
%! im(585,136) = 8;
%! im(593,135) = 7;
%! im(310,46) = 6;
%! expected = [585, 136; 310, 46];
%! assert (houghpeaks (im, 2), expected)

%!test
%! I = max (0, phantom ());
%! H = hough (I);
%! P0 = [585, 136; 310, 46; 595, 136; 522, 104; 373, 46];
%! assert (houghpeaks (H, 5), P0)

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
%! figure, imshow (mat2gray (H), [],"XData",theta,"YData",rho);
%! title ("Hough transform of edge image \n 2 peaks marked");
%! axis on; xlabel("theta [degrees]"); ylabel("rho [pixels]");
%! peaks = houghpeaks (H, 2);
%! peaks_rho = rho(peaks(:,1))
%! peaks_theta = theta(peaks(:,2))
%! hold on;
%! plot(peaks_theta,peaks_rho,"sr");
%! hold off;
