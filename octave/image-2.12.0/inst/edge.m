## Copyright (C) 1999 Andy Adler <adler@sce.carleton.ca>
## Copyright (C) 2008 Søren Hauberg <soren@hauberg.org>
## Copyright (C) 2015 Hartmut Gimpel <hg_code@gmx.de>
## Copyright (C) 2015 Carnë Draug <carandraug@octave.org>
##
## This program is free software; you can redistribute it and/or
## modify it under the terms of the GNU General Public License as
## published by the Free Software Foundation; either version 3 of the
## License, or (at your option) any later version.
##
## This program is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
## General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program; if not, see
## <http:##www.gnu.org/licenses/>.

## -*- texinfo -*-
## @deftypefn {Function File} {[@var{bw}, @var{thresh}] =} edge (@var{im}, @var{method}, @dots{})
## Find edges using various methods.
##
## The image @var{im} must be 2 dimensional and grayscale.  The @var{method}
## must be a string with the string name.  The other input arguments are
## dependent on @var{method}.
##
## @var{bw} is a binary image with the identified edges.  @var{thresh} is
## the threshold value used to identify those edges.  Note that @var{thresh}
## is used on a filtered image and not directly on @var{im}.
##
## @seealso{fspecial}
##
## @end deftypefn
## @deftypefn  {Function File} {} edge (@var{im}, @qcode{"Canny"})
## @deftypefnx {Function File} {} edge (@var{im}, @qcode{"Canny"}, @var{thresh})
## @deftypefnx {Function File} {} edge (@var{im}, @qcode{"Canny"}, @var{thresh}, @var{sigma})
## Find edges using the Canny method.
##
## @var{thresh} is two element vector for the hysteresis thresholding.
## The lower and higher threshold values are the first and second elements
## respectively.  If it is a scalar value, the lower value is set to
## @code{0.4 * @var{thresh}}.
##
## @var{sigma} is the standard deviation to be used on the Gaussian filter
## that is used to smooth the input image prior to estimating gradients.
## Defaults to @code{sqrt (2)}.
##
## @end deftypefn
## @deftypefn  {Function File} {} edge (@var{im}, @qcode{"Kirsch"})
## @deftypefnx {Function File} {} edge (@var{im}, @qcode{"Kirsch"}, @var{thresh})
## @deftypefnx {Function File} {} edge (@var{im}, @qcode{"Kirsch"}, @var{thresh}, @var{direction})
## @deftypefnx {Function File} {} edge (@var{im}, @qcode{"Kirsch"}, @var{thresh}, @var{direction}, @var{thinning})
## Find edges using the Kirsch approximation to the derivatives.
##
## Edge points are defined as points where the length of the gradient exceeds
## a threshold @var{thresh}.
##
## @var{thresh} is the threshold used and defaults to twice the square root of the
## mean of the gradient squared of @var{im}.
##
## @var{direction} is the direction of which the gradient is
## approximated and can be @qcode{"vertical"}, @qcode{"horizontal"},
## or @qcode{"both"} (default).
##
## @var{thinning} can be the string @qcode{"thinning"} (default) or
## @qcode{"nothinning"}.  This controls if a simple thinning procedure
## is applied to the edge image such that edge points also need to have
## a larger gradient than their neighbours. The resulting "thinned"
## edges are only one pixel wide.
##
## @end deftypefn
## @deftypefn  {Function File} {} edge (@var{im}, @qcode{"Lindeberg"})
## @deftypefnx {Function File} {} edge (@var{im}, @qcode{"Lindeberg"}, @var{sigma})
## Find edges using the the differential geometric single-scale edge
## detector by Tony Lindeberg.
##
## @var{sigma} is the scale (spread of Gaussian filter) at which the edges
## are computed. Defaults to @code{2}.
##
## This method does not use a threshold value and therefore does not return
## one.
##
## @end deftypefn
## @deftypefn  {Function File} {} edge (@var{im}, @qcode{"LoG"})
## @deftypefnx {Function File} {} edge (@var{im}, @qcode{"LoG"}, @var{thresh}, @var{sigma})
## Find edges by convolving with the Laplacian of Gaussian (LoG) filter,
## and finding zero crossings.
##
## Only zero crossings where the filter response is larger than @var{thresh}
## are retained.  @var{thresh} is automatically computed as 0.75*@math{M},
## where @math{M} is the mean of absolute value of LoG filter response.
##
## @var{sigma} sets the spread of the LoG filter. Default to @code{2}.
##
## @end deftypefn
## @deftypefn  {Function File} {} edge (@var{im}, @qcode{"Prewitt"}, @dots{})
## Find edges using the Prewitt approximation to the derivatives.
##
## This method is the same as Kirsch except a different approximation
## gradient is used.
##
## @end deftypefn
## @deftypefn  {Function File} {} edge (@var{im}, @qcode{"Roberts"})
## @deftypefnx {Function File} {} edge (@var{im}, @qcode{"Roberts"}, @var{thresh})
## @deftypefnx {Function File} {} edge (@var{im}, @qcode{"Roberts"}, @var{thresh}, @var{thinning})
## Find edges using the Roberts approximation to the derivatives.
##
## This method is similar to Kirsch except a different approximation
## gradient is used, and the default @var{thresh} is multiplied by sqrt(1.5).
## In addition, there it does not accept the @var{direction} argument.
##
## @end deftypefn
## @deftypefn {Function File} {} edge (@var{im}, @qcode{"Sobel"}, @dots{})
## Find edges using the Sobel approximation to the derivatives.
##
## This method is the same as Kirsch except a different approximation
## gradient is used.
##
## @end deftypefn
## @deftypefn {Function File} {} edge (@var{im}, @qcode{"zerocross"}, @var{thresh}, @var{filter})
## Find edges by convolving with a user-supplied filter and finding zero
## crossings.
##
## @end deftypefn
## @deftypefn {Function File} {} edge (@var{im}, @qcode{"Andy"}, @var{thresh}, @var{params})
## Find edges by the original (Andy Adlers) @code{edge} algorithm.
## The steps are
## @enumerate
## @item
## Do a Sobel edge detection and to generate an image at
## a high and low threshold.
## @item
## Edge extend all edges in the LT image by several pixels,
## in the vertical, horizontal, and 45 degree directions.
## Combine these into edge extended (EE) image.
## @item
## Dilate the EE image by 1 step.
## @item
## Select all EE features that are connected to features in
## the HT image.
## @end enumerate
## 
## The parameters for the method is given in a vector:
## @table @asis
## @item params(1)==0 or 4 or 8
## Perform x connected dilatation (step 3).
## @item params(2)
## Dilatation coefficient (threshold) in step 3.
## @item params(3)
## Length of edge extention convolution (step 2).
## @item params(4)
## Coefficient of extention convolution in step 2.
## @end table
## defaults = [8, 1, 3, 3]
##
## @end deftypefn

function [bw_out, thresh, varargout] = edge (im, method = "sobel", varargin)

  if (nargin < 1 || nargin > 5)
    print_usage ();
  endif

  if (! isgray (im))
    error ("edge: IM must be a grayscale image");
  endif

  method = tolower (method);
  switch (method)
    case {"sobel", "prewitt", "kirsch"}
      [bw, thresh] = edge_kirsch_prewitt_sobel (im, method, varargin{:});
    case "roberts"
      [bw, thresh, varargout{1:2}] = edge_roberts (im, varargin{:});
    case "log"
      [bw, thresh] = edge_log (im, varargin{:});
    case "zerocross"
      [bw, thresh] = edge_zerocross (im, varargin{:});
    case "canny"
      [bw, thresh] = edge_canny (im, varargin{:});
    case "lindeberg"
      [bw] = edge_lindeberg (im, varargin{:});
    case "andy"
      [bw, thresh] = edge_andy (im, varargin{:});
    otherwise
      error ("edge: unsupported edge detector `%s'", method);
  endswitch

  if (nargout == 0)
    imshow (bw);
  else
    bw_out = bw;
  endif
endfunction

function [bw, thresh] = edge_kirsch_prewitt_sobel (im, method, varargin)
  if (numel (varargin) > 3)
    error ("edge: %s method takes at most 3 extra arguments", method);
  endif

  ## This supports thinning, direction, and thresh arguments in any
  ## order.  Matlab madness I tell you.
  varargin = tolower (varargin);
  direction_mask = (strcmp (varargin, "both")
                    | strcmp (varargin, "horizontal")
                    | strcmp (varargin, "vertical"));
  thinning_mask = (strcmp (varargin, "thinning")
                   | strcmp (varargin, "nothinning"));
  thresh_mask = cellfun (@(x) isnumeric (x) && (isscalar (x) || isempty (x)),
                         varargin);

  if (! all (direction_mask | thinning_mask | thresh_mask))
    error ("edge: %s method takes only THRESH, DIRECTION, and THINNING arguments",
           method);
  endif

  if (nnz (direction_mask) > 1)
    error ("edge: more than 1 direction argument defined");
  elseif (any (direction_mask))
    direction = varargin{direction_mask};
  else
    direction = "both";
  endif

  if (nnz (thinning_mask) > 1)
    error ("edge: more than 1 thinning argument defined");
  elseif (any (thinning_mask)
          && strcmp (varargin{thinning_mask}, "nothinning"))
    thinning = false;
  else
    thinning = true;
  endif

  if (nnz (thresh_mask) > 1)
    error ("edge: more than 1 threshold argument defined");
  elseif (any (thresh_mask))
    thresh = varargin{thresh_mask};
  else
    thresh = [];
  endif

  ## For better speed, the calculation is done with
  ## squared edge strenght values (strength2) and
  ## squared threshold values (thresh2).

  h1 = fspecial (method);
  h1 ./= sum (abs (h1(:)));  # normalize h1
  im = im2double (im);
  switch (direction)
    case "horizontal"
      strength2 = imfilter (im, h1, "replicate").^2;
    case "vertical"
      strength2 = imfilter (im, h1', "replicate").^2;
    case "both"
      strength2 = (imfilter (im, h1, "replicate").^2
                  + imfilter (im, h1', "replicate").^2);
    otherwise
      error ("edge: unknown DIRECTION `%s' for %s method", direction, method);
  endswitch

  if (isempty (thresh))
    thresh2 = 4 * mean (strength2(:));
    thresh = sqrt (thresh2);
  else
    thresh2 = thresh .^ 2;
  endif

  if (thinning)
    ## Keep edge strengths for use in non-maximum
    ## suppresion in the simple_thinning step.
    strength2(strength2<=thresh2) = 0;
    bw = simple_thinning (strength2);
  else
    bw = strength2 > thresh2;
  endif
endfunction

function [bw, thresh, g45, g135] = edge_roberts (im, varargin)
  if (numel (varargin) > 2)
    error ("edge: Roberts method takes at most 2 extra arguments");
  endif

  ## This supports thinning, direction, and thresh arguments in any
  ## order.  Matlab madness I tell you.
  varargin = tolower (varargin);
  thinning_mask = (strcmp (varargin, "thinning")
                   | strcmp (varargin, "nothinning"));
  thresh_mask = cellfun (@(x) isnumeric (x) && (isscalar (x) || isempty (x)),
                         varargin);

  if (! all (thinning_mask | thresh_mask))
    error ("edge: roberts method takes only THRESH and THINNING arguments");
  endif

  if (nnz (thinning_mask) > 1)
    error ("edge: more than 1 thinning argument defined");
  elseif (any (thinning_mask)
          && strcmp (varargin{thinning_mask}, "nothinning"))
    thinning = false;
  else
    thinning = true;
  endif

  if (nnz (thresh_mask) > 1)
    error ("edge: more than 1 threshold argument defined");
  elseif (any (thresh_mask))
    thresh = varargin{thresh_mask};
  else
    thresh = [];
  endif

  h1 = [1 0; 0 -1] ./ 2;
  h2 = [0 1; -1 0] ./ 2;
  g45 = imfilter (im, h1, "replicate");
  g135 = imfilter (im, h2, "replicate");
  strength2 = g45.^2 + g135.^2;

  if (isempty (thresh))
    thresh2 = 6 * mean (strength2(:));
    thresh = sqrt (thresh2);
  else
    thresh2 = thresh .^ 2;
  endif

  if (thinning)
    ## Keep edge strengths for use in non-maximum
    ## suppresion in the simple_thinning step.
    strength2(strength2<=thresh2) = 0;
    bw = simple_thinning (strength2);
  else
    bw = strength2 > thresh2;
  endif
endfunction

function [bw, thresh] = edge_log (im, thresh = [], sigma = 2)
  if (nargin > 1 && (! isnumeric (thresh) || all (numel (thresh) != [0 1])))
    error ("edge: THRESH for LoG method must be a numeric scalar or empty");
  endif
  if (nargin > 2 && (! isnumeric (sigma) || ! isscalar (sigma)))
    error ("edge: SIGMA for LoG method must be a numeric scalar");
  endif

  f = fspecial ("log", (2 * ceil (3*sigma)) +1, sigma);
  g = conv2 (im, f, "same");

  if (isempty (thresh))
    thresh = 0.75*mean(abs(g(:)));
  endif

  bw = (abs (g) > thresh) & zerocrossings (g);
endfunction

function [bw, thresh] = edge_zerocross (im, thresh, f)
  ## because filter is a required argument, so is thresh
  if (nargin != 3)
    error ("edge: a FILTER and THRESH are required for the zerocross method");
  elseif (! isnumeric (thresh) || all (numel (thresh) != [0 1]))
    error ("edge: THRESH for zerocross method must be a numeric scalar or empty");
  elseif (! isnumeric (f))
    error ("edge: FILTER for zerocross method must be numeric");
  endif

  g = conv2 (im, f, "same");
  if (isempty (thresh))
    thresh = mean (abs (g(:)));
  endif
  bw = (abs (g) > thresh) & zerocrossings(g);
endfunction

function [bw, thresh] = edge_canny (im, thresh = [], sigma = sqrt (2))
  if (nargin > 1 && (! isnumeric (thresh) || all (numel (thresh) != [0 1 2])))
    error ("edge: THRESH for Canny method must have 0, 1, or 2 elements");
  endif
  if (nargin > 2 && (! isnumeric (sigma) || ! isscalar (sigma)))
    error ("edge: SIGMA for Canny method must be a numeric scalar");
  endif

  ## Gaussian filtering to change the edge scale.
  ## Treat each dimensions separately for performance.
  gauss = fspecial ("gaussian", [1 (8*ceil(sigma))], sigma);
  im = im2double (im);
  J = imfilter (im, gauss, "replicate");
  J = imfilter (J, gauss', "replicate");

  ## edge detection with Prewitt filter (treat dimensions separately)
  p = [1 0 -1]/2;
  Jx = imfilter (J, p, "replicate");
  Jy = imfilter (J, p', "replicate");
  Es = sqrt (Jx.^2 + Jy.^2);
  Es_max = max (Es(:));
  if (Es_max > 0)
    Es ./= Es_max;
  endif
  Eo = pi - mod (atan2 (Jy, Jx) - pi, pi);

  if (isempty (thresh))
    tmp = mean(abs(Es(:)));
    thresh = [0.4*tmp, tmp];
  elseif (numel (thresh) == 1)
    thresh = [0.4*thresh thresh];
  else
    thresh = thresh(:).'; # always return a row vector
  endif
  bw = nonmax_suppress(Es, Eo, thresh(1), thresh(2));
endfunction

function [bw] = edge_lindeberg (im, sigma = 2)
  if (nargin > 1 && (! isnumeric (sigma) || ! isscalar (sigma)))
    error ("edge: SIGMA for Lindeberg method must be a numeric scalar");
  endif

  ## Filters for computing the derivatives
  Px   = [-1 0 1; -1 0 1; -1 0 1];
  Py   = [1 1 1; 0 0 0; -1 -1 -1];
  Pxx  = conv2 (Px,  Px, "full");
  Pyy  = conv2 (Py,  Py, "full");
  Pxy  = conv2 (Px,  Py, "full");
  Pxxx = conv2 (Pxx, Px, "full");
  Pyyy = conv2 (Pyy, Py, "full");
  Pxxy = conv2 (Pxx, Py, "full");
  Pxyy = conv2 (Pyy, Px, "full");
  ## Change scale
  L = imsmooth (double (im), "Gaussian", sigma);
  ## Compute derivatives
  Lx   = conv2 (L, Px,   "same");
  Ly   = conv2 (L, Py,   "same");
  Lxx  = conv2 (L, Pxx,  "same");
  Lyy  = conv2 (L, Pyy,  "same");
  Lxy  = conv2 (L, Pxy,  "same");
  Lxxx = conv2 (L, Pxxx, "same");
  Lyyy = conv2 (L, Pyyy, "same");
  Lxxy = conv2 (L, Pxxy, "same");
  Lxyy = conv2 (L, Pxyy, "same");
  ## Compute directional derivatives
  Lvv  = Lx.^2.*Lxx + 2.*Lx.*Ly.*Lxy + Ly.^2.*Lyy;
  Lvvv = Lx.^3.*Lxxx + 3.*Lx.^2.*Ly.*Lxxy ...
         + 3.*Lx.*Ly.^2.*Lxyy + 3.*Ly.^3.*Lyyy;
  ## Perform edge detection
  bw = zerocrossings (Lvv) & Lvvv < 0;
endfunction

## The 'andy' edge detector that was present in older versions of 'edge'.
function [imout, thresh] = edge_andy (im, thresh, param2)
   [n,m]= size(im);
   xx= 2:m-1;
   yy= 2:n-1;

   filt= [1 2 1;0 0 0; -1 -2 -1]/8;  tv= 2;
   imo= conv2(im, rot90(filt), 'same').^2 + conv2(im, filt, 'same').^2;
   if nargin<2 || thresh==[];
      thresh= sqrt( tv* mean(mean( imo(yy,xx) ))  );
   end
#     sum( imo(:)>thresh ) / prod(size(imo))
   dilate= [1 1 1;1 1 1;1 1 1]; tt= 1; sz=3; dt=3;
   if nargin>=3
      # 0 or 4 or 8 connected dilation
      if length(param2) > 0
         if      param2(1)==4 ; dilate= [0 1 0;1 1 1;0 1 0];
         elseif  param2(1)==0 ; dilate= 1;
         end
      end
      # dilation threshold
      if length(param2) > 2; tt= param2(2); end
      # edge extention length
      if length(param2) > 2; sz= param2(3); end
      # edge extention threshold
      if length(param2) > 3; dt= param2(4); end
      
   end
   fobliq= [0 0 0 0 1;0 0 0 .5 .5;0 0 0 1 0;0 0 .5 .5 0;0 0 1 0 0; 
                      0 .5 .5 0 0;0 1 0 0 0;.5 .5 0 0 0;1 0 0 0 0];
   fobliq= fobliq( 5-sz:5+sz, 3-ceil(sz/2):3+ceil(sz/2) );

   xpeak= imo(yy,xx-1) <= imo(yy,xx) & imo(yy,xx) > imo(yy,xx+1) ;
   ypeak= imo(yy-1,xx) <= imo(yy,xx) & imo(yy,xx) > imo(yy+1,xx) ;

   imht= ( imo >= thresh^2 * 2); # high threshold image   
   imht(yy,xx)= imht(yy,xx) & ( xpeak | ypeak );
   imht([1,n],:)=0; imht(:,[1,m])=0;

%  imlt= ( imo >= thresh^2 / 2); # low threshold image   
   imlt= ( imo >= thresh^2 / 1); # low threshold image   
   imlt(yy,xx)= imlt(yy,xx) & ( xpeak | ypeak );
   imlt([1,n],:)=0; imlt(:,[1,m])=0;

# now we edge extend the low thresh image in 4 directions

   imee= ( conv2( imlt, ones(2*sz+1,1)    , 'same') > tt ) | ...
         ( conv2( imlt, ones(1,2*sz+1)    , 'same') > tt ) | ...
         ( conv2( imlt, eye(2*sz+1)       , 'same') > tt ) | ...
         ( conv2( imlt, rot90(eye(2*sz+1)), 'same') > tt ) | ...
         ( conv2( imlt, fobliq            , 'same') > tt ) | ...
         ( conv2( imlt, fobliq'           , 'same') > tt ) | ...
         ( conv2( imlt, rot90(fobliq)     , 'same') > tt ) | ...
         ( conv2( imlt, flipud(fobliq)    , 'same') > tt );
#  imee(yy,xx)= conv2(imee(yy,xx),ones(3),'same') & ( xpeak | ypeak );
   imee= conv2(imee,dilate,'same') > dt; #

%  ff= find( imht==1 );
%  imout = bwselect( imee, rem(ff-1, n)+1, ceil(ff/n), 8);  
   imout = imee;
endfunction

## An auxiliary function that performs a very simple thinning.
## Strength is an image containing the edge strength.
## bw contains a 1 in (r,c) if
##  1) strength(r,c) is greater than both neighbours in the
##     vertical direction, OR
##  2) strength(r,c) is greater than both neighbours in the
##     horizontal direction.
## Note the use of OR.
function bw = simple_thinning(strength)
  [r c] = size(strength);
  x = ( strength > [ zeros(r,1) strength(:,1:end-1) ] & ...
        strength > [ strength(:,2:end) zeros(r,1) ] );
  y = ( strength > [ zeros(1,c); strength(1:end-1,:) ] & ...
        strength > [ strength(2:end,:); zeros(1,c) ] );
  bw = x | y;
endfunction

## Auxiliary function. Finds the zero crossings of the 
## 2-dimensional function f. (By Etienne Grossmann)
function z = zerocrossings(f)
  z0 = f<0;                 ## Negative
  [R,C] = size(f);
  z = zeros(R,C);
  z(1:R-1,:) |= z0(2:R,:);  ## Grow
  z(2:R,:) |= z0(1:R-1,:);
  z(:,1:C-1) |= z0(:,2:C);
  z(:,2:C) |= z0(:,1:C-1);

  z &= !z0;                  ## "Positive zero-crossings"?
endfunction


## Test the madness of arbitrary order of input for prewitt, kirsch,
## and sobel methods.
%!test
%! im = [
%!   249   238   214   157   106    69    60    90   131   181   224   247   252   250   250
%!   250   242   221   165   112    73    62    91   133   183   225   248   252   250   251
%!   252   246   228   173   120    78    63    90   130   181   224   248   253   251   251
%!   253   248   232   185   132    87    62    80   116   170   217   244   253   251   252
%!   253   249   236   198   149   101    66    71   101   155   206   238   252   252   252
%!   254   250   240   210   164   115    73    69    92   143   196   232   252   253   252
%!    70    70    68    61    49    36    24    22    26    38    52    63    70    70    70
%!     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
%!    62    63    62    59    51    42    33    25    22    26    36    45    56    60    62
%!   252   253   252   246   221   190   157   114    90    90   118   157   203   235   248
%!   251   253   254   251   233   209   182   136   103    92   107   139   185   225   245
%!   251   253   254   253   243   227   206   163   128   108   110   133   175   217   242
%!   252   253   254   254   249   241   228   195   164   137   127   139   172   212   239
%! ] / 255;
%!
%! methods = {"kirsch", "prewitt", "sobel"};
%! for m_i = 1:numel (methods)
%!   method = methods{m_i};
%!
%!   bw = edge (im, method, 0.2, "both", "thinning");
%!   assert (edge (im, method, 0.2), bw)
%!
%!   args = perms ({0.2, "both", "thinning"});
%!   for i = 1:rows (args)
%!     assert (edge (im, method, args{i,:}), bw)
%!   endfor
%!
%!   bw = edge (im, method, 0.2, "vertical", "nothinning");
%!   args = perms ({0.2, "vertical", "nothinning"});
%!   for i = 1:rows (args)
%!     assert (edge (im, method, args{i,:}), bw)
%!   endfor
%!
%!   bw = edge (im, method, 0.2, "vertical", "thinning");
%!   args = perms ({0.2, "vertical"});
%!   for i = 1:rows (args)
%!     assert (edge (im, method, args{i,:}), bw)
%!   endfor
%!
%!   bw = edge (im, method, 0.2, "both", "nothinning");
%!   args = perms ({0.2, "nothinning"});
%!   for i = 1:rows (args)
%!     assert (edge (im, method, args{i,:}), bw)
%!   endfor
%! endfor

%!error <more than 1 threshold argument>
%!  bw = edge (rand (10), "sobel", 0.2, 0.4)
%!error <more than 1 thinning argument>
%!  bw = edge (rand (10), "sobel", "thinning", "nothinning")
%!error <more than 1 direction argument>
%!  bw = edge (rand (10), "sobel", "both", "both")
%!error <only THRESH, DIRECTION, and THINNING arguments>
%!  bw = edge (rand (10), "sobel", [0.2 0.7], "both", "thinning")

%!error <more than 1 threshold argument>
%!  bw = edge (rand (10), "kirsch", 0.2, 0.4)
%!error <more than 1 thinning argument>
%!  bw = edge (rand (10), "kirsch", "thinning", "nothinning")
%!error <more than 1 direction argument>
%!  bw = edge (rand (10), "kirsch", "both", "both")
%!error <only THRESH, DIRECTION, and THINNING arguments>
%!  bw = edge (rand (10), "kirsch", [0.2 0.7], "both", "thinning")

%!error <more than 1 threshold argument>
%!  bw = edge (rand (10), "prewitt", 0.2, 0.4)
%!error <more than 1 thinning argument>
%!  bw = edge (rand (10), "prewitt", "thinning", "nothinning")
%!error <more than 1 direction argument>
%!  bw = edge (rand (10), "prewitt", "both", "both")
%!error <only THRESH, DIRECTION, and THINNING arguments>
%!  bw = edge (rand (10), "prewitt", [0.2 0.7], "both", "thinning")

%!test
%! im = [
%!   249   238   214   157   106    69    60    90   131   181   224   247   252   250   250
%!   250   242   221   165   112    73    62    91   133   183   225   248   252   250   251
%!   252   246   228   173   120    78    63    90   130   181   224   248   253   251   251
%!   253   248   232   185   132    87    62    80   116   170   217   244   253   251   252
%!   253   249   236   198   149   101    66    71   101   155   206   238   252   252   252
%!   254   250   240   210   164   115    73    69    92   143   196   232   252   253   252
%!    70    70    68    61    49    36    24    22    26    38    52    63    70    70    70
%!     0     0     0     0     0     0     0     0     0     0     0     0     0     0     0
%!    62    63    62    59    51    42    33    25    22    26    36    45    56    60    62
%!   252   253   252   246   221   190   157   114    90    90   118   157   203   235   248
%!   251   253   254   251   233   209   182   136   103    92   107   139   185   225   245
%!   251   253   254   253   243   227   206   163   128   108   110   133   175   217   242
%!   252   253   254   254   249   241   228   195   164   137   127   139   172   212   239
%! ] / 255;
%!
%! bw = edge (im, "roberts", .2, "thinning");
%! assert (edge (im, "roberts", 0.2), bw)
%! assert (edge (im, "roberts", "thinning", 0.2), bw)
%!
%! bw = edge (im, "roberts", .2, "nothinning");
%! assert (edge (im, "roberts", "nothinning", 0.2), bw)

%!error <more than 1 threshold argument>
%!  bw = edge (rand (10), "roberts", 0.2, 0.4)
%!error <more than 1 thinning argument>
%!  bw = edge (rand (10), "roberts", "thinning", "nothinning")
%!error <only THRESH and THINNING arguments>
%!  bw = edge (rand (10), "roberts", "both", "thinning")

## test how canny threshold arguments are always a row vector
%!test
%! im = rand (10);
%! [~, thresh] = edge (im, "canny");
%! assert (size (thresh), [1 2])
%! [~, thresh] = edge (im, "canny", [.2 .6]);
%! assert (thresh, [.2 .6])
%! [~, thresh] = edge (im, "canny", [.2; .6]);
%! assert (thresh, [.2 .6])

## test SOBEL edge detector
%!test
%! in = zeros (5);
%! in(3,3) = 1;
%!
%! E = logical ([
%!    0 0 0 0 0
%!    0 0 1 0 0
%!    0 1 0 1 0
%!    0 0 1 0 0
%!    0 0 0 0 0]);
%! assert (edge (in), E)
%! assert (edge (uint8 (in.*100)), E)
%! assert (edge (in, "sobel"), E)
%! assert (edge (in, "sobel", 0), E)
%! assert (edge (in, "sobel", 1), false (5))
%!
%! [E, auto_thresh] = edge (in);
%! assert (auto_thresh, 0.2449, 1e-4)
%!
%! V = logical([
%!    0 0 0 0 0
%!    0 1 0 1 0
%!    0 1 0 1 0
%!    0 1 0 1 0
%!    0 0 0 0 0]);
%! assert (edge (in, "sobel", 0, "vertical"), V)
%!
%! H = logical ([
%!    0 0 0 0 0
%!    0 1 1 1 0
%!    0 0 0 0 0
%!    0 1 1 1 0
%!    0 0 0 0 0]);
%! assert (edge (in, "sobel", 0, "horizontal"), H)
%!
%! V = false (5);
%! V(3,2) = true;
%! V(3,4) = true;
%! assert (edge (in, "sobel", [], "vertical"), V)
%!
%! H = false (5);
%! H(2,3) = true;
%! H(4,3) = true;
%! assert (edge (in, "sobel", [], "horizontal"), H)

%!test
%! A = ones (5);
%! A(3, 3) = 0;
%! expected = logical ([
%!    0  0  0  0  0
%!    0  0  1  0  0
%!    0  1  0  1  0
%!    0  0  1  0  0
%!    0  0  0  0  0]);
%! assert (edge (A), expected)

## test PREWITT edge detector
%!test
%! in = zeros (5);
%! in(3, 3) = 1;
%!
%! E = logical ([
%!    0 0 0 0 0
%!    0 1 0 1 0
%!    0 0 0 0 0
%!    0 1 0 1 0
%!    0 0 0 0 0]);
%!
%! assert (edge (in, "prewitt"), E)
%!
%! [~, auto_thresh] = edge (in, "prewitt");
%! assert (auto_thresh, 0.2309, 1e-4)
%!
%! V = logical([
%!    0 0 0 0 0
%!    0 1 0 1 0
%!    0 1 0 1 0
%!    0 1 0 1 0
%!    0 0 0 0 0]);
%! assert (edge (in, "prewitt", 0, "vertical"), V)
%!
%! H = logical ([
%!    0 0 0 0 0
%!    0 1 1 1 0
%!    0 0 0 0 0
%!    0 1 1 1 0
%!    0 0 0 0 0]);
%! assert (edge (in, "prewitt", 0, "horizontal"), H)

## test ROBERTS edge detector
%!test
%! in = zeros (5);
%! in(3,3) = 1;
%! in(3,4) = 0.9;
%!
%! E = logical ([
%!    0 0 0 0 0
%!    0 0 1 0 0
%!    0 0 1 0 0
%!    0 0 0 0 0
%!    0 0 0 0 0]);
%!
%! assert (edge (in, "roberts"), E)
%!
%! [~, auto_thresh] = edge (in, "roberts");
%! assert (auto_thresh, 0.6591, 1e-4)
%!
%! E45 = [0     0      0     0  0
%!        0  -0.5  -0.45     0  0
%!        0     0   0.50  0.45  0
%!        0     0      0     0  0
%!        0     0      0     0  0];
%! E135 = [0    0      0      0  0
%!         0    0  -0.50  -0.45  0
%!         0  0.5   0.45      0  0
%!         0    0      0      0  0
%!         0    0      0      0  0];
%!
%! [~, ~, erg45, erg135] = edge (in, "roberts");
%! assert (erg45, E45)
%! assert (erg135, E135)

## test CANNY edge detector
%!xtest
%! ## The edge image is correct and Matlab compatible so those should
%! ## pass.  However, the threshold values used to generate the edge
%! ## image are not the same as Matlab.
%!
%! in_8 = fspecial ("gaussian", [8 8], 2);
%! in_8 /= in_8(4,4);
%! in_8_uint8 = im2uint8 (in_8);
%!
%! ## Matlab changed their implementation of the Canny method in
%! ## release 2011a.  We are compatible with their new implementation
%! ## but for testing purposes, this is the expected result for the
%! ## old implementation.
%! out_8_old = logical ([
%!  0   0   0   0   0   0   0   0
%!  0   0   0   1   1   0   0   0
%!  0   0   1   0   0   1   0   0
%!  0   1   0   0   0   0   1   0
%!  0   1   0   0   0   0   1   0
%!  0   0   1   0   0   1   0   0
%!  0   0   0   1   1   0   0   0
%!  0   0   0   0   0   0   0   0]);
%!
%! out_8 = logical ([
%!  0   0   0   0   0   0   0   0
%!  0   1   1   1   1   1   0   0
%!  0   1   0   0   0   1   0   0
%!  0   1   0   0   0   1   0   0
%!  0   1   0   0   0   1   0   0
%!  0   1   1   1   1   1   0   0
%!  0   0   0   0   0   0   0   0
%!  0   0   0   0   0   0   0   0]);
%! out_thresh = [0.34375 0.859375];
%!
%! [obs_edge, obs_thresh] = edge (in_8, "Canny");
%! assert (obs_edge, out_8)
%! assert (obs_thresh, out_thresh)
%!
%! [obs_edge_givethresh, obs_thresh_givethresh] ...
%!    = edge (in_8, "Canny", out_thresh);
%! assert (obs_edge_givethresh, out_8)
%! assert (obs_thresh_givethresh, out_thresh)
%!
%! [obs_edge_uint8, obs_thresh_uint8] = edge (in_8_uint8, "Canny");
%! assert (obs_edge_uint8, out_8)
%! assert (obs_thresh_uint8, out_thresh)

%!xtest
%! ## The edge image is correct and Matlab compatible so those should
%! ## pass.  However, the threshold values used to generate the edge
%! ## image are not the same as Matlab.
%!
%! in_9 = fspecial ("gaussian", [9 9], 2);
%! in_9 /= in_9(5,5);
%!
%! ## Matlab changed their implementation of the Canny method in
%! ## release 2011a.  We are compatible with their new implementation
%! ## but for testing purposes, this is the expected result for the
%! ## old implementation.
%! out_9_old = logical ([
%!  0   0   0   0   0   0   0   0   0
%!  0   0   0   0   0   0   0   0   0
%!  0   0   0   1   1   1   0   0   0
%!  0   0   1   0   0   0   1   0   0
%!  0   0   1   0   0   0   1   0   0
%!  0   0   1   0   0   0   1   0   0
%!  0   0   0   1   1   1   0   0   0
%!  0   0   0   0   0   0   0   0   0
%!  0   0   0   0   0   0   0   0   0]);
%!
%! out_9 = logical ([
%!  0   0   0   0   0   0   0   0   0
%!  0   0   1   1   1   1   0   0   0
%!  0   1   1   0   0   1   1   0   0
%!  0   1   0   0   0   0   1   0   0
%!  0   1   0   0   0   0   1   0   0
%!  0   1   1   0   0   1   1   0   0
%!  0   0   1   1   1   1   0   0   0
%!  0   0   0   0   0   0   0   0   0
%!  0   0   0   0   0   0   0   0   0]);
%! out_thresh = [0.35 0.875];
%!
%! [obs_edge, obs_thresh] = edge (in_9, "Canny");
%! assert (obs_edge, out_9)
%! assert (obs_thresh, out_thresh)
%!
%! [obs_edge_givethresh, obs_thresh_givethresh] ...
%!    = edge (in_9, "Canny", out_thresh);
%! assert (obs_edge_givethresh, out_9)
%! assert (obs_thresh_givethresh, out_thresh)
