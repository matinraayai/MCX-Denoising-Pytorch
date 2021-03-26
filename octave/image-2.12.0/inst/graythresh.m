## Copyright (C) 2004 Antti Niemistö <antti.niemisto@tut.fi>
## Copyright (C) 2007 Søren Hauberg
## Copyright (C) 2012-2013 Carnë Draug <carandraug+dev@gmail.com>
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
## @deftypefn  {Function File} {[@var{level}, @var{sep}] =} graythresh (@var{img})
## @deftypefnx {Function File} {[@var{level}, @var{sep}] =} graythresh (@var{img}, @var{method}, @var{options})
## @deftypefnx {Function File} {[@var{level}, @var{sep}] =} graythresh (@var{hist}, @dots{})
## Compute global image threshold.
##
## Given an image @var{img} finds the optimal threshold value @var{level} for
## conversion to a binary image with @code{im2bw}.  Color images are converted
## to grayscale before @var{level} is computed.  An image histogram @var{hist}
## can also be used to allow for preprocessing of the histogram.
##
## The optional argument @var{method} is the algorithm to be used (default's to
## Otsu).  Some methods may have other @var{options} and/or return an extra
## value @var{sep} (see each entry for details).  The available @var{method}s are:
##
## @table @asis
## @item Otsu (default)
## Implements Otsu's method as described in @cite{Nobuyuki Otsu (1979). "A
## threshold selection method from gray-level histograms", IEEE Trans. Sys.,
## Man., Cyber. 9 (1): 62-66}. This algorithm chooses the threshold to minimize
## the intraclass variance of the black and white pixels.
##
## The second output, @var{sep} represents the ``goodness'' (or separability) of
## the threshold at @var{level}.  It is a value within the range [0 1], the
## lower bound (zero) being attainable by, and only by, histograms having a
## single constant gray level, and the upper bound being attainable by, and only
## by, two-valued pictures.
##
## @item concavity
## Find a global threshold for a grayscale image by choosing the threshold to
## be in the shoulder of the histogram @cite{A. Rosenfeld, and P. De La Torre
## (1983). "Histogram concavity analysis as an aid in threshold selection", IEEE
## Transactions on Systems, Man, and Cybernetics, 13: 231-235}.
##
## @item intermodes
## This assumes a bimodal histogram and chooses the threshold to be the mean of
## the two peaks of the bimodal histogram @cite{J. M. S. Prewitt, and M. L.
## Mendelsohn (1966). "The analysis of cell images", Annals of the New York
## Academy of Sciences, 128: 1035-1053}.
##
## Images with histograms having extremely unequal peaks or a broad and flat
## valley are unsuitable for this method.
##
## @item intermeans
## Iterative procedure based on the iterative intermeans algorithm of @cite{T.
## Ridler, and S. Calvard (1978). "Picture thresholding using an iterative
## selection method", IEEE Transactions on Systems, Man, and Cybernetics, 8: 630-632}
## and @cite{H. J. Trussell (1979). "Comments on 'Picture thresholding using an
## iterative selection method'", IEEE Transactions on Systems, Man, and Cybernetics,
## 9: 311}.
##
## Note that several implementations of this method exist. See the source code
## for details.
##
## @item MaxEntropy
## Implements Kapur-Sahoo-Wong (Maximum Entropy) thresholding method based on the
## entropy of the image histogram @cite{J. N. Kapur, P. K. Sahoo, and A. C. K. Wong
## (1985). "A new method for gray-level picture thresholding using the entropy
## of the histogram", Graphical Models and Image Processing, 29(3): 273-285}.
##
## @item MaxLikelihood
## Find a global threshold for a grayscale image using the maximum likelihood
## via expectation maximization method @cite{A. P. Dempster, N. M. Laird, and D. B.
## Rubin (1977). "Maximum likelihood from incomplete data via the EM algorithm",
## Journal of the Royal Statistical Society, Series B, 39:1-38}.
##
## @item mean
## The mean intensity value. It is mostly used by other methods as a first guess
## threshold.
##
## @item MinError
## An iterative implementation of Kittler and Illingworth's Minimum Error
## thresholding @cite{J. Kittler, and J. Illingworth (1986). "Minimum error
## thresholding", Pattern recognition, 19: 41-47}.
##
## This implementation seems to converge more often than the original.
## Nevertheless, sometimes the algorithm does not converge to a solution. In
## that case a warning is displayed and defaults to the initial estimate of the
## mean method.
##
## @item minimum
## This assumes a bimodal histogram and chooses the threshold to be in the
## valley of the bimodal histogram.  This method is also known as the mode
## method @cite{J. M. S. Prewitt, and M. L. Mendelsohn (1966). "The analysis of
## cell images", Annals of the New York Academy of Sciences, 128: 1035-1053}.
##
## Images with histograms having extremely unequal peaks or a broad and flat
## valley are unsuitable for this method.
##
## @item moments
## Find a global threshold for a grayscale image using moment preserving
## thresholding method @cite{W. Tsai (1985). "Moment-preserving thresholding:
## a new approach", Computer Vision, Graphics, and Image Processing, 29: 377-393}
##
## @item percentile
## Assumes a specific fraction of pixels (set at @var{options}) to be background.
## If no value is given, assumes 0.5 (equal distribution of background and foreground)
## @cite{W Doyle (1962). "Operation useful for similarity-invariant pattern
## recognition", Journal of the Association for Computing Machinery 9: 259-267}
## @end table
##
## @seealso{im2bw, otsuthresh}
## @end deftypefn

## Notes:
##  * The following methods were adapted from http://www.cs.tut.fi/~ant/histthresh/
##      intermodes    percentile      minimum
##      MaxEntropy    MaxLikelihood   intermeans
##      moments       minerror        concavity
##  * Carnë Draug implemented and vectorized the Otsu's method
##  * Carnë Draug vectorized percentile and moments.
##  * missing methods from ImageJ
##      Yen           triangle        RenyiEntropy
##      Shanbhag      Li              Huang
##      ImageJ

function [varargout] = graythresh (img, algo = "otsu", varargin)
  ## Input checking
  if (nargin < 1 || nargin > 3)
    print_usage();
  elseif (nargin > 2 && !any (strcmpi (algo, {"percentile"})))
    error ("graythresh: algorithm `%s' does not accept any options.", algo);
  else
    hist_in = false;
    if (! isnumeric (img))
      error ("graythresh: IMG must be numeric");
    elseif (size (img, 3) == 3)
      ## If the image is RGB convert it to grayscale
      img = rgb2gray (img);
    elseif (isfloat(img) && isvector (img) && !issparse (img) &&
            isreal (img) && all (img >= 0))
      ## FIXME: we need to rething this approach.  This means that
      ##        "images" with a single row or vector will be handled
      ##        as histograms.  Also, previously we relied on having
      ##        values outside [0 1] range to distinguish between an
      ##        image and a histogram but for Matlab compatibility we
      ##        need to clip the values so there's backwards
      ##        incompatibility issues when addressing this.  Because
      ##        previously any histogram of integer class would be
      ##        identified as an image, we added the 'isfloat' logic
      ##        to limit the issue to images of floating point
      ##        images.  Best fix is to implement histthresh to handle
      ##        histograms and this function simply prepares an
      ##        histogram from a user image and then calls histthresh.
      hist_in = true;
      ihist   = img;
    endif
    ## ... else it should a gray image so do nothing.  The only thing
    ## we need to worry is on clipping values outside the [0 1] range
    ## for floating point images.  But that is done later when it
    ## converts to uint8.
  endif

  ## the "mean" is the simplest of all, we can get rid of it right here
  if (strcmpi (algo, "mean"))
    varargout{1} = mean (im2double(img)(:));
    return
  endif

  ## we only need to do this if the input is an image. If an histogram, is
  ## supplied, no need to do any of this
  if (!hist_in)
    ## if image is uint we do nothing. If it's int, then we convert to uint since
    ## it may mess up calculations later. If it's double we need some bins so we
    ## choose uint8 by default... but should we adjust the histogram before?
    if (isa (img, "uint16") || isa (img, "uint8"))
      ## do nothing
    elseif (isa (img, "int16"))
      img     = im2uint16 (img);
    else
      img = im2uint8 (img);
    endif
    ihist = hist (img(:), 0:intmax (class (img)));
  endif

  switch tolower (algo)
    case {"concavity"},     thresh = concavity      (ihist);
    case {"intermeans"},    thresh = intermeans     (ihist, floor (mean (img (:))));
    case {"intermodes"},    thresh = intermodes     (ihist);
    case {"maxentropy"},    thresh = maxentropy     (ihist);
    case {"maxlikelihood"}, thresh = maxlikelihood  (ihist);
    case {"minerror"},      thresh = minerror_iter  (ihist, floor (mean (img (:))));
    case {"minimum"},       thresh = minimum        (ihist);
    case {"moments"},       thresh = moments        (ihist);
    case {"otsu"},          thresh = otsu           (ihist, nargout > 1);
    case {"percentile"},    thresh = percentile     (ihist, varargin{:});
    otherwise, error ("graythresh: unknown method '%s'", algo);
  endswitch
  ## normalize the threshold value to the [0 1] range
  if (numel (ihist) > 1)
    thresh{1} = double (thresh{1}) / (numel (ihist) - 1);
  endif

  ## some algorithms may return more than one value...
  for i = 1:numel (thresh)
    varargout{i} = thresh{i};
  endfor
endfunction

function [thresh] = otsu (ihist, compute_good)
  ## this method is quite well explained at
  ## http://www.labbookpages.co.uk/software/imgProc/otsuThreshold.html
  ##
  ## It does not, however, explain how to compute the goodness of threshold and
  ## there's not many pages explaining it either. For that, one really needs to
  ## check the paper.
  ##
  ## The implementation on the link above assumes that threshold is to be
  ## made for values "greater or equal than" but that is not the case (in im2bw
  ## and also not ImageJ) so we subtract 1 at the end.

  if (numel (ihist) == 1 || sum (ihist) == 0)
    thresh{1} = 0;
    thresh{2} = 0;
    return;
  endif
  
  bins  = 0:(numel (ihist) - 1);
  total = sum (ihist);
  ## b = black, w = white
  b_totals  = cumsum ([0 ihist(1:end-1)]);
  b_weights = b_totals / total;
  b_means   = [0 cumsum(bins(1:end-1) .* ihist(1:end-1))] ./ b_totals;

  w_totals  = total - b_totals;
  w_weights = w_totals / total;
  w_means   = (cumsum (bins(end:-1:1) .* ihist(end:-1:1)) ./ w_totals(end:-1:1))(end:-1:1);

  ## between class variance (its maximum is the best threshold)
  bcv       = b_weights .* w_weights .* (b_means - w_means).^2;

  max_bcv = max (bcv);
  if (isnan (max_bcv))
    ## Couldn't measure variance.  This will happen if for example all
    ## values are the same.  See bug #45333.
    thresh{1} = 0;
    thresh{2} = 0;
  else
    ## In case there's more than one place with best maximum (for
    ## example, a group of empty bins), we select the one in the
    ## center (this is compatible with ImageJ).
    thresh{1} = (mean (find (bcv == max_bcv))) - 2;
    ## We subtract 2, once for the 1 based indexes and another for the
    ## greater than or equal problem.

    if (compute_good)
      ## Basically we need to divide the between class variance by the
      ## total variance which is a single value, independent of the
      ## threshold.  From the paper, last of the equation 12:
      ##     eta = sigma²b / sigma²t
      ## where b = between and t = total
      norm_hist      = ihist / total;
      total_mean     = sum (bins .* norm_hist);
      total_variance = sum (((bins - total_mean).^2) .* norm_hist);
      thresh{2}      = max (bcv) / total_variance;
    endif
  endif
endfunction

function level = moments (y)
  n = numel (y) - 1;

  ## The threshold is chosen such that partial_sumA(y,t)/partial_sumA(y,n)
  ## is closest to x0.
  sumY = sum (y);
  Avec = cumsum (y) / sumY;

  sumB = partial_sumB (y,n);
  sumC = partial_sumC (y,n);
  sumD = partial_sumD (y,n);
  ## The following finds x0.
  x2 = (sumB*sumC - sumY*sumD) / (sumY*sumC - sumB^2);
  x1 = (sumB*sumD - sumC^2) / (sumY*sumC - sumB^2);
  x0 = .5 - (sumB/sumY + x2/2) / sqrt (x2^2 - 4*x1);

  ## And finally the threshold
  [~, ind] = min (abs (Avec-x0));
  level{1} = ind-1;
endfunction

function T = maxentropy(y)
  n = numel (y) - 1;

  warning ("off", "Octave:divide-by-zero", "local");

  ## The threshold is chosen such that the following expression is minimized.
  sumY = sum (y);
  negY = negativeE (y, n);
  for j = 0:n
    sumA = partial_sumA (y, j);
    negE = negativeE (y, j);
    sum_diff = sumY - sumA;
    vec(j+1) = negE/sumA - log10 (sumA) + (negY-negE)/(sum_diff) - log10 (sum_diff);
  end

  [~,ind] = min (vec);
  T{1} = ind-1;
endfunction

function [T] = intermodes (y)
  ## checked with ImageJ and is slightly different but not by much
  n = numel (y) - 1;

  % Smooth the histogram by iterative three point mean filtering.
  iter = 0;
  while ~bimodtest(y)
    h = ones(1,3)/3;
    y = conv2(y,h,'same');
    iter = iter+1;
    % If the histogram turns out not to be bimodal, set T to zero.
    if iter > 10000;
      T{1} = 0;
      return
    end
  end

  % The threshold is the mean of the two peaks of the histogram.
  ind = 0;
  for k = 2:n
    if y(k-1) < y(k) && y(k+1) < y(k)
      ind = ind+1;
      TT(ind) = k-1;
    end
  end
  T{1} = floor(mean(TT));
endfunction

## The threshold is chosen such that 50% (in case of p = 0.5) of
## pixels lie in each category.
function [T] = percentile (y, p = 0.5)
  Avec = cumsum (y) / sum (y);
  [~, ind] = min (abs (Avec - p));
  T{1} = ind -1;
endfunction


function T = minimum(y)
  n = numel (y) - 1;

  % Smooth the histogram by iterative three point mean filtering.
  iter = 0;
  while ~bimodtest(y)
    h = ones(1,3)/3;
    y = conv2(y,h,'same');
    iter = iter+1;
    % If the histogram turns out not to be bimodal, set T to zero.
    if iter > 10000;
      T{1} = 0;
      return
    end
  end

  peakfound = false;
  for k = 2:n
    if y(k-1) < y(k) && y(k+1) < y(k)
      peakfound = true;
    end
    if peakfound && y(k-1) >= y(k) && y(k+1) >= y(k)
      T{1} = k-1;
      return
    end
  end
endfunction

function [Tout] = minerror_iter (y, T)
  n = numel (y) - 1;

  Tprev = NaN;

  warning ("off", "Octave:divide-by-zero", "local");

  sumA = partial_sumA (y, n);
  sumB = partial_sumB (y, n);
  sumC = partial_sumC (y, n);
  while T ~= Tprev
    % Calculate some statistics.
    sumAT = partial_sumA (y, T);
    sumBT = partial_sumB (y, T);
    sumCT = partial_sumC (y, T);
    sumAdiff = sumA - sumAT;

    mu = sumBT/sumAT;
    nu = (sumB-sumBT)/(sumAdiff);
    p = sumAT/sumA;
    q = (sumAdiff) / sumA;
    sigma2 = sumCT/sumAT-mu^2;
    tau2 = (sumC-sumCT) / (sumAdiff) - nu^2;

    % The terms of the quadratic equation to be solved.
    w0 = 1/sigma2-1/tau2;
    w1 = mu/sigma2-nu/tau2;
    w2 = mu^2/sigma2 - nu^2/tau2 + log10((sigma2*q^2)/(tau2*p^2));

    % If the next threshold would be imaginary, return with the current one.
    sqterm = w1^2-w0*w2;
    if sqterm < 0
      warning('MINERROR:NaN','Warning: th_minerror_iter did not converge.')
      break
    endif

    % The updated threshold is the integer part of the solution of the
    % quadratic equation.
    Tprev = T;
    T = floor((w1+sqrt(sqterm))/w0);

    % If the threshold turns out to be NaN, return with the previous threshold.
    if isnan(T)
      warning('MINERROR:NaN','Warning: th_minerror_iter did not converge.')
      T = Tprev;
    end
  endwhile
  Tout{1} = T;
endfunction
#{
  ## this is an implementation of the original minerror algorithm but seems
  ## to converge less often than the iterative version. This one is also from the
  ## HistThresh toolbox
  function T = th_minerror(I,n)
    if nargin == 1
      n = 255;
    end

    I = double(I);

    % Calculate the histogram.
    y = hist(I(:),0:n);

    % The threshold is chosen such that the following expression is minimized.
    for j = 0:n
      mu = partial_sumB(y,j)/partial_sumA(y,j);
      nu = (partial_sumB(y,n)-partial_sumB(y,j))/(partial_sumA(y,n)-partial_sumA(y,j));
      p = partial_sumA(y,j)/partial_sumA(y,n);
      q = (partial_sumA(y,n)-partial_sumA(y,j)) / partial_sumA(y,n);
      sigma2 = partial_sumC(y,j)/partial_sumA(y,j)-mu^2;
      tau2 = (partial_sumC(y,n)-partial_sumC(y,j)) / (partial_sumA(y,n)-partial_sumA(y,j)) - nu^2;
      vec(j+1) = p*log10(sqrt(sigma2)/p) + q*log10(sqrt(tau2)/q);
    end

    vec(vec==-inf) = NaN;
    [minimum,ind] = min(vec);
    T = ind-1;
  endfunction
#}

function Tout = maxlikelihood (y)

  n = numel (y) - 1;

  ## initial estimate for the threshold is found with the minimum algorithm
  T = minimum (y){1};

  sumY = sum (y);

  sumB  = partial_sumB (y, n);
  sumC  = partial_sumC (y, n);

  sumAT = partial_sumA (y, T);
  sumBT = partial_sumB (y, T);
  sumCT = partial_sumC (y, T);
  ## initial values for the statistics
  mu     = sumBT / sumAT;
  nu     = (sumB - sumBT) / (sumY - sumAT);
  p      = sumAT / sumY;
  q      = (sumY - sumAT) / sumY;
  sigma2 = sumCT / sumAT - mu^2;
  tau2   = (sumC - sumCT) / (sumY - sumAT) - nu^2;

  ## Return if sigma2 or tau2 are zero, to avoid division by zero
  if (sigma2 == 0 || tau2 == 0)
    Tout{1} = T;
    return
  endif

  do
    ## we store the previous values for comparison at the end (we will stop when
    ## they stop changing)
    mu_prev     = mu;
    nu_prev     = nu;
    p_prev      = p;
    q_prev      = q;
    sigma2_prev = nu;
    tau2_prev   = nu;

    for i = 0:n
      phi(i+1) = p/sqrt((sigma2)) * exp(-((i-mu)^2) / (2*sigma2)) / ...
                (p/sqrt(sigma2)   * exp(-((i-mu)^2) / (2*sigma2)) + ...
                (q/sqrt(tau2))    * exp(-((i-nu)^2) / (2*tau2)));
    endfor
    ind   = 0:n;
    gamma = 1-phi;
    F     = phi*y';
    G     = gamma*y';

    mu      = ind.*phi*y'/F;
    nu      = ind.*gamma*y'/G;
    p       = F / sumY;
    q       = G / sumY;
    sigma2  = ind.^2.*phi*y'/F - mu^2;
    tau2    = ind.^2.*gamma*y'/G - nu^2;

  until (abs (mu - mu_prev)         <= eps || abs (nu - nu_prev)     <= eps || ...
         abs (p  - p_prev)          <= eps || abs (q - q_prev)       <= eps || ...
         abs (sigma2 - sigma2_prev) <= eps || abs (tau2 - tau2_prev) <= eps)

  ## the terms of the quadratic equation to be solved
  w0 = 1/sigma2-1/tau2;
  w1 = mu/sigma2-nu/tau2;
  w2 = mu^2/sigma2 - nu^2/tau2 + log10((sigma2*q^2)/(tau2*p^2));

  ## If the threshold would be imaginary, return with threshold set to zero
  sqterm = w1^2-w0*w2;
  if (sqterm < 0)
    Tout{1} = 0;
    return
  endif

  ## The threshold is the integer part of the solution of the quadratic equation
  Tout{1} = floor((w1+sqrt(sqterm))/w0);
endfunction

function Tout = intermeans (y, T)
  n = numel (y) - 1;

  Tprev = NaN;

  % The threshold is found iteratively. In each iteration, the means of the
  % pixels below (mu) the threshold and above (nu) it are found. The
  % updated threshold is the mean of mu and nu.
  sumY = sum (y);
  sumB = partial_sumB (y, n);
  while T ~= Tprev
    sumAT = partial_sumA (y, T);
    sumBT = partial_sumB (y, T);

    mu = sumBT/sumAT;
    nu = (sumB-sumBT)/(sumY-sumAT);
    Tprev = T;
    T = floor((mu+nu)/2);
  end
  Tout{1} = T;
endfunction

function T = concavity (h)
  n = numel (h) - 1;
  H = hconvhull(h);

  % Find the local maxima of the difference H-h.
  lmax = flocmax(H-h);

  % Find the histogram balance around each index.
  for k = 0:n
    E(k+1) = hbalance(h,k);
  end

  % The threshold is the local maximum with highest balance.
  E = E.*lmax;
  [dummy ind] = max(E);
  T{1} = ind-1;
endfunction

################################################################################
## Auxiliary functions from HistThresh toolbox http://www.cs.tut.fi/~ant/histthresh/
################################################################################

## partial sums from C. A. Glasbey, "An analysis of histogram-based thresholding
## algorithms," CVGIP: Graphical Models and Image Processing, vol. 55, pp. 532-537, 1993.
function x = partial_sumA (y, j)
  x = sum (y(1:j+1));
endfunction
function x = partial_sumB (y, j)
  ind = 0:j;
  x   = ind*y(1:j+1)';
endfunction
function x = partial_sumC (y, j)
  ind = 0:j;
  x = ind.^2*y(1:j+1)';
endfunction
function x = partial_sumD (y, j)
  ind = 0:j;
  x = ind.^3*y(1:j+1)';
endfunction

## Test if a histogram is bimodal.
function b = bimodtest(y)
  len = length(y);
  b = false;
  modes = 0;

  % Count the number of modes of the histogram in a loop. If the number
  % exceeds 2, return with boolean return value false.
  for k = 2:len-1
    if y(k-1) < y(k) && y(k+1) < y(k)
      modes = modes+1;
      if modes > 2
        return
      end
    end
  end

  % The number of modes could be less than two here
  if modes == 2
    b = true;
  end
endfunction

## Find the local maxima of a vector using a three point neighborhood.
function y = flocmax(x)
%  y    binary vector with maxima of x marked as ones

  len = length(x);
  y = zeros(1,len);

  for k = 2:len-1
    [dummy,ind] = max(x(k-1:k+1));
    if ind == 2
      y(k) = 1;
    end
  end
endfunction

## Calculate the balance measure of the histogram around a histogram index.
function E = hbalance(y,ind)
%  y    histogram
%  ind  index about which balance is calculated
%
% Out:
%  E    balance measure
%
% References:
%
% A. Rosenfeld and P. De La Torre, "Histogram concavity analysis as an aid
% in threshold selection," IEEE Transactions on Systems, Man, and
% Cybernetics, vol. 13, pp. 231-235, 1983.
%
% P. K. Sahoo, S. Soltani, and A. K. C. Wong, "A survey of thresholding
% techniques," Computer Vision, Graphics, and Image Processing, vol. 41,
% pp. 233-260, 1988.

  n = length(y)-1;
  E = partial_sumA(y,ind)*(partial_sumA(y,n)-partial_sumA(y,ind));
endfunction

## Find the convex hull of a histogram.
function H = hconvhull(h)
  % In:
  %  h    histogram
  %
  % Out:
  %  H    convex hull of histogram
  %
  % References:
  %
  % A. Rosenfeld and P. De La Torre, "Histogram concavity analysis as an aid
  % in threshold selection," IEEE Transactions on Systems, Man, and
  % Cybernetics, vol. 13, pp. 231-235, 1983.

  len = length(h);
  K(1) = 1;
  k = 1;

  % The vector K gives the locations of the vertices of the convex hull.
  while K(k)~=len

    theta = zeros(1,len-K(k));
    for i = K(k)+1:len
      x = i-K(k);
      y = h(i)-h(K(k));
      theta(i-K(k)) = atan2(y,x);
    end

    maximum = max(theta);
    maxloc = find(theta==maximum);
    k = k+1;
    K(k) = maxloc(end)+K(k-1);

  end

  % Form the convex hull.
  H = zeros(1,len);
  for i = 2:length(K)
    H(K(i-1):K(i)) = h(K(i-1))+(h(K(i))-h(K(i-1)))/(K(i)-K(i-1))*(0:K(i)-K(i-1));
  end
endfunction

## Entropy function. Note that the function returns the negative of entropy.
function x = negativeE(y,j)
  ## used by the maxentropy method only
  y = y(1:j+1);
  y = y(y~=0);
  x = sum(y.*log10(y));
endfunction

## these were tested with ImageJ
%!shared img, histo
%! ## this is the old default.img that came with GNU Octave. While the current
%! ## is very very similar, is off just enough for us to get precision errors
%! img = uint8 (reshape ([138 138 138 142 142 138 142 138 138 117 105 81 69 61 53 40 49 45 40 36 40 45 53 49 65 73 121 166 210 243 247 247 247 239 235 178 154 170 150 150 162 174 190 190 194 186 178 170 154 182 198 174 117 138 138 142 138 142 142 146 142 138 138 130 109 97 81 73 69 57 53 53 57 61 61 69 73 77 105 121 158 219 243 243 247 243 243 243 206 150 158 158 158 150 158 182 186 190 194 186 174 190 206 198 162 138 142 138 142 146 138 142 142 138 146 142 134 142 130 121 101 97 85 85 81 81 81 85 93 85 73 57 61 93 150 194 215 239 243 243 243 223 166 138 158 158 154 142 162 178 190 190 198 186 182 186 174 162 182 146 142 138 142 142 146 142 146 146 146 146 142 142 142 134 125 101 85 73 65 69 73 73 57 40 53 49 57 69 85 125 166 182 178 178 174 150 130 121 146 146 150 142 166 182 190 182 174 166 162 170 194 198 138 138 146 146 138 146 146 146 146 142 150 146 146 142 130 93 65 45 45 49 45 40 49 40 49 49 49 49 61 81 113 142 150 154 154 146 142 134 125 125 138 134 125 146 162 178 178 178 166 186 202 206 186 142 142 142 134 142 146 142 150 142 146 142 146 146 130 81 53 49 49 45 49 40 36 36 32 36 36 36 53 73 89 125 150 146 134 138 146 138 146 138 142 117 117 113 117 146 166 174 178 182 178 178 170 146 142 142 138 142 146 142 142 146 150 138 146 142 130 73 49 40 49 57 65 69 73 61 61 53 57 53 61 77 77 97 113 138 134 130 138 142 150 146 150 134 138 121 121 101 121 150 158 154 142 150 162 166 178 138 138 146 142 142 142 142 146 146 142 142 130 73 57 49 36 49 65 77 85 89 85 81 81 81 85 93 93 97 105 117 125 150 158 154 162 162 166 154 134 150 130 125 113 138 182 174 154 130 178 227 239 239 134 138 142 138 142 142 146 146 138 150 125 61 49 32 32 45 49 57 65 85 101 105 101 101 109 125 117 113 109 138 134 125 166 178 170 162 150 170 162 170 150 146 150 138 125 162 186 182 142 206 247 247 243 138 138 138 138 142 142 146 146 146 130 85 45 45 36 40 53 45 57 69 97 125 130 130 134 138 146 142 134 142 158 138 117 146 174 170 174 178 170 174 170 166 154 162 158 130 134 170 178 158 190 243 247 247 142 142 142 142 142 146 146 142 138 89 53 45 40 45 45 49 57 77 93 125 138 150 154 158 158 162 154 150 166 174 142 73 125 174 178 174 182 182 178 178 174 166 174 174 162 125 154 170 174 170 227 247 251 142 138 142 142 142 142 142 138 105 61 40 40 32 40 40 49 61 89 117 146 154 158 162 170 170 174 162 166 174 182 150 65 146 166 174 186 198 198 198 190 178 178 174 174 158 134 154 198 194 174 202 251 251 146 142 142 142 146 150 138 134 69 40 40 36 32 40 45 45 65 101 134 150 158 166 174 178 174 174 174 170 170 174 142 73 150 162 178 194 202 202 194 194 178 178 154 134 125 138 154 198 194 186 190 243 251 150 146 146 146 146 150 130 109 53 45 28 40 40 36 32 49 73 101 130 154 162 170 170 170 178 182 178 178 174 158 142 121 146 158 178 174 186 190 186 186 174 146 105 109 113 130 150 178 202 190 186 243 251 146 146 146 146 150 142 109 73 49 40 32 40 40 45 40 53 69 93 130 154 162 170 174 178 182 182 186 182 178 154 146 130 138 142 150 170 182 178 174 166 150 117 97 105 113 130 150 150 174 182 190 243 251 146 146 154 146 150 134 105 53 40 45 45 40 40 36 36 40 69 105 134 162 170 174 178 182 182 182 186 190 186 178 170 158 154 150 162 182 182 174 174 174 150 113 109 113 113 130 150 162 186 186 190 239 251 154 150 146 150 146 125 77 49 36 40 36 40 36 28 40 36 77 113 138 150 170 170 174 186 190 190 190 194 190 186 194 190 170 162 174 194 174 182 170 170 158 121 113 113 113 146 158 170 210 215 215 206 243 150 146 150 150 150 113 57 49 40 45 45 49 49 40 32 45 85 113 142 170 178 174 182 194 190 194 194 198 198 198 210 210 182 162 170 190 182 186 170 170 162 130 121 113 121 146 154 150 198 215 206 210 215 150 150 150 150 150 105 49 45 40 49 49 57 40 49 49 53 85 121 158 182 178 174 182 198 194 194 194 194 202 202 194 186 174 154 162 166 178 174 170 170 170 158 117 113 130 150 154 121 182 194 206 215 206 158 150 150 150 146 97 45 36 49 49 49 40 40 49 49 65 97 130 154 174 174 174 186 194 194 194 194 198 198 186 170 158 154 158 138 158 162 170 190 182 174 170 138 138 142 154 134 142 146 170 206 219 215 150 150 158 158 150 85 36 40 40 40 40 45 45 49 49 65 97 130 146 166 166 174 182 190 194 194 194 194 190 182 162 158 150 158 182 186 178 198 206 198 190 174 154 174 174 142 142 170 170 166 202 223 219 158 150 150 150 146 85 40 45 40 40 36 45 53 45 49 53 93 117 130 154 162 174 190 186 194 194 194 190 186 178 162 162 170 174 182 198 210 206 210 198 198 182 170 178 174 158 154 194 194 174 198 210 215 150 154 158 150 150 85 49 45 40 40 32 36 53 40 45 53 81 109 142 158 158 174 178 182 190 190 194 190 190 178 170 174 178 186 190 190 206 215 202 206 194 186 178 182 174 154 170 198 210 186 186 202 215 150 154 150 154 150 97 45 40 40 40 36 36 45 40 45 73 89 113 142 158 158 174 174 182 186 186 194 186 182 178 174 170 105 166 206 186 190 202 198 194 190 182 182 174 166 154 162 198 215 202 182 202 219 154 150 154 150 146 117 61 45 45 45 36 53 53 49 53 77 93 101 125 158 162 174 174 178 174 186 190 182 182 186 182 182 77 125 198 194 186 190 190 178 178 178 162 162 162 154 186 210 227 210 190 206 223 154 150 154 150 154 138 65 45 45 45 40 49 49 40 53 65 77 89 113 150 158 166 166 170 178 182 186 182 170 170 170 162 81 117 186 190 186 182 178 186 174 166 162 150 130 154 194 227 227 219 202 202 219 154 154 150 154 146 146 89 45 40 45 40 49 49 36 40 57 65 89 109 138 146 158 158 170 170 178 182 178 162 150 158 154 113 146 186 182 178 182 178 170 170 162 146 138 138 146 202 223 231 219 210 190 215 130 130 130 130 130 130 109 45 53 40 32 36 40 45 53 61 65 81 97 117 130 138 150 158 158 178 170 162 158 138 142 150 146 166 178 174 174 170 170 170 162 158 138 117 117 142 202 223 239 223 215 186 206 61 61 65 69 69 65 57 36 40 36 32 40 40 53 57 53 57 69 93 105 109 130 138 142 154 162 150 138 142 125 121 150 162 170 170 166 170 170 170 166 162 138 121 113 130 170 202 223 227 231 202 178 182 45 49 45 40 40 40 45 45 45 45 36 40 32 49 61 61 57 65 73 81 101 109 121 130 142 146 121 89 93 117 113 134 154 174 166 162 166 170 170 162 154 150 142 150 223 186 194 215 231 227 206 182 174 49 40 45 45 49 49 45 49 49 49 49 40 36 45 57 69 65 61 65 69 85 93 109 109 117 109 89 57 57 81 97 113 154 162 166 162 170 158 158 162 154 162 174 231 239 178 186 210 231 239 210 194 178 49 36 49 45 49 49 49 45 45 49 49 36 40 40 45 36 53 53 53 57 57 69 69 73 69 61 57 45 45 65 89 105 125 142 146 150 150 154 162 170 174 223 235 247 231 178 178 206 227 227 223 198 190 40 53 36 45 40 40 40 40 45 40 40 45 45 45 45 40 53 49 49 45 53 45 32 36 36 36 36 40 49 45 61 73 89 93 97 113 125 142 186 202 239 239 243 251 239 198 166 194 215 235 227 215 202 40 45 36 32 36 40 40 45 40 40 45 49 45 49 45 49 40 40 45 49 40 45 45 45 49 49 32 40 49 40 49 57 69 81 101 134 170 206 235 243 243 239 247 251 247 210 170 186 202 231 231 227 210 49 45 49 40 40 40 49 45 40 40 45 45 45 40 45 45 45 49 40 49 40 49 45 45 36 40 40 45 45 45 45 65 121 150 210 239 243 243 247 243 243 247 251 251 239 223 178 174 194 219 239 231 219 36 45 45 40 40 49 40 45 49 49 40 40 45 49 40 40 45 49 45 40 49 45 40 40 40 49 40 45 40 49 49 121 162 215 247 247 247 247 247 243 247 251 251 251 247 239 223 194 186 202 215 210 210 36 45 45 40 40 49 40 45 32 36 49 36 45 49 40 40 45 40 36 40 45 45 40 40 40 36 45 32 40 49 57 121 142 215 243 247 243 247 243 247 251 251 251 251 247 247 247 227 186 194 190 190 182 40 32 45 32 45 40 45 45 49 45 40 45 49 36 40 45 32 40 45 45 49 45 45 45 45 53 49 53 45 45 40 69 97 186 239 243 247 247 247 251 251 251 251 251 243 243 231 202 202 206 206 186 170 53 40 40 40 40 40 36 32 32 36 45 53 49 32 36 32 36 32 40 49 40 40 45 40 40 53 45 49 49 40 32 40 49 138 219 235 247 247 251 251 251 251 251 247 243 235 198 206 210 198 190 186 186 73 69 61 57 61 49 53 40 49 45 40 49 49 49 57 57 53 49 53 53 45 40 45 40 45 49 45 49 45 40 32 53 69 101 215 231 247 247 247 247 251 251 251 243 235 219 194 202 202 186 186 190 194], [53 40]));
%!assert (graythresh (img, "percentile"),      142/255);
%!assert (graythresh (img, "percentile", 0.5), 142/255);
%!assert (graythresh (img, "moments"),         142/255);
%!assert (graythresh (img, "minimum"),          93/255);
%!assert (graythresh (img, "maxentropy"),      150/255);
%!assert (graythresh (img, "intermodes"),       99/255);
%!assert (graythresh (img, "otsu"),            114.5/255);
%! histo = hist (img(:), 0:255);
%!assert (graythresh (histo, "otsu"),          114.5/255);

## for the mean our results differ from matlab because we do not calculate it
## from the histogram. Our results should be more accurate.
%!assert (graythresh (img, "mean"), 0.51445615982, 0.000000001);  # here our results differ from ImageJ

## Test for bug #45333
%!test
%! im = repmat (0.5, 100, 100);
%! [t, g] = graythresh (im);
%! assert (t, 0)
%! assert (g, 0)

## test for bug #51976
## Values outside the range [0 1] in floating images, should be clipped
%!test
%! im = [-2  1  0; 43  .5 .2];
%! clip_im = [ 0  1  0;  1  .5 .2];
%! t = graythresh (clip_im);
%! assert (graythresh (im), t)
%! assert (graythresh (single (im)), t)

## test for bug #45333
%!test
%! H(1) = 100;
%! assert (graythresh (H), 0)
