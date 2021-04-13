## Copyright (C) 2018 Avinoam Kalma <a.kalma@gmail.com>
## Copyright (C) 2018 David Miguel Susano Pinto <carandraug@octave.org>
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
## @deftypefn {} {[@var{level}, @var{sep}] =} otsuthresh (@var{hist})
## Compute global image threshold for histogram using Otsu's method.
##
## Given an image histogram @var{hist} finds the optimal threshold
## value @var{level} for conversion to a binary image with
## @code{im2bw}.
##
## The Otsu's method chooses the threshold value that minimises the
## intraclass variance between two classes, the background and
## foreground. The method is described in @cite{Nobuyuki Otsu
## (1979). "A threshold selection method from gray-level histograms",
## IEEE Trans. Sys., Man., Cyber. 9 (1): 62-66}.
##
## The second output, @var{sep} represents the ``goodness'' (or
## separability) of the threshold at @var{level}.  It is a value
## within the range [0 1], the lower bound (zero) being attainable by,
## and only by, histograms having a single constant grey level, and
## the upper bound being attainable by, and only by, two-valued
## pictures.
##
## @seealso{graythresh, im2bw}
## @end deftypefn

function [varargout] = otsuthresh (hist)

  if (nargin != 1)
    print_usage ();
  endif

  if (! isvector (hist) || ! isnumeric (hist) || ! isreal (hist)
      || any (isinf (hist)) || any (isnan (hist)) || any (hist < 0)
      || any (hist != fix (hist)))
    error ("otsuthresh: HIST must be a vector of non-negative integers");
  endif

  hist = double (hist);
  [varargout{1:nargout}] = graythresh (hist(:).', "otsu");
endfunction

%!test
%! histo = zeros (1, 256);
%! histo([ 29  33  37  41  46  50  54  58  62  66  70  74  78  82 ...
%!         86  90  94  98 102 106 110 114 118 122 126 131 135 139 ...
%!        143 147 151 155 159 163 167 171 175 179 183 187 191 195 ...
%!        199 203 207 211 216 220 224 228 232 236 240 244 248 252]) = ...
%!   [2 27 51 144 132 108 43 29 22 21 22 20 10 16 17 12 13 14 12 13 ...
%!    15 25 19 20 23 37 23 65 92 84 87 54 50 54 33 73 76 64 57 58 47 ...
%!    48 30 27 22 20 20 11 12 12 11 7 17 31 37 31];
%! assert (otsuthresh (histo), 114.5/255)

%!test
%! I = max (phantom (), 0);
%! H = imhist (I);
%! assert (otsuthresh (H), 178/255)
%! assert (otsuthresh (H'), 178/255)
%! H = imhist (I, 10);
%! assert (otsuthresh (H), 170/255)

%!assert (otsuthresh (100), 0)
%!assert (otsuthresh (zeros (256, 1)), 0)
%!assert (otsuthresh (zeros (5, 1)), 0)

%!assert (otsuthresh (uint8 ([10 20 30])), 0.5)
%!assert (otsuthresh (int32 ([100 200 300])), 0.5)
%!assert (otsuthresh (int32 ([100 200])), 0)
%!assert (otsuthresh (single ([10 20 30 40])), 1/3);
%!assert (otsuthresh (uint16 ([10 20 30 40 50 60 70 80 90 100])), 5/9)
%!assert (otsuthresh (int16 ([10 20 30 40 50 60 70 80 90 100])), 5/9)
%!assert (otsuthresh (int16 (1:255)), 156/254)
%!assert (otsuthresh (int16 (1:1023)), 631/1022)
%!assert (otsuthresh (int8 (1:1023)), 541/1022)

%!test
%! warning ("off", "Octave:data-file-in-path", "local");
%! S = load ("penny.mat");
%! h = imhist (uint8 (S.P));
%! assert (otsuthresh (h), 94/255);

%!test
%! I = max (phantom (), 0);
%! h = imhist (I, 5);
%! assert (otsuthresh (h), 0.625);

%!error id=Octave:invalid-fun-call  otsuthresh ()
%!error id=Octave:invalid-fun-call  otsuthresh (ones (10), 5)
%!error <HIST must be a vector of non-negative integers> otsuthresh ([])
%!error <HIST must be a vector of non-negative integers> otsuthresh ([Inf 10])
%!error <HIST must be a vector of non-negative integers> otsuthresh ([10 NA])
%!error <HIST must be a vector of non-negative integers> otsuthresh ([10 NaN])
%!error <HIST must be a vector of non-negative integers> otsuthresh (zeros (5))
%!error <HIST must be a vector of non-negative integers> otsuthresh ([10 -10])
%!error <HIST must be a vector of non-negative integers> otsuthresh ("foo")


%!demo
%! I = max (phantom (), 0);
%! figure; imshow (I);
%! title ("Original image");
%! h = imhist (I);
%! t = otsuthresh (h);
%! J = im2bw (I);
%! figure; imshow (J);
%! title_line = sprintf ("Black and white image after thresholding, t=%g",
%!                       t*255);
%! title (title_line);

%!demo
%! warning ("off", "Octave:data-file-in-path", "local");
%! S = load ("penny.mat");
%! I = uint8 (S.P);
%! figure; imshow (I);
%! title ("Original penny image");
%! h = imhist (I);
%! t = otsuthresh (h);
%! J = im2bw (I);
%! figure; imshow (J);
%! title_line = sprintf ("Black and white penny image after thresholding, t=%g",
%!                       t*255);
%! title (title_line);
%! I = 255 - I;
%! figure; imshow(I);
%! title ("Negative penny image");
%! h = imhist (I);
%! t = otsuthresh (h);
%! J = im2bw (I);
%! figure; imshow (J);
%! title_line = sprintf ("Black and white negative penny image after thresholding, t=%g",
%!                       t*255);
%! title (title_line);
