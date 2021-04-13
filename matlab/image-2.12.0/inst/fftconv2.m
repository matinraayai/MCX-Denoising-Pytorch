## Copyright (C) 2013-2015 Carnë Draug <carandraug@octave.org>
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
## <http://www.gnu.org/licenses/>.
##
## This file incorporates work covered by the following copyright and
## permission notice:
##
## Copyright (C) 2004 Stefan van der Walt <stefan@sun.ac.za>
## All rights reserved.
##
## Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are met:
##
##     1 Redistributions of source code must retain the above copyright notice,
##       this list of conditions and the following disclaimer.
##     2 Redistributions in binary form must reproduce the above copyright
##       notice, this list of conditions and the following disclaimer in the
##       documentation and/or other materials provided with the distribution.
##
## THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ''AS IS''
## AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
## IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
## ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE FOR
## ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
## DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
## SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
## CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
## OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
## OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## -*- texinfo -*-
## @deftypefn  {Function File} {} fftconv2 (@var{a}, @var{b})
## @deftypefnx {Function File} {} fftconv2 (@var{v1}, @var{v2}, @var{a})
## @deftypefnx {Function File} {} fftconv2 (@dots{}, @var{shape})
## Convolve 2 dimensional signals using the FFT.
##
## This method is faster but less accurate than @var{conv2} for large @var{a}
## and @var{b}.  It also uses more memory. A small complex component will be 
## introduced even if both @var{a} and @var{b} are real.
## @seealso{conv2, fftconv, fft, ifft}
## @end deftypefn

function X = fftconv2 (varargin)
  if (nargin < 2)
    print_usage ();
  endif

  nargs = nargin; # nargin minus the shape option
  if (ischar (varargin{end}))
    shape = varargin{end};
    nargs--;
  else
    shape = "full";
  endif

  rowcolumn = false;
  if (nargs == 2)
    ## usage: fftconv2(a, b[, shape])
    a = varargin{1};
    b = varargin{2};
  elseif (nargs == 3)
    ## usage: fftconv2 (v1, v2, a[, shape])
    rowcolumn = true;
    if (! isnumeric (varargin{3}) && ! islogical (varargin{3}))
      error ("fftconv2: A must be a numeric or logical array");
    endif
    v1      = vec (varargin{1});
    v2      = vec (varargin{2}, 2);
    orig_a  = varargin{3};
  else
    print_usage ();
  endif

  if (rowcolumn)
    a = fftconv2 (orig_a, v2);
    b = v1;
  endif

  ra = rows(a);
  ca = columns(a);
  rb = rows(b);
  cb = columns(b);

  A = fft2 (padarray (a, [rb-1 cb-1], "post"));
  B = fft2 (padarray (b, [ra-1 ca-1], "post"));

  X = ifft2 (A.*B);

  if (rowcolumn)
    rb = rows (v1);
    ra = rows (orig_a);
    cb = columns (v2);
    ca = columns (orig_a);
  endif

  switch (tolower (shape))
    case "full",
      ## do nothing
    case "same",
      r_top = ceil ((rb + 1) / 2);
      c_top = ceil ((cb + 1) / 2);
      X = X(r_top:r_top + ra - 1, c_top:c_top + ca - 1);
    case "valid",
      X = X(rb:ra, cb:ca);
    otherwise
      error ("fftconv2: unknown convolution SHAPE `%s'", shape);
  endswitch

endfunction

## usage: fftconv2(a,b,[, shape])
%!test
%! a = repmat (1:10, 5);
%! b = repmat (10:-1:3, 7);
%! assert (fftconv2 (a, b), conv2 (a, b), 1.8e4*eps)
%! assert (fftconv2 (b, a), conv2 (b, a), 1.8e4*eps)
%! assert (fftconv2 (a, b, "full"), conv2 (a, b, "full"), 1.8e4*eps)
%! assert (fftconv2 (b, a, "full"), conv2 (b, a, "full"), 1.8e4*eps)
%! assert (fftconv2 (a, b, "same"), conv2 (a, b, "same"), 1.8e4*eps)
%! assert (fftconv2 (b, a, "same"), conv2 (b, a, "same"), 1.8e4*eps)
%! assert (isempty (fftconv2 (a, b, "valid")));
%! assert (fftconv2 (b, a, "valid"),  conv2 (b, a, "valid"), 1e4*eps)

## usage: fftconv2(v1, v2, a[, shape])
%!test
%! x = 1:4;
%! y = 4:-1:1;
%! a = repmat(1:10, 5);
%! assert (fftconv2 (x, y, a),          conv2 (x, y, a),          1e4*eps)
%! assert (fftconv2 (x, y, a, "full"),  conv2 (x, y, a, "full"),  1e4*eps)
%! assert (fftconv2 (x, y, a, "same"),  conv2 (x, y, a, "same"),  1e4*eps)
%! assert (fftconv2 (x, y, a, "valid"), conv2 (x, y, a, "valid"), 1e4*eps)

%!demo
%! ## Draw a cross
%! z = zeros (101, 101);
%! z(50, :) = 1;
%! z(:, 50) = 1;
%! subplot (1, 3, 1)
%! imshow (z);
%! title ("Original thin cross")
%!
%! ## Draw a sinc blob
%! b = getheight (strel ("ball", 10, 1));
%! subplot (1, 3, 2)
%! imshow (b);
%! title ("Sync blob")
%!
%! ## Convolve the cross with the blob
%! fc = real (fftconv2 (z, b, "same"));
%! subplot (1, 3, 3)
%! imshow (fc, [min(fc(:)) max(fc(:))])
%! title ("Convolution in the frequency domain")
