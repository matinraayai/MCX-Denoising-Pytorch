## Copyright (C) 2014 Carnë Draug <carandraug@octave.org>
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

## -*- texinfo -*-
## @deftypefn  {Function File} {[@var{peaksnr}] =} psnr (@var{A}, @var{ref})
## @deftypefnx {Function File} {[@var{peaksnr}] =} psnr (@var{A}, @var{ref}, @var{peak})
## @deftypefnx {Function File} {[@var{peaksnr}, @var{snr}] =} psnr (@dots{})
## Compute peak signal-to-noise ratio.
##
## Computes the peak signal-to-noise ratio for image @var{A}, using
## @var{ref} as reference.  @var{A} and @var{ref} must be of same size
## and class.
##
## The optional value @var{peak} defines the highest value for the image
## and its default is class dependent (see @code{getrangefromclass}).
##
## The second output @var{snr} is the simple signal-to-noise ratio.
##
## @seealso{immse}
## @end deftypefn

function [peaksnr, snr] = psnr (A, ref, peak)

  if (nargin < 2 || nargin > 3)
    print_usage ();
  elseif (! size_equal (A, ref))
    error ("psnr: A and REF must be of same size");
  elseif (! strcmp (class (A), class (ref)))
    error ("psnr: A and REF must have same class");
  end

  if (nargin < 3)
    peak = getrangefromclass (A)(2);
  elseif (! isscalar (peak))
    error ("psnr: PEAK must be a scalar value");
  endif

  ## simpler way to keep single precision if they are single
  if (isinteger (A))
    A   = double (A);
    ref = double (ref);
  endif

  mse = immse (A, ref);

  peaksnr = 10 * log10 ((peak.^2) / mse);
  if (nargout > 1)
    snr = 10 * log10 ((sumsq (A(:)) / numel (A)) / mse);
  endif

endfunction

