## Copyright (C) 2000 Paul Kienzle <pkienzle@users.sf.net>
## Copyright (C) 2004 Stefan van der Walt <stefan@sun.ac.za>
## Copyright (C) 2012 Carlo de Falco
## Copyright (C) 2012 Carnë Draug <carandraug@octave.org>
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
## @deftypefn  {Function File} {} imnoise (@var{A}, @var{type})
## @deftypefnx {Function File} {} imnoise (@dots{}, @var{options})
## Add noise to image.
##
## @end deftypefn
## @deftypefn {Function File} {} imnoise (@var{A}, "gaussian", @var{mean}, @var{variance})
## Additive gaussian noise with @var{mean} and @var{variance} defaulting to 0
## and 0.01.
##
## @end deftypefn
## @deftypefn {Function File} {} imnoise (@var{A}, "poisson")
## Creates poisson noise in the image using the intensity value of each pixel as
## mean.
##
## @end deftypefn
## @deftypefn {Function File} {} imnoise (@var{A}, "salt & pepper", @var{density})
## Create "salt and pepper"/"lost pixels" in @var{density}*100 percent of the
## image.  @var{density} defaults to 0.05.
##
## @end deftypefn
## @deftypefn {Function File} {} imnoise (@var{A}, "speckle", @var{variance})
## Multiplicative gaussian noise with @var{B} = @var{A} + @var{A} * noise with
## mean 0 and @var{variance} defaulting to 0.04.
##
## @seealso{rand, randn, randp}
## @end deftypefn

function A = imnoise (A, stype, a, b)
  ## we do not set defaults right at the start because they are different
  ## depending on the method used to generate noise

  if (nargin < 2 || nargin > 4)
    print_usage;
  elseif (! isimage (A))
    error ("imnoise: first argument must be an image.");
  elseif (! ischar (stype))
    error ("imnoise: second argument must be a string with name of noise type.");
  endif

  in_class  = class (A);
  fix_class = false;      # for cases when we need to use im2double

  switch (lower (stype))
    case "poisson"
      switch (in_class)
        case ("double")
          A = randp (A * 1e12) / 1e12;
        case ("single")
          A = single (randp (A * 1e6) / 1e6);
        case {"uint8", "uint16"}
          A = cast (randp (A), in_class);
        otherwise
          A = imnoise (im2double (A), "poisson");
          fix_class = true;
      endswitch

    case "gaussian"
      A         = im2double (A);
      fix_class = true;
      if (nargin < 3), a = 0.00; endif
      if (nargin < 4), b = 0.01; endif
      A = A + (a + randn (size (A)) * sqrt (b));
      ## Variance of Gaussian data with mean 0 is E[X^2]

    case {"salt & pepper", "salt and pepper"}
      if (nargin < 3), a = 0.05; endif
      noise = rand (size (A));
      if (isfloat (A))
        black = 0;
        white = 1;
      else
        black = intmin (in_class);
        white = intmax (in_class);
      endif
      A(noise <= a/2)   = black;
      A(noise >= 1-a/2) = white;

    case "speckle"
      A         = im2double (A);
      fix_class = true;
      if (nargin < 3), a = 0.04; endif
      A = A .* (1 + randn (size (A)) * sqrt (a));

    otherwise
      error ("imnoise: unknown or unimplemented type of noise `%s'", stype);
  endswitch

  if (fix_class)
    A = imcast (A, in_class);
  elseif (isfloat (A))
    ## this includes not even cases where the noise made it go outside of the
    ## [0 1] range, but also images that were already originally outside that
    ## range. This is by design and matlab compatibility. And we do this after
    ## fixing class because the imcast functions already take care of such
    ## adjustment
    A(A < 0) = cast (0, class (A));
    A(A > 1) = cast (1, class (A));
  endif

endfunction

%!assert(var(imnoise(ones(10)/2,'gaussian')(:)),0.01,0.005) # probabilistic
%!assert(length(find(imnoise(ones(10)/2,'salt & pepper')~=0.5)),5,10) # probabilistic
%!assert(var(imnoise(ones(10)/2,'speckle')(:)),0.01,0.005) # probabilistic

%!test
%! A = imnoise (.5 * ones (100), 'poisson');
%! assert (class (A), 'double')
%!test
%! A = imnoise (.5 * ones (100, 'single'), 'poisson');
%! assert (class (A), 'single')
%!test
%! A = imnoise (128 * ones (100, 'uint8'), 'poisson');
%! assert (class (A), 'uint8')
%!test
%! A = imnoise (256 * ones (100, 'uint16'), 'poisson');
%! assert (class (A), 'uint16')

%!demo
%!  A = imnoise (2^7 * ones (100, 'uint8'), 'poisson');
%!  subplot (2, 2, 1)
%!  imshow (A)
%!  title ('uint8 image with poisson noise')
%!  A = imnoise (2^15 * ones (100, 'uint16'), 'poisson');
%!  subplot (2, 2, 2)
%!  imshow (A)
%!  title ('uint16 image with poisson noise')
%!  A = imnoise (.5 * ones (100), 'poisson');
%!  subplot (2, 2, 3)
%!  imshow (A)
%!  title ('double image with poisson noise')
%!  A = imnoise (.5 * ones (100, 'single'), 'poisson');
%!  subplot (2, 2, 4)
%!  imshow (A)
%!  title ('single image with poisson noise')
