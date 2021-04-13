## Copyright (C) 2018 Ricardo Fantin da Costa <ricardofantin@gmail.com>
##
## This program is free software; you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
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
## @deftypefn {} {} colorangle (@var{rgb1}, @var{rgb2})
## Compute angle between RGB colors in degrees.
##
## Colors are represented as 3 element row vectors, for their RGB
## values. The angle between @var{rgb1} and @var{rgb2} is defined as:
##
## @tex
## $$
## cos (ANGLE) = \frac{RGB1 \cdot RGB2}{|RGB1| |RGB2|}
## $$
## @end tex
## @ifnottex
## @example
## @group
##                    dot (@var{rgb1}, @var{rgb2})
## cos (@var{angle}) = ---------------------------
##                norm (@var{rgb1}) * norm (@var{rgb2})
## @end group
## @end example
## @end ifnottex
##
## This is a binary operator so standard automatic broadcasting rules
## apply.
##
## @end deftypefn

## Author: Ricardo Fantin da Costa
## Created: 2018-03-26

function angles = colorangle (rgb1, rgb2)
  if (nargin != 2)
    print_usage ();
  endif

  rgb1 = check_rgb (rgb1, "RGB1");
  rgb2 = check_rgb (rgb2, "RGB2");
  if (rows (rgb1) != rows (rgb2) && (rows (rgb1) != 1 && rows (rgb2) != 1))
    error ("Octave:invalid-input-arg",
           "colorangle: RGB1 and RGB2 must have one or same number of colors");
  endif

  norm1 = sqrt (sumsq (rgb1, 2));
  norm2 = sqrt (sumsq (rgb2, 2));

  ## Would be nice if dot() had automatic broadcasting, see
  ## https://savannah.gnu.org/bugs/index.php?55077. In the mean time,
  ## we do this.
  if (rows (rgb1) == rows (rgb2))
    dot_products = dot (rgb1, rgb2, 2);
  elseif (rows (rgb1) > rows (rgb2))
    dot_products = rgb1 * rgb2.';
  else
    dot_products = rgb2 * rgb1.';
  endif
  warning ("off", "Octave:divide-by-zero", "local");
  angles = rad2deg (acos (dot_products ./ (norm1 .* norm2)));

  ## For Matlab compatibility, return 0 instead of NaN for this cases.
  angles(norm1 == 0 & norm2 == 0) = 0;

  ## Complex values may come out of acos.  This will happen if acos
  ## input is larger than 1, which may happen due to floating point
  ## error, as in `colorangle ([1 1 1], [1 1 1])`
  angles = real (angles);
endfunction

function rgb = check_rgb (rgb, name)
  validateattributes (rgb, {"numeric"}, {"real"}, "colorangle", name);
  if (numel (rgb) == 3)
    ## For Matlab compatibility, if this is a single rgb color, accept
    ## a vector in any dimension.
    rgb = rgb(:).';
  elseif (columns (rgb) != 3)
    error ("Octave:invalid-input-arg",
           "colorangle: %s must be a 3 element or Nx3 array", name);
  endif
endfunction

%!error id=Octave:invalid-fun-call colorangle ()
%!error id=Octave:invalid-fun-call colorangle (1, 2, 3)
%!error <RGB1 must be a 3 element or Nx3 array> colorangle (2, 3)
%!error <RGB1 must be a 3 element or Nx3 array> colorangle ([1, 2], [3, 4])
%!error id=Octave:expected-real colorangle ([1, 2, 3j], [4, 5, 6])
%!error id=Octave:expected-real colorangle ([1, 2, 3], [4j, 5, 6])
%!error id=Octave:invalid-type colorangle ("abc", "def")

%!test
%! assert (colorangle ([0 0 0], [0 1 0]), NaN)
%! assert (colorangle ([0 0 0], [0 1 1]), NaN)
%! assert (colorangle ([0 1 0], [0 0 0]), NaN)
%! assert (colorangle ([1 1 0], [0 0 0]), NaN)
%! assert (colorangle ([1 1 1], [1 1 1]), 0)

## This is for Matlab compatibility.  If one colour is [0 0 0], then
## it's at the origin and there's no angle to the other colour. Both
## Octave and Matlab return NaN in this case.  The thing is what to do
## when both colours are [0 0 0].  There's no angle to measure, hence
## NaN, but they're at the same position hence zero.
%!assert (colorangle ([0 0 0], [0 0 0]), 0)

%!assert (colorangle ([1 0 0], [-1 0 0]), 180)
%!assert (colorangle ([0 0 1], [1 0 0]), 90)
%!assert (colorangle ([0; 0; 1], [1 0 0]), 90)
%!assert (colorangle ([0, 0, 1], [1; 0; 0]), 90)

%!assert (colorangle ([0.5 0.61237 -0.61237], [0.86603 0.35355 -0.35355]), 30.000270917, 1e-4)
%!assert (colorangle ([0.1582055390, 0.2722362096, 0.1620813305], [0.0717 0.1472 0.0975]), 5.09209927, 1e-6)
%!assert (colorangle ([0.0659838500, 0.1261619536, 0.0690643667], [0.0717 0.1472 0.0975]), 5.10358588, 1e-6)
%!assert (colorangle ([0.436871170, 0.7794672250, 0.4489702582], [0.0717 0.1472 0.0975]), 5.01339769, 1e-6)

%!test
%! a = [1 0 0];
%! b = [1 1 0];
%! expected = colorangle (a, b);
%! assert (colorangle (a.', b.'), expected)
%! assert (colorangle (a, b.'), expected)
%! assert (colorangle (a.', b), expected)
%! assert (colorangle (vec (a, 3), b.'), expected)

%!assert (colorangle ([1 0 0; 0 1 1], [1 1 1; 2 3 4]),
%!        [colorangle([1 0 0], [1 1 1]); colorangle([0 1 1], [2 3 4])])

%!test
%! a = [1 0 0; 0.5 1 0; 0 1 1; 1 1 1];
%! b = [0 1 0];
%! expected = zeros (4, 1);
%! for i = 1:4
%!   expected(i) = colorangle (a(i,:), b);
%! endfor
%! assert (colorangle (a, b), expected)
%! assert (colorangle (b, a), expected)

%!xtest
%! a = [1 2 3];
%! b = [2 3 4];
%! c = [5 6 7];
%! d = [3 1 1];
%!
%! ac = colorangle (c, a);
%! bc = colorangle (b, c);
%! ad = colorangle (a, d);
%! bd = colorangle (b, d);
%!
%! assert (colorangle (a, cat (3, c, d)),
%!         cat (3, [ac ad]))
%!
%! assert (colorangle (cat (3, a, b), cat (3, c, d)),
%!         cat (3, [ac cd]))
%!
%! assert (colorangle (cat (1, a, b), cat (3, c, d)),
%!         reshape ([ac bc ad bd], [2 2]))
