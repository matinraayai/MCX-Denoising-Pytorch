## Copyright (C) 2006 Søren Hauberg <soren@hauberg.org>
## Copyright (C) 2013 Carnë Draug <carandraug@octave.org>
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
## @deftypefn {Function File} {} label2rgb (@var{L})
## @deftypefnx{Function File} {} label2rgb (@var{L}, @var{cmap})
## @deftypefnx{Function File} {} label2rgb (@var{L}, @var{cmap}, @var{background})
## @deftypefnx{Function File} {} label2rgb (@var{L}, @var{cmap}, @var{background}, @var{order})
## Convert labeled image into RGB.
##
## The labeled image @var{L} is converted into an RGB image using the
## colormap @var{cmap}.  The label number of each region is used to select
## the color from @var{cmap} which can be specified as:
##
## @itemize @bullet
## @item @var{N}-by-3 colormap matrix where N must be larger than or equal
## to the highest label number;
## @item name of a function that returns a colormap;
## @item handle for a function that returns a colormap (defaults to @code{jet}).
## @end itemize
##
## In a labeled image, zero valued pixels are considered background and
## are colored according to the color @var{background}.  It can be specified
## as an RGB triplet values (3 element vector of values between 0 and 1), or
## by name:
##
## @itemize @bullet
## @item @qcode{"w"} or @qcode{"white"} (default)
## @item @qcode{"b"} or @qcode{"blue"}.
## @item @qcode{"c"} or @qcode{"cyan"}.
## @item @qcode{"g"} or @qcode{"green"}.
## @item @qcode{"k"} or @qcode{"black"}.
## @item @qcode{"m"} or @qcode{"magenta"}.
## @item @qcode{"r"} or @qcode{"red"}.
## @item @qcode{"y"} or @qcode{"yellow"}.
## @end itemize
##
## The option @var{order} must be a string with values @qcode{"shuffle"} or
## @qcode{"noshuffle"} (default). If shuffled, the colors in @var{cmap} are
## permuted randomly before the image conversion.
##
## The output RGB image is always of class uint8.
##
## @seealso{bwconncomp, bwlabel, colormap, ind2rgb}
## @end deftypefn

function rgb = label2rgb (L, cmap = @jet, background = "w", order = "noshuffle")

  if (nargin < 1 || nargin > 4)
    print_usage ();
  elseif (! isimage (L) || ndims (L) > 4 || size (L, 3) != 1 ||
          any (L(:) != fix (L(:))) || any (L(:) < 0))
    error ("label2rgb: L must be a labelled image");
  elseif (! ischar (cmap) && ! isa (cmap, "function_handle") && ! iscolormap (cmap))
    error ("label2rgb: CMAP must be a colormap, colormap name, or colormap function");
  elseif (! ischar (background) && ! iscolormap (background))
    error("label2rgb: BACKGROUND must be a colorname or a RGB triplet");
  elseif (! any (strcmpi (order, {"noshuffle", "shuffle"})))
    error("label2rgb: ORDER must be either 'noshuffle' or 'shuffle'");
  endif

  ## Convert map to a matrix if needed
  num_objects = max (L(:));
  if (ischar (cmap) || isa (cmap, "function_handle"))
    ## cast to double because of bug #44070
    cmap = feval (cmap, double (num_objects));
  endif

  num_colors = rows (cmap);
  if (num_objects > num_colors)
    error ("label2rgb: CMAP has not enough colors (%i) for all objects (%i) in L",
           num_colors, num_objects);
  endif

  background = handle_colorspec ("label2rgb", background);

  ## Should we shuffle the colormap?
  if (strcmpi (order, "shuffle"))
    ## Matlab does the shuffling "pseudorandomly". We don't know how it
    ## actually does the shuffling since it is not documented but using
    ## the same labeled image and colormap, Matlab always returns the same.
    cmap = cmap(randperm (num_colors), :);
  endif

  ## Check if the background color is in the colormap
  idx = find (ismember (cmap, background, "rows"));
  if (! isempty (idx))
    if (isscalar (idx))
      warning ("label2rgb: region %i has the same color as background", idx);
    else
      idx_list = sprintf ("%i, ", idx(1:end-1));
      idx_list = sprintf ("%s, and %i", idx_list, idx(end));
      warning ("label2rgb: regions %s, have the same color as background",
               idx_list);
    endif
  endif

  ## We will use ind2rgb for the conversion. An indexed image is interpreted
  ## differently depending if it's an integer or floating point image. We make
  ## sure we pass an integer image where value of zero is the color in the
  ## first row of the colormap (if it was a floating point image, the image
  ## could not have zero values, and a value of 1 is the color in the first
  ## row of the colormap).
  if (! isinteger (L))
    if     (num_objects <= intmax ("uint8")),  L = uint8  (L);
    elseif (num_objects <= intmax ("uint16")), L = uint16 (L);
    elseif (num_objects <= intmax ("uint32")), L = uint32 (L);
    else,                                      L = uint64 (L);
    endif
  endif

  ## Insert the background color at the head of the colormap
  rgb  = ind2rgb (L, [background; cmap]);
  rgb  = im2uint8 (rgb);
endfunction

%!function map = test_colormap ()
%!  map = [0 0 0; 0.5 0.5 0.5; 0.125 0.125 0.125];
%!endfunction

%!shared in, out, cmap
%! in  = [  0    1    1    0    2    2    0    3    3
%!          0    1    1    0    2    2    0    3    3];
%!
%! out = [255    0    0  255  128  128  255   32   32
%!        255    0    0  255  128  128  255   32   32];
%! out(:,:,2) = out(:,:,3) = out(:,:,1);
%! out = uint8(out);
%!
%! cmap = [0 0 0; 0.5 0.5 0.5; 0.125 0.125 0.125];
%!assert (label2rgb (in, cmap),            out);
%!assert (label2rgb (uint8 (in), cmap),    out);
%!assert (label2rgb (in, "test_colormap"), out);
%!assert (label2rgb (in, @test_colormap),  out);
%!
%! out(find (in == 0)) = 0;
%!assert (label2rgb (in, cmap, "cyan"),    out);
%!assert (label2rgb (in, cmap, [0 1 1]),   out);
%!
%! in(1) = 10;
%!error label2rgb (in, cmap);
%!error label2rgb (in, cmap, 89);
%!error label2rgb (in, cmap, "g", "wrong");
