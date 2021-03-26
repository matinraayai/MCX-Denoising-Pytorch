## Copyright (C) 2015-2018 Carnë Draug <carandraug@octave.org>
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
## @deftypefn  {Function File} {} viscircles (@var{centers}, @var{radii})
## @deftypefnx {Function File} {} viscircles (@var{hax}, @var{centers}, @var{radii})
## @deftypefnx {Function File} {} viscircles (@dots{}, @var{property}, @var{value})
## @deftypefnx {Function File} {@var{h} =} viscircles (@dots{})
## Draw circles on figure.
##
## Circles are specified by a Nx2 matrix @var{centers} with x,y
## coordinates per row, and a N length vector @var{radii}.
##
## @example
## ## draw circles at [10 20] and [-10 -20] coordinates
## ## with radius of 10 and 20 respectively
## viscircles ([10 20; -10 -20], [10 20])
## @end example
##
## The appearance of the drawn circles can be configured with the
## following properties names:
##
## @table @asis
## @item @qcode{"Color"}
## The color of the circle.  Defaults to @qcode{"red"}. Can be defined
## via the color names or RGB triplets.  See the help text for
## @code{plot} for further details on specifying colors in figures.
##
## @item @qcode{"LineStyle"}
##
## The line style of the circle.  Defaults to @qcode{"-"} (solid
## line).  See the help text for @code{plot} for possible values.
##
## @item @qcode{"LineWidth"}
## The width of the circle line.  Defaults to 2.
##
## @item @qcode{"EnhanceVisibility"}
## Enhance visibility by drawing a white circle under the colored
## circle.  Must be a logical value.  Defaults to true.
##
## @end table
##
## @seealso{plot, line}
## @end deftypefn

function h = viscircles (varargin)

  if (nargin < 2)
    print_usage ();
  endif

  args_ind = 1;
  if (mod (nargin, 2) == 0)
    hax = gca ();
  else
    hax = varargin{args_ind++};
    if (! ishandle (hax))
      error ("viscircles: HAX is not an axes graphics handle");
    endif
  endif

  centers = varargin{args_ind++};
  radii   = varargin{args_ind++};
  if (columns (centers) != 2)
    error ("viscircles: CENTERS must be a Nx2 matrix");
  elseif (! isvector (radii))
    error ("viscircles: RADII must be a vector");
  elseif (rows (centers) != numel (radii))
    error ("viscircles: RADII length must be equal to the rows of CENTERS");
  endif

  p = inputParser ();
  p.FunctionName = "viscircles";

  ## Original version of viscircles had EdgeColor and
  ## DrawBackgroundCircle parameters.  They have been renamed Color
  ## and EnhanceVisibility on later Matlab versions but keep them for
  ## backwards compatibility.
  params = struct ("old_name", {"DrawBackgroundCircle"; "EdgeColor"},
                   "new_name", {"EnhanceVisibility"; "Color"},
                   "parser_args", {{true, @isbool};
                                   {"red", @(c) isfloat (c) || ischar (c)}});
  for idx = 1:numel(params)
    param = params(idx);
    p.addParamValue (param.old_name, param.parser_args{:});
    p.addParamValue (param.new_name, param.parser_args{:});
  endfor

  p.addParamValue ("LineStyle", "-", @ischar);
  p.addParamValue ("LineWidth", 2, @isnumeric);

  ## FIXME: we use "numel (varargin)" instead of end to work around
  ##        https://savannah.gnu.org/bugs/index.php?44779
  p.parse (varargin{args_ind:numel (varargin)});

  ## Results is write-protected but we may need to modify them
  options = p.Results;

  ## Check if the user used the old parameter names and remap to the
  ## new names.  Error if both are being set.
  for idx = 1:numel(params)
    param = params(idx);
    if (! any (strcmp (param.old_name, p.UsingDefaults)))
      if (! any (strcmp (param.new_name, p.UsingDefaults)))
        error ("viscircles: both '%s' (deprecated) and '%s' parameters set",
               param.old_name, param.new_name);
      endif
      options.(param.new_name) = p.Results.(param.old_name);
    endif
  endfor

  theta = linspace (0, 2*pi, 100);
  x = radii(:).' .* cos (theta(:)) .+ centers(:,1).';
  y = radii(:).' .* sin (theta(:)) .+ centers(:,2).';

  hold_was_on = ishold (hax);
  unwind_protect
    hold (hax, "on");
    h_tmp = hggroup (hax);

    if (options.EnhanceVisibility)
      line (hax, x, y, "Parent", h_tmp,
            "Color", "white", "LineStyle", "-",
            "LineWidth", options.LineWidth + 1);
    endif
    line (hax, x, y, "Parent", h_tmp,
          "Color", options.Color,
          "LineWidth", options.LineWidth,
          "LineStyle", options.LineStyle);

  unwind_protect_cleanup
    if (! hold_was_on)
      hold (hax, "off");
    endif
  end_unwind_protect

  if (nargout)
    h = h_tmp;
  endif
endfunction

%!demo
%! centers = randi ([0 100], 5, 2);
%! radii = randi ([10 100], 5, 1);
%! axis equal
%! viscircles (centers, radii,
%!             "Color", "magenta",
%!             "LineStyle", ":",
%!             "LineWidth", 5);
%! title ("5 random circles");
%! #----------------------------------------------
%! # the figure window shows 5 circles with random
%! # radii and positions

%!test # old undocumented property
%! h = viscircles ([0 0], 1, "EdgeColor", "black");
%! assert (get (get (h, "children")(1), "color"), [0 0 0])

%!test # old undocumented property
%! h = viscircles ([0 0], 1, "DrawBackgroundCircle", false);
%! assert (numel (get (h, "children")), 1)

%!error <both 'EdgeColor' \(deprecated\) and 'Color'> ...
%!      viscircles ([0 0], 1, "Color", "magenta", "EdgeColor", "black")

%!test
%! centers = randi ([0 100], 5, 2);
%! radii = randi ([0 100], 5, 1);
%! h = viscircles (centers, radii);
%! close;

%!test
%! centers = randi ([0 100], 5, 2);
%! radii = randi ([0 100], 5, 1);
%! figure ();
%! h = viscircles (gca (), centers, radii);
%! close;

%!test
%! centers = randi ([0 100], 5, 2);
%! radii = randi ([0 100], 5, 1);
%! h = viscircles (centers, radii, "Color", "magenta",
%!                 "LineStyle", ":", "LineWidth", 5);
%! close;

%!test
%! centers = randi ([0 100],5,2);
%! radii = randi ([0 100],5,1);
%! figure ();
%! h = viscircles (centers, radii, "Color", "magenta",
%!                 "LineStyle", ":", "LineWidth", 5);
%! close;
