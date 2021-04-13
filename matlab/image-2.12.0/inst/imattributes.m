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
## <http:##www.gnu.org/licenses/>.

## -*- texinfo -*-
## @deftypefn  {Function File} {} imattributes ()
## @deftypefnx {Function File} {} imattributes (@var{himage})
## Get information about image attributes.
##
## Return attributes for the image in the current figure or in the image
## handle @var{himage}.  Returns a struct with the fields:
##
## @table @asis
## @item @qcode{"Width"}
## Number of columns.
##
## @item @qcode{"Height"}
## Number of rows.
##
## @item @qcode{"Class"}
## Note that some classes are converted to double for display.
##
## @item @qcode{"Image type"}
## One of @qcode{"binary"}, @qcode{"truecolor"}, @qcode{"intensity"}, or
## @qcode{"indexed"}.
##
## @item  @qcode{"Minimum intensity"}
## @itemx @qcode{"Maximum intensity"}
## These values are not returned for images of type @qcode{"truecolor"}
## and @qcode{"binary"}.
##
## For indexed images, the returned values are the lowest and highest index
## for the colormap, @emph{not} the used index for the lowest or highest
## intensity or their values.  This weird behaviour is kept for Matlab
## compatibility.
## @end table
##
## This function is meant to be used in an interactive session, and not
## programatically.  The properties of an image should be measured from the
## image variable itself not from the figure object.  In addition this
## function is purposely Matlab incompatible on their return value which
## returns a cell array of strings which is only useful for display.
##
## @end deftypefn

function attr = imattributes (imgh = gcf ())

  if (nargin > 1)
    print_usage ();
  elseif (isa (imgh, "imagemodel"))
    ## FIXME we don't even have a imagemodel class yet but when we do, this
    ##       is already here
    error ("imattributes: support for imagemodel objects not yet implemented");
  endif

  while (! isempty (get (imgh, "children")))
    imgh = get (imgh, "children");
  endwhile
  cdata = get (imgh, "cdata");
  cdatamapping = get (imgh, "cdatamapping");

  if (isbool (cdata))
    img_type = "binary";
  elseif (ndims (cdata) == 3)
    img_type = "truecolor";
  elseif (strcmpi (cdatamapping, "direct"))
    img_type = "indexed";
  else
    img_type = "intensity";
  endif

  ## Implementation note: this function returns a struct while Matlab returns
  ## a cell array of strings (even for the numeric values).  It is completely
  ## useless in programs, so I can only assume it is meant to be used
  ## interactively.  If so, a cell array is useless for us because Octave does
  ## not display cell arrays columns aligned but a struct looks good.

  attr = struct (
    "Width (columns)", columns (cdata),
    "Height (rows)", rows (cdata),
    "Class", class (cdata),
    "Image type", img_type
  );

  ## Matlab compatibility: for indexed images, we still give the lowest
  ## and highest index to the colormap.
  if (! any (strcmp (img_type, {"binary", "truecolor"})))
    attr = setfield (attr, "Minimum intensity", min (cdata(:)));
    attr = setfield (attr, "Maximum intensity", max (cdata(:)));
  endif

endfunction

%!shared x, map, img, rgb, bw
%! [x, map] = imread ("default.img");
%! rgb = ind2rgb (x, map);
%! img = ind2gray (x, map);
%! bw = im2bw (img);

%!test
%! h = imshow (img);
%! a = imattributes (h);
%! assert ([a.("Height (rows)") a.("Width (columns)")], [53 40]);
%! assert (a.Class, "uint8");
%! assert (a.("Image type"), "intensity");
%! assert (a.("Minimum intensity"), uint8 (28));
%! assert (a.("Maximum intensity"), uint8 (250));

## FIXME this is a bug upstream, the original class is not always preserved
%!xtest
%! h = imshow (rgb);
%! a = imattributes (h);
%! assert ([a.("Height (rows)") a.("Width (columns)")], [53 40]);
%! assert (a.Class, "uint8");
%! assert (a.("Image type"), "truecolor");
%! assert (isfield (a, "Minimum intensity"), false);
%! assert (isfield (a, "Maximum intensity"), false);

%!test
%! h = imshow (bw);
%! a = imattributes (h);
%! assert ([a.("Height (rows)") a.("Width (columns)")], [53 40]);
%! assert (a.Class, "logical");
%! assert (a.("Image type"), "binary");
%! assert (isfield (a, "Minimum intensity"), false);
%! assert (isfield (a, "Maximum intensity"), false);

%!test
%! h = imshow (x, map);
%! a = imattributes (h);
%! assert ([a.("Height (rows)") a.("Width (columns)")], [53 40]);
%! assert (a.Class, "uint8");
%! assert (a.("Image type"), "indexed");
%! assert (a.("Minimum intensity"), uint8 (0));
%! assert (a.("Maximum intensity"), uint8 (55));

%!test
%! h = imshow (img);
%! a1 = imattributes ();
%! a2 = imattributes (h);
%! assert (a1, a2);

