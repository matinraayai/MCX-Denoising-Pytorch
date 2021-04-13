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
## @deftypefn  {Function File} {} labelmatrix (@var{cc})
## Create labelled matrix from bwconncomp structure.
##
## Uses the structure as returned by the @code{bwconncomp} function to create
## a label matrix, where each individual object is assigned a positive number.
## A value of zero corresponds to the background.
##
## The class of the output matrix is dependent on the number of objects, being
## uint, uint16, uint32, or double, whichever is enough.
##
## @seealso{bwconncomp, bwlabel, bwlabeln, label2rgb, rgb2label}
## @end deftypefn


function labelled = labelmatrix (cc)

  if (nargin != 1)
    print_usage ();
  elseif (! isstruct (cc) && ! all (isfield (cc, {"Connectivity", "ImageSize",
                                                  "NumObjects", "PixelIdxList"})))
    error ("labelmatrix: CC must be a struct as returned by bwconncomp");
  endif

  n_obj = cc.NumObjects;
  if     (n_obj <        256), cl = "uint8";
  elseif (n_obj <      65536), cl = "uint16";
  elseif (n_obj < 4294967296), cl = "uint32";
  else,                        cl = "double";
  endif

  labels = repelems (1:n_obj, [1:n_obj; cellfun("numel", cc.PixelIdxList)]);
  ind = cell2mat (cc.PixelIdxList');

  labelled = zeros (cc.ImageSize, cl);
  labelled(ind) = labels;

endfunction

%!test
%! cc = struct ();
%! cc.Connectivity = 8;
%! cc.ImageSize = [7 7];
%! cc.NumObjects = 4;
%! cc.PixelIdxList = {[1;2], [5;7;12;13;14], [22;23], [26;32;33;36;37;38]};
%!
%! l = uint8 ([
%!   1  0  0  3  0  4  0
%!   1  0  0  3  0  4  0
%!   0  0  0  0  0  4  0
%!   0  0  0  0  4  0  0
%!   2  2  0  4  4  0  0
%!   0  2  0  0  0  0  0
%!   2  2  0  0  0  0  0
%! ]);
%! assert (labelmatrix (cc), l)
