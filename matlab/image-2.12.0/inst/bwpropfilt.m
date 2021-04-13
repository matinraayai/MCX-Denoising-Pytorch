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
## @deftypefn  {Function File} {} bwpropfilt (@var{bw}, @var{attrib})
## @deftypefnx {Function File} {} bwpropfilt (@var{bw}, @var{I}, @var{attrib})
## @deftypefnx {Function File} {} bwpropfilt (@dots{}, @var{range})
## @deftypefnx {Function File} {} bwpropfilt (@dots{}, @var{n})
## @deftypefnx {Function File} {} bwpropfilt (@dots{}, @var{n}, @var{keep})
## @deftypefnx {Function File} {} bwpropfilt (@dots{}, @dots{}, @var{conn})
## Filter objects from image based on their properties.
##
## Returns a logical matrix with the objects of @var{bw} filtered based
## on the specific property @var{attrib}.  The possible values for @var{attrib}
## are all the properties from @command{regionprops} that return a scalar
## value, e.g., Area, Extent, and MaxIntensity, but not PixelValues, basic, and
## BoundingBox.  For certain attributes, such as MaxIntensity and
## WeightedCentroid, the grayscale image @var{I} must also be specified.
##
## To filter objects with a value on a specific interval, @var{range} must be
## a two-element vector with the interval @code{[@var{low} @var{high}]}
## (values are inclusive).
##
## Alternatively, a scalar @var{n} will select the objects with the N highest
## values.  The @var{keep} option defaults to @qcode{"largest"} but can also
## be set to @qcode{"smallest"} to select the N objects with lower values.
##
## The last optional argument, @var{conn}, can be a connectivity matrix, or
## the number of elements connected to the center (see @command{conndef}).
##
## @seealso{bwareaopen, bwareafilt, bwlabel, bwlabeln, bwconncomp, regionprops}
## @end deftypefn

function bwfiltered = bwpropfilt (bw, varargin)
  if (nargin < 3 || nargin > 6)
    print_usage ();
  endif

  if (ischar (varargin{1}))
    no_gray  = true;
    attrib   = varargin{1};
    next_idx = 2;
  else
    no_gray  = false;
    img      = varargin{1};
    attrib   = varargin{2};
    next_idx = 3;
  endif

  valid_nargin = @(x) numel (varargin) >= x;

  if (isscalar (varargin{next_idx}))
    ## Get the N largest or smallest
    in_range = false;
    n_keep = varargin{next_idx};
    next_idx++;

    if (valid_nargin (next_idx) && ischar (varargin{next_idx}))
      keep = tolower (varargin{next_idx});
      if (! any (strcmpi (keep, {"largest", "smallest"})))
        error ("bwpropfilt: KEEP must be `largest' or `smallest'");
      endif
      next_idx++;
    else
      keep = "largest";
    endif

  elseif (numel (varargin{next_idx}) == 2)
    in_range = true;
    range = varargin{next_idx};
    next_idx++;
  else
    error ("bwpropfilt: N and RANGE must have 1 or 2 elements respectively");
  endif

  if (valid_nargin (next_idx))
    conn = conndef (varargin{next_idx++});

    if (valid_nargin (next_idx))
      print_usage ();
    endif
  else
    ## Non-documented default
    conn = conndef (ndims (bw), "maximal");
  endif

  cc = bwconncomp (bw, conn);
  if (no_gray)
    stats = regionprops (cc, {"PixelIdxList", attrib});
  else
    stats = regionprops (cc, img, {"PixelIdxList", attrib});
  endif

  n_objs  = numel (stats);
  idxs    = {stats.PixelIdxList};
  attribs = [stats.(attrib)];

  if (in_range)
    filtered_idxs = idxs(attribs >= range(1) & attribs <= range(2));
  else
    [~, sorted_idxs] = sort (attribs, "descend");
    switch (keep)
      case "largest",
        filtered_idxs = idxs(sorted_idxs(1:min (n_keep, n_objs)));
      case "smallest",
        filtered_idxs = idxs(sorted_idxs(max (1, n_objs - n_keep +1):end));
    endswitch
  endif

  bwfiltered = false (size (bw));
  bwfiltered(cat (1, filtered_idxs{:})(:)) = true;

endfunction

