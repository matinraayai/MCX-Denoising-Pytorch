## Copyright (C) 2013 Carnë Draug <carandraug+dev@gmail.com>
##
## This program is free software; you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program; if not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*-
## @deftypefn  {Function File} {} montage (@var{I})
## @deftypefnx {Function File} {} montage (@var{X}, @var{cmap})
## @deftypefnx {Function File} {} montage (@var{filenames})
## @deftypefnx {Function File} {} montage (@dots{}, @var{param1}, @var{value1}, @dots{})
## @deftypefnx {Function File} {@var{h} =} montage (@dots{})
## Create montage from multiple images.
##
## The created montage will be of a single large image built from the 4D matrix
## @var{I}.  @var{I} must be a MxNx1x@var{P} or MxNx3x@var{P} matrix for a
## grayscale and binary, or RGB image with @var{P} frames.
##
## Alternatively, @var{X} can be a MxNx1x@var{P} indexed image with @var{P}
## frames, with the colormap @var{cmap}, or a cell array of @var{filenames}
## for multiple images.
##
## @table @asis
## @item DisplayRange
## A vector with 2 or 0 elements setting the highest and lowest value for
## display range.  It is interpreted like the @var{limits} argument to
## @code{imshow}.  Of special significance is that an empty array uses the
## image minimum and maximum values for limits.  Defaults to the limits of
## the image data type, i.e., the range returned by @code{getrangefromclass}.
##
## @item Indices
## A vector with the image indices to be displayed.  Defaults to all images,
## i.e., @code{1:size (@var{I}, 4)}.
##
## @item Size
## Sets the montage layout size.  Must be a 2 element vector setting
## [@var{nRows} @var{nCols}].  A value of NaN will be adjusted to the required
## value to display all images.  If both values are NaN (default), it will
## find the most square layout capable of displaying all of the images.
##
## @item MarginColor
## Sets color for the margins between panels.  Defaults to white.  Must be a
## 1 or 3 element vector for grayscale or RGB images.
##
## @item MarginWidth
## Sets width for the margins between panels.  Defaults to 0 pixels.  Note that
## the margins are only between panels.
##
## @item BackgroundColor
## Sets the montage background color. Defaults to black.  Must be a
## 1 or 3 element vector for grayscale or RGB images.  This will only affect
## montages with more panels than images.
## @end table
##
## The optional return value H is a graphics handle to the created plot.
##
## @seealso{imshow, padarray, permute, reshape}
## @end deftypefn

function h = montage (images, varargin)

  if (nargin < 1)
    print_usage ();
  endif

  if (iscellstr (images))
    ## we are using cellfun instead of passing the "all" option
    ## to file_in_loadpath, so we know which of the figures was
    ## not found, and provide a more meaningful error message
    fullpaths = cellfun (@file_in_loadpath, images(:), "UniformOutput", false);
    lost      = cellfun (@isempty, fullpaths);
    if (any (lost))
      badpaths = strjoin (images(:)(lost), "\n");
      error ("montage: unable to find files:\n%s", badpaths);
    endif
    ## supporting grayscale, indexed, and truecolor images in
    ## the same list, complicates things a bit. Also, don't forget
    ## some images may be multipage
    infos = cellfun (@imfinfo, fullpaths, "UniformOutput", false);
    nImg  = sum (cellfun (@numel, infos));  # number of images

    ## in case of multipage images, different pages may have different sizes,
    ## so we must check the height and width of each page of each image
    height = infos{1}(1).Height;
    width  = infos{1}(1).Width;
    if (any (cellfun (@(x) any (arrayfun (@(y) y.Height != height || y.Width != width, x)), infos)))
      error ("montage: all images must have the same size.");
    endif

    ## To save on memory, we find the image with highest bitdepth, and
    ## create a matrix with the correct dimensions, where the images will
    ## be read into. We could read all the images and maps with a single
    ## cellfun call, and that would be a bit simpler than deducing
    ## from imfinfo but then we'd end up taking up the double of memory
    ## as we reorganize and convert the images

    ## a multipage image could have a mixture of colortypes, so we must
    ## check the type of every single one. If they are all grayscale, that's
    ## nice, it will be one dimension less, otherwise 3rd dimension is for
    ## rgb values. Also, any indexed image will be later converted to
    ## truecolor with rgb2ind so it will count as double
    if (any (cellfun (@(x) any (strcmp ({x(:).ColorType}, "indexed")), infos)))
      cl = "double";
      convt = @im2double;
    else
      maxbd = max (cellfun (@(x) max ([x(:).BitDepth]), infos));
      switch (maxbd)
        case {8}
          cl = "uint8";
          convt = @im2uint8;
        case {16}
          ## includes both uint18 and int16
          cl = "uint16";
          convt = @im2uint16;
        otherwise
          ## includes maxbd == 32 plus anything else. In case of anything
          ## unexpected, better play safe and use the one with more precision
          cl = "double";
          convt = @im2double;
      endswitch
    endif

    if (all (cellfun (@(x) all (strcmp ({x(:).ColorType}, "grayscale")), infos)))
      images = zeros (height, width, 1, nImg, cl);
    else
      images = zeros (height, width, 3, nImg, cl);
    endif

    nRead = 0; # count of pages already read
    for idx = 1:numel (infos)
      img_info   = infos{idx};
      nPages     = numel (img_info);
      nRead     += nPages;
      page_range = nRead+1-nPages:nRead;
      filepath = img_info(1).Filename;

      ## we won't be handling the alpha channel, but matlab doesn't either
      if (size (images, 3) == 1 || all (strcmp ({img_info(:).ColorType}, "truecolor")))
        ## sweet, no problems for sure
        [images(:,:,:,page_range), map] = imread (filepath, 1:nPages);
      else
        [tmp_img, map] = imread (filepath, 1:nPages);
        if (! isempty (map))
          ## an indexed image. According to TIFF 6 specs, an indexed
          ## multipage image can have different colormaps for each page.
          ## How does imread behave with such images? And do they actually
          ## exist out there?
          tmp_img = ind2rgb (ind, map)
        elseif (size (tmp_img, 3) == 1)
          ## must be a grayscale image, propagate values to all channels
          tmp_img = repmat (tmp_img, [1 1 3 nPages])
        else
          ## must be a truecolor image, do nothing
        endif
        images(:,:,:,page_range) = tmp_img;
      endif
    endfor

  ## we can't really distinguish between a multipage indexed and normal
  ## image. So we'll assume it's an indexed if there's a second argument
  ## that is a colormap
  elseif (isimage (images) && nargin > 1 && iscolormap (varargin{1}))
    images = ind2rgb (images, varargin{1});
    varargin(1) = [];
  elseif (isimage (images))
    ## all is nice
  else
    print_usage ();
  endif

  [height, width, channels, nImg] = size (images);

  p = inputParser ();
  p.FunctionName = "montage";

  ## FIXME: inputParser was first implemented in the general package in the
  ##        old @class type which allowed for a very similar interface to
  ##        Matlab.  classdef was implemented in the upcoming 4.0 release,
  ##        which enabled inputParser to be implemented exactly the same and
  ##        it is now part of Octave core.  To prevent issues while all this
  ##        versions are available, we check if the inputParser being used
  ##        is in a @inputParser directory.
  ##
  ##        Remove all this checks once the general package has been released
  ##        again without the @inputParser class

  im_type_range = getrangefromclass (images);
  if (strfind (which ("inputParser"), ["@inputParser" filesep "inputParser.m"]))
    p = p.addParamValue ("DisplayRange",    im_type_range, @(x) isnumeric (x) && any (numel (x) == [0 2]));
    p = p.addParamValue ("Indices",         1:nImg, @(x) isindex (x, nImg));
    p = p.addParamValue ("Size",            [NaN NaN], @(x) isnumeric (x) && numel (x) == 2);
    p = p.addParamValue ("MarginWidth",     0, @(x) isnumeric (x) && isscalar (x));
    p = p.addParamValue ("MarginColor",     im_type_range(2), @(x) isnumeric (x) && any (numel (x) == [1 3]));
    p = p.addParamValue ("BackgroundColor", im_type_range(1), @(x) isnumeric (x) && any (numel (x) == [1 3]));
    p = p.parse (varargin{:});
  else
    p.addParamValue ("DisplayRange",    im_type_range, @(x) isnumeric (x) && any (numel (x) == [0 2]));
    p.addParamValue ("Indices",         1:nImg, @(x) isindex (x, nImg));
    p.addParamValue ("Size",            [NaN NaN], @(x) isnumeric (x) && numel (x) == 2);
    p.addParamValue ("MarginWidth",     0, @(x) isnumeric (x) && isscalar (x));
    p.addParamValue ("MarginColor",     im_type_range(2), @(x) isnumeric (x) && any (numel (x) == [1 3]));
    p.addParamValue ("BackgroundColor", im_type_range(1), @(x) isnumeric (x) && any (numel (x) == [1 3]));
    p.parse (varargin{:});
  endif

  ## remove unnecessary images
  images = images(:,:,:,p.Results.Indices);
  nImg   = size (images, 4);

  ## 1) calculate layout of the montage
  nRows = p.Results.Size(1);
  nCols = p.Results.Size(2);
  if (isnan (nRows) && isnan (nCols))
    ## We must find the smallest layout that is most square. The most square
    ## are the ones with smallest difference between height and width. And to
    ## find the smallest layout for each number of columns, is the one that
    ## requires less rows. The smallest layout will be used as a mask to choose
    ## the minimum from hxW_diff
    v_heights = cumsum (linspace (height, nImg*height, nImg));
    v_widths  = cumsum (linspace (width,  nImg*width,  nImg));

    HxW_diff     = abs (v_heights' - v_widths);
    small_layout = ((1:nImg)' .* (1:nImg)) >= nImg;
    small_layout = logical (diff (padarray (small_layout, 1, 0, "pre")));
    HxW_diff(! small_layout) = Inf;
    ## When there is more than one layout that returns equal "squariness",
    ## we must pick the one with most columns.  This is because monitors
    ## have more horizontal space, so a wider image will be better.  Hence
    ## the "last".
    [nRows, nCols] = find (HxW_diff == min (HxW_diff(:)), 1, "last");

  elseif (isnan (nRows))
    nRows = ceil (nImg/nCols);
  elseif (isnan (nCols))
    nCols = ceil (nImg/nRows);
  elseif (nCols * nRows < nImg)
    error ("montage: size of %ix%i is not enough for image with %i frames.",
           nRows, nCols, nImg);
  endif

  ## 2) build the image
  margin_width = p.Results.MarginWidth;
  back_color   = fix_color (p.Results.BackgroundColor, channels);
  disp_img = zeros (height*nRows + margin_width*(nRows-1),
                    width *nCols + margin_width*(nCols-1),
                    channels, class (images)) + back_color;

  ## find the start and end coordinates for each of the images
  xRows = start_end (nRows, height, margin_width);
  xCols = start_end (nCols, width,  margin_width);

  ## Using reshape and permute to build the final image turned out to
  ## be quite a problem. So yeah... we'll use a for loop. Anyway, the number
  ## of images on a montage is never very high, this function is unlikely
  ## to be the speed bottleneck for any usage, and will make the code
  ## much more readable.
  iRow = iCol = 1;
  for iImg = 1:nImg
    if (iCol > nCols)
      iCol = 1;
      iRow++;
    endif
    rRow = xRows(1,iRow):xRows(2,iRow); # range of rows
    rCol = xCols(1,iCol):xCols(2,iCol); # range of columns
    disp_img(rRow,rCol,:) = images(:,:,:,iImg);
    iCol++;
  endfor

  ## 3) color margins as required
  margin_color = fix_color (p.Results.MarginColor, channels);
  if (margin_width > 0 && any (margin_color != back_color))
    mRows = linspace (xRows(2,1:end-1) +1, xRows(1,2:end) -1, margin_width) (:);
    mCols = linspace (xCols(2,1:end-1) +1, xCols(1,2:end) -1, margin_width) (:);

    ## a function that can be used to brodcast assignment
    bd_ass = @(x, y) subsasgn (x, struct ("type", "()", "subs", {{":"}}), y);
    disp_img(mRows,:,:) = bsxfun (bd_ass, disp_img(mRows,:,:), margin_color);
    disp_img(:,mCols,:) = bsxfun (bd_ass, disp_img(:,mCols,:), margin_color);
  endif

  ## 4) display the image
  tmp_h = imshow (disp_img, p.Results.DisplayRange);

  if (nargout > 0)
    h = tmp_h;
  endif
endfunction

## given number of elements (n), the length or each, and the border length
## between then, returns start and end coordinates for each of the elements
function [coords] = start_end (n, len, bord)
  coords      = bord * (0:(n-1)) + len * (0:(n-1)) + 1;
  coords(2,:) = coords(1,:) + len - 1;
endfunction

## color values can be given in grayscale or RGB values and may not match
## the values of the image.  This function will make the color match the
## image and give a 1x1x(1||3) vector that can be used for broadcasting
function color = fix_color (color, img_channels)
  if (numel (color) != img_channels)
    if (img_channels == 3)
      color = repmat (color, [3 1]);
    elseif (img_channels == 1)
      color = rgb2gray (reshape (color, [1 1 3]));
    else
      error ("montage: image has an unknown number (%d) of channels.", img_channels);
    endif
  endif
  color = reshape (color, [1 1 img_channels]);
endfunction

%!function cdata = montage_cdata (varargin)
%!  h = figure ();
%!  set (h, "visible", "off");
%!  mh = montage (varargin{:});
%!  cdata = get (mh, "cdata");
%!  close (h);
%!endfunction

## Test automatic distribution of panels
%!test
%! im = uint8 (ones (2, 2, 1, 5)) .* reshape ([1 2 3 4 5], [1 1 1 5]);
%! cdata = montage_cdata (im);
%! expected = uint8 ([
%!   1 1 2 2 3 3
%!   1 1 2 2 3 3
%!   4 4 5 5 0 0
%!   4 4 5 5 0 0
%! ]);
%! assert (cdata, expected)

%!test
%! im = uint8 (ones (2, 4, 1, 6)) .* reshape ([1 2 3 4 5 6], [1 1 1 6]);
%! cdata = montage_cdata (im);
%! expected = uint8 ([
%!   1 1 1 1 2 2 2 2
%!   1 1 1 1 2 2 2 2
%!   3 3 3 3 4 4 4 4
%!   3 3 3 3 4 4 4 4
%!   5 5 5 5 6 6 6 6
%!   5 5 5 5 6 6 6 6
%! ]);
%! assert (cdata, expected)
