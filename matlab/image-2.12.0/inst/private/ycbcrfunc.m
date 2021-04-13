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

## Private function for ycbcr2rgb and rgb2ycbcr functions which are
## very similar

function out = ycbcrfunc (func, in, standard)

  img = false; # was input an image?
  if (iscolormap (in))
    ## do nothing, it's a colormap
  elseif (isrgb (in))
    img = true;
    ## we shape it as a colormap (2D matrix) so we can use matrix multiplcation
    nRows = rows (in);
    nCols = columns (in);
    in    = reshape (in, [nRows*nCols 3]);
  else
    error ("%s: input must be a colormap (Nx3) or RGB image (NxMx3)", func);
  endif

  if (ischar (standard))
    if (strcmpi (standard, "601")) # for ITU-R BT.601
      Kb = 0.114;
      Kr = 0.299;
    elseif (strcmpi (standard, "709")) # for ITU-R BT.709
      Kb = 0.0722;
      Kr = 0.2126;
    elseif (strcmpi (standard, "2020")) # for ITU-R BT.2020
      Kb = 0.0593;
      Kr = 0.2627;
    else
      error ("%s: unknown standard `%s'", func, standard);
    endif
  elseif (isnumeric (standard) && numel (standard) == 2)
    Kb = standard(1);
    Kr = standard(2);
  else
    error ("%s: must specify a standard (string), or Kb and Kr values", func);
  endif

  ## the color matrix for the conversion. Derived from:
  ##    Y  = Kr*R + (1-Kr-Kb)*G + kb*B
  ##    Cb = (1/2) * ((B-Y)/(1-Kb))
  ##    Cr = (1/2) * ((R-Y)/(1-Kr))
  ## It expects RGB values in the range [0 1], and returns Y in the
  ## range [0 1], and Cb and Cr in the range [-0.5 0.5]
  cmat = [  Kr            (1-Kr-Kb)            Kb
          -(Kr/(2-2*Kb)) -(1-Kr-Kb)/(2-2*Kb)   0.5
            0.5          -(1-Kr-Kb)/(2-2*Kr) -(Kb/(2-2*Kr)) ];

  cls = class (in);
  if (! isfloat (in))
    in  = im2double (in);
  endif

  ## note that these blocks are the inverse of one another. Changes
  ## in one will most likely require a change on the other
  if (strcmp (func, "rgb2ycbcr"))
    ## convert to YCbCr colorspace
    out = in * cmat';
    ## rescale Cb and Cr to range [0 1]
    out(:, [2 3]) += 0.5;
    ## the actual range of Cb, Cr and Y will be smaller. The values at the
    ## extremes are named footroom and headroom. Cb, Cr, and Y have different
    ## ranges for footroom and headroom
    ##
    ##    Cb and Cr: footroom -> [0       16/255[
    ##    Y        : footroom -> [0       16/255[
    ##    Cb and Cr: headroom -> ]240/255 1     ]
    ##    Y        : headroom -> ]235/255 1     ]
    ##
    ## So we first compress the values to the actual available range (the whole
    ## [0 1] interval minus headroom and footroom), and then shift forward
    out(:,1)     = (out(:,1) * 219/255) + 16/255;
    out(:,[2 3]) = (out(:,[2 3]) * 224/255) + 16/255;

  elseif (strcmp (func, "ycbcr2rgb"))
    ## just the inverse of the rgb2ycbcr conversion
    in(:,[2 3])  = (in(:,[2 3]) - 16/255) / (224/255);
    in(:,1)      = (in(:,1) - 16/255) / (219/255);
    in(:,[2 3]) -= 0.5;
    out          = in * inv (cmat');
  else
    error ("internal error for YCbCr conversion. Unknown function %s", func);
  endif

  switch (cls)
    case {"single", "double"}
      ## do nothing. All is good
    case "uint8"
      out = im2uint8 (out);
    case "uint16"
      out = im2uint16 (out);
    otherwise
      error ("%s: unsupported image class %s", func, cls);
  endswitch

  if (img)
    ## put the image back together
    out = reshape (out, [nRows nCols 3]);
  endif

endfunction
