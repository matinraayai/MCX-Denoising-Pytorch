## Copyright (C) 2010, 2013 Carnë Draug <carandraug@octave.org>
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
## @deftypefn  {Function File} {} mmgradm (@var{img})
## @deftypefnx {Function File} {} mmgradm (@var{img}, @var{se_dil})
## @deftypefnx {Function File} {} mmgradm (@var{img}, @var{se_dil}, @var{se_ero})
## Perform morphological gradient.
##
## The matrix @var{img} must be numeric whose gradients is calculated, while
## @var{se_dil} and @var{se_ero} are the structuring elements for the dilation
## and erosion respectively.  They can be a:
## @itemize @bullet
## @item
## strel object;
## @item
## array of strel objects as returned by `@@strel/getsequence';
## @item
## matrix of 0's and 1's.
## @end itemize
##
## The @var{se_dil} and @var{se_ero} default to the elementary cross, i.e.:
## @example
## [ 0 1 0
##   1 1 1
##   0 1 0];
## @end example
##
## The basic morphological gradient corresponds to a matrix erosion
## subtracted to its dilation, which is equivalent to:
## @example
## imdilate (img, se_dil) - imerode (img, se_ero)
## @end example
##
## To perform the half-gradients by erosion or dilation, or the internal or
## external gradients, simply pass an empty matrix as structuring element:
## @example
## mmgradm (img, [], se_ero) # half-gradient by erosion or internal gradient
## mmgradm (img, se_dil, []) # half-gradient by dilation or external gradient
## @end example
##
## @seealso{imerode, imdilate, imopen, imclose, imtophat, imbothat}
## @end deftypefn

function grad = mmgradm (img, se_dil = strel ("diamond", 1), se_ero = strel ("diamond", 1))

  ## This function does not exist in Matlab. It is meant to be compatible
  ## with the mmgradm function from the SDC morphology toolbox

  if (nargin < 1 || nargin > 3)
    print_usage ();
  elseif (! isimage (img))
    error("imtophat: IMG must be a numeric matrix");
  endif
  se_dil = prepare_strel ("mmgradm", se_dil);
  se_ero = prepare_strel ("mmgradm", se_ero);

  dilated   = imdilate  (img, se_dil);
  eroded    = imerode   (img, se_ero);

  if (islogical (img))
    grad  = dilated & ! eroded;
  else
    grad  = dilated - eroded;
  endif
endfunction
