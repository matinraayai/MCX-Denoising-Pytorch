## Copyright (C) 2014 Carnë Draug <carandraug+dev@gmail.com>
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
## @deftypefn  {Function File} {} imcast (@var{img}, @var{type})
## @deftypefnx {Function File} {} imcast (@var{img}, @var{type}, "indexed")
## Convert image to specific data type @var{type}.
##
## Converts a valid image @var{img} into another class @var{type}.  A valid
## image must be of class logical, uint8, uint16, int16, single, or double.
##
## If the image is indexed, the last argument may be the string
## @qcode{"indexed"}.  An indexed image may not be of class int16,
## single, or logical (see @code{isind} for details).
##
## Details on how the conversion is performed is class dependent, and
## can be seen on the help text of @code{im2double}, @code{im2single},
## @code{im2uint8}, @code{im2uint16}, and @code{im2int16}.  For the
## case of logical, conversion to logical is equivalent to simply
## casting @var{img} to logical data type.  Conversion from logical,
## true elements are converted to 1 in the case of floating point, or
## the maximum value in the case of integer types.
##
## @seealso{im2bw, im2uint8, im2double, im2int16, im2single, im2uint16}
## @end deftypefn

function imout = imcast (img, outcls, varargin)

  if (nargin < 2 || nargin > 3)
    print_usage ();
  elseif (nargin == 3 && ! strcmpi (varargin{1}, "indexed"))
    error ("imcast: third argument must be the string \"indexed\"");
  endif

  incls = class (img);
  if (strcmp (outcls, incls))
    imout = img;
    return
  endif

  ## we are dealing with indexed images
  if (nargin == 3)
    if (! isind (img))
      error ("imcast: input should have been an indexed image but it is not.");
    endif

    ## Check that the new class is enough to hold all the previous indices
    ## If we are converting to floating point, then we don't bother
    ## check the range of indices. Also, note that indexed images of
    ## integer class are always unsigned.

    ## we will be converting a floating point image to integer class
    if (strcmp (outcls, "single") || strcmp (outcls, "double"))
      if (isinteger (img))
        imout = cast (img, outcls) +1;
      else
        imout = cast (img, outcls);
      endif

    ## we will be converting an indexed image to integer class
    else
      if (isinteger (img) && intmax (incls) > intmax (outcls) && max (img(:)) > intmax (outcls))
          error ("imcast: IMG has too many colours '%d' for the range of values in %s",
            max (img(:)), outcls);
      elseif (isfloat (img))
        imax = max (img(:)) -1;
        if (imax > intmax (outcls))
          error ("imcast: IMG has too many colours '%d' for the range of values in %s",
            imax, outcls);
        endif
        img -= 1;
      endif
      imout = cast (img, outcls);
    endif

  ## we are dealing with "normal" images
  else
    problem = false; # did we found a bad conversion?
    switch (incls)

      case {"double", "single"}
        switch (outcls)
          case "uint8",  imout = uint8  (img * 255);
          case "uint16", imout = uint16 (img * 65535);
          case "int16",  imout = int16 (double (img * uint16 (65535)) -32768);
          case {"double", "single"}, imout = cast (img, outcls);
          case "logical", imout = logical (img);
          otherwise, problem = true;
        endswitch

      case {"uint8"}
        switch (outcls)
          case "double", imout = double (img) / 255;
          case "single", imout = single (img) / 255;
          case "uint16", imout = uint16 (img) * 257; # 257 comes from 65535/255
          case "int16",  imout = int16 ((double (img) * 257) -32768); # 257 comes from 65535/255
          case "logical", imout = logical (img);
          otherwise, problem = true;
        endswitch

      case {"uint16"}
        switch (outcls)
          case "double", imout = double (img) / 65535;
          case "single", imout = single (img) / 65535;
          case "uint8",  imout = uint8 (img / 257); # 257 comes from 65535/255
          case "int16",  imout = int16 (double (img) -32768);
          case "logical", imout = logical (img);
          otherwise, problem = true;
        endswitch

      case {"logical"}
        switch (outcls)
          case {"double", "single"}
            imout = cast (img, outcls);
          case {"uint8", "uint16", "int16"}
            imout = repmat (intmin (outcls), size (img));
            imout(img) = intmax (outcls);
          otherwise
            problem = true;
        endswitch

      case {"int16"}
        switch (outcls)
          case "double", imout = (double (img) + 32768) / 65535;
          case "single", imout = (single (img) + 32768) / 65535;
          case "uint8",  imout = uint8 ((double (img) + 32768) / 257); # 257 comes from 65535/255
          case "uint16", imout = uint16 (double (img) + 32768);
          case "logical", imout = logical (img);
          otherwise, problem = true;
        endswitch

      otherwise
        error ("imcast: unknown image of class \"%s\"", incls);

    endswitch
    if (problem)
      error ("imcast: unsupported TYPE \"%s\"", outcls);
    endif
  endif

endfunction

%!test
%! im = randi ([0 255], 40, "uint8");
%! assert (imcast (im, "uint8"), im2uint8 (im))
%! assert (imcast (im, "uint16"), im2uint16 (im))
%! assert (imcast (im, "single"), im2single (im))
%! assert (imcast (im, "uint8", "indexed"), im2uint8 (im, "indexed"))
%! assert (imcast (im, "uint16", "indexed"), im2uint16 (im, "indexed"))
%! assert (imcast (im, "single", "indexed"), im2single (im, "indexed"))

%!test
%! im = randi ([1 256], 40, "double");
%! assert (imcast (im, "uint8"), im2uint8 (im))
%! assert (imcast (im, "uint8", "indexed"), im2uint8 (im, "indexed"))
%! assert (imcast (im, "single", "indexed"), im2single (im, "indexed"))

%!test
%! im = randi ([0 65535], 40, "uint16");
%! assert (imcast (im, "uint8"), im2uint8 (im))
%! assert (imcast (im, "single"), im2single (im))
%! assert (imcast (im, "single", "indexed"), im2single (im, "indexed"))

%!test
%! im = randi ([1 255], 40, "double");
%! assert (imcast (im, "uint8", "indexed"), im2uint8 (im, "indexed"))
%! assert (imcast (im, "single", "indexed"), im2single (im, "indexed"))

%!test
%! im = rand (40);
%! assert (imcast (im, "uint8"), im2uint8 (im))

%!error <unknown image of class> imcast (randi (127, 40, "int8"), "uint8")
%!error <unsupported TYPE> imcast (randi (255, 40, "uint8"), "uint32")
%!error <unsupported TYPE> imcast (randi (255, 40, "uint8"), "not a class")
%!error <range of values> imcast (randi ([0 65535], 40, "uint16"), "uint8", "indexed")

%!assert (imcast ([0 1 .2; 2 -0 Inf], "logical"), logical ([0 1 1; 1 0 1]))
