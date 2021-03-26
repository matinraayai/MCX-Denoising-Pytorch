## Copyright (C) 2015 Carnë Draug <carandraug+dev@gmail.com>
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

## Private function for functions that convert between color spaces, i.e.,
## rgb2xyz, xyz2rgb, xyz2lab, lab2xyz, rgb2lab and lab2rgb.  This reverts
## a colormap type into the same shape and class as it was in the input.
## (But setting the keep_class flag to 1 tells this function to not change
## the class back.)  The other flags are meant to come from  complementary
## private function colorspace_conversion_input_check()
##
## adapted by: Hartmut Gimpel, 2015

function rv = colorspace_conversion_revert (rv, cls, sz, is_im, is_nd, is_int, keep_class)
  if (is_im)
    if (is_nd)
      rv = reshape (rv, [sz(1:2) sz(4) sz(3)]);
      rv = permute (rv, [1 2 4 3]);
    else
      rv = reshape (rv, sz);
    endif
  endif
  if (is_int && ~keep_class)
    rv *= intmax (cls);
  endif
endfunction
