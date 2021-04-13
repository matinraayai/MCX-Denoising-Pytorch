## Copyright (C) 2018 Martin Janda <janda.martin1@gmail.com>
## 
## This program is free software: you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see
## <https://www.gnu.org/licenses/>.

## -*- texinfo -*- 
## @deftypefn {} disp (@var{r})
##
## @seealso{}
## @end deftypefn

function disp (r)
  printf("%s with properties:\n", class (r));
  printf("\n");
  printf("         XWorldLimits: [%d %d]\n", r.XWorldLimits);
  printf("         YWorldLimits: [%d %d]\n", r.YWorldLimits);
  printf("            ImageSize: [%d %d]\n", r.ImageSize);
  printf("  PixelExtentInWorldX: %d\n", r.PixelExtentInWorldX);
  printf("  PixelExtentInWorldY: %d\n", r.PixelExtentInWorldY);
  printf("  ImageExtentInWorldX: %d\n", r.ImageExtentInWorldX);
  printf("  ImageExtentInWorldY: %d\n", r.ImageExtentInWorldY);
  printf("     XIntrinsicLimits: [%d %d]\n", r.XIntrinsicLimits);
  printf("     YIntrinsicLimits: [%d %d]\n", r.YIntrinsicLimits);
endfunction
