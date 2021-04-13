## Copyright (C) 2017 Hartmut Gimpel <hg_code@gmx.de>
##
## This program is free software; you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*-
## @deftypefn  {} {} deconvwnr (@var{I}, @var{PSF})
## @deftypefnx {} {} deconvwnr (@var{I}, @var{PSF}, @var{NSR})
## Apply Wiener deconvolution filter.
##
## The Wiener deconvolution algorithm estimates the original image
## from a deteriorated image @var{I}.  It approximately undoes the
## filtering (e.g. blurring) that deteriorated the original image with
## a linear filter @var{PSF} ("point spread function") and additional
## additive noise.  The amount of noise is specified by the parameter
## @var{NSR} ("noise to signal ratio"), also known as "regularisation
## parameter" K.  This @var{NSR} parameter is meant as the ratio
## between the noise variance and the, unknown, original image
## variance.  The resulting image is optimal in the sense that it
## minimises the mean square error to the original image.
##
## The input image @var{I} can be of class uint8, uint16, int16,
## single, or double.  The output image has the same class and size as
## @var{I}.
##
## The filter @var{PSF} should be a float array.  It can
## have any size that is smaller or equal to the size of @var{I}.
##
## The noise parameter @var{NSR} must be non-negative and can either
## be given as a single float number, or as a float array
## (in Fourier domain) of the same size as @var{I}.  It defaults to 0
## (zero) which produces the, generally bad quality, direct inverse
## filtering.
##
## @seealso{wiener2}
## @end deftypefn

## The algorithm is taken from the book "Digital Image Processing"
## by R. C. Gonzalez and R. E. Woods, Prentice Hall, 3rd edition 2007.
## Equation (5.8-6) in chapter 5.8 "Minimum Mean Square Error (Wiener) Filtering".

function deconvolved = deconvwnr (img, psf, varargin)

  ## Defaults
  nsr = 0;

  if (nargin < 2 || nargin > 3)
    print_usage ();

  elseif (nargin == 3) # deconvwnr (I, PSF, NSR)
    nsr = varargin{1};

    ## TODO: The additional Matlab syntax deconvwnr (I, PSF, NCORR, ICORR)
    ## is not yet implemented.
  endif

  if (! isimage (img) || islogical (img))
    error ("deconvwnr: I must be an non-logical image");
  endif

  if (! isreal (psf) || ! isfloat (psf))
    error("deconvwnr: PSF must be real and float");
  elseif (ndims (psf) > ndims (img))
    error ("deconvwnr: PSF must have less dimensions than I");
  endif

  ## Check length of each dimension but must allow PSF to have less
  ## dimensions than IMG.
  if (any (size (psf) > (size (img)(1:ndims (psf)))))
    error ("deconvwnr: PSF dimensions length must not be longer than I");
  endif

  if (! isfloat (nsr) || any (nsr < 0))
    error ("deconvwnr: NSR must be non-negative and float");
  elseif (numel (nsr) != 1 && ! size_equal (nsr, img))
    error ("deconvwnr: NSR must be a scalar or array of same size as I");
  endif

  cls = class (img);
  if (! isa (img, "double"))
    img = im2double (img);
  endif

  ## Allow psf and nsr inputs to be of class single too, but cast them
  ## to double for calculations.  This behavior is Octave-only, Matlab
  ## requires everything to be double and the user needs to explictely
  ## cast them to double.
  if (isa (psf, "single"))
    psf = double (psf);
  endif

  if (isa (nsr, "single"))
    nsr = double (nsr);
  endif

  deconvolved = wiener_deconvolution (img, psf, nsr);
  deconvolved = imcast (deconvolved, cls);

endfunction

## Actually perform the deconvolution.  All input must be real and of
## class double.  Variables names in capital letters are meant in
## Fourier space.
function deconvolved = wiener_deconvolution (im, psf, K)
  ## This is the equation from the book cited above:
  ##
  ##   DECONV = 1 / PSF  *  [ |PSF|^ 2 / ( |PSF|^ 2 + K )  ] * IM
  ##
  ## using |PSF|^ 2 = PSF * conj(PSF) this can be transformed to
  ##
  ##   DECONV = [ conj(PSF) /  ( |PSF|^ 2 + K )  ] * IM
  ##
  ## (This is the Wiener deconvolution filter given in
  ## https://en.wikipedia.org/wiki/Wiener_deconvolution#Definition ,
  ## it avoids divisions by maybe-zero-valued PSF).
  PSF = psf2otf (psf, size (im));
  PSF_abs_sq = PSF .* conj (PSF);

  ## Make sure the denominator is non-zero (PSF_abs_sq is
  ## non-negative)
  K(K==0) = eps;

  FILTER = conj (PSF) ./ (PSF_abs_sq + K);
  deconvolved = ifftn (FILTER .* fftn (im));

  ## Both im and psf are real so the output should also be.
  deconvolved = real (deconvolved);
endfunction

%!shared im0, psf0, im0_out, psf1, im2, out2_0, out2_1, im3
%! im0 = ones (5, 5);
%! psf0 = ones (3, 3);
%! im0_out = 0.11111 .* ones (5, 5);
%! psf1 = [1 0 0; 0 1 0; 0 0 1];
%! im2 = checkerboard (2, 2, 2);
%! out2_0 = [
%!   -0.4713   -0.2786    0.4229    0.5161   -0.2759   -0.4685    0.5131    0.4199;
%!   -0.4713   -0.2786    0.4229    0.5161   -0.2759   -0.4685    0.5131    0.4199;
%!    0.5161    0.4229   -0.2786   -0.4713    0.4199    0.5131   -0.4685   -0.2759;
%!    0.5161    0.4229   -0.2786   -0.4713    0.4199    0.5131   -0.4685   -0.2759;
%!   -0.4713   -0.2786    0.4229    0.5161   -0.2759   -0.4685    0.5131    0.4199;
%!   -0.4713   -0.2786    0.4229    0.5161   -0.2759   -0.4685    0.5131    0.4199;
%!    0.5161    0.4229   -0.2786   -0.4713    0.4199    0.5131   -0.4685   -0.2759;
%!    0.5161    0.4229   -0.2786   -0.4713    0.4199    0.5131   -0.4685   -0.2759];
%! out2_1 = [
%!   -0.0000    0.8481    0.4288   -0.4194    0.0000    0.2765    0.1373   -0.1392;
%!    0.5623   -0.0000   -0.4194    0.1429    0.5623    0.0000   -0.1392    0.4231;
%!    0.1429   -0.4194         0    0.5623    0.4231   -0.1392         0    0.5623;
%!   -0.4194    0.4288    0.8481         0   -0.1392    0.1373    0.2765         0;
%!   -0.0000    0.8481    0.4288   -0.4194    0.0000    0.2765    0.1373   -0.1392;
%!    0.5623   -0.0000   -0.4194    0.1429    0.5623    0.0000   -0.1392    0.4231;
%!    0.1429   -0.4194         0    0.5623    0.4231   -0.1392         0    0.5623;
%!   -0.4194    0.4288    0.8481         0   -0.1392    0.1373    0.2765         0];
%! im3 = rot90 (diag (0.5.*ones (1,8)) + diag (ones(1,7), 1));

## test input syntax:
%!error deconvwnr ()
%!error deconvwnr (ones (5))
%!assert (deconvwnr (ones (5), ones (3)))
%!assert (deconvwnr (ones (5), ones (3), 0.7))
%!assert (deconvwnr (ones (5), ones (3), 0.5 .* ones (5)))
%!assert (deconvwnr (ones (5, 5, 5), ones (3)))
%!error <NSR must be non-negative> deconvwnr (ones (5), ones (3), -0.7)
%!error <PSF dimensions length must not be longer than I>
%!      deconvwnr (ones (5), ones (7))
%!error <PSF dimensions length must not be longer than I>
%!      deconvwnr (ones (5, 8, 2), ones (6, 5))

## test dimensions and classes:
%!assert (deconvwnr (im0, psf0), im0_out, 1e-5)
%!assert (deconvwnr (im0, single (psf0)), im0_out, 1e-5)
%!assert (class (deconvwnr  (im0, psf0)), "double")
%!assert (deconvwnr (single (im0), psf0), single (im0_out), 1e-5)
%!assert (class (deconvwnr  (single (im0), psf0)), "single")
%!assert (deconvwnr (im2uint8 (im0), psf0), im2uint8 (im0_out))
%!assert (class (deconvwnr  (im2uint8 (im0), psf0)), "uint8")
%!assert (deconvwnr (im2uint16 (im0), psf0), im2uint16 (im0_out))
%!assert (class (deconvwnr  (im2uint16 (im0), psf0)), "uint16")
%!assert (deconvwnr (im2int16 (im0), psf0), im2int16 (im0_out))
%!assert (class (deconvwnr  (im2int16 (im0), psf0)), "int16")
%!error deconvwnr (true (5), ones (3))

## test calculation results:
%!test
%! assert (deconvwnr (im0, psf0, 0.01), im0_out, 1e-4)
%! assert (deconvwnr (im0, psf1, 0.01), 0.333.*ones (5), 1e-4)

%!test
%! im1 = magic (5)./25;
%! out1_0 = [
%!   -0.0820    0.5845   -0.4293    0.2372   -0.0214;
%!    0.6241   -0.5877    0.2768    0.0182   -0.0424;
%!   -0.5481    0.3164    0.0578   -0.2009    0.6637;
%!    0.1580    0.0974   -0.1613    0.7033   -0.5085;
%!    0.1370   -0.1217    0.5449   -0.4689    0.1976];
%! out1_1 = [
%!   -0.2959   -0.1363    0.4038    0.7595    0.1347;
%!   -0.0191    0.3269    0.8768    0.0559   -0.3748;
%!    0.2481    0.7979    0.1731   -0.4517    0.0982;
%!    0.7210    0.2904   -0.5305    0.0194    0.3654;
%!    0.2116   -0.4132   -0.0575    0.4826    0.6422];
%! assert (deconvwnr (im1, psf0, 0.01), out1_0, 1e-4)
%! assert (deconvwnr (im1, psf1, 0.01), out1_1, 1e-4)

%!test
%! assert (deconvwnr (im2, psf0, 0.01), out2_0, 1e-4)
%! assert (deconvwnr (im2, psf1, 0.01), out2_1, 1e-4)

%!test
%! out3_0_x = [
%!   -1.1111    1.0556   -0.4444   -0.1111    0.5556   -0.9444    0.8889    0.0556;
%!    1.0556   -0.7778    0.2222    0.5556   -1.2778    1.2222    0.0556   -0.7778;
%!   -0.4444    0.2222    0.2222   -0.9444    1.2222   -0.2778   -0.4444    0.7222;
%!   -0.1111    0.5556   -0.9444    0.8889    0.0556   -0.4444    0.3889   -0.4444;
%!    0.5556   -1.2778    1.2222    0.0556   -0.7778    0.7222   -0.4444    0.2222;
%!   -0.9444    1.2222   -0.2778   -0.4444    0.7222   -0.7778    0.5556    0.2222;
%!    0.8889    0.0556   -0.4444    0.3889   -0.4444    0.5556   -0.1111   -0.9444;
%!    0.0556   -0.7778    0.7222   -0.4444    0.2222    0.2222   -0.9444    1.2222];
%! out3_0_01 = [
%!   -0.5064    0.2140    0.1101   -0.0993    0.0297   -0.1942    0.3223    0.0772;
%!    0.2140   -0.0659    0.0375    0.0891   -0.4109    0.4783    0.2202   -0.2860;
%!    0.1101    0.0375   -0.0525   -0.3208    0.5721    0.0034   -0.1743    0.0939;
%!   -0.0993    0.0891   -0.3208    0.4624    0.0936   -0.1150   -0.1395   -0.0135;
%!    0.0297   -0.4109    0.5721    0.0936   -0.2566   -0.0027    0.1101    0.1341;
%!   -0.1942    0.4783    0.0034   -0.1150   -0.0027   -0.0659    0.2542   -0.0819;
%!    0.3223    0.2202   -0.1743   -0.1395    0.1101    0.2542   -0.3023   -0.3371;
%!    0.0772   -0.2860    0.0939   -0.0135    0.1341   -0.0819   -0.3371    0.6794];
%! out3_0_00001 = [
%!   -1.1087    1.0520   -0.4419   -0.1112    0.5532   -0.9410    0.8864    0.0557;
%!    1.0520   -0.7746    0.2213    0.5537   -1.2742    1.2190    0.0565   -0.7759;
%!   -0.4419    0.2213    0.2211   -0.9418    1.2196   -0.2767   -0.4433    0.7195;
%!   -0.1112    0.5537   -0.9418    0.8870    0.0557   -0.4428    0.3864   -0.4425;
%!    0.5532   -1.2742    1.2196    0.0557   -0.7755    0.7188   -0.4419    0.2220;
%!   -0.9410    1.2190   -0.2767   -0.4428    0.7188   -0.7746    0.5544    0.2206;
%!    0.8864    0.0565   -0.4433    0.3864   -0.4419    0.5544   -0.1121   -0.9418;
%!    0.0557   -0.7759    0.7195   -0.4425    0.2220    0.2206   -0.9418    1.2201];
%! out3_0_3 = [
%!   -0.0893   -0.0089    0.0446   -0.0357   -0.0268    0.0268    0.0893    0.0446;
%!   -0.0089    0.0223   -0.0089   -0.0357   -0.0089    0.1473    0.1161    0.0179;
%!    0.0446   -0.0089   -0.0357   -0.0089    0.1607    0.0804   -0.0089   -0.0357;
%!   -0.0357   -0.0357   -0.0089    0.1652    0.0804   -0.0179   -0.0714    0.0045;
%!   -0.0268   -0.0089    0.1607    0.0804   -0.0179   -0.0446    0.0446   -0.0000;
%!    0.0268    0.1473    0.0804   -0.0179   -0.0446    0.0223    0.0268   -0.0000;
%!    0.0893    0.1161   -0.0089   -0.0714    0.0446    0.0268   -0.1071   -0.0446;
%!    0.0446    0.0179   -0.0357    0.0045    0.0000   -0.0000   -0.0446    0.1652];
%! out3_1_x = [
%!   -0.3333    0.1667   -0.6667   -0.3333    0.3333    0.1667    0.3333    0.1667;
%!    0.1667   -0.3333   -0.3333    0.3333    0.1667    0.3333    0.1667    0.3333;
%!   -0.6667   -0.3333    0.6667    0.1667    0.3333    0.1667    0.3333    0.1667;
%!   -0.3333    0.3333    0.1667   -0.3333    0.1667    0.3333    0.1667   -0.6667;
%!    0.3333    0.1667    0.3333    0.1667    0.6667    0.1667   -0.6667   -0.3333;
%!    0.1667    0.3333    0.1667    0.3333    0.1667   -0.3333   -0.3333    0.3333;
%!    0.3333    0.1667    0.3333    0.1667   -0.6667   -0.3333   -0.3333    0.1667;
%!    0.1667    0.3333    0.1667   -0.6667   -0.3333    0.3333    0.1667    0.6667];
%! out3_1_01 = [
%!   -0.1868    0.1548   -0.5994   -0.2997    0.3097    0.1548    0.3097    0.1548;
%!    0.1548   -0.2997   -0.2997    0.3097    0.1548    0.3097    0.1548    0.3097;
%!   -0.5994   -0.2997    0.4965    0.1548    0.3097    0.1548    0.3097    0.1548;
%!   -0.2997    0.3097    0.1548   -0.1247    0.1548    0.3097    0.1548   -0.5994;
%!    0.3097    0.1548    0.3097    0.1548    0.4965    0.1548   -0.5994   -0.2997;
%!    0.1548    0.3097    0.1548    0.3097    0.1548   -0.2997   -0.2997    0.3097;
%!    0.3097    0.1548    0.3097    0.1548   -0.5994   -0.2997   -0.1868    0.1548;
%!    0.1548    0.3097    0.1548   -0.5994   -0.2997    0.3097    0.1548    0.4343];
%! out3_1_00001 = [
%!   -0.3331    0.1667   -0.6666   -0.3333    0.3333    0.1667    0.3333    0.1667;
%!    0.1667   -0.3333   -0.3333    0.3333    0.1667    0.3333    0.1667    0.3333;
%!   -0.6666   -0.3333    0.6664    0.1667    0.3333    0.1667    0.3333    0.1667;
%!   -0.3333    0.3333    0.1667   -0.3330    0.1667    0.3333    0.1667   -0.6666;
%!    0.3333    0.1667    0.3333    0.1667    0.6664    0.1667   -0.6666   -0.3333;
%!    0.1667    0.3333    0.1667    0.3333    0.1667   -0.3333   -0.3333    0.3333;
%!    0.3333    0.1667    0.3333    0.1667   -0.6666   -0.3333   -0.3331    0.1667;
%!    0.1667    0.3333    0.1667   -0.6666   -0.3333    0.3333    0.1667    0.6663];
%! out3_1_3 = [
%!   -0.0089    0.0625   -0.1250   -0.0625    0.1250    0.0625    0.1250    0.0625;
%!    0.0625   -0.0625   -0.0625    0.1250    0.0625    0.1250    0.0625    0.1250;
%!   -0.1250   -0.0625    0.1339    0.0625    0.1250    0.0625    0.1250    0.0625;
%!   -0.0625    0.1250    0.0625    0.0982    0.0625    0.1250    0.0625   -0.1250;
%!    0.1250    0.0625    0.1250    0.0625    0.1339    0.0625   -0.1250   -0.0625;
%!    0.0625    0.1250    0.0625    0.1250    0.0625   -0.0625   -0.0625    0.1250;
%!    0.1250    0.0625    0.1250    0.0625   -0.1250   -0.0625   -0.0089    0.0625;
%!    0.0625    0.1250    0.0625   -0.1250   -0.0625    0.1250    0.0625    0.0268];
%! assert (deconvwnr (im3, psf0), out3_0_x, 1e-4)
%! assert (deconvwnr (im3, psf0, 0.1), out3_0_01, 1e-4)
%! assert (deconvwnr (im3, psf0, 0.0001), out3_0_00001, 1e-4)
%! assert (deconvwnr (im3, psf0, 3), out3_0_3, 1e-4)
%! assert (deconvwnr (im3, psf1), out3_1_x, 1e-4)
%! assert (deconvwnr (im3, psf1, 0.1), out3_1_01, 1e-4)
%! assert (deconvwnr (im3, psf1, 0.0001), out3_1_00001, 1e-4)
%! assert (deconvwnr (im3, psf1, 3), out3_1_3, 1e-4)

%!test
%! im_rgb = cat (3, im2, im3, magic (8)./64);
%! out_rgb_0(:, :, 1) = out2_0;
%! out_rgb_0(:, :, 2) = [
%!   -0.9255    0.7869   -0.2553   -0.1154    0.3801   -0.6906    0.7000    0.0651;
%!    0.7869   -0.5407    0.1534    0.4141   -1.0064    0.9816    0.1222   -0.6335;
%!   -0.2553    0.1534    0.1343   -0.7453    1.0211   -0.1936   -0.3586    0.5209;
%!   -0.1154    0.4141   -0.7453    0.7468    0.0675   -0.3247    0.2023   -0.2996;
%!    0.3801   -1.0064    1.0211    0.0675   -0.6045    0.4711   -0.2553    0.2032;
%!   -0.6906    0.9816   -0.1936   -0.3247    0.4711   -0.5407    0.4692    0.1052;
%!    0.7000    0.1222   -0.3586    0.2023   -0.2553    0.4692   -0.1868   -0.7477;
%!    0.0651   -0.6335    0.5209   -0.2996    0.2032    0.1052   -0.7477    1.0630];
%! out_rgb_0(:, :, 3) = [
%!   -0.8118    0.8805    0.8341   -0.7963   -0.6343    0.8222    0.7757   -0.6188;
%!    0.5720   -0.4151   -0.3687    0.5565    0.3945   -0.3567   -0.3103    0.3791;
%!    0.2007   -0.0438    0.0026    0.1852    0.0232    0.0146    0.0610    0.0078;
%!   -0.6880    0.7568    0.7104   -0.6725   -0.5105    0.6984    0.6520   -0.4951;
%!    0.6079   -0.5392   -0.5856    0.6234    0.7854   -0.5975   -0.6439    0.8008;
%!    0.1051    0.0519    0.0983    0.0896   -0.0724    0.1102    0.1566   -0.0879;
%!   -0.2662    0.4231    0.4696   -0.2817   -0.4437    0.4815    0.5279   -0.4592;
%!    0.7317   -0.6629   -0.7093    0.7471    0.9091   -0.7213   -0.7677    0.9246];
%! out_rgb_1(:, :, 1) = out2_1;
%! out_rgb_1(:, :, 2) = [
%!   -0.3110    0.1654   -0.6593   -0.3297    0.3308    0.1654    0.3308    0.1654;
%!    0.1654   -0.3297   -0.3297    0.3308    0.1654    0.3308    0.1654    0.3308;
%!   -0.6593   -0.3297    0.6418    0.1654    0.3308    0.1654    0.3308    0.1654;
%!   -0.3297    0.3308    0.1654   -0.3016    0.1654    0.3308    0.1654   -0.6593;
%!    0.3308    0.1654    0.3308    0.1654    0.6418    0.1654   -0.6593   -0.3297;
%!    0.1654    0.3308    0.1654    0.3308    0.1654   -0.3297   -0.3297    0.3308;
%!    0.3308    0.1654    0.3308    0.1654   -0.6593   -0.3297   -0.3110    0.1654;
%!    0.1654    0.3308    0.1654   -0.6593   -0.3297    0.3308    0.1654    0.6323];
%! out_rgb_1(:, :, 3) = [
%!   -0.0240    0.3338    0.3335    0.0329    0.0344    0.1564    0.3942    0.0913;
%!    0.7871    0.6512   -0.5394   -0.2225    0.7287    0.5905   -0.3619   -0.2809;
%!    0.1333   -0.7196    0.2335    1.0291    0.0749   -0.5421    0.1728    0.9708;
%!   -0.2201    0.4109    0.6487   -0.1632   -0.1617    0.4716    0.4713   -0.1048;
%!    0.4430   -0.1331   -0.1334    0.4999    0.5014   -0.3106   -0.0727    0.5582;
%!   -0.6326    0.1654    0.8803    0.2633   -0.6910    0.1047    1.0577    0.2049;
%!    0.6191    0.7001   -0.2523   -0.3905    0.5607    0.8776   -0.3130   -0.4489;
%!    0.2469   -0.0561    0.1818    0.3038    0.3052    0.0047    0.0043    0.3621];
%! assert (deconvwnr (im_rgb, psf0, 0.01), out_rgb_0, 1e-4)
%! assert (deconvwnr (im_rgb, psf1, 0.01), out_rgb_1, 1e-4)

%!test
%! ## Test that psf and nsr can be of class single, but are usually
%! ## internally as doubles.  Matlab requires everything all to be
%! ## double so this is Matlab incompatible behaviour by design.
%! nsr = 0.1;
%! psf1_recast = double (single (psf1));
%! nsr_recast = double (single (0.1));
%! deconvolved = deconvwnr (im2, psf1_recast, nsr_recast);
%! assert (deconvwnr (im2, single (psf1), single (nsr)), deconvolved)
%! assert (deconvwnr (im2, single (psf1), nsr_recast), deconvolved)
%! assert (deconvwnr (im2, psf1_recast, single (nsr)), deconvolved)


## show instructive demo:
%!demo
%! I = phantom ();
%! figure, imshow (I);
%! title ("Original image");
%! psf = fspecial ("motion", 30, 15);
%! blurred = imfilter (I, psf, "conv");
%! figure, imshow (blurred);
%! title ("Image with added motion blur");
%! var_noise = 0.00005;
%! blurred_noisy = imnoise (blurred, "gaussian", 0, var_noise);
%! figure, imshow (blurred_noisy);
%! title ("Image with motion blur and added Gaussian noise");
%! estimated_nsr = var_noise / (var(blurred_noisy(:)) - var_noise);
%! J = deconvwnr (blurred_noisy, psf, estimated_nsr);
%! figure, imshow (J)
%! title ({"restored image after Wiener deconvolution",
%!           "with known PSF and estimated NSR"});
