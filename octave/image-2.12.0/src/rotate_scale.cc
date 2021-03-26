// Copyright (C) 2003 Andy Adler <adler@sce.carleton.ca>
//
// This program is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation; either version 3 of the License, or (at your option) any later
// version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
// details.
//
// You should have received a copy of the GNU General Public License along with
// this program; if not, see <http://www.gnu.org/licenses/>.

/*
 * ROTATE_SCALE: rotate and scale a matrix using bilinear interpolation
 * imo= block(im, xregs, yregs);
 */

#include <octave/oct.h>

void
calc_rotation_params(
              double x0l,double y0l,double x0r,double y0r,
              double x1l,double y1l,double x1r,double y1r,
              double* Tx_x, double* Ty_x,
              double* Tx_y, double* Ty_y,
              double* Tx_1, double* Ty_1 );
void
do_interpolation (
              double Tx_x, double Ty_x,
              double Tx_y, double Ty_y,
              double Tx_1, double Ty_1,
              int x0max, int y0max,// initial size
              int x1max, int y1max,// output size
              const double * img0,
              double       * img1 );

DEFUN_DLD (rotate_scale, args, , "\
-*- texinfo -*-\n\
@deftypefn {Loadable Function} {@var{im1} =} rotate_scale(@var{im0}, @var{lm0}, @var{lm1}, @var{out_size})\n\
Arbitrary rotation and scaling of a gray-scale image using fast bilinear interpolation.\n\
\n\
The image @var{im0} will be rotated and scaled such that the landmark points in\n\
@var{lm0} in the image will be placed in the landmark points in @var{lm1} in\n\
the output image @var{im1}. The landmark points are given as a 2 by 2 matrix\n\
where the first row contains the x-coordinates of the landmarks, and the second\n\
row contains the y-coordinates.\n\
\n\
The size of the output image is given in the vector @var{out_size}.\n\
\n\
The following example shows basic usage of the function\n\
@example\n\
im0 = zeros(100); im0(25:75, 25:75)=1;\n\
im1 = rotate_scale( im0, [40,60; 50,50], [60,90; 60,90], [120,120]);\n\
@end example\n\
@seealso{imrotate, imresize}\n\
@end deftypefn\n\
") {
   octave_value_list retval;
   if (args.length() < 4 ||
       !args(0).is_matrix_type() ||
       !args(1).is_matrix_type() ||
       !args(2).is_matrix_type() ||
       !args(3).is_matrix_type())
     print_usage ();

   Matrix im0( args(0).matrix_value() );
   const double * im0p = im0.data();
   Matrix lm0( args(1).matrix_value() );
   Matrix lm1( args(2).matrix_value() );
   ColumnVector out_size( args(3).vector_value() );

   int inp_hig= im0.rows();
   int inp_wid= im0.cols();

   int out_hig= (int) out_size(0);
   int out_wid= (int) out_size(1);
   Matrix im1( out_hig, out_wid);
   double * im1p = im1.fortran_vec();

   double Tx_x; double Ty_x;
   double Tx_y; double Ty_y;
   double Tx_1; double Ty_1;
   calc_rotation_params(
          lm0(0,0), lm0(1,0), lm0(0,1), lm0(1,1),
          lm1(0,0), lm1(1,0), lm1(0,1), lm1(1,1),
          & Tx_x, & Ty_x,
          & Tx_y, & Ty_y,
          & Tx_1, & Ty_1 );
    
   do_interpolation( Tx_x, Ty_x, Tx_y, Ty_y, Tx_1, Ty_1,
                  inp_wid, inp_hig, out_wid, out_hig,
                  im0p, im1p );

   retval(0) = im1;
   return retval;
}

inline double sqr(double a) { return (a)*(a); }

void
calc_rotation_params(
              double x1l,double y1l,double x1r,double y1r,
              double x0l,double y0l,double x0r,double y0r,
              double* Tx_x, double* Ty_x,
              double* Tx_y, double* Ty_y,
              double* Tx_1, double* Ty_1 
              )
{
    double d0= sqrt( sqr(x0l-x0r) + sqr(y0l-y0r) );
    double d1= sqrt( sqr(x1l-x1r) + sqr(y1l-y1r) );
    double dr= d1/d0;

    double a0= atan2( y0l-y0r , x0l-x0r );
    double a1= atan2( y1l-y1r , x1l-x1r );
    double ad= a1-a0;
    double dr_cos_ad= dr*cos(ad);
    double dr_sin_ad= dr*sin(ad);

    double x0m= (x0l+x0r)/2;
    double y0m= (y0l+y0r)/2;
    double x1m= (x1l+x1r)/2;
    double y1m= (y1l+y1r)/2;

    *Tx_x=  dr_cos_ad;
    *Ty_x=  dr_sin_ad;
    *Tx_y= -dr_sin_ad;
    *Ty_y=  dr_cos_ad;
    *Tx_1=  x1m - dr_cos_ad*x0m + dr_sin_ad*y0m;
    *Ty_1=  y1m - dr_sin_ad*x0m - dr_cos_ad*y0m;
}    


void
do_interpolation (
              double Tx_x, double Ty_x,
              double Tx_y, double Ty_y,
              double Tx_1, double Ty_1,
              int x0max, int y0max,// initial size
              int x1max, int y1max,// output size
              const double * img0,
              double       * img1 
            )
{

    for (int i=0; i< x1max; i++) {
        for (int j=0; j< y1max; j++) {
            double x0i= Tx_x * i + Tx_y * j + Tx_1;
            double y0i= Ty_x * i + Ty_y * j + Ty_1;

            if ( x0i < 0       )    x0i= 0;
            else
            if (x0i >= x0max-1 )    x0i= x0max - 1.00001; 

            if ( y0i < 0       )    y0i= 0;
            else
            if (y0i >= y0max-1 )    y0i= y0max - 1.00001; 

            int x0idx= (int) x0i;
            int y0idx= (int) y0i;

            double frac_r= x0i- x0idx; 
            double frac_l= 1 - frac_r;
            double frac_d= y0i- y0idx; 
            double frac_u= 1 - frac_d;

            int pix_lu= (y0idx+0) + (x0idx+0) * y0max ;
            int pix_ru= (y0idx+0) + (x0idx+1) * y0max ;
            int pix_ld= (y0idx+1) + (x0idx+0) * y0max ;
            int pix_rd= (y0idx+1) + (x0idx+1) * y0max ;
               
            img1[ i*y1max + j ]=
               frac_l*frac_u* img0[ pix_lu ] +
               frac_r*frac_u* img0[ pix_ru ] +
               frac_l*frac_d* img0[ pix_ld ] +
               frac_r*frac_d* img0[ pix_rd ];

        }
    }
}

