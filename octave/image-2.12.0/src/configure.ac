## Copyright (C) 2020 David Miguel Susano Pinto <carandraug@octave.org>
##
## Copying and distribution of this file, with or without modification,
## are permitted in any medium without royalty provided the copyright
## notice and this notice are preserved.  This file is offered as-is,
## without any warranty.

AC_PREREQ([2.67])
AC_INIT([Octave-Forge image package], [2.12.0])

AC_CONFIG_HEADERS([config.h])

AC_PROG_SED

AC_PATH_PROG([OCTAVE], [octave])
if test -z "$OCTAVE"; then
  AC_MSG_ERROR([*** 'octave' not found.])
fi

AC_PATH_PROG([MKOCTFILE], [mkoctfile])
if test -z "$MKOCTFILE"; then
  AC_MSG_ERROR([*** 'mkoctfile' not found.])
fi

dnl CXX is the compiler including options such as -std=c++11.
dnl CXXFLAGS are flags for CXX coming from the environment.
dnl XTRA_CXXFLAGS are extra flags that we want to set ourselves.


dnl We will not actually use CXX in the build, we only use mkoctfile
dnl which picks up the dnl CXX environment variable and defaults to what
dnl was used to build Octave.  However, that may not be enough for our
dnl C++11 dependency (Octave 4.0 did not require C++11) so we set CXX
dnl ourselves, check if any extra flag was needed, and then use that
dnl on MKOCTFILE.

CXX=`${MKOCTFILE} -p CXX`
image_save_CXX="$CXX"

AC_PROG_CXX
if test "x$ac_prog_cxx_stdcxx" != "xcxx11"; then
  AC_MSG_ERROR([** could not find a C++11 compiler])
fi

CXX11_SWITCH=$(echo "$CXX" | $SED "s,^$image_save_CXX,,")
MKOCTFILE="$MKOCTFILE $CXX11_SWITCH"
CXX="$image_save_CXX"


dnl Pick up any warnings as soon as possible.
XTRA_CXXFLAGS="-Wall"


AC_LANG(C++)


image_save_CXXFLAGS="$CXXFLAGS"
CXXFLAGS="$CXXFLAGS -I`$MKOCTFILE -p OCTINCLUDEDIR`"

## Starting with Octave 4.2, functions have been moved into the octave
## namespace.  While the old ones still exist and are deprecated,
## warnings scare users.  So check for their correct location and wrap
## them if needed.

## feval was moved into octave:: during 4.3 development series.
AC_CACHE_CHECK(
  [whether feval is in the octave namespace],
  [octave_image_cv_feval_in_octave_namespace],
  [AC_COMPILE_IFELSE(
    [AC_LANG_PROGRAM([#include <parse.h>],
                     [octave::feval ("eye")])],
    [octave_image_cv_feval_in_octave_namespace=yes],
    [octave_image_cv_feval_in_octave_namespace=no])])

if test "$octave_image_cv_feval_in_octave_namespace" = yes; then
  AC_DEFINE([HAVE_FEVAL_IN_OCTAVE_NAMESPACE], [1],
            [Define if octave::feval exists as replacement to feval.])
fi

## Sometime during 4.3 development, the is_* style methods were
## deprecated in favour of others whose names match the functions
## in the Octave language:
##   is_bool_type    -> islogical
##   is_complex_type -> iscomplex
##   is_empty        -> isempty
##   is_float_type   -> isfloat
##   is_numeric_type -> isnumeric
##   is_real_type    -> isreal
## Warnings are scary to users, so use the new style if it's available.
AC_CACHE_CHECK([whether octave value has Octave style is* methods],
   [octave_image_cv_octave_style_is_functions],
   [AC_COMPILE_IFELSE(
     [AC_LANG_PROGRAM([#include <ov.h>],
                      [octave_value foo = octave_value ();
                       bool is_result;
                       is_result = foo.iscomplex ();
                       is_result = foo.isempty ();
                       is_result = foo.isfloat ();
                       is_result = foo.islogical ();
                       is_result = foo.isnumeric ();
                       is_result = foo.isreal ();])],
     [octave_image_cv_octave_style_is_functions=yes],
     [octave_image_cv_octave_style_is_functions=no])]
)

if test "$octave_image_cv_octave_style_is_functions" = yes; then
  AC_DEFINE([HAVE_OCTAVE_STYLE_IS_FUNCTIONS], [1],
            [Define if octave_value has is* style like the Octave language.])
fi

## Since Octave 6.0 the exception throw in C++ has its own message.
## Check for that.  In older Octave versions there's a single error
## message at a time (see iptcheckconn code).
AC_CACHE_CHECK([whether octave::execution_exception has message member],
  [octave_image_cv_execution_exception_has_message],
  [AC_COMPILE_IFELSE(
    [AC_LANG_PROGRAM([#include <error.h>],
                     [try
                        {
                          error("test");
                        }
                      catch (octave::execution_exception& e)
                        {
                          std::string msg = e.message ();
                        }])],
    [octave_image_cv_execution_exception_has_message=yes],
    [octave_image_cv_execution_exception_has_message=no])]
)

if test "$octave_image_cv_execution_exception_has_message" = yes; then
  AC_DEFINE([HAVE_OCTAVE_EXCEPTION_MESSAGE], [1],
            [Define if octave_exception has message member.])
fi


CXXFLAGS="$image_save_CXXFLAGS"


## Test for gcc bug #65843 https://gcc.gnu.org/bugzilla/show_bug.cgi?id=65843
## which shows up as Octave bug #45096 https://savannah.gnu.org/bugs/?45096
AC_CACHE_CHECK([whether templated lambda functions accept '&const int'],
  [_cv_template_lambda_accepts_ref_const_inst],
  [AC_COMPILE_IFELSE(
    [AC_LANG_PROGRAM([[
      template<class T>
      void test (T b)
      {
        const int a = b;
        [&] () { return a, a; }();
      }
      ]], [[ test (1); ]])],
    [_cv_template_lambda_accepts_ref_const_inst=yes],
    [_cv_template_lambda_accepts_ref_const_inst=no])
])
if test $_cv_template_lambda_accepts_ref_const_inst = no; then
  AC_MSG_WARN([
    Your C++ compiler (are you using GCC 5.0 or 5.1?) has a bug that
    prevents it from building the Octave Forge image package.  But you
    can fix it very easily.  See https://savannah.gnu.org/bugs/?45096
    for details on working around it.
  ])
fi

## Some functions have been moving between Octave core and the Image
## package so make sure we are not shadowing anything on this
## installation.
##
##  im2double - moved into core for core version 4.2
##  ntsc2rgb - moved into the image package for core version 4.4
##  rgb2ntsc - moved into the image package for core version 4.4

M_FILES=""
m4_foreach([FNAME], [im2double, ntsc2rgb, rgb2gray, rgb2ntsc], [
  AC_MSG_CHECKING([Octave core for function FNAME])
  HAS_FUNCTION=`$OCTAVE -qfH --eval "printf ('%i', exist ('FNAME'))"`
  if test "$HAS_FUNCTION" == 0; then
    AC_MSG_RESULT([no])
    M_FILES="${M_FILES} FNAME.m"
  else
    AC_MSG_RESULT([yes])
  fi
])

AC_SUBST([XTRA_CXXFLAGS])
AC_SUBST([M_FILES])

AC_CONFIG_FILES([Makefile])

AC_OUTPUT
