// Copyright (C) 2014 Carnë Draug <carandraug@octave.org>
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 3 of the
// License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, see
// <http://www.gnu.org/licenses/>.

#include <octave/oct.h>

#include "config.h"

#include "connectivity.h"
using namespace octave::image;

// The conndef() function is really really simple and could have easily
// been a m file (actually it once was, check the hg log if it ever needs
// to be recovered) but then it would be awkward to call it from oct
// functions so we made a C++ class for it.

DEFUN_DLD(conndef, args, , "\
-*- texinfo -*-\n\
@deftypefn  {Loadable Function} {} conndef (@var{conn})\n\
@deftypefnx {Loadable Function} {} conndef (@var{mask})\n\
@deftypefnx {Loadable Function} {} conndef (@var{ndims}, @var{type})\n\
Create connectivity array.\n\
\n\
Creates a matrix of for morphological operations, where elements with\n\
a value of 1 are considered connected to the center element (a\n\
connectivity array).\n\
\n\
It can be specified by the number of dimensions, @var{ndims}, and\n\
@var{type} which must be one of the following strings:\n\
\n\
@table @asis\n\
@item @qcode{\"minimal\"}\n\
Neighbours touch the central element on a (@var{ndims}-1)-dimensional\n\
surface.\n\
\n\
@item @qcode{\"maximal\"}\n\
Neighbours touch the central element in any way. Equivalent to\n\
@code{ones (repmat (3, 1, @var{ndims}))}.\n\
\n\
@end table\n\
\n\
the number of connected elements to the center element, @var{conn},\n\
in which case the following are valid:\n\
\n\
@table @asis\n\
@item 4\n\
Two-dimensional 4-connected neighborhood.\n\
\n\
@item 8\n\
Two-dimensional 8-connected neighborhood.\n\
\n\
@item 6\n\
Three-dimensional 6-connected neighborhood.\n\
\n\
@item 18\n\
Three-dimensional 18-connected neighborhood.\n\
\n\
@item 26\n\
Three-dimensional 26-connected neighborhood.\n\
\n\
@end table\n\
\n\
or a connectivity array itself, in which case it checks for its validity\n\
and returns itself.  In such case, it is equivalent to @code{iptcheckconn}.\n\
\n\
@seealso{iptcheckconn, strel}\n\
@end deftypefn")
{
  const octave_idx_type nargin = args.length ();

  if (nargin < 1 || nargin > 2)
    print_usage ();

  connectivity conn;
  if (nargin == 1)
    conn = conndef (args(0));
  else
    {
      const octave_idx_type ndims = args(0).uint_value (true);
      if (ndims < 1)
        error ("conndef: NDIMS must be a positive integer");
      const std::string type = args(1).string_value ();
      try
        {
          conn = connectivity (ndims, type);
        }
      catch (invalid_connectivity& e)
        {
          error ("conndef: TYPE %s", e.what ());
        }
    }

  // we must return an array of class double
  return octave_value (NDArray (conn.mask));
}

/*

%!assert (conndef (1, "minimal"), [1; 1; 1]);
%!assert (conndef (2, "minimal"), [0 1 0; 1 1 1; 0 1 0]);

%!test
%! C = zeros (3, 3, 3);
%! C(:,2,2) = 1;
%! C(2,:,2) = 1;
%! C(2,2,:) = 1;
%! assert (conndef (3, "minimal"), C);

%!test
%! C = zeros (3, 3, 3, 3);
%! C(:,:,2,1) = [0   0   0
%!               0   1   0
%!               0   0   0];
%! C(:,:,1,2) = [0   0   0
%!               0   1   0
%!               0   0   0];
%! C(:,:,2,2) = [0   1   0
%!               1   1   1
%!               0   1   0];
%! C(:,:,3,2) = [0   0   0
%!               0   1   0
%!               0   0   0];
%! C(:,:,2,3) = [0   0   0
%!               0   1   0
%!               0   0   0];
%! assert (conndef (4, "minimal"), C);

%!assert (conndef (1, "maximal"), ones (3, 1));
%!assert (conndef (2, "maximal"), ones (3, 3));
%!assert (conndef (3, "maximal"), ones (3, 3, 3));
%!assert (conndef (4, "maximal"), ones (3, 3, 3, 3));

%!assert (nnz (conndef (3, "minimal")), 7)
%!assert (nnz (conndef (4, "minimal")), 9)
%!assert (nnz (conndef (5, "minimal")), 11)
%!assert (nnz (conndef (6, "minimal")), 13)

%!assert (find (conndef (3, "minimal")), [5 11 13 14 15 17 23](:))
%!assert (find (conndef (4, "minimal")), [14 32 38 40 41 42 44 50 68](:))
%!assert (find (conndef (5, "minimal")),
%!        [   41   95  113  119  121  122  123  125  131  149  203](:))
%!assert (find (conndef (6, "minimal")),
%!        [  122  284  338  356  362  364  365  366  368  374  392  446  608](:))

%!error conndef ()
%!error <must be a positive integer> conndef (-2, "minimal")
%!error conndef (char (2), "minimal")
%!error conndef ("minimal", 3)
%!error <TYPE must be "maximal" or "minimal"> conndef (3, "invalid")
%!error <CONN must be in the set \[4 6 8 18 26\]> conndef (10)

%!assert (conndef (2, "minimal"), conndef (4))
%!assert (conndef (2, "maximal"), conndef (8))
%!assert (conndef (3, "minimal"), conndef (6))
%!assert (conndef (3, "maximal"), conndef (26))

%!assert (conndef (18), reshape ([0 1 0 1 1 1 0 1 0
%!                                1 1 1 1 1 1 1 1 1
%!                                0 1 0 1 1 1 0 1 0], [3 3 3]))
*/

// PKG_ADD: autoload ("iptcheckconn", which ("conndef"));
// PKG_DEL: autoload ("iptcheckconn", which ("conndef"), "remove");
DEFUN_DLD(iptcheckconn, args, , "\
-*- texinfo -*-\n\
@deftypefn  {Loadable Function} {} iptcheckconn (@var{conn}, @var{func}, @var{var})\n\
@deftypefnx {Loadable Function} {} iptcheckconn (@var{conn}, @var{func}, @var{var}, @var{pos})\n\
Check if argument is valid connectivity.\n\
\n\
If @var{conn} is not a valid connectivity argument, gives a properly\n\
formatted error message.  @var{func} is the name of the function to be\n\
used on the error message, @var{var} the name of the argument being\n\
checked (for the error message), and @var{pos} the position of the\n\
argument in the input.\n\
\n\
A valid connectivity argument must be either double or logical.  It must\n\
also be either a scalar from set [4 6 8 18 26], or a symmetric matrix\n\
with all dimensions of size 3, with only 0 or 1 as values, and 1 at its\n\
center.\n\
\n\
@seealso{conndef}\n\
@end deftypefn")
{
  const octave_idx_type nargin = args.length ();

  if (nargin < 3 || nargin > 4)
    print_usage ();

  const std::string func = args(1).string_value ();
  const std::string var = args(2).string_value ();

  int pos = 0;
  if (nargin > 3)
    {
      pos = args(3).int_value ();
      if (pos < 1)
        error ("iptcheckconn: POS must be a positive integer");
    }

  std::string err_msg;
#if defined HAVE_OCTAVE_EXCEPTION_MESSAGE
  try
    {
      const connectivity conn = conndef (args(0));
    }
  catch (invalid_connectivity& e)
    {
      err_msg = e.what ();
    }
  catch (octave::execution_exception& e)
    {
      err_msg = e.message ();
    }
#else
  {
    octave::unwind_protect frame;
    frame.protect_var (buffer_error_messages);
    buffer_error_messages++;
    try
      {
        const connectivity conn = conndef (args(0));
      }
    catch (invalid_connectivity& e)
      {
        err_msg = e.what ();
      }
    catch (octave::execution_exception& e)
      {
        err_msg = last_error_message ();
      }
  }
#endif

  if (! err_msg.empty ())
    {
      // We get the error message from conndef and then parse it to
      // get the issue with the connectivity so we can throw it again
      // formatted appropriately for iptcheckconn.  This parsing of
      // the error message is not nice but:1) we don't want to
      // duplicate the logic of conndef in iptcheckconn; 2) we prefer
      // to use conndef and only have this function for Matlab
      // compatibility; 3) this code is only used when conn is invalid
      // so won't be happening many times (meaning performance here is
      // not important); 4) we have plenty of tests to ensure that the
      // commit message "surgery" will continue to work as expected.

      const std::string token = "CONN ";
      std::string::size_type n = err_msg.find(token);
      if (n == std::string::npos)
        error ("iptcheckconn: CONN is invalid but failed to parse error");
      err_msg = err_msg.substr (n + token.size ());
      if (pos == 0)
        error ("%s: %s %s", func.c_str (), var.c_str (), err_msg.c_str ());
      else
        error ("%s: %s, at pos %i, %s",
               func.c_str (), var.c_str (), pos, err_msg.c_str ());
    }
  return octave_value ();
}

/*
// the complete error message should be "expected error <.> but got none",
// but how to escape <> within the error message?

%!test iptcheckconn ( 4, "func", "var")
%!test iptcheckconn ( 6, "func", "var")
%!test iptcheckconn ( 8, "func", "var")
%!test iptcheckconn (18, "func", "var")
%!test iptcheckconn (26, "func", "var")

%!test iptcheckconn (1, "func", "var")
%!test iptcheckconn (ones (3, 1), "func", "var")
%!test iptcheckconn (ones (3, 3), "func", "var")
%!test iptcheckconn (ones (3, 3, 3), "func", "var")
%!test iptcheckconn (ones (3, 3, 3, 3), "func", "var")

%!error <func: VAR must be in the set \[4 6 8 18 26\]>
%!      iptcheckconn (3, "func", "VAR");
%!error <func: VAR center is not true>
%!      iptcheckconn ([1 1 1; 1 0 1; 1 1 1], "func", "VAR");
%!error <func: VAR must either be a logical array or a numeric scalar>
%!      iptcheckconn ([1 2 1; 1 1 1; 1 1 1], "func", "VAR");
%!error <func: VAR is not symmetric relative to its center>
%!      iptcheckconn ([0 1 1; 1 1 1; 1 1 1], "func", "VAR");
%!error <func: VAR is not 1x1, 3x1, 3x3, or 3x3x...x3>
%!      iptcheckconn (ones (3, 3, 3, 4), "func", "VAR");
*/
