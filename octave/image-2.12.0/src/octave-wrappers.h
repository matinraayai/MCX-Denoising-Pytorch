// Copyright (C) 2017 Carnë Draug <carandraug@octave.org>
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

// Core is going through a bunch of changes, and moving a lot of
// functions into the octave namespace and deprecating thw old
// functions.  We want to be compatible with older versions and we
// don't want to scare users with deprecation warnings so we have our
// own wrappers so nothing breaks.
//
// We don't want to have a file per function we need to wrap; we don't
// want to repeat the wrapper in each file that needs it; we don't
// want to disable the deprecation warnings (so that we get warnings
// next time we something else gets deprecated); and we don't want to
// include all needed headers.
//
// It is the job of the file that includes this to include the
// required headers, at least as long as core only changes the
// namespace and not the header file.
//
// This wrappers are all temporary until we no longer support the
// Octave version that made the change.

#include "config.h"

namespace octave_image
{
  // Temporary wrapper until we no longer support Octave 4.2 (bug #50180)
#if defined WANTS_FEVAL && ! defined HAS_FEVAL
#define HAS_FEVAL 1
  inline octave_value_list
  feval (const std::string& name,
         const octave_value_list& args,
         int nargout = 0)
  {
#if defined HAVE_FEVAL_IN_OCTAVE_NAMESPACE
    return octave::feval (name, args, nargout);
#else
    return ::feval (name, args, nargout);
#endif
  }
#endif

  // Temporary wrapper until we no longer support Octave 4.2
#if defined WANTS_OCTAVE_IMAGE_VALUE && ! defined HAS_OCTAVE_IMAGE_VALUE
#define HAS_OCTAVE_IMAGE_VALUE 1
  class
  value : public octave_value
  {
  public:
    value (const octave_value& a) : octave_value (a) {}

#if ! defined HAVE_OCTAVE_STYLE_IS_FUNCTIONS

    bool islogical (void) const
    { return is_bool_type (); }

    bool iscomplex (void) const
    { return is_complex_type (); }

    bool isempty (void) const
    { return is_empty (); }

    bool isfloat (void) const
    { return is_float_type (); }

    bool isnumeric (void) const
    { return is_numeric_type (); }

    bool isreal (void) const
    { return is_real_type (); }

#endif
  };
#endif
}
