## Copyright (C) 2015 Carnë Draug <carandraug@octave.org>
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

## -*- texinfo -*-
## @deftypefn  {Function File} {[@var{path}, @var{flag}] =} imgetfile ()
## @deftypefnx {Function File} {[@dots{}] =} imgetfile (@var{options}, @dots{})
## Open GUI dialogue to select image files.
##
## The GUI dialogue opened is exactly the same as @code{uigetfile} but uses
## recognized image file formats as file filter.  All other options from
## @code{uigetfile} are accepted.
##
## The return value @var{path} is a string with the full filepath of the
## selected file.  If the @qcode{"MultiSelect"} option is selected, then a
## cell array of strings is returned.
##
## @var{flag} is a logical value, true if there was any issue with file
## selection such as the user closing or cancelling the dialogue with selecting
## a file.  It has a value of false otherwise.
##
## @example
## [filepath, flag] = imgetfile ();
## if (flag)
##   error ("A file must be selected");
## endif
## @end example
##
## There is no guarantee that @code{imread} will be capable to read all
## files selected via this dialogue.  Possible reasons for this are:
##
## @itemize @bullet
## @item
## the filter uses all the extensions as obtained from @code{imformats}.
## This only means that a format with such extensions is registered, not
## necessarily that read or write support has been implemented;
##
## @item
## the dialogue also has a "All files (*)" filter, so a user is actually
## able to select any file;
##
## @item
## the file may be corrupt;
##
## @item
## it is the file content, and not the file extension, that defines the file
## format.  A file may have a file extension that is equal to an image file
## format and not actually be an image file.
##
## @end itemize
##
## The opposite is also true.  @code{imread} is able to guess the file
## format even when the file extension is incorrect (or even in the absence
## of a file extension).  A file that is filtered out may still be readable
## by @code{imread} or @code{imfinfo}.
##
## @seealso{imformats, uigetfile}
## @end deftypefn

function [filepath, flag] = imgetfile (varargin)

  ext = strcat ("*.", [imformats().ext]);
  im_filter = strjoin (ext, ";");
  [fname, fpath] = uigetfile ({im_filter, "Image files"}, varargin{:});

  if (isequal (fname, 0))
    flag = true;
    m = strcmpi (varargin, "MultiSelect");
    if (any (m) && strcmpi (varargin{find (m) +1}, "on"))
      filepath = {};
    else
      filepath = "";
    endif
  else
    flag = false;
    filepath = fullfile (fpath, fname);
  endif
endfunction

## Remove from test statistics.  No real tests possible.
%!assert (1)

