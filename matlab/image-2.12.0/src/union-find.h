// Copyright (C) 2011 Jordi Gutiérrez Hermoso <jordigh@octave.org>
// Copyright (C) 2014 Carnë Draug <carandraug@octave.org>
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

#include <vector>

struct voxel
{
  octave_idx_type rank;
  octave_idx_type parent;

  voxel () = default;
};

class union_find
{
  // Union-find data structure, see e.g.
  // http://en.wikipedia.org/wiki/Union-find

private:
  std::vector<voxel> voxels;

public:

  explicit union_find (octave_idx_type s) : voxels (s) {};

  // Use only when adding new elements for the first time
  void
  add (const octave_idx_type idx)
  {
    voxels[idx].parent  = idx;
    voxels[idx].rank    = 0;
    return;
  }

  // Give the root representative id for this object
  octave_idx_type
  find (const octave_idx_type idx)
  {
    voxel* elt = &voxels[idx];
    if (elt->parent != idx)
      elt->parent = find (elt->parent);

    return elt->parent;
  }

  //Given two objects, unite the sets to which they belong
  void
  unite (const octave_idx_type idx1, const octave_idx_type idx2)
  {
    octave_idx_type root1 = find (idx1);
    octave_idx_type root2 = find (idx2);

    //Check if any union needs to be done, maybe they already are
    //in the same set.
    if (root1 != root2)
      {
        voxel* v1 = &voxels[root1];
        voxel* v2 = &voxels[root2];
        if (v1->rank > v2->rank)
          v1->parent = root2;
        else if (v1->rank < v2->rank)
          v2->parent = root1;
        else
          {
            v2->parent = root1;
            v1->rank++;
          }
      }
  }

  std::vector<octave_idx_type>
  get_ids (const NDArray& L) const
  {
    std::vector<octave_idx_type> ids;
    const double* v = L.fortran_vec ();

    for (size_t i = 0; i < voxels.size (); i++)
      if (v[i])
        ids.push_back (i);

    return ids;
  };

};
