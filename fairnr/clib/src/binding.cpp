// Copyright (c) Facebook, Inc. and its affiliates.
// 
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "intersect.h"
#include "octree.h"
#include "sample.h"
#include "face_areas_normals.h"
#include "point_mesh.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cloud_intersect", &cloud_intersect);
  m.def("ball_intersect", &ball_intersect);
  m.def("aabb_intersect", &aabb_intersect);
  m.def("svo_intersect", &svo_intersect);
  m.def("triangle_intersect", &triangle_intersect);

  m.def("uniform_ray_sampling", &uniform_ray_sampling);
  m.def("inverse_cdf_sampling", &inverse_cdf_sampling);

  m.def("build_octree", &build_octree);

  m.def("face_areas_normals_forward", &FaceAreasNormalsForward);
  m.def("face_areas_normals_backward", &FaceAreasNormalsBackward);


  // PointFace distance functions
  m.def("point_face_dist_forward", &PointFaceDistanceForward);
  m.def("point_face_dist_backward", &PointFaceDistanceBackward);
  m.def("face_point_dist_forward", &FacePointDistanceForward);
  m.def("face_point_dist_backward", &FacePointDistanceBackward);
  m.def("point_face_array_dist_forward", &PointFaceArrayDistanceForward);
  m.def("point_face_array_dist_backward", &PointFaceArrayDistanceBackward);
}