import numpy as np
from typing import Any, Dict, Hashable, Mapping, Optional, Sequence, Tuple, Union, Collection

from monai.transforms import Randomizable, MapTransform
from monai.config import KeysCollection

from ..utils.main import is_mesh, is_image_2d
from transforms_templates.items.items import PointsItem

# TODO: 
#  ToPointCloud:
#  - features:
#    -option alpha channel
#    - labels can be optional
#    - colors can be optional
#  wrap _to_xxx as class too

class ToPointCloud(MapTransform):
    """
    """

    def __init__(
        self,
        keys: KeysCollection,
        new_keys: KeysCollection = None,
        method: str = 'centers',
        normals: bool =True,  # append normals to features
        # TODO:
        # - extract features from vertices (mean) or faces
        # - extract labels from faces
        features_from_vertices: KeysCollection = None,
        labels_from_faces: KeysCollection = None

    ) -> None:
        """
        """
        super().__init__(keys)

        if new_keys is None:
            new_keys = keys
        assert len(new_keys) == len(keys)
        assert method in ['centers', 'vertices']

        self.method = method
        self.normals = normals
        self.features_from_vertices = features_from_vertices
        self.labels_from_faces = labels_from_faces
        self.new_keys = new_keys

        if self.method == 'centers':
            self.converter = _to_points_cloud_by_centers
        else:
            self.converter = _to_points_cloud_by_vertices


    def __call__(self, d):

        for key, new_key in zip(self.keys, self.new_keys):
            item = d[key]
            assert is_mesh(item)
            data = self.converter(item, normals=self.normals, features_from_vertices=self.features_from_vertices, labels_from_faces=self.labels_from_faces)
            del d[key]
            d[new_key] = PointsItem(data)
        return d


# TODO: wrap as class too
def _to_points_cloud_by_centers(item, normals=False, features_from_vertices=None, labels_from_faces=None):
    """
    Extract points as center of faces.
    """
    #assert x.is_colors_from_vertices
    #assert not x.is_labels_from_vertices

    d = {}

    mesh = item.data

    points = np.array(mesh.vertices, dtype=np.float32)
    faces = np.array(mesh.faces)

    faces_xyz = points[faces]
    d['points'] = np.mean(faces_xyz, axis=1)

    features = None
    features_original_keys = []
    if features_from_vertices is not None:
        features_original_keys = list(features_from_vertices)
        vertex_features = np.array([item.vertex_features[key] for key in features_from_vertices], dtype=np.float32).T
        assert len(points) == len(vertex_features)

        features = vertex_features[faces]
        features = np.mean(features, axis=1)

    if normals:
        features_original_keys += ['nx', 'ny', 'nz']
        face_normals = np.array(mesh.face_normals, dtype=np.float32)
        if features is None:
            features = face_normals
        else:
            features = np.hstack([features, face_normals])

    d['features'] = features

    for key in ['points', 'features']:
        if d[key] is not None:
            assert len(faces) == len(d[key]), key

    if labels_from_faces is not None:
        d['labels_keys'] = list(labels_from_faces)
        for key in labels_from_faces:
            d[key] = item.face_features[key]
            assert len(faces) == len(d[key]), key

    d['features_original_keys'] = features_original_keys
    return d


# def _to_points_cloud_by_vertices(x, normals=False):
#     # TODO: labels can be optional
#     # TODO: colors can be optional

#     raise NotImplementedError
#     mesh = x.data

#     points = np.array(mesh.vertices, dtype=np.float32)
#     colors = x.colors
#     labels = x.labels

#     assert len(points) == len(colors)
#     is_multilabels = isinstance(labels, (list, tuple))
#     if is_multilabels:
#         for l in labels:
#             assert len(points) == len(l)
#     else:
#         assert len(points) == len(labels)

#     d = {'points': points, 'colors': colors, 'labels': labels}

#     if normals:
#         d['normals'] = np.array(mesh.vertex_normals, dtype=np.float32)
#     return d
