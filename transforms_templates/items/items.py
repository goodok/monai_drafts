from pathlib import Path
from ..utils.log import log
import trimesh
import lzma

class ItemBase():
    def __init__(self, data, *args, **kwargs):
        self.data = data

class MeshItem(ItemBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO: work around with it
        self.is_colors_from_vertices = True
        self.is_labels_from_vertices = False

    @classmethod
    def from_file(cls, fn, need_lzma=False, **kwargs):
        assert Path(fn).exists(), f'File not found {fn}'
        if need_lzma is True:
            mesh = trimesh.load_mesh(lzma.open(str(fn)), file_type='ply', process=False)
        else:
            mesh = trimesh.load_mesh(str(fn), file_type='ply', process=False)
        o = cls(mesh)
        return o

    def aplly_affine(self, affine_mat):
        "Apply affine transformations"
        self.data.apply_transform(affine_mat)

    def  __str__(self):
        return str(self.data)

    def describe(self, vertices=True, features=True):
        print( f'{self.data}\n')
        points = self.data.vertices
        indent = '  '
        if vertices:
            log('x', points[:, 0], indent=indent)
            log('y', points[:, 1], indent=indent)
            log('z', points[:, 1], indent=indent)
        if features:
            vertex_features = getattr(self, 'vertex_features', None)
            if vertex_features is not None:
                print(indent + 'vertex_features:')
                for key in vertex_features.keys():
                    log(key, vertex_features[key], indent + '  ')
            face_features = getattr(self, 'face_features', None)
            if face_features is not None:
                print(indent + 'face_features:')
                for key in face_features.keys():
                    log(key, face_features[key], indent + '  ')


class PointsItem(ItemBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __str__(self):
        _size = self.data['points'].shape
        return f"(n: {_size[0]})"

    def describe_xyz(self):
        data = self.data
        log('x', data['points'][:, 0])
        log('y', data['points'][:, 1])
        log('z', data['points'][:, 2])

    def describe(self):
        self.describe_xyz()
        data = self.data
        keys = ['features']
        if 'labels_keys' in data:
            keys += data['labels_keys']
        for key in keys:
            log(key, data[key])

        keys = ['features_original_keys', 'labels_keys']
        for key in keys:
            if key in data:
                print(f"{key}: {data[key]}")
        
