from monai.transforms import LoadDatad, LoadPNG
from monai.config import KeysCollection
from monai.transforms import Transform

from transforms_templates.items.items import MeshItem


class LoadPLY(Transform):
    def __init__(self, load_vertex_features=True, load_face_features=True, need_lzma=False):
        self.load_vertex_features = load_vertex_features
        self.load_face_features = load_face_features
        self.need_lzma = need_lzma

    def __call__(self, filename):
        o = MeshItem.from_file(filename, need_lzma=self.need_lzma)

        if self.load_vertex_features:
            o.vertex_features = self.get_features(o.data, kind='vertex')
        if self.load_face_features:
            o.face_features = self.get_features(o.data, kind='face')

        # TODO: fill meta with something
        return [o, {'meta': 1}]

    def get_features(self, mesh, kind='vertex'):
        res = {}
        d = mesh.metadata['ply_raw'][kind]
        for key in d['properties'].keys():
            if key not in ['x', 'y', 'z', 'vertex_indices']:
                res[key] = d['data'][key]
        return res



# LoadDatad - inherites form TransformMap (Transform for dictionaries)
# defects:
#     --- put ouput with the same input key, 'mesh' e.g.
class LoadPLYd(LoadDatad):
    def __init__(
        self,
        keys: KeysCollection,
        meta_key_postfix: str = "meta_dict",
        overwriting: bool = False,
        load_vertex_features = True,
        load_face_features = True,
        need_lzma=False,
    ) -> None:
        """
        """
        loader = LoadPLY(load_vertex_features=True, load_face_features=load_face_features, need_lzma=need_lzma)
        super().__init__(keys, loader, meta_key_postfix, overwriting)
