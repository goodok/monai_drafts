from pathlib import Path
import trimesh
from monai.data import Dataset

import hydra

from transforms_templates.utils.log import log

@hydra.main(config_path="configs", config_name="config")
def my_app(cfg=None):

    ds = get_dataset(cfg.paths.data)

    print("ds[0]:")     
    print(ds[0])
    print()

    tfms = hydra.utils.instantiate(cfg.transforms)

    if cfg.debug.transforms:
        print("tfms.transforms:")
        for tfm in tfms.transforms:
            print('  ', tfm.__class__.__name__)
        print()

    # do transforms
    o = tfms(ds[0], debug=cfg.debug.transforms)



def get_dataset(dir_data):
    # Setup data directory
    DIR_DATA = Path(dir_data)
    assert DIR_DATA.exists(), f'Directory is not found {DIR_DATA}'

    # dataset
    train_ply_files = sorted((DIR_DATA / 'corrupted').glob('*.ply'))

    def get_corresponding_png_file(fn, foreshortening='top', kind='rgb'):
        assert foreshortening in ['top', 'left', 'right']
        assert kind in ['rgb', 'depth']
        base = fn.with_suffix('').name
        return fn.parent.parent / 'renders' / f'{base}_{foreshortening}_{kind}.png'

    def get_corresponding_meta_info_file(fn, foreshortening='top'):
        assert foreshortening in ['top', 'left', 'right']
        base = fn.with_suffix('').name
        return fn.parent.parent / 'renders' / f'{base}_{foreshortening}_meta_info.json'

    def get_files_dict(fn):
        return {'mesh': fn, 'image2d': get_corresponding_png_file(fn), 'image2d_meta_dict': get_corresponding_meta_info_file(fn)}
        

    data_dicts = [get_files_dict(fn) for fn in train_ply_files]
    ds = Dataset(data=data_dicts)

    o = ds[0]
    assert o['mesh'].exists()
    assert o['image2d'].exists()
    assert o['image2d_meta_dict'].exists()

    return ds

if __name__ == "__main__":
    my_app()
