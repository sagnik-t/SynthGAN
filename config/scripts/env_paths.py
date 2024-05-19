import site
from pathlib import Path

root = Path.cwd()

src_path = root / 'src'
models_path = root / 'src' / 'models'
layers_path = root / 'src' / 'models' / 'layers'
utils_path = root / 'src' / 'utils'
config_path = root / 'config'

env_paths = [src_path, models_path, layers_path, utils_path, config_path]

site_pckgs_dir = Path(site.getsitepackages()[0])

pth_file = Path(site_pckgs_dir / 'env_paths.pth')

Path.touch(pth_file)

with open(pth_file, 'w') as file:
    for path in env_paths:
        file.write(path.as_posix() + '\n')