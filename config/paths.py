import os

import toml
from pathlib import Path


class Config:
    def __init__(self, cfg_name: str = 'config.toml') -> None:
        self.root_dir = Path(__file__).resolve().parent.parent
        cfg = toml.load(self.root_dir / 'config' / cfg_name)
        p = cfg['paths']

        # Optional input-data root override.
        data_override = os.environ.get('PRIMATE_ALIGN_DIR')
        data_root = Path(data_override or p.get('data_root', 'data')).expanduser()
        if not data_root.is_absolute():
            data_root = self.root_dir / data_root
        data_root = data_root.resolve()

        self.data_dir = data_root
        self.things_dir = data_root / 'things'
        self.macq_dir = data_root / 'macaque'
        self.timeavg_dir = self.macq_dir / 'time_averaged'
        self.timeres_dir = self.macq_dir / 'time_resolved'
        self.mri_dir = data_root / 'human' / 'mri'
        self.dnn_dir = data_root / 'dnn'
        self.fig_dir    = self.root_dir / p['fig_dir']
        self.results_dir = self.root_dir / p['results_dir']
        self.cache_file  = self.root_dir / p['cache_file']
        self.categories_tsv = self.things_dir / 'Categories_final_20200131_fixedUniqueID.tsv'

        # Expose other config sections
        self.analysis = cfg.get('analysis', {})
        self.hyperparameters = cfg.get('hyperparameters', {})
        self.plotting = cfg.get('plotting', {})

    # Helpers
    def get_macq_paths(self, monkey: str) -> dict:
        return {
            'data': self.timeavg_dir / f'monkey{monkey}.npy',
            'stiminfo': self.timeavg_dir / f'monkey{monkey}_stiminfo.csv',
        }

    def get_macq_tr_paths(self, monkey: str) -> dict:
        return {
            'data': self.timeres_dir / f'monkey{monkey}_tr.npy',
            'stiminfo': self.timeres_dir / f'monkey{monkey}_tr_stiminfo.csv',
        }

    def get_mri_paths(self, sub: str) -> dict:
        return {
            'betas': self.mri_dir / 'betas_csv' / f'sub-{sub}_ResponseData.h5',
            'stiminfo': self.mri_dir / f'sub-{sub}_stiminfo.csv',
            'brainmask': self.mri_dir / 'brainmasks' / f'sub-{sub}_space-T1w_brainmask.nii.gz',
            'voxmeta': self.mri_dir / 'betas_csv' / f'sub-{sub}_VoxelMetadata.csv',
        }

    def get_annotations_path(self) -> Path:
        """Hand-coded image labels shipped with the repo (preferred), else data tree."""
        candidates = [
            self.root_dir / 'data' / 'annotations.csv',
            Path(self.things_dir) / 'annotations.csv',
            Path(self.data_dir) / 'things' / 'annotations.csv',
        ]
        for path in candidates:
            if path.exists():
                return path
        return candidates[0]


config = Config()
