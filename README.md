# Shared and Distinct High-Dimensional Object Spaces in Human and Macaque Inferotemporal Cortex

## Setup

Run commands from the repository root:

```bash
conda env create -f environment.yml
conda activate primate_align
python functions/srf/build_extension.py
```

The extension build requires a C compiler. Settings and worker counts are in
`config/config.toml`; adjust workers to the available CPUs and memory.

## Data

Run `fetch/download_data.ipynb` to download and prepare the inputs. For the
THINGS image password, obtain `password_images.txt` from the
[THINGS OSF repository](https://osf.io/jum2f/) and paste it into
`things_img_password` in the notebook.

Data default to `data/`. An alternative location can be set through `data_root`
in `config/config.toml` or the `PRIMATE_ALIGN_DIR` environment variable before
starting Jupyter or running analyses. When using another location, also copy
the bundled `data/features/vis_props/` folder to `<data_root>/features/vis_props/`.
Annotations are included in `data/annotations.csv`.

DNN extraction code and its separate environment are in `analysis/dnn/`.
Set its input and output paths in `analysis/dnn/layer_config.toml`.
Extraction inputs should contain only the 8,640 images listed in the annotations;
feature rows must follow sorted stimulus-name order. DINOv3 requires access to
its Hugging Face weights. Generate DNN features before DNN encoding or
residualized decoding.

## Analyses

Main commands, in order:

```bash
python model/CCA/param_gridsearch.py
python model/CCA/CCA_families.py
python model/CCA/param_crossview.py
python model/CCA/CCA_ridge.py

python model/SRF/SRF.py
python analysis/nmf/ridge_guard.py

python model/feature/ridge_component.py
python model/feature/ridge_species_weighted.py
python model/feature/species_modality.py
python analysis/category/category_asymm.py
python analysis/symbol/decode.py --regions it v1 v4 --residualized
```

Visualizations and additional analyses are under `viz/` and `analysis/`.

## License

See [LICENSE](LICENSE). The vendored PySRF implementation and attribution are
described in [functions/srf/README.md](functions/srf/README.md).
