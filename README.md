# PenPRS: Penalized Regression for Inference of Polygenic Risk Scores

:hammer: **Under Construction** :construction:

```
-->-->-->-->->->->->-------<-<-<--<-<--<-<--<--

    ▗▄▄▖  ▗▄▄▄▖ ▗▖  ▗▖ ▗▄▄▖  ▗▄▄▖   ▗▄▄▖
    ▐▌ ▐▌ ▐▌    ▐▛▚▖▐▌ ▐▌ ▐▌ ▐▌ ▐▌ ▐▌
    ▐▛▀▘  ▐▛▀▀▘ ▐▌ ▝▜▌ ▐▛▀▘  ▐▛▀▚▖  ▝▀▚▖
    ▐▌    ▐▙▄▄▖ ▐▌  ▐▌ ▐▌    ▐▌ ▐▌ ▗▄▄▞▘

Penalized Regression for Polygenic Risk Scores
  Version: 0.0.1 | Release date: January 2025
      Authors: Shadi Zabad & Jack Song
             McGill University
-->-->-->-->->->->->-------<-<-<--<-<--<-<--<--
```

## Installation

To install the package from `GitHub`, use the following command:

```bash
pip install git+https://github.com/shz9/penprs.git
```

## Usage

```python
import magenpy as mgp
from penprs.model.Lasso import Lasso

# Load the data
gdl = mgp.GWADataLoader(mgp.tgp_eur_data_path(),
                        sumstats_files=mgp.ukb_height_sumstats_path(),
                        sumstats_format="fastgwa")

# Compute LD matrix:
ld_block_url = "https://bitbucket.org/nygcresearch/ldetect-data/raw/ac125e47bf7ff3e90be31f278a7b6a61daaba0dc/EUR/fourier_ls-all.bed"
gdl.compute_ld('block',
               ld_blocks_file=ld_block_url,
               dtype='int16',
               compute_spectral_properties=True,
               output_dir='temp/block_ld/')

# Initialize Lasso model:
lasso_model = Lasso(gdl, lam=100)

# Perform model fit:
lasso_model.fit()

```

## Citation

If you use this package in your research, please cite the following paper:

```bibtex

@article {Song2025.01.28.25321292,
	author = {Song, Junyi and Zabad, Shadi and Yang, Archer and Li, Yue},
	title = {Sparse Polygenic Risk Score Inference with the Spike-and-Slab LASSO},
	elocation-id = {2025.01.28.25321292},
	year = {2025},
	doi = {10.1101/2025.01.28.25321292},
	publisher = {Cold Spring Harbor Laboratory Press},
	URL = {https://www.medrxiv.org/content/early/2025/01/29/2025.01.28.25321292},
	eprint = {https://www.medrxiv.org/content/early/2025/01/29/2025.01.28.25321292.full.pdf},
	journal = {medRxiv}
}

```
