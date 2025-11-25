# PenPRS: Penalized Regression for Inference of Polygenic Risk Scores

**PenPRS** is a Python package that includes summary statistics based sparse and accurate penalized regression models for the inference of Polygenic Risk Scores (PRS) using Linkage Disequilibrium (LD) and GWAS Summary Statistics. The models currently available include:
* LASSO
* SSL (Spike-and-slab LASSO PRS) 
    * SSLAlpha (Alpha-heritability modeled SSL)

PenPRS integrates with [`magenpy`](https://github.com/shz9/magenpy) for data harmonization and LD loading and computation. Details related to model methodology and specifications are present in the paper.
## Reference 

>Song, J., Zabad, S., Yang, A., Gravel, S., & Li, Y. (2025). **Sparse polygenic risk score inference with the spike-and-slab LASSO**. *Bioinformatics*. https://doi.org/10.1093/bioinformatics/btaf578

## Installation 

To install the package from `PyPI`, use the following command:

```bash
pip install penprs
```

## Quickstart

```python
import magenpy as mgp
from penprs.model.Lasso import Lasso
from penprs.model.SSL import SSL
from penprs.model.SSLAlpha import SSLAlpha

# Load a genotype reference and GWAS summary statistics data (chromosome 22)
gdl = mgp.GWADataLoader(mgp.tgp_eur_data_path(),
                        sumstats_files=mgp.ukb_height_sumstats_path(),
                        sumstats_format="fastgwa")

# Compute block LD matrix:
ld_block_url = "https://bitbucket.org/nygcresearch/ldetect-data/raw/ac125e47bf7ff3e90be31f278a7b6a61daaba0dc/EUR/fourier_ls-all.bed"
gdl.compute_ld('block',
               ld_blocks_file=ld_block_url,
               dtype='int16',
               compute_spectral_properties=True,
               output_dir='temp/block_ld/')

# Initialize respective models
lasso_model = Lasso(gdl, lam=100)
ssl_model = SSL(gdl, l0=750, l1=10)

# Perform single fit using model initialized arguments
lasso_model.fit()   # fit using lam=100
ssl_model.fit()     # fit using l0=750, l1=10
```

### Viewing model effect size estimates (`BETA`):
```python
# Viewing the effect size estimates (BETA) for LASSO
lasso_model.to_table().head()
```
| CHR | SNP        | POS      | A1 | A2 |   BETA   |
|-----|------------|----------|----|----|----------|
| 22  | rs131538   | 16871137 | A  | G  |  0.001439|
| 22  | rs9605903  | 17054720 | C  | T  | -0.000000|
| 22  | rs5746647  | 17057138 | G  | T  | -0.001534|
| 22  | rs16980739 | 17058616 | T  | C  | -0.001342|
| 22  | rs9605923  | 17065079 | A  | T  |  0.006215|

### Pathwise and Warm-Start fashioned fit
```python
# To solve in a path-wise or warm-start fashion
# save_intermediate=True to save effect size estimate per ladder step
lasso_model_pw = Lasso(gdl)
ssl_model_ws = SSL(gdl)

# Performs path-wise fit across the default 20 step ladder
lasso_model_pw.pathwise_fit(save_intermediate=True) 

# Performs warm-start fit across the default 20 step ladder
ssl_model_ws.warm_start_fit(save_intermediate=True)  
```
## Command Line Interface (CLI)

You can fit the `SSL` model with Grid-Search (GS) from the command line using `penprs_fit`:

```bash
penprs_fit -l path_to_train_ld \
		-s path_to_train_sumstats \
		--output-dir path_to_output_dir \
		-m SSL \              # model: options = LASSO, SSL, SSLAlpha
		--hyp-search GS \     # hyperparameter search: options = GS (Grid Search), WS (Warm Start)
		--validation-ld-panel path_to_validation_ld \
		--validation-sumstats path_to_validation_sumstats \
		--use-symmetric-ld \
		--grid-metric pseudo_validation \
		--threads 4
```

For more information on the command line arguments, see `penprs_fit --help`

## Citation

If you use this package in your research, please cite the following paper:

```bibtex
@article{10.1093/bioinformatics/btaf578,
    author = {Song, Junyi and Zabad, Shadi and Yang, Archer and Gravel, Simon and Li, Yue},
    title = {Sparse Polygenic Risk Score Inference with the Spike-and-Slab LASSO},
    journal = {Bioinformatics},
    pages = {btaf578},
    year = {2025},
    month = {10},
    doi = {10.1093/bioinformatics/btaf578},
    url = {https://doi.org/10.1093/bioinformatics/btaf578},
    eprint = {https://academic.oup.com/bioinformatics/advance-article-pdf/doi/10.1093/bioinformatics/btaf578/64738987/btaf578.pdf},
}
```
