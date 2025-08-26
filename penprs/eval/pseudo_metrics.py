import numpy as np
from viprs.eval.pseudo_metrics import _match_variant_stats

def pseudo_r2(test_gdl, prs_beta_table):
    """
    `pseudo_r2` metric as in the viprs package: https://github.com/shz9/viprs
    Provides better validation curves and variable selection when used to 
    evaluate R-Squared in grid search than pearson_pseudo_r
    
    Compute the R-Squared metric (proportion of variance explained) for a given
    PRS using standardized marginal betas from an independent test set.
    Here, we follow the pseudo-validation procedures outlined in Mak et al. (2017) and
    Yang and Zhou (2020), where the proportion of phenotypic variance explained by the PRS
    in an independent validation cohort can be approximated with:

    R2(PRS, y) ~= 2*r'b - b'Sb

    Where `r` is the standardized marginal beta from a validation/test set,
    `b` is the posterior mean for the effect size of each variant and `S` is the LD matrix.

    :param test_gdl: An instance of `GWADataLoader` with the summary statistics table initialized.
    :param prs_beta_table: A pandas DataFrame with the PRS effect sizes. Must contain
    the columns: CHR, SNP, A1, A2, BETA.
    """

    std_beta, prs_beta, q = _match_variant_stats(test_gdl, prs_beta_table)

    rb = np.sum((prs_beta.T * std_beta).T, axis=0)
    bsb = np.sum(prs_beta*q, axis=0)

    return 2*rb - bsb


def pseudo_pearson_r2(test_gdl, prs_beta_table):
    """
    `pseudo_pearson_r2` metric as in the viprs package: https://github.com/shz9/viprs
    Used during results evaluation as pseudo_r2 above can be sensitive to sparsified LD
    especially on external methods.

    Perform pseudo-validation of the inferred effect sizes by comparing them to
    standardized marginal betas from an independent validation set. Here, we follow the pseudo-validation
    procedures outlined in Mak et al. (2017) and Yang and Zhou (2020), where
    the correlation between the PRS and the phenotype in an independent validation
    cohort can be approximated with:

    Corr(PRS, y) ~= r'b / sqrt(b'Sb)

    Where `r` is the standardized marginal beta from a validation set,
    `b` is the posterior mean for the effect size of each variant and `S` is the LD matrix.

    :param test_gdl: An instance of `GWADataLoader` with the summary statistics table initialized.
    :param prs_beta_table: A pandas DataFrame with the PRS effect sizes. Must contain
    the columns: CHR, SNP, A1, A2, BETA.
    """

    std_beta, prs_beta, q = _match_variant_stats(test_gdl, prs_beta_table)

    rb = np.sum((prs_beta.T * std_beta).T, axis=0)
    bsb = np.sum(prs_beta * q, axis=0)

    return (rb / np.sqrt(bsb))**2