import numpy as np
import pandas as pd
import os.path as osp
import logging
from viprs.utils.compute_utils import expand_column_names


class PRSModel:
    """
    A base class for PRS models.

    This class defines the basic structure and methods
    that are common to most PRS models. Specifically, this class provides methods and interfaces
    for initialization, harmonization, prediction, and fitting of PRS models.

    The class is generic is designed to be inherited and extended by
    a variety of PRS models, such as `Lasso` and `SSL`.

    :ivar gdl: A GWADataLoader object containing harmonized GWAS summary statistics and
    Linkage-Disequilibrium (LD) matrices.
    :ivar Nj: A numpy array containing the sample sizes per variant.
    :ivar shapes: The shapes of the variant arrays (e.g. the number of variants per chromosome).
    :ivar _sample_size: The maximum per-SNP sample size.
    :ivar betas: The coefficients for the variants.
    """

    def __init__(self,
        gdl,
        fix_params=None,
        tracked_params=None,
        float_precision='float32',
        lambda_min=None,
        verbose=True,
        low_memory=False,
        dequantize_on_the_fly=False,
        threads=1):
        """
        Initialize the PRS model.
        :param gdl: An instance of `GWADataLoader`.
        :param fix_params: A dictionary of fixed parameters.
        :param tracked_params: A list of parameters to track during the optimization.
        :param float_precision: The float precision to use for the model.
        :param lambda_min: The minimum value for the regularization parameter.
        :param verbose: Whether to print progress messages.
        :param low_memory: Whether to use low memory mode.
        :param dequantize_on_the_fly: Whether to dequantize the LD matrix on the fly.
        :param threads: The number of threads to use for parallel computation.
        """

        # ---------- General properties: ----------

        self.fix_params = fix_params or {}
        self.tracked_params = tracked_params or ['objective']
        self.float_precision = float_precision
        self.verbose = verbose
        self.gdl = gdl
        self.low_memory = low_memory
        self.threads = threads
        self.history = None

        # ----------------- Sanity checks -----------------

        assert gdl.ld is not None, "The LD matrices must be initialized in the GWADataLoader object."
        assert gdl.sumstats_table is not None, ("The summary statistics must be "
                                                "initialized in the GWADataLoader object.")

        # -------------------- The data --------------------


        self.chromosome = gdl.chromosomes[0]
        self.shape = self.gdl.shapes[self.chromosome]

        # >>>>>> Summary statistics <<<<<<
        ss_table = gdl.sumstats_table[self.chromosome]
        # Sample size per SNP:
        self.n_per_snp = ss_table.n_per_snp.astype(self.float_precision)
        self.std_beta = ss_table.get_snp_pseudo_corr().astype(self.float_precision)

        # Determine the overall sample size:
        self._sample_size = np.max(self.n_per_snp)

        # >>>>>> LD matrices <<<<<<

        ld_mat = self.gdl.ld[self.chromosome]

        # Determine how to load the LD data:
        if dequantize_on_the_fly and np.issubdtype(ld_mat.stored_dtype, np.integer):
            ld_dtype = ld_mat.stored_dtype
        else:
            if dequantize_on_the_fly:
                logging.warning("Dequantization on the fly is only supported for "
                                "integer data types. Ignoring this flag.")

            ld_dtype = float_precision
            dequantize_on_the_fly = False

        self.ld_data, self.ld_indptr, self.ld_left_bound = ld_mat.load_data(
                return_symmetric=not low_memory,
                dtype=ld_dtype
        )

        # ------------- Data-dependent properties -------------

        self.dequantize_on_the_fly = dequantize_on_the_fly

        if self.dequantize_on_the_fly:
            info = np.iinfo(self.ld_data.dtype)
            self.dequantize_scale = 1. / info.max
        else:
            self.dequantize_scale = 1.

        if lambda_min is None or lambda_min == "infer":
            self.lambda_min = self.gdl.ld[self.chromosome].get_lambda_min('min')
        else:
            self.lambda_min = lambda_min

        # ----------------- Model parameters -----------------

        # Coefficients:
        self.betas = None
        # Multiplication of coefficients with LD matrix:
        self.q = None
        # Coefficient differences (between iterations)
        self.betas_diff = None


    @property
    def m(self) -> int:
        """

        !!! seealso "See Also"
            * [n_snps][penprs.model.PRSModel.PRSModel.n_snps]

        :return: The number of variants in the model.
        """
        return self.gdl.m

    @property
    def n(self) -> int:
        """
        :return: The number of samples in the dataset.
        """
        return self._sample_size

    @property
    def n_snps(self) -> int:
        """
        !!! seealso "See Also"
            * [m][penprs.model.PRSModel.PRSModel.m]

        :return: The number of variants in the model.
        """
        return self.m

    @property
    def nnz(self) -> int:
        """
        :return: The number of non-zero coefficients.
        """
        assert self.betas is not None, "The coefficients BETA are not set. Call `.fit()` first."
        return np.count_nonzero(self.betas, axis=0)

    @property
    def max_betas_diff(self) -> float:
        """
        :return: The maximum absolute difference in the coefficients between iterations.
        """
        assert self.betas_diff is not None, "The coefficients BETA are not set. Call `.fit()` first."
        return np.max(np.abs(self.betas_diff))

    def _compute_lambda_max(self):
        """
        Compute the maximum LASSO penalty based on the input data.
        """
        return np.max(np.abs(self.n_per_snp * self.std_beta))

    def initialize(self, theta_0=None, param_0=None):

        self.initialize_hyperparameters(theta_0)
        self.initialize_model_parameters(param_0)
        self.init_history()

    def init_history(self):
        """
        Initialize the history dictionary.
        """

        self.history = {

        }

        for tt in self.tracked_params:
            if isinstance(tt, str):
                self.history[tt] = []
            elif callable(tt):
                self.history[tt.__name__] = []

        # Ensure that the objective is always tracked:
        if 'objective' not in self.history:
            self.history['objective'] = []

    def initialize_hyperparameters(self, theta_0=None):
        """
        Initialize the hyperparameters for the model.
        """
        raise NotImplementedError

    def initialize_model_parameters(self, param_0=None):

        param_0 = param_0 or {}

        self.q = np.zeros(self.n_snps, dtype=self.float_precision)

        if 'betas' in param_0:
            self.betas = param_0['betas'].astype(self.float_precision)
            self.p_gamma = np.count_nonzero(self.betas)
        else:
            self.betas = np.zeros(self.n_snps, dtype=self.float_precision)
            self.p_gamma = 0

        self.betas_diff = np.zeros(self.n_snps, dtype=self.float_precision)

    def get_coefficients(self):
        """
        Get the coefficients for the variants.
        :return: The coefficients for the variants.
        """
        return self.betas

    def fit(self, *args, **kwargs):
        """
        A genetic method to fit the PRS model.
        :raises NotImplementedError: If the method is not implemented in the child class.
        """
        raise NotImplementedError

    def objective(self):
        """
        :return: The objective function.
        """
        raise NotImplementedError

    def mse(self):
        """
        :return: The mean squared error.
        """

        return (1. - 2.*self.std_beta.dot(self.betas) + np.multiply(self.q, self.betas).sum(axis=0) +
            (1 + self.lambda_min)*(self.betas**2).sum(axis=0))

    def penalty(self):
        """
        :return: The penalty term.
        :raises NotImplementedError: If the method is not implemented in the child class.
        """

        raise NotImplementedError


    def predict(self, test_gdl=None):
        """
        Given the inferred effect sizes, predict the phenotype for the training samples in
        the GWADataLoader object or new test samples. If `test_gdl` is not provided, genotypes
        from training samples will be used (if available).

        :param test_gdl: A GWADataLoader object containing genotype data for new test samples.
        :raises ValueError: If the posterior means for BETA are not set. AssertionError if the GWADataLoader object
        does not contain genotype data.
        """

        if self.betas is None:
            raise ValueError("The coefficients BETA are not set. Call `.fit()` first.")

        if test_gdl is None:
            assert self.gdl.genotype is not None, "The GWADataLoader object must contain genotype data."
            test_gdl = self.gdl
            betas = self.betas
        else:
           betas = self.harmonize_data(gdl=test_gdl)

        return test_gdl.predict(betas)

    def harmonize_data(self, gdl=None, parameter_table=None):
        """
        Harmonize the inferred effect sizes with a new GWADataLoader object. This method is useful
        when the user wants to predict on new samples or when the effect sizes are inferred from a
        different set of samples. The method aligns the effect sizes with the SNP table in the
        GWADataLoader object.

        :param gdl: An instance of `GWADataLoader` object.
        :param parameter_table: A `pandas` DataFrame of variant effect sizes.

        :return: A tuple of the harmonized posterior inclusion probability, posterior mean for the effect sizes,
        and posterior variance for the effect sizes.

        """

        if gdl is None and parameter_table is None:
            return

        if gdl is None:
            gdl = self.gdl

        if parameter_table is None:
            parameter_table = self.to_table()
        else:
            parameter_table = parameter_table.loc[parameter_table['CHR'] == self.chromosome, ]

        snp_table = gdl.to_snp_table(col_subset=['SNP', 'A1', 'A2'],
            per_chromosome=True)[self.chromosome]


        from magenpy.utils.model_utils import merge_snp_tables

        beta_cols = expand_column_names('BETA', self.betas.shape)

        # Merge the effect table with the GDL SNP table:
        c_df = merge_snp_tables(snp_table, parameter_table, how='left',
                                signed_statistics=beta_cols)

        if len(c_df) < len(snp_table):
            raise ValueError("The parameter table could not aligned with the reference SNP table. "
                             "This may due to conflicts/errors in use of reference vs. "
                             "alternative alleles.")

        # Obtain the values for the coefficients:
        c_df[beta_cols] = c_df[beta_cols].fillna(0.)

        return c_df[beta_cols].values

    def to_table(self,
        col_subset=('CHR', 'SNP', 'POS', 'A1', 'A2'),
        prune=False):

        assert self.betas is not None, "The effect sizes are not set. Call `.fit()` first."

        if prune:
            # Check that there is only one set of effect sizes:
            assert self.betas.shape == (self.n_snps,)

        table = self.gdl.to_summary_statistics_table(col_subset=col_subset)

        from viprs.utils.compute_utils import expand_column_names

        table = pd.concat([
            table,
            pd.DataFrame(self.betas,
                         columns=expand_column_names('BETA', self.betas.shape),
                         index=table.index)
        ], axis=1)

        if prune:
            table = table.loc[table['BETA'] != 0, ].reset_index(drop=True)

        return table

    def to_history_table(self):
        """
        :return: A `pandas` DataFrame containing the history of tracked
        parameters as a function of the number of iterations.
        """
        return pd.DataFrame(self.history)

    def to_hyperparameter_table(self):
        """
        :return: A `pandas` DataFrame containing the hyperparameters of the model.
        """
        raise NotImplementedError

    def update_history(self):
        """
        Update the history dictionary with the current values of the
        tracked parameters and objectives.
        """

        for tt in self.tracked_params:

            if isinstance(tt, str) and hasattr(self, tt):
                attr = getattr(self, tt)
                if callable(attr):
                    self.history[tt].append(attr())
                else:
                    self.history[tt].append(attr)
            elif callable(tt):
                self.history[tt.__name__].append(tt(self))

    def pseudo_validate(self, test_gdl, metric='r2'):
        """
        Evaluate the prediction accuracy of the inferred PRS using external GWAS summary statistics.

        :param test_gdl: A `GWADataLoader` object with the external GWAS summary statistics and LD matrix information.
        :param metric: The metric to use for evaluation. Options: 'r2' or 'pearson_correlation'.

        :return: The pseudo-validation metric.
        """

        from viprs.eval.pseudo_metrics import pseudo_r2, pseudo_pearson_r

        metric = metric.lower()

        assert self.betas is not None, "The posterior means for BETA are not set. Call `.fit()` first."

        param_table = self.to_table()

        if metric in ('pearson_correlation', 'corr', 'r'):
            return pseudo_pearson_r(test_gdl, param_table)
        elif metric == 'r2':
            return pseudo_r2(test_gdl, param_table)
        else:
            raise KeyError(f"Pseudo validation metric ({metric}) not recognized. "
                           f"Options are: 'r2' or 'pearson_correlation'.")

    def select_best_model(self, validation_gdl=None, criterion='objective'):

        """
        From the grid of models that were fit to the data, select the best
        model according to the specified `criterion`. If the criterion is the objective,
        the model with the highest objective will be selected. If the criterion is
        validation or pseudo-validation, the model with the highest R-squared on the
        validation set will be selected.

        :param validation_gdl: An instance of `GWADataLoader` containing data from the validation set.
        Must be provided if criterion is `validation` or `pseudo_validation`.
        :param criterion: The criterion for selecting the best model.
        Options are: (`objective`, `validation`, `pseudo_validation`)
        """

        # Sanity checks:
        assert criterion in ('objective', 'validation', 'pseudo_validation')
        assert self.betas is not None, "The effect sizes are not set. Call `.fit()` first."
        assert len(self.betas.shape) > 1, "Multiple models must be fit to select the best model."
        assert self.betas.shape[1] > 1, "Multiple models must be fit to select the best model."

        if criterion == 'objective':
            # TODO: Make sure the .objective() method
            # accommodates multiple betas.
            best_model_idx = np.argmax(self.objective())
        elif criterion == 'validation':

            assert validation_gdl is not None
            assert validation_gdl.sample_table is not None
            assert validation_gdl.sample_table.phenotype is not None

            from viprs.eval.continuous_metrics import r2

            # TODO: Fix the predict method
            prs = self.predict(test_gdl=validation_gdl)
            prs_r2 = np.array([r2(prs[:, i], validation_gdl.sample_table.phenotype)
                                for i in range(self.betas.shape[1])])
            best_model_idx = np.argmax(prs_r2)

        elif criterion == 'pseudo_validation':

            pseudo_r2 = self.pseudo_validate(validation_gdl, metric='r2')
            best_model_idx = np.argmax(np.nan_to_num(pseudo_r2, nan=0., neginf=0., posinf=0.))

        if int(self.verbose) > 1:
            logging.info(f"> Based on the {criterion} criterion, selected model: {best_model_idx}")
            logging.info("> Model details:\n")

        # -----------------------------------------------------------------------

        # Update the model parameters:
        self.betas = self.betas[:, best_model_idx]
        self.q = self.q[:, best_model_idx]

        # -----------------------------------------------------------------------

        return best_model_idx

    def set_model_parameters(self, parameter_table):
        """
        Parses a pandas dataframe with model parameters and assigns them
        to the corresponding class attributes.

        For example:
            * Columns with `BETA`, will be assigned to `self.betas`.

        :param parameter_table: A pandas dataframe.
        """

        self.betas = self.harmonize_data(parameter_table=parameter_table)

    def read_inferred_parameters(self, f_names, sep=r"\s+"):
        """
        Read a file with the inferred parameters.
        :param f_names: A path (or list of paths) to the file with the effect sizes.
        :param sep: The delimiter for the file(s).
        """

        if isinstance(f_names, str):
            f_names = [f_names]

        param_table = []

        for f_name in f_names:
            param_table.append(pd.read_csv(f_name, sep=sep))

        if len(param_table) > 0:
            param_table = pd.concat(param_table)
            self.set_model_parameters(param_table)
        else:
            raise FileNotFoundError

    def write_inferred_parameters(self, f_name, sep="\t"):
        """
        A convenience method to write the inferred posterior for the effect sizes to file.

        TODO:
            * Support outputting scoring files compatible with PGS catalog format:
            https://www.pgscatalog.org/downloads/#dl_scoring_files

        :param f_name: The filename where to write the effect sizes
        :param sep: The delimiter for the file (tab by default).
        """

        tables = self.to_table()

        if '.fit' not in f_name:
            ext = '.fit'
        else:
            ext = ''

        try:
            tables.to_csv(f_name + ext, sep=sep, index=False)
        except Exception as e:
            raise e
