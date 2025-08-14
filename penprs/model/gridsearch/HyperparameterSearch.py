import pandas as pd
import numpy as np
from multiprocessing import shared_memory
from pprint import pprint
from joblib import Parallel, delayed
from loky import get_reusable_executor
import copy
from tqdm import tqdm
import time

from sslprs.model.SSL import SSL

def fit_model_fixed_params(model, fixed_params, scale_l0_by_l1=False, shm_data=None, **fit_kwargs):
    """

    Perform model fitting using a set of fixed set of (hyper)-parameters.
    This is a helper function to allow users to use the `multiprocessing` module
    to fit PRS models in parallel.

    :param model: A PRS model object that implements a `.fit()` method and takes `fix_params` as an attribute.
    :param fixed_params: A dictionary of fixed parameters to use for the model fitting.
    :param shm_data: A dictionary of shared memory data to use for the model fitting. This is primarily used to
    share LD data across multiple processes.
    :param fit_kwargs: Key-word arguments to pass to the `.fit()` method of the PRS model.

    :return: The fitted PRS model. If the fitting fails, the function returns `None`.
    """
    
    model.fix_params = fixed_params

    if scale_l0_by_l1:
        model.fix_params["l0"] = model.fix_params["l0"]*model.fix_params["l1"]
        # print(model.fix_params["l0"])

    if shm_data is not None:

        model.ld_data = {}

        try:
            ld_data_shm = shared_memory.SharedMemory(name=shm_data['shm_name'])
            model.ld_data[shm_data['chromosome']] = np.ndarray(
                shape=shm_data['shm_shape'],
                dtype=shm_data['shm_dtype'],
                buffer=ld_data_shm.buf
            )
        except FileNotFoundError:
            raise Exception("LD data not found in shared memory.")

    try:
        model.fit(**fit_kwargs)
    except Exception as e:
        model = None
    finally:
        if shm_data is not None:
            ld_data_shm.close()
            model.ld_data = None

    if model is not None:
        return {
            'coef_table': model.to_table()[['CHR', 'SNP', 'POS', 'A1', 'A2', 'BETA']],
            'hyp_table': model.to_theta_table(),
            'training_objective': model.obj_f()
        }

def combine_coefficient_tables(coef_tables, coef_col='BETA'):
    """
    Combine a list of coefficient tables (output from a PRS model) into a single
    table that can be used for downstream tasks, such scoring and evaluation. Note that
    this implementation assumes that the coefficients tables were generated for the same
    set of variants, from a grid-search or similar procedure.

    :param coef_tables: A list of pandas dataframes containing variant information as well as
    inferred coefficients.
    :param coef_col: The name of the column containing the coefficients.
    :return: A single pandas dataframe with the combined coefficients. The new coefficient columns will be
    labelled as BETA_0, BETA_1, etc.
    """

    # Sanity checks:
    assert all([coef_col in t.columns for t in coef_tables]), "All tables must contain the coefficient column."
    assert all([len(t) == len(coef_tables[0]) for t in coef_tables]), "All tables must have the same number of rows."

    if len(coef_tables) == 1:
        return coef_tables[0]

    ref_table = coef_tables[0].copy()
    ref_table.rename(columns={coef_col: f'{coef_col}_0'}, inplace=True)

    # Extract the coefficients from the other tables:
    return pd.concat([ref_table, *[t[[coef_col]].rename(columns={coef_col: f'{coef_col}_{i}'})
                                   for i, t in enumerate(coef_tables[1:], 1)]], axis=1)

class HyperparameterSearch(object):

    def __init__(self,
                 gdl,
                 model=None,
                 criterion = "pseudo_validation",
                 validation_gdl=None,
                 verbose=False,
                 n_jobs=1,
                 threads=1):

        self.gdl = gdl
        self.n_jobs = n_jobs
        self.threads=threads

        # print(self.threads)
        if model is None:
            self.model = SSL(gdl, threads = self.threads)
        else:
            import inspect
            if inspect.isclass(model):
                self.model = model(gdl, threads = self.threads)
            else:
                self.model = model

        self.validation_result = None

        self.criterion = criterion
        self._validation_gdl = validation_gdl

        self.verbose = verbose
        self.model.verbose = verbose

        if self._validation_gdl is not None:
            self._validation_gdl.verbose = verbose

        if self.criterion == 'pseudo_validation':
            assert self._validation_gdl is not None
            assert self._validation_gdl.sumstats_table is not None

    def to_validation_table(self):
        """
        Summarize the validation results in a pandas table.
        """
        if self.validation_result is None:
            raise Exception("Validation result is not set!")
        elif len(self.validation_result) < 1:
            raise Exception("Validation result is not set!")

        return pd.DataFrame(self.validation_result)

    def write_validation_result(self, v_filename, sep="\t"):
        """
        After performing hyperparameter search, write a table
        that records that value of the objective for each combination
        of hyperparameters.
        :param v_filename: The filename for the validation table.
        :param sep: The separator for the validation table
        """

        v_df = self.to_validation_table()
        v_df.to_csv(v_filename, index=False, sep=sep)

    def multi_objective(self, models):
        """
        This method evaluates multiple PRS models simultaneously. This can be faster for
        some evaluation criteria, such as the validation R^2, because we only need to
        multiply the inferred effect sizes with the genotype matrix only once.

        :param models: A list of PRS models that we wish to evaluate.
        """

        if len(models) == 1:
            return self.objective(models[0])

        elif self.criterion == 'pseudo_validation':
            return [m.pseudo_validate(self._validation_gdl) for m in models]

    def objective(self, model):
        """
        A method that takes the result of fitting the model
        and returns the desired objective (either `ELBO`, `pseudo_validation`, or `validation`).
        :param model: The PRS model to evaluate
        """

        if self.criterion == 'pseudo_validation':
            return model.pseudo_validate(self._validation_gdl)
    def _evaluate_models(self):
        """
        This method evaluates multiple PRS models to determine their relative performance based on the
        criterion set by the user. The criterion can be the training objective (e.g. ELBO in the case of VIPRS),
        pseudo-validation or validation using held-out test data.

        :return: The metrics associated with each model setup.
        """

        assert self._training_objective is not None
        assert self._model_coefs is not None

        if self.criterion == 'training_objective':
            metrics = self._training_objective
        elif self.criterion == 'pseudo_validation':
            from penprs.eval.pseudo_metrics import pseudo_r2

            metrics = pseudo_r2(self._validation_gdl, self._model_coefs)

        return metrics
    
    def fit(self):
        raise NotImplementedError
    
class GridSearch(HyperparameterSearch):
    """
    Hyperparameter search using Grid Search
    """

    def __init__(self,
                 gdl,
                 grid,
                 model=None,
                 criterion='pseudo_validation',
                 validation_gdl=None,
                 verbose=False,
                 scale_l0_by_l1 = False,
                 n_jobs=1,
                 threads=1):

        """
        Perform hyperparameter search using grid search
        :param gdl: A GWADataLoader object
        :param grid: A HyperParameterGrid object
        :param model: A `PRSModel`-derived object (e.g. VIPRS).
        :param criterion: The objective function for the grid search (ELBO or validation).
        :param validation_gdl: If the objective is validation, provide the GWADataLoader object for the validation
        dataset.
        :param verbose: Detailed messages and print statements.
        :param n_jobs: The number of processes to use for the grid search
        """

        super().__init__(gdl, model=model, criterion=criterion,
                         validation_gdl=validation_gdl,
                         verbose=verbose,
                         n_jobs=n_jobs,
                         threads=threads)

        self.grid = grid
        self.scale_l0_by_l1 = scale_l0_by_l1

    def fit(self, max_iter=1000, abs_tol=1e-6):

        """
        Perform grid search over the hyperparameters to determine the
        best model based on the criterion set by the user. This utility method
        performs model fitting across the grid of hyperparameters, potentially in parallel
        if `n_jobs` is greater than 1.

        :param max_iter: The maximum number of iterations to run for each model fit.
        :param f_abs_tol: The absolute tolerance for the function convergence criterion.
        :param x_abs_tol: The absolute tolerance for the parameter convergence criterion.

        :return: The best model based on the criterion set by the user.
        """

        #keep track of fitting and validation time
        fit_valid_time = {}

        fit_start_time = time.time()

        print("> Performing Grid Search over the following grid:")
        print(self.grid.to_table())

        if self.n_jobs > 1:
            # Only create the shared memory object if the number of processes is more than 1.
            # Otherwise, this would be a waste of resources.

            # ----------------- Copy the LD data to shared memory -----------------
            ld_data_arr = self.model.ld_data
            # Create a shared memory block for the array
            shm = shared_memory.SharedMemory(create=True, size=ld_data_arr.nbytes)

            # Create a NumPy array backed by the shared memory block
            shared_array = np.ndarray(ld_data_arr.shape, dtype=ld_data_arr.dtype, buffer=shm.buf)

            np.copyto(shared_array, ld_data_arr)

            del ld_data_arr
            self.model.ld_data = None

            shm_args = {
                'shm_name': shm.name,
                'chromosome': self.model.chr,
                'shm_shape': shared_array.shape,
                'shm_dtype': shared_array.dtype
            }

        else:
            shm_args = None

        # --------------------------------------------------------------------
        # Perform grid search:

        grid = self.grid.combine_grids()

        parallel = Parallel(n_jobs=self.n_jobs, backend="multiprocessing")

        with parallel:

            fitted_models = parallel(
                delayed(fit_model_fixed_params)(self.model, g, self.scale_l0_by_l1, shm_args,
                                                max_iter=max_iter,
                                                abs_tol=abs_tol)
                for g in tqdm(grid, desc="Fitting Models")
            )

        # Clean up after performing model fit:
        self.model.ld_data = None  # To minimize memory usage with validation/pseudo-validation

        # Close and unlink shared memory objects:
        if shm_args is not None:
            shm.close()
            shm.unlink()

        fit_end_time = time.time()

        fit_valid_time["Fit_time"] = round(fit_end_time - fit_start_time, 2)

        # --------------------------------------------------------------------
        # Post-process the results and determine the best model:

        valid_start_time = time.time()

        assert not all([fm is None for fm in fitted_models]), "None of the models converged successfully."

        self._model_coefs = combine_coefficient_tables([fm['coef_table'] for fm in fitted_models if fm is not None])
        self._model_hyperparams = [fm['hyp_table'] for fm in fitted_models if fm is not None]
        self._training_objective = [fm['training_objective'] for fm in fitted_models if fm is not None]

        # 2) Perform evaluation on the models that converged:
        eval_metrics = self._evaluate_models()

        # 3) Combine all the results together into a single table (populate records in
        # self.validation_result):

        self.validation_result = []
        success_counter = 0

        for i, vr in enumerate(grid):
            if fitted_models[i] is not None:
                vr['Converged'] = True
                vr['training_objective'] = self._training_objective[success_counter]
                if self.criterion != 'training_objective':
                    vr[self.criterion] = eval_metrics[success_counter]
                success_counter += 1
            else:
                vr['Converged'] = False
                vr['training_objective'] = np.NaN
                if self.criterion != 'training_objective':
                    vr[self.criterion] = np.NaN

            self.validation_result.append(vr)

        # --------------------------------------------------------------------
        # Determine and return the best model:
        validation_table = self.to_validation_table()
        best_idx = np.argmax(validation_table[self.criterion].values)

        print("> Grid search identified the best hyperparameters as:")
        pprint(grid[best_idx])

        self.model.fix_params = grid[best_idx]
        self.model.initialize()
        self.model.betas=fitted_models[best_idx]['coef_table']["BETA"]
        # self.model.set_model_parameters(fitted_models[best_idx]['coef_table'])

        valid_end_time = time.time()

        fit_valid_time["Validation_time"] = valid_end_time - valid_start_time

        return self.model, fit_valid_time, validation_table

    # def select_best_model(self):

    #     if len(self.fit_results) > 1:
    #         res_objectives = self.multi_objective(self.fit_results)
    #     else:
    #         raise Exception("Error: Convergence was achieved for less than 2 models.")

    #     if self.criterion == 'pseudo_validation':
    #         for i in range(len(self.validation_result)):
    #             self.validation_result[i]['Pseudo_Validation_Corr'] = res_objectives[i]

    #     if self.criterion == "pseudo_validation":
    #         best_idx = np.argmax(res_objectives)
    #         best_params = self.params[best_idx]

    #     print("> Grid search identified the best hyperparameters as:")
    #     pprint(best_params)

    #     print("> Refitting the model with the best hyperparameters...")

    #     self.model.fix_params = best_params
    #     self.model.fit()

    #     return self
    def to_table(self):
        return self.model.to_table()