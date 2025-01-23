import numpy as np
import pandas as pd
from tqdm import tqdm
import logging

from .PRSModel import PRSModel
from ..utils.exceptions import OptimizationDivergence
from .co.coord_opt import cpp_update_beta_lasso

class Lasso(PRSModel):

    def __init__(self,
                 gdl,
                 lam=None,
                 **prs_model_kwargs):

        super().__init__(gdl, **prs_model_kwargs)
        self.lam = lam

    def initialize_hyperparameters(self, theta_0=None):

        if theta_0 is None:
            if self.fix_params is not None:
                theta_0 = self.fix_params
            else:
                theta_0 = {}

        # Default settings for the Lasso hyperparameters:
        lam_max = self._compute_lambda_max()

        if 'lambda' in theta_0:
            self.lam = theta_0['lambda']
        elif 'min_lambda_frac' in theta_0:
            self.lam = theta_0['min_lambda_frac'] * lam_max
        elif self.lam is None:
            # By default, scale lambda_max by 10^-3:
            self.lam = 1e-3 * lam_max

        self.lambda_min = np.dtype(self.float_precision).type(self.lambda_min)
        self.lam = np.dtype(self.float_precision).type(self.lam)

    def _coord_ascent_step(self):
        """
        Perform a single iteration of the coordinate ascent algorithm
        for the Lasso model, given the current settings of the
        coefficients and hyperparameters.
        """

        cpp_update_beta_lasso(
            self.ld_left_bound, self.ld_indptr, self.ld_data,
            self.std_beta,
            self.betas, self.betas_diff,
            self.q, self.n_per_snp,
            self.lam,
            self.lambda_min,
            self.dequantize_scale,
            self.threads,
            self.low_memory
        )

    def penalty(self):
        """
        :return: The Lasso penalty.
        """
        return self.lam * np.sum(np.abs(self.betas), axis=0)

    def objective(self):

        return -.5*self.n*self.mse() - self.penalty()

    def fit(self,
            max_iter=1000,
            theta_0=None,
            param_0=None,
            continued=False,
            min_iter=3,
            f_abs_tol=1e-4,
            x_abs_tol=1e-4,
            disable_pbar=False):

        # initialize parameters
        if not continued:
            self.initialize(theta_0, param_0)
            self.update_history()

        pbar = tqdm(range(max_iter),
                    disable=not self.verbose or disable_pbar,
                    desc=f'Chromosome {self.chromosome} ({self.n_snps} variants)')

        curr_obj = prev_obj = self.history['objective'][-1]
        stop_iter = False

        for i in pbar:

            if stop_iter:
                pbar.set_postfix({
                    'Final objective': f"{curr_obj:.3f}",
                    'Nonzero': self.nnz
                })
                pbar.n = i - 1
                pbar.total = i - 1
                pbar.refresh()
                pbar.close()
                break

            # Run the coordinate ascent step:
            self._coord_ascent_step()
            # Update the tracked parameters (including objectives):
            self.update_history()

            # Extract the current value for the objective:
            curr_obj = self.history['objective'][-1]

            # Update the progress bar:
            pbar.set_postfix(
                {
                    'Objective': f"{curr_obj:.3f}",
                }
            )

            if i > min_iter:
                # Check for convergence:
                if not np.isfinite(curr_obj):
                    raise OptimizationDivergence(f"Stopping at iteration {i}: "
                                     f"The optimization algorithm is not converging!\n"
                                     f"The objective is undefined ({curr_obj}).")
                elif self.mse() < 0:
                    raise OptimizationDivergence(f"Stopping at iteration {i}: "
                        f"The optimization algorithm is not converging!\n"
                        f"The MSE is negative ({self.mse()}).")
                elif np.isclose(prev_obj, curr_obj, atol=f_abs_tol, rtol=0.):
                    stop_iter = True
                elif self.max_betas_diff < x_abs_tol:
                    stop_iter = True

        return self

    def pathwise_fit(self,
                     lambda_path=None,
                     path_steps=20,
                     max_lambda_frac=.99,
                     largest_first=True,
                     save_intermediate=False,
                     theta_0=None,
                     param_0=None,
                     **fit_kwargs):

        """
        Fit the model using a warm-start strategy, where the model is fit
        multiple times with different values of the l0 hyperparameter.

        :param lambda_path: A list of values for the lambda hyperparameter to use
            in the warm-start strategy. If None, a default ladder of values
            will be used.
        :param path_steps: The number of steps to use in the ladder of lambda values.
        :param max_lambda_frac: The maximum value of the lambda hyperparameter.
        :param largest_first: If True, the ladder of lambda values will be
            sorted in descending order.
        :param save_intermediate: If True, the model parameters will be saved
            for each `l0` value.
        :param theta_0: The initial values for the hyperparameters.
        :param param_0: The initial values for the model parameters.
        :param fit_kwargs: Additional keyword arguments to pass to the `fit`
        """

        assert lambda_path is not None or path_steps is not None, (
            "Either `lambda_path` or `path_steps` must be provided."
        )

        self.initialize(theta_0=theta_0, param_0=param_0)

        if lambda_path is None:
            # Generate a default ladder of lambda values
            # by interpolating from lambda_max to
            # the current lambda value.
            lam_max = max_lambda_frac*self._compute_lambda_max()
            lambda_path = np.logspace(np.log2(self.lam),
                np.log2(lam_max),
                num=path_steps,
                base=2)

        lambda_path = np.sort(lambda_path)
        if largest_first:
            lambda_path = lambda_path[::-1]

        if save_intermediate:
            betas = np.empty((self.n_snps, len(lambda_path)), dtype=self.float_precision)
            qs = np.empty((self.n_snps, len(lambda_path)), dtype=self.float_precision)

        pbar = tqdm(range(len(lambda_path)),
                    total=len(lambda_path),
                    disable=not self.verbose,
                    desc=f'Chromosome {self.chromosome} ({self.n_snps} variants)')

        for i in pbar:

            self.lam = lambda_path[i]
            if i == 0:
                self.update_history()

            self.fit(continued=True, disable_pbar=True, **fit_kwargs)

            pbar.set_postfix(
                {
                    'Objective': f"{self.objective():.3f}",
                    'Nonzero': self.nnz,
                    'Lambda': f"{self.lam:.2f}"
                }
            )

            if save_intermediate:
                betas[:, i] = self.betas
                qs[:, i] = self.q

        if save_intermediate:
            self.betas = betas
            self.q = qs
            self.lam = lambda_path

        return self

    def select_best_model(self, validation_gdl=None, criterion='objective'):

        # Call the parent method:
        best_idx = super().select_best_model(validation_gdl=validation_gdl, criterion=criterion)

        # Update the hyperparameter:
        self.lam = self.lam[best_idx]

    def to_hyperparameter_table(self):
        """
        :return: A pandas DataFrame containing the hyperparameters
            of the model.
        """

        return pd.DataFrame([
            {'Parameter': 'lambda', 'Value': self.lam},
            {'Parameter': 'Nonzero coefficients', 'Value': self.nnz},
            {'Parameter': 'lambda_min', 'Value': self.lambda_min},
        ])
