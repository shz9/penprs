import numpy as np
import pandas as pd
from tqdm import tqdm
import logging

from .PRSModel import PRSModel
from ..utils.exceptions import OptimizationDivergence
from .co.coord_opt import cpp_update_delta, cpp_update_beta_ssl, cpp_update_var

class SSL(PRSModel):

    def __init__(self,
                 gdl,
                 l0=None,
                 l1=None,
                 hyperparam_update_freq=10,
                 unknown_var = False,
                 **prs_model_kwargs):

        super().__init__(gdl, **prs_model_kwargs)

        # Set SSL-specific hyperparameters:
        self.l0 = l0
        self.l1 = l1
        self.theta = None
        self.delta = None
        self.p_gamma = None  # Number of non-zero coefficients
        self.var = None
        self._a_theta_prior = None
        self._b_theta_prior = None
        self.hyperparam_update_freq = hyperparam_update_freq

        self.unknown_var = unknown_var

    def initialize_hyperparameters(self, theta_0=None):

        if theta_0 is None:
            if self.fix_params is not None:
                theta_0 = self.fix_params
            else:
                theta_0 = {}
        else:
            if self.fix_params is not None:
                for key, value in self.fix_params.items():
                    theta_0.setdefault(key, value)

        if 'b' in theta_0:
            self._b_theta_prior = theta_0['b']
        elif self._b_theta_prior is None:
            self._b_theta_prior = self.n_snps

        if 'a' in theta_0:
            self._a_theta_prior = theta_0['a']
        elif self._a_theta_prior is None:
            # Set 'a' so that the prior mean of theta is 0.05:
            self._a_theta_prior = (0.05/(1. - 0.05))*self._b_theta_prior

        # Default settings for the SSL hyperparameters:
        lam_max = self._compute_lambda_max()

        if 'l1' in theta_0:
            self.l1 = theta_0['l1']
        elif 'min_lambda_frac' in theta_0:
            self.l1 = theta_0['min_lambda_frac']*lam_max
        elif self.l1 is None:
            # By default, scale lambda_max by 10^-3:
            self.l1 = 1e-3*lam_max

        if 'l0' in theta_0:
            self.l0 = theta_0['l0']
        elif 'max_lambda_frac' in theta_0:
            self.l0 = theta_0['max_lambda_frac']*lam_max
        elif self.l0 is None:
            # By default set lambda_0 to 10% of lambda_max
            # or 100*l1:
            self.l0 = max(0.1*lam_max, 100*self.l1)

        if self.unknown_var:
            self.min_var, self.init_var = self._init_min_var(self.n)
        else:
            self.min_var, self.init_var = 0., 0.


        if self.unknown_var:
            self.var = self.init_var
        elif 'var' in theta_0:
            self.var = theta_0['var']
        else:
            self.var = 1.

        if 'theta' in theta_0:
            self.theta = theta_0['theta']
        else:
            # Set theta based on the mean of the Beta prior:
            self.theta = self._a_theta_prior / (self._a_theta_prior + self._b_theta_prior)


        self._a_theta_prior = np.dtype(self.float_precision).type(self._a_theta_prior)
        self._b_theta_prior = np.dtype(self.float_precision).type(self._b_theta_prior)
        self.l0 = np.dtype(self.float_precision).type(self.l0)
        self.l1 = np.dtype(self.float_precision).type(self.l1)
        self.var = np.dtype(self.float_precision).type(self.var)
        self.lambda_min = np.dtype(self.float_precision).type(self.lambda_min)
        self.theta = np.dtype(self.float_precision).type(self.theta)

        self._update_delta()

    def _init_min_var(self, n):
        """
        Initialize the minimum variance for the unknown variance case.
        """

        from scipy.stats import chi2

        df = 3
        sigquant = 0.9
        sigest=1
        qchi = chi2.ppf(1 - sigquant, df)
        ncp = (sigest ** 2) * qchi / df
        min_var = (sigest ** 2) / n
        init_var = df * ncp / (df + 2)

        return min_var, init_var

    def _p_star(self):
        """
        Compute the p-star quantity for the spike-and-slab prior.
        """
        return 1./(1. + ((1. - self.theta)/self.theta)*(self.l0/self.l1) *
                   np.exp(-np.abs(self.betas) * (self.l0-self.l1)))

    def _lambda_star(self):
        """
        Compute the lambda-star quantity for the spike-and-slab prior.
        """
        p_star = self._p_star()
        return self.l1*p_star + self.l0*(1. - p_star)

    def _update_theta(self):
        """
        Update the theta parameter based on the current value of the coefficients
        and hyperparameters.
        """

        self.theta = (self._a_theta_prior + self.nnz) / (
            self._a_theta_prior + self._b_theta_prior + self.n_snps
        )

    def _update_delta(self):
        """
        Update the hard-thresholding delta parameter based
        on the current value of the coefficients and hyperparameters.
        """

        # NOTE: Due to a bug in cython dynamic type
        # inference, we need to cast the variables to float64
        # here and then cast the result back to the desired precision
        self.delta = np.dtype(self.float_precision).type(
            cpp_update_delta(
                np.dtype('float64').type(self.n),
                np.dtype('float64').type(self.l0),
                np.dtype('float64').type(self.l1),
                np.dtype('float64').type(self.theta),
                np.dtype('float64').type(self.var)
            )
        )

    def _coord_ascent_step(self, update_var=False):
        """
        Perform a single iteration of the coordinate ascent algorithm
        for the SSL model, given the current settings of the
        coefficients and hyperparameters.
        """

        # Wrap the hyperparameters in numpy arrays to pass by reference
        # to cython/c++:
        delta = np.array([self.delta],dtype=self.float_precision)
        theta = np.array([self.theta],dtype=self.float_precision)
        var = np.array([self.var],dtype=self.float_precision)
        p_gamma = np.array([self.p_gamma], dtype=np.int32)

        cpp_update_beta_ssl(
            self.ld_left_bound, self.ld_indptr, self.ld_data,
            self.std_beta,
            self.betas, self.betas_diff,
            self.q, self.n_per_snp,
            p_gamma, theta, delta, var,
            self.n, self.l0, self.l1,
            self._a_theta_prior, self._b_theta_prior,
            self.hyperparam_update_freq,
            self.lambda_min,
            self.init_var, self.min_var, update_var,
            self.dequantize_scale,
            self.threads,
            self.low_memory
        )

        # Retrieve updated values
        self.delta = np.dtype(self.float_precision).type(delta[0])
        self.theta = np.dtype(self.float_precision).type(theta[0])
        self.var = np.dtype(self.float_precision).type(var[0])
        self.p_gamma = p_gamma[0]

    def _update_var(self):
        # self.var = self.n/(self.n+2) * self.mse()
        self.var = np.dtype(self.float_precision).type(
            cpp_update_var(
                np.dtype('float64').type(self.n),
                np.dtype('float64').type(self.betas),
                np.dtype('float64').type(self.std_beta),
                np.dtype('float64').type(self.q),
                np.dtype('float64').type(self.lambda_min)
            )
        )


    def penalty(self):
        """
        Compute the contribution of the penalty for the SSL objective
        using the current value of the coefficients and hyperparameters.
        """

        p_star_0 = -self.n_snps*np.log(1. + ((1. - self.theta)/self.theta)*(self.l0/self.l1))

        return -self.l1*np.abs(self.betas).sum() + p_star_0 - np.log(self._p_star()).sum(axis=0)

    def log_likelihood(self):
        """
        Compute the log-likelihood of the SSL model given the current
        value of the coefficients and hyperparameters.
        """
        return -self.n/(2*self.var) * self.mse() - (self.n+2) * np.log(np.sqrt(self.var))

    def objective(self):
        """
        The objective function for the SSL model.
        """
        return self.log_likelihood() + self.penalty()

    def fit(self,
            max_iter=1000,
            theta_0=None,
            param_0=None,
            continued=False,
            min_iter=3,
            f_abs_tol=1e-4,
            x_abs_tol=1e-4,
            update_var=False,
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
                    'Nonzero': self.p_gamma
                })
                pbar.n = i - 1
                pbar.total = i - 1
                pbar.refresh()
                pbar.close()
                
                self.prev_n_it = i-1
                break

            # Run the coordinate ascent step:
            self._coord_ascent_step(update_var=update_var)
            # Update the tracked parameters (including objectives):
            self.update_history()

            # Extract the current value for the objective:
            curr_obj = self.history['objective'][-1]

            # Update the progress bar:
            pbar.set_postfix(
                {
                    'Objective': f"{curr_obj:.3f}",
                    'Nonzero': self.p_gamma
                }
            )

            if i > min_iter:
                # Check for convergence:
                if not np.isfinite(curr_obj):
                    raise OptimizationDivergence(f"Stopping at iteration {i}: "
                                                 "The optimization algorithm is not converging!\n"
                                                 f"The objective is undefined ({curr_obj}).")
                # elif self.mse() < 0:
                #     raise OptimizationDivergence(f"Stopping at iteration {i}: "
                #                                  "The optimization algorithm is not converging!\n"
                #                                  f"The MSE is negative ({self.mse()}).")
                elif np.isclose(prev_obj, curr_obj, atol=f_abs_tol, rtol=0.):
                    stop_iter = True
                elif self.max_betas_diff < x_abs_tol:
                    stop_iter = True

        return self

    def warm_start_fit(self,
                        l0_ladder=None,
                        ladder_steps=20,
                        max_lambda_frac=.99,
                        pathwise_fit = False,
                        pathwise_decrease_l1 = False,
                        save_intermediate=False,
                        theta_0=None,
                        param_0=None,
                        **fit_kwargs):

        """
        Fit the model using a warm-start strategy, where the model is fit
        multiple times with different values of the l0 hyperparameter.

        :param l0_ladder: A list of values for the l0 hyperparameter to use
            in the warm-start strategy. If None, a default ladder of values
            will be used.
        :param ladder_steps: The number of steps to use in the ladder of l0 values.
        :param max_lambda_frac: The multiplier to use to compute the maximum
            lambda value for the default ladder of l0 values.
        :param save_intermediate: If True, the model parameters will be saved
            for each `l0` value.
        :param theta_0: Initial values for the hyperparameters.
        :param param_0: Initial values for the model parameters.
        :param fit_kwargs: Additional keyword arguments to pass to the `fit`
        """

        assert l0_ladder is not None or ladder_steps is not None, (
            "Either `l0_ladder` or `ladder_steps` must be provided."
        )

        if self.betas is None:
            self.initialize(theta_0=theta_0, param_0=param_0)

        if l0_ladder is None:

            # Generate a default ladder of l0 values by
            # interpolating from `l1` to lam_max on a log2 scale:
            max_penalty = max_lambda_frac*self._compute_lambda_max()


            # run warm-start fit in a pathwise style or default 
            if pathwise_fit:

                l0_ladder = np.logspace(np.log2(self.l1),
                np.log2(max_penalty),
                num=ladder_steps,
                base=2)

                l0_ladder = np.sort(l0_ladder)
                l0_ladder = l0_ladder[::-1]

                if pathwise_decrease_l1:
                    self.l0 = 0.99*max_penalty
            else:

                l0_ladder = np.logspace(
                    np.log2(self.l1),
                    np.log2(max_penalty),
                    num=ladder_steps,
                    base=2)

        if save_intermediate:
            betas = np.empty((self.n_snps, len(l0_ladder)), dtype=self.float_precision)
            qs = np.empty((self.n_snps, len(l0_ladder)), dtype=self.float_precision)

        pbar = tqdm(range(len(l0_ladder)),
                    total=len(l0_ladder),
                    disable=not self.verbose,
                    desc=f'Chromosome {self.chromosome} ({self.n_snps} variants)')

        for i in pbar:
            
            if pathwise_decrease_l1:
                self.l1 = l0_ladder[i]
            else:
                self.l0 = l0_ladder[i]

            if i == 0:
                self.update_history()

            update_var = False

            if i !=0 and self.unknown_var:
                if self.prev_n_it < 100 :
                    update_var = True
                    self._update_var()

                    if self.var < self.min_var:
                        self.var = self.init_var
                        update_var = False
                else:
                    update_var = False
                    if self.prev_n_it == 1000:
                        self.var = self.init_var

            self.fit(continued=True, disable_pbar=True, update_var=update_var, **fit_kwargs)

            pbar.set_postfix(
                {
                    'Objective': f"{self.objective():.3f}",
                    'Nonzero': self.p_gamma,
                    'l0': f"{self.l0:.2f}",
                    'var': f"{self.var:.3f}"
                }
            )

            if save_intermediate:
                betas[:, i] = self.betas
                qs[:, i] = self.q

        if save_intermediate:
            self.betas = betas
            self.q = qs
            self.l0 = l0_ladder

        return self

    def select_best_model(self, validation_gdl=None, criterion='objective'):

        # Call the parent method:
        best_idx = super().select_best_model(validation_gdl=validation_gdl, criterion=criterion)

        # Update the hyperparameter:
        self.l0 = self.l0[best_idx]

        # Update the rest of the hyperparameters:
        self._update_theta()
        self._update_delta()

    def to_hyperparameter_table(self):

        return pd.DataFrame([
            {'Parameter': 'delta', 'Value': self.delta},
            {'Parameter': 'theta', 'Value': self.theta},
            {'Parameter': 'var', 'Value': self.var},
            {'Parameter': 'l0', 'Value': self.l0},
            {'Parameter': 'l1', 'Value': self.l1},
            {'Parameter': 'Nonzero coefficients', 'Value': self.nnz},
            {'Parameter': 'lambda_min', 'Value': self.lambda_min},
            {'Parameter': 'a (theta prior)', 'Value': self._a_theta_prior},
            {'Parameter': 'b (theta prior)', 'Value': self._b_theta_prior},
        ])
