import numpy as np
import pandas as pd
import itertools


class HyperparameterGrid(object):

    def __init__(self,
                l0_grid=None,
                l0_steps=None,
                l1_grid=None,
                l1_steps=None,
                var_grid = None,
                var_steps = None,
                a_grid = None,
                a_steps = None,
                lambda_max= None,
                scale_l0_by_l1=False,
                n_snps=12500):

        self.n_snps = n_snps
        self.lambda_max = lambda_max
        self.scale_l0_by_l1 = scale_l0_by_l1
        self._search_params = []

        # Initialize the grid for l0:
        self.l0 = l0_grid
        if self.l0 is not None:
            self._search_params.append('l0')

            #ensure correctly parsed into list of values
            if isinstance(self.l0, str):
                self.l0 = np.array([np.float64(l0.strip()) for l0 in self.l0.split(",")])
        elif l0_steps is not None:
            self.generate_l0_grid(steps=l0_steps)


        # Initialize the grid for l1:
        self.l1 = l1_grid
        if self.l1 is not None:
            self._search_params.append('l1')

            #ensure correctly parsed into list of values
            if isinstance(self.l1, str):
                self.l1 = np.array([np.float64(l1.strip()) for l1 in self.l1.split(",")])
        elif l1_steps is not None:
            self.generate_l1_grid(steps=l1_steps)

        # Initialize the grid for var:
        self.var = var_grid
        if self.var is not None:
            self._search_params.append('var')

            #ensure correctly parsed into list of values
            if isinstance(self.var, str):
                self.var = np.array([np.float64(var.strip()) for var in self.var.split(",")])
        elif var_steps is not None:
            self.generate_var_grid(steps=var_steps)

        # Initialize the grid for a:
        self.a = a_grid
        if self.a is not None:
            self._search_params.append('a')

            #ensure correctly parsed into list of values
            if isinstance(self.a, str):
                self.a = np.array([np.int32(a.strip()) for a in self.a.split(",")])
        elif a_steps is not None:
            self.generate_a_grid(steps=a_steps)


    def generate_l0_grid(self, steps=50):
        """
        Generate a grid of values for the `l0` spike penalty hyperparameter.

        :param steps: The number of steps for the l0 grid.
        """

        assert steps > 0
        if self.lambda_max is not None:
            if self.scale_l0_by_l1:
                self.l0 = np.logspace(0, 2, steps)
            else:
                self.l0 = np.linspace(0.01,1,steps=steps) * self.lambda_max
        else:

            start = 200
            end = 2000
            
            self.l0 = np.linspace(start, end, steps)

        if 'l0' not in self._search_params:
            self._search_params.append('l0')
    
    def generate_l0_per_of_lm_grid(self):
        self.l0 = self.l0 * self.lambda_max

    def generate_l1_grid(self, steps=3):
        """
        Generate a grid of values for the `l1` slab penalty hyperparameter.

        :param steps: The number of steps for the l1 grid.
        """

        assert steps > 0

        # print(self.lambda_max)
        if self.lambda_max is not None:
            eps = 0.001
            self.l1 = np.logspace(np.log10(self.lambda_max), np.log10(eps * self.lambda_max), steps)
        else:
            half_range = (steps - 1) // 2
            if steps % 2 == 1:
                exponents = np.linspace(-half_range, half_range, steps)
            else:
                exponents = np.concatenate([np.linspace(-half_range-1, 0, steps//2+1), np.linspace(1, half_range, steps//2-1)])

            self.l1 = 10 ** exponents 

        if 'l1' not in self._search_params:
            self._search_params.append('l1')
    
    def generate_var_grid(self, steps=3):
        """
        Generate a grid of values for the `var` or sigma2 error variance.

        :param steps: The number of steps for the var grid.
        """

        assert steps > 0

        start = 0.6
        end = 1.1

        self.var = np.concatenate([np.linspace(start, 1, steps//2), np.linspace(1, end, steps//2+1)])

        if 'var' not in self._search_params:
            self._search_params.append('var')

    def generate_a_grid(self, steps=6):
        """
        Generate a grid of values for the `l1` slab penalty hyperparameter.

        :param steps: The number of steps for the l1 grid.
        """

        assert steps > 0

        self.a = np.linspace(1, 0.5*self.n_snps, steps)

        if 'a' not in self._search_params:
            self._search_params.append('a')

    def combine_grids(self):
        """
        Weave together the different hyperparameter grids and return a list of
        dictionaries where the key is the hyperparameter name and the value is
        value for that hyperparameter.

        :return: A list of dictionaries containing the hyperparameter values.
        :raises ValueError: If all the grids are empty.

        """
        hyp_names = [name for name, value in self.__dict__.items()
                     if value is not None and name in self._search_params]

        if len(hyp_names) > 0:
            hyp_values = itertools.product(*[hyp_grid for hyp_name, hyp_grid in self.__dict__.items()
                                             if hyp_grid is not None and hyp_name in hyp_names])

            return [dict(zip(hyp_names, hyp_v)) for hyp_v in hyp_values]
        else:
            raise ValueError("All the grids are empty!")

    def to_table(self):
        """
        :return: The hyperparameter grid as a pandas `DataFrame`.
        """

        combined_grids = self.combine_grids()
        if combined_grids:
            return pd.DataFrame(combined_grids)
