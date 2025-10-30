import numpy as np
import magenpy as mgp
from penprs.model.Lasso import Lasso
from penprs.model.SSL import SSL
from viprs.model.vi.e_step_cpp import check_blas_support, check_omp_support
from functools import partial
import shutil
import pytest


@pytest.fixture(scope='module')
def gdl_object():
    """
    Initialize a GWADataLoader using data pre-packaged with magenpy.
    Make this data loader available to all tests.
    """
    gdl = mgp.GWADataLoader(mgp.tgp_eur_data_path(),
                            sumstats_files=mgp.ukb_height_sumstats_path(),
                            sumstats_format='fastgwa',
                            backend='xarray')

    ld_block_url = ("https://bitbucket.org/nygcresearch/ldetect-data/raw/"
                    "ac125e47bf7ff3e90be31f278a7b6a61daaba0dc/EUR/fourier_ls-all.bed")
    gdl.compute_ld('block', gdl.output_dir, ld_blocks_file=ld_block_url)

    gdl.harmonize_data()

    yield gdl

    # Clean up after tests are done:
    gdl.cleanup()
    shutil.rmtree(gdl.temp_dir)
    shutil.rmtree(gdl.output_dir)


@pytest.fixture(scope='module')
def lasso_model(gdl_object):
    """
    Initialize a basic Lasso model using GWAS sumstats data pre-packaged with magenpy.
    Make this data loader available to all tests.
    """
    return Lasso(gdl_object)


@pytest.fixture(scope='module')
def ssl_model(gdl_object):
    """
    Initialize a basic SSL model using GWAS sumstats data pre-packaged with magenpy.
    Make this data loader available to all tests.
    """
    return SSL(gdl_object)

class TestSSL(object):

    def test_init(self,
                  ssl_model: SSL,
                  gdl_object: mgp.GWADataLoader):

        assert ssl_model.m == gdl_object.m

        ssl_model.initialize()

        # Check the input data:
        for p in (ssl_model.std_beta, ssl_model.n_per_snp):
            assert p.shape == (ssl_model.m, )

        # Check the LD data:
        assert ssl_model.ld_indptr.shape == (ssl_model.m + 1, )
        assert ssl_model.ld_left_bound.shape == (ssl_model.m, )
        assert ssl_model.ld_data.shape == (ssl_model.ld_indptr[-1], )

        # Check hyperparameters:
        assert np.all(np.asarray(ssl_model.l0) >= ssl_model.l1)
        assert 0. < ssl_model.var
        assert 0 <= ssl_model.theta <=1
        assert ssl_model._a_theta_prior > 0
        assert ssl_model._b_theta_prior > 0

    def test_fit(self, ssl_model: SSL):

        ssl_model.fit(max_iter=10)

        assert ssl_model.betas.shape == (ssl_model.m,)

        # Test that the following methods are working properly:
        ssl_model.to_table()
        ssl_model.to_history_table()
        ssl_model.mse()
        ssl_model.log_likelihood()
        ssl_model.penalty()
        ssl_model.objective()

class TestLasso(object):

    def test_init(self,
                  lasso_model: Lasso,
                  gdl_object: mgp.GWADataLoader):
        """
        Verify initialization and hyperparameter setup for the Lasso model.
        """

        # Check that dimensions match
        assert lasso_model.m == gdl_object.m

        # Initialize model
        lasso_model.initialize()

        # Check input data arrays
        for p in (lasso_model.std_beta, lasso_model.n_per_snp):
            assert p.shape == (lasso_model.m, )

        # Check LD-related arrays
        assert lasso_model.ld_indptr.shape == (lasso_model.m + 1, )
        assert lasso_model.ld_left_bound.shape == (lasso_model.m, )
        assert lasso_model.ld_data.shape == (lasso_model.ld_indptr[-1], )

        # Hyperparameter checks
        assert lasso_model.lam is not None
        assert lasso_model.lam >= 0
        assert lasso_model.lambda_min >= 0

    def test_fit(self, lasso_model: Lasso):
        """
        Test that Lasso fitting, objective computation, and reporting methods work.
        """

        # Run a short fit to ensure flow executes
        lasso_model.fit(max_iter=10)

        # Betas should be correctly shaped
        assert lasso_model.betas.shape == (lasso_model.m,)

        # Test that main methods execute successfully
        lasso_model.to_hyperparameter_table()
        lasso_model.to_history_table()
        lasso_model.mse()
        lasso_model.penalty()
        lasso_model.objective()

        # Check objective consistency
        lasso_model.penalty()
        lasso_model.mse()
        lasso_model.objective()

@pytest.mark.xfail(not check_blas_support(), reason="BLAS library not found!")
def test_check_blas_support():
    assert check_blas_support()


@pytest.mark.xfail(not check_omp_support(), reason="OpenMP library not found!")
def test_check_omp_support():
    assert check_omp_support()
