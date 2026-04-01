import random

import gpytorch
import numpy as np
import torch
from botorch.models import ModelListGP
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.kernels import MaternKernel, RBFKernel, RQKernel as RationalQuadraticKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from gpytorch.priors import GammaPrior
from gpytorch.settings import fast_pred_var
from torch import nn
from torch.optim import Adam, LBFGS


class FeatureNet(nn.Sequential):
    def __init__(self, in_dim, hidden_dims=(128, 64), latent_dim=16, dropout=0.1):
        layers = []
        prev = in_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            prev = hidden_dim
        layers.append(nn.Linear(prev, latent_dim))
        super().__init__(*layers)
        self.latent_dim = latent_dim


class ExactDKLModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, feature_extractor, kernel_type="matern", nu=1.5):
        super().__init__(train_x, train_y, likelihood)
        self.likelihood = likelihood
        self.feat = feature_extractor
        self.mean_module = ConstantMean()

        latent_dim = feature_extractor.latent_dim
        if kernel_type == "matern":
            base_kernel = MaternKernel(nu=nu, ard_num_dims=latent_dim)
        elif kernel_type == "rq":
            base_kernel = RationalQuadraticKernel(ard_num_dims=latent_dim)
            base_kernel.register_prior("alpha_prior", GammaPrior(1.5, 1.0), "alpha")
        elif kernel_type == "rbf":
            base_kernel = RBFKernel(ard_num_dims=latent_dim)
        else:
            raise ValueError(f"Unknown kernel {kernel_type}")

        self.covar_module = ScaleKernel(base_kernel)

    def forward(self, x):
        latent = self.feat(x)
        mean = self.mean_module(latent)
        covar = self.covar_module(latent)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def posterior(self, X, observation_noise=False, **kwargs):
        self.eval()
        self.likelihood.eval()
        with torch.no_grad(), fast_pred_var():
            posterior = self(X)
            if observation_noise:
                posterior = self.likelihood(posterior)
        return GPyTorchPosterior(posterior)


def _set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def train_exact_dkl_full(
    X_train,
    y_train,
    feature_args,
    gp_args,
    dtype=torch.double,
    adam_lr=1e-4,
    adam_epochs=100,
    lbfgs_lr=0.5,
    lbfgs_iters=20,
    seed=42,
):
    _set_seed(seed)

    X = torch.as_tensor(X_train, dtype=dtype)
    y = torch.as_tensor(y_train, dtype=dtype).reshape(-1)

    feature_extractor = FeatureNet(**feature_args).to(dtype=dtype)
    likelihood = GaussianLikelihood().to(dtype=dtype)
    model = ExactDKLModel(X, y, likelihood, feature_extractor, **gp_args).to(dtype=dtype)
    mll = ExactMarginalLogLikelihood(likelihood, model)

    optimizer = Adam(model.parameters(), lr=adam_lr)
    for _ in range(adam_epochs):
        model.train()
        likelihood.train()
        optimizer.zero_grad()
        loss = -mll(model(X), y)
        loss.backward()
        optimizer.step()

    optimizer = LBFGS(model.parameters(), lr=lbfgs_lr, max_iter=lbfgs_iters, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = -mll(model(X), y)
        loss.backward()
        return loss

    model.train()
    likelihood.train()
    optimizer.step(closure)
    model.eval()
    likelihood.eval()
    return model


def fit_two_dkl_models(
    X_train,
    tc_z,
    mod_z,
    feature_args_tc,
    feature_args_mod,
    gp_args_tc,
    gp_args_mod,
    train_kwargs_tc,
    train_kwargs_mod,
):
    model_tc = train_exact_dkl_full(
        X_train,
        tc_z,
        feature_args=feature_args_tc,
        gp_args=gp_args_tc,
        **train_kwargs_tc,
    )
    model_mod = train_exact_dkl_full(
        X_train,
        -mod_z,
        feature_args=feature_args_mod,
        gp_args=gp_args_mod,
        **train_kwargs_mod,
    )
    return model_tc, model_mod


def build_mobo_surrogate(model_tc, model_mod):
    model = ModelListGP(model_tc, model_mod)
    for submodel in model.models:
        submodel.eval()
        submodel.likelihood.eval()
    return model
