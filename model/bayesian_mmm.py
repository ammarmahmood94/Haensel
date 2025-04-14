#!/usr/bin/env python


import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as at


# Model Set up

def build_mmm_model(df_processed, channel_means, spend, trend, peak_flag, revenue):

    with pm.Model() as mmm_model:

        # Data blocks
        pm.Data("spend", spend)         # shape: (T, 7)
        pm.Data("trend", trend)                       # shape: (T,)
        pm.Data("peak_flag", peak_flag)               # shape: (T,)

        T = spend.shape[0]
        n_channels = spend.shape[1]
        L_max = 52  # max lag

        # Priors
        intercept = pm.Normal("intercept", mu=1, sigma=2)
        beta_trend = pm.Normal("beta_trend", mu=0, sigma=1)
        beta_peak = pm.Normal("beta_peak", mu=0, sigma=10)

        # ⬅️ Use channel_means for HalfNormal sigma priors
        beta_media = pm.HalfNormal("beta_media", sigma=channel_means.values, shape=n_channels)
        alpha_media = pm.Beta("alpha_media", alpha=1, beta=1, shape=n_channels)
        sigma = pm.HalfNormal("sigma", sigma=1)

        # Adstock
        media_contribs = []
        for i in range(n_channels):
            x_i = mmm_model["spend"][:, i]
            lagged = [at.concatenate([at.zeros(j), x_i[:-j]]) if j > 0 else x_i for j in range(L_max)]
            lagged_stack = at.stack(lagged)
            decay_weights = alpha_media[i] ** at.arange(L_max)
            adstock_i = at.dot(decay_weights, lagged_stack)
            media_contribs.append(beta_media[i] * adstock_i)

        total_media_effect = at.sum(at.stack(media_contribs), axis=0)

        trend_data = mmm_model["trend"]
        peak_flag_data = mmm_model["peak_flag"]
        control_effect = beta_trend * trend_data + beta_peak * peak_flag_data

        mu = intercept + control_effect + total_media_effect

        sales_obs = pm.Normal("sales_obs", mu=mu, sigma=sigma, observed=df_processed['scaled_revenue'])

        return mmm_model