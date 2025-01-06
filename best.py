import arviz as az
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import seaborn as sns
import sys

def run_best(grp1_vals, grp1_name, grp2_vals, grp2_name, tf_name):

    df1 = pd.DataFrame({"aupr": grp1_vals, "group": grp1_name})
    df2 = pd.DataFrame({"aupr": grp2_vals, "group": grp2_name})
    indv = pd.concat([df1, df2]).reset_index()

    mu_m = indv.aupr.mean()
    mu_s = indv.aupr.std() * 2

    with pm.Model() as model:
        group1_mean = pm.Normal("group1_mean", mu=mu_m, sigma=mu_s)
        group2_mean = pm.Normal("group2_mean", mu=mu_m, sigma=mu_s)

    sigma_low = 10**-1
    sigma_high = 10

    with model:
        group1_std = pm.Uniform("group1_std", lower=sigma_low, upper=sigma_high)
        group2_std = pm.Uniform("group2_std", lower=sigma_low, upper=sigma_high)

    with model:
        nu_minus_one = pm.Exponential("nu_minus_one", 1 / 29.0)
        nu = pm.Deterministic("nu", nu_minus_one + 1)
        nu_log10 = pm.Deterministic("nu_log10", np.log10(nu))

    with model:
        lambda_1 = group1_std**-2
        lambda_2 = group2_std**-2
        group1 = pm.StudentT("drug", nu=nu, mu=group1_mean, lam=lambda_1, observed=grp1_vals)
        group2 = pm.StudentT("placebo", nu=nu, mu=group2_mean, lam=lambda_2, observed=grp2_vals)

    with model:
        diff_of_means = pm.Deterministic("difference of means", group1_mean - group2_mean)
        diff_of_stds = pm.Deterministic("difference of stds", group1_std - group2_std)
        effect_size = pm.Deterministic(
            "effect size", diff_of_means / np.sqrt((group1_std**2 + group2_std**2) / 2)
        )

    with model:
        idata = pm.sample()

    az.plot_posterior(
        idata,
        var_names=["difference of means", "difference of stds", "effect size"],
        ref_val=0,
        color="#87ceeb",
    );
    plt.savefig(f"{tf_name}_{grp1_name}_vs_{grp2_name}.png")
    plt.close()

    posterior_diff = az.extract(idata, var_names=["difference of means"])
    evid_ratio_gr = np.array(len(np.where(posterior_diff > 0)[0])) / np.array(len(np.where(posterior_diff < 0)[0]))
    evid_ratio_lt = np.array(len(np.where(posterior_diff < 0)[0])) / np.array(len(np.where(posterior_diff > 0)[0]))

    summary_df = az.summary(idata, var_names=["difference of means"])
    summary_df["comparison"] = f"{grp1_name}_vs_{grp2_name}"
    summary_df["tf"] = tf_name
    summary_df["evid_ratio_gr"] = evid_ratio_gr
    summary_df["evid_ratio_lt"] = evid_ratio_lt
    print(f"Summary for {tf_name} {grp1_name} vs {grp2_name}:\n", summary_df)
    summary_df.to_csv(f"{tf_name}_{grp1_name}_vs_{grp2_name}.csv")

def logit(vals):
    eps = np.finfo(vals.dtype).eps
    cvals = np.clip(vals, 0+eps, 1-eps)
    return cvals / (1-cvals)

def run_all_best(df):

    motif_types = np.unique(df.motif.values)
    tfs = np.unique(df.tf.values)

    for tf in tfs:
        this_df = df[df.tf == tf]
        for i,motif_type_i in enumerate(motif_types):
            for j,motif_type_j in enumerate(motif_types):
                if i < j:
                    i_vals = logit(this_df.value[this_df.motif == motif_type_i].values)
                    j_vals = logit(this_df.value[this_df.motif == motif_type_j].values)
                    run_best(i_vals, motif_type_i, j_vals, motif_type_j, tf)


def main():
    print(f"Running on PyMC v{pm.__version__}")

    df = pd.read_csv('auprs.txt')
    print(df.head())
    run_all_best(df)


if __name__ == "__main__":
    main()
