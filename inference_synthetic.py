#!/usr/bin/env python3

"""
Example training/testing script
"""

import pathlib
import string

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as skl_mets

from unsupervised_multimodal_trajectory_modeling.linear_gaussian import (
    marginalizable_mixture_model as mixmodel,
)
from unsupervised_multimodal_trajectory_modeling.util import (
    util_state_space as util,
)

import data_synthetic as data
import util as util_plotting

plt.rcParams["figure.autolayout"] = True
plt.rcParams["legend.loc"] = "upper right"
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 12

pwd = pathlib.Path(__file__).absolute().parent
alpha = 1.0
n_clusters = 3
save_figs = True
show_figs = True


def main():
    """runs example training and prediction"""

    # gather and process datasets
    ztrain_orig, xtrain, ctrain, *_ = data.get_data(1000)
    ztrain, std_param = util.standardize(ztrain_orig, return_params=True)

    ztest_orig, xtest, ctest, *_ = data.get_data(
        1000, rng=np.random.default_rng(0)
    )
    ztest = util.standardize(ztest_orig, params=std_param)

    # train mixture model
    best_mdl = mixmodel.MMLinGaussSS_marginalizable(
        n_clusters=n_clusters,
        states=ztrain,
        observations=xtrain,
        init="k-means",
        # k-means-all doesn't make sense for trajectories
        # of differing lengths
    ).train_with_multiple_random_starts(n_starts=100, use_cache=True)
    data.set_model_correspondence(best_mdl, ctrain)

    # test model on full data
    predicted_ctest = np.array(
        [
            best_mdl.correspondence[c]
            for c in best_mdl.mle_cluster_assignment(
                states=ztest, observations=xtest
            )
        ]
    )

    print("Confusion matrix |".ljust(79, "-"))
    print(
        skl_mets.confusion_matrix(
            np.array(list(string.ascii_uppercase))[ctest], predicted_ctest
        )
    )

    # test model with states missing
    predicted_ctest_no_hidden = np.array(
        [
            best_mdl.correspondence[c]
            for c in best_mdl.mle_cluster_assignment(
                states=np.nan * np.ones_like(ztest), observations=xtest
            )
        ]
    )

    print(
        "Confusion matrix - predictions from measurements only |".ljust(
            79, "-"
        )
    )

    print(
        cmat_test_no_hidden := skl_mets.confusion_matrix(
            np.array(list(string.ascii_uppercase))[ctest],
            predicted_ctest_no_hidden,
        )
    )

    util_plotting.plot_matrix(
        cmat_test_no_hidden,
        yticks=list(string.ascii_uppercase)[:3],
        ylabel="true",
        xticks=list(string.ascii_uppercase)[:3],
        xlabel="predicted",
        rotate_xlabels=False,
        fmt_str="{:d}",
        savename=pwd.joinpath("figures").joinpath(
            "{}_confusion_matrix_test_no_hidden.png".format(data.name)
        ),
        show_figure=False,
    )

    # test model with follow-ups missing
    predicted_ctest_init = np.array(
        [
            best_mdl.correspondence[c]
            for c in best_mdl.mle_cluster_assignment(
                states=util.mask_all_but_time_i(ztest, 0),
                observations=util.mask_all_but_time_i(xtest, 0),
            )
        ]
    )

    print(
        "Confusion matrix - predictions from initial states "
        "and measurements only |".ljust(79, "-")
    )
    print(
        skl_mets.confusion_matrix(
            np.array(list(string.ascii_uppercase))[ctest],
            predicted_ctest_init,
        )
    )

    # test model using only initial measurements
    predicted_ctest_init_meas = np.array(
        [
            best_mdl.correspondence[c]
            for c in best_mdl.mle_cluster_assignment(
                states=util.mask_all_but_time_i(ztest, 0),
                observations=np.nan * np.ones_like(xtest),
            )
        ]
    )

    print(
        "Confusion matrix - predictions from initial "
        "states and measurements only |".ljust(79, "-")
    )
    print(
        skl_mets.confusion_matrix(
            np.array(list(string.ascii_uppercase))[ctest],
            predicted_ctest_init_meas,
        )
    )

    util_plotting.plot_matrix(
        skl_mets.confusion_matrix(
            np.array(list(string.ascii_uppercase))[ctest],
            predicted_ctest_init_meas,
        ),
        yticks=list(string.ascii_uppercase)[:3],
        ylabel="true",
        xticks=list(string.ascii_uppercase)[:3],
        xlabel="predicted",
        fmt_str="{:d}",
    )

    util.plot_metric_vs_clusters_over_time(
        metric=ztest[..., 0],
        assignments=predicted_ctest,
        metric_name="First latent dimension",
        title="",
        savename=pwd.joinpath("figures").joinpath(
            "metric_v_cluster_over_time.png"
        ),
        show=False,
    )

    util.histograms_by_cluster(
        metrics=np.column_stack([ztest[0], xtest[0]]),
        metric_names=[f"z dim={i}" for i in range(ztest.shape[-1])]
        + [f"x dim={i}" for i in range(xtest.shape[-1])],
        clusters=predicted_ctest,
        μσ_overlay=best_mdl.get_initial_means_and_stds(),
        mean_overlay=False,
        title="",
        savename=str(
            pwd.joinpath("figures").joinpath("histograms_by_cluster.png")
        ),
        show=False,
        nbins=15,
    )


if __name__ == "__main__":
    main()
