#!/usr/bin/env python3

"""
Example training/testing script
"""

import pathlib
import string

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as skl_mets

from unsupervised_multimodal_trajectory_modeling.nonlinear import (
    state_space_model_mixture as mixmodel,
    StateSpaceKNN as ss_knn,
    StateSpaceHybrid as ss_hybrid,
)
from unsupervised_multimodal_trajectory_modeling.util import (
    util_state_space as util,
)

import data_synthetic as data

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

    for model_spec in [ss_knn, ss_hybrid]:
        print(model_spec())

        # train mixture model
        best_mdl = mixmodel.StateSpaceMixtureModel(
            n_clusters=n_clusters,
            data=(ztrain, xtrain),
            component_model=model_spec,
            # k-means-all doesn't make sense for trajectories
            # of differing lengths
        ).fit(init="kmeans", n_iter=100, use_cache=True)
        data.set_model_correspondence(best_mdl, ctrain)

        # test model on full data
        print("Confusion matrix |".ljust(79, "-"))
        print(
            skl_mets.confusion_matrix(
                np.array(list(string.ascii_uppercase))[ctest],
                best_mdl.predict(data=(ztest, xtest)),
            )
        )


if __name__ == "__main__":
    main()
