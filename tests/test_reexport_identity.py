"""Identity checks: sklplus re-exports must be the same object as sklearn."""
import importlib

import pytest
from sklearn import (
    discriminant_analysis as sk_da,
    dummy as sk_dummy,
    ensemble as sk_ens,
    gaussian_process as sk_gp,
    kernel_ridge as sk_kr,
    linear_model as sk_lm,
    naive_bayes as sk_nb,
    neighbors as sk_nn,
    neural_network as sk_nn_net,
    svm as sk_svm,
    tree as sk_tree,
)

from sklplus import (
    discriminant_analysis as sp_da,
    dummy as sp_dummy,
    ensemble as sp_ens,
    gaussian_process as sp_gp,
    kernel_ridge as sp_kr,
    linear_model as sp_lm,
    naive_bayes as sp_nb,
    neighbors as sp_nn,
    neural_network as sp_nn_net,
    svm as sp_svm,
    tree as sp_tree,
)


def test_logistic_regression_is_sklearn():
    assert sp_lm.LogisticRegression is sk_lm.LogisticRegression


# One representative per submodule
IDENTITY_SAMPLES = [
    (sp_lm, sk_lm, "LogisticRegression"),
    (sp_tree, sk_tree, "DecisionTreeClassifier"),
    (sp_nn, sk_nn, "KNeighborsClassifier"),
    (sp_svm, sk_svm, "SVC"),
    (sp_nb, sk_nb, "GaussianNB"),
    (sp_da, sk_da, "LinearDiscriminantAnalysis"),
    (sp_nn_net, sk_nn_net, "MLPClassifier"),
    (sp_gp, sk_gp, "GaussianProcessClassifier"),
    (sp_dummy, sk_dummy, "DummyClassifier"),
    (sp_kr, sk_kr, "KernelRidge"),
    (sp_ens, sk_ens, "RandomForestClassifier"),
]


@pytest.mark.parametrize("sp_mod,sk_mod,name", IDENTITY_SAMPLES, ids=[t[2] for t in IDENTITY_SAMPLES])
def test_reexport_identity(sp_mod, sk_mod, name):
    assert getattr(sp_mod, name) is getattr(sk_mod, name)


def test_kmeans_is_sklearn():
    from sklearn.cluster import KMeans as sk_km
    from sklplus.cluster import KMeans as sp_km
    assert sp_km is sk_km


def test_kmodes_importable():
    from sklplus.cluster import KModes
    from kmodes.kmodes import KModes as upstream
    assert KModes is upstream
