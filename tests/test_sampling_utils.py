import numpy as np
from imblearn.over_sampling import SMOTE as ImbSMOTE
from sklearn.datasets import make_classification

from sklearnplus.linear_model import LogisticRegression
from sklearnplus.pipeline import ImbPipeline
from sklearnplus.sampling import SMOTE


def test_smote_is_imblearn():
    assert SMOTE is ImbSMOTE


def test_imbpipeline_smote_logistic():
    X, y = make_classification(
        n_samples=120,
        n_features=8,
        n_informative=4,
        weights=[0.8, 0.2],
        random_state=0,
    )
    pipe = ImbPipeline(
        [
            ("sample", SMOTE(random_state=0)),
            ("clf", LogisticRegression(max_iter=500)),
        ]
    )
    pipe.fit(X, y)
    pred = pipe.predict(X)
    assert pred.shape == (len(y),)


def test_pipeline_make_pipeline_identity():
    from sklearn.pipeline import Pipeline as SkPipeline
    from sklearn.pipeline import make_pipeline as sk_make_pipeline
    from sklearnplus.pipeline import Pipeline, make_pipeline

    assert Pipeline is SkPipeline
    assert make_pipeline is sk_make_pipeline


def test_compose_impute_metrics_samples():
    from sklearn.compose import ColumnTransformer as SkCT
    from sklearn.decomposition import PCA as SkPCA
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.feature_selection import SelectKBest as SkSelectKBest
    from sklearn.impute import SimpleImputer as SkSimpleImputer
    from sklearn.metrics import accuracy_score as sk_acc
    from sklearn.model_selection import train_test_split as sk_tts

    from sklearnplus.compose import ColumnTransformer
    from sklearnplus.decomposition import PCA
    from sklearnplus.feature_selection import SelectKBest
    from sklearnplus.impute import SimpleImputer
    from sklearnplus.metrics import accuracy_score
    from sklearnplus.model_selection import train_test_split

    assert ColumnTransformer is SkCT
    assert SimpleImputer is SkSimpleImputer
    assert PCA is SkPCA
    assert SelectKBest is SkSelectKBest
    assert train_test_split is sk_tts
    assert accuracy_score is sk_acc
