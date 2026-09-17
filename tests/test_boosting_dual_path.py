def test_xgb_dual_path_same_object():
    from sklearnplus.ensemble import XGBClassifier as A
    from sklearnplus.xgboost import XGBClassifier as B
    assert A is B


def test_lgbm_dual_path_same_object():
    from sklearnplus.ensemble import LGBMClassifier as A
    from sklearnplus.lightgbm import LGBMClassifier as B
    assert A is B


def test_catboost_dual_path_same_object():
    from sklearnplus.ensemble import CatBoostClassifier as A
    from sklearnplus.catboost import CatBoostClassifier as B
    assert A is B
