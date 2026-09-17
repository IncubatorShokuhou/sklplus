def test_xgb_dual_path_same_object():
    from sklplus.ensemble import XGBClassifier as A
    from sklplus.xgboost import XGBClassifier as B
    assert A is B


def test_lgbm_dual_path_same_object():
    from sklplus.ensemble import LGBMClassifier as A
    from sklplus.lightgbm import LGBMClassifier as B
    assert A is B


def test_catboost_dual_path_same_object():
    from sklplus.ensemble import CatBoostClassifier as A
    from sklplus.catboost import CatBoostClassifier as B
    assert A is B
