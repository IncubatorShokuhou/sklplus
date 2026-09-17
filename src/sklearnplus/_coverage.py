"""CI-only public symbol checklist. Not a user-facing registry API."""
PUBLIC_SYMBOLS: list[tuple[str, str]] = [
    # linear_model
    ("sklearnplus.linear_model", "LogisticRegression"),
    ("sklearnplus.linear_model", "LinearRegression"),
    ("sklearnplus.linear_model", "Ridge"),
    ("sklearnplus.linear_model", "RidgeClassifier"),
    ("sklearnplus.linear_model", "Lasso"),
    ("sklearnplus.linear_model", "ElasticNet"),
    ("sklearnplus.linear_model", "Lars"),
    ("sklearnplus.linear_model", "LassoLars"),
    ("sklearnplus.linear_model", "OrthogonalMatchingPursuit"),
    ("sklearnplus.linear_model", "BayesianRidge"),
    ("sklearnplus.linear_model", "ARDRegression"),
    ("sklearnplus.linear_model", "PassiveAggressiveRegressor"),
    ("sklearnplus.linear_model", "RANSACRegressor"),
    ("sklearnplus.linear_model", "TheilSenRegressor"),
    ("sklearnplus.linear_model", "HuberRegressor"),
    ("sklearnplus.linear_model", "SGDClassifier"),
    # tree
    ("sklearnplus.tree", "DecisionTreeClassifier"),
    ("sklearnplus.tree", "DecisionTreeRegressor"),
    # neighbors
    ("sklearnplus.neighbors", "KNeighborsClassifier"),
    ("sklearnplus.neighbors", "KNeighborsRegressor"),
    # svm
    ("sklearnplus.svm", "SVC"),
    ("sklearnplus.svm", "SVR"),
    # naive_bayes
    ("sklearnplus.naive_bayes", "GaussianNB"),
    # discriminant_analysis
    ("sklearnplus.discriminant_analysis", "LinearDiscriminantAnalysis"),
    ("sklearnplus.discriminant_analysis", "QuadraticDiscriminantAnalysis"),
    # neural_network
    ("sklearnplus.neural_network", "MLPClassifier"),
    ("sklearnplus.neural_network", "MLPRegressor"),
    # gaussian_process
    ("sklearnplus.gaussian_process", "GaussianProcessClassifier"),
    # dummy
    ("sklearnplus.dummy", "DummyClassifier"),
    ("sklearnplus.dummy", "DummyRegressor"),
    # kernel_ridge
    ("sklearnplus.kernel_ridge", "KernelRidge"),
    # ensemble (sklearn-only)
    ("sklearnplus.ensemble", "RandomForestClassifier"),
    ("sklearnplus.ensemble", "RandomForestRegressor"),
    ("sklearnplus.ensemble", "ExtraTreesClassifier"),
    ("sklearnplus.ensemble", "ExtraTreesRegressor"),
    ("sklearnplus.ensemble", "AdaBoostClassifier"),
    ("sklearnplus.ensemble", "AdaBoostRegressor"),
    ("sklearnplus.ensemble", "GradientBoostingClassifier"),
    ("sklearnplus.ensemble", "GradientBoostingRegressor"),
    ("sklearnplus.ensemble", "BaggingClassifier"),
    ("sklearnplus.ensemble", "BaggingRegressor"),
    ("sklearnplus.ensemble", "StackingClassifier"),
    ("sklearnplus.ensemble", "StackingRegressor"),
    ("sklearnplus.ensemble", "VotingClassifier"),
    ("sklearnplus.ensemble", "VotingRegressor"),
    ("sklearnplus.ensemble", "CalibratedClassifierCV"),

    # boosting dual-path (dedicated modules)
    ("sklearnplus.xgboost", "XGBClassifier"),
    ("sklearnplus.xgboost", "XGBRegressor"),
    ("sklearnplus.lightgbm", "LGBMClassifier"),
    ("sklearnplus.lightgbm", "LGBMRegressor"),
    ("sklearnplus.catboost", "CatBoostClassifier"),
    ("sklearnplus.catboost", "CatBoostRegressor"),
    # boosting also via ensemble
    ("sklearnplus.ensemble", "XGBClassifier"),
    ("sklearnplus.ensemble", "XGBRegressor"),
    ("sklearnplus.ensemble", "LGBMClassifier"),
    ("sklearnplus.ensemble", "LGBMRegressor"),
    ("sklearnplus.ensemble", "CatBoostClassifier"),
    ("sklearnplus.ensemble", "CatBoostRegressor"),
]
