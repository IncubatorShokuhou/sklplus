"""P0 preprocessors: CleanColumnNames, RareCategoryGrouper, DateFeatureExtractor."""

import pandas as pd

from sklearnplus.preprocessing import (
    CleanColumnNames,
    DateFeatureExtractor,
    RareCategoryGrouper,
)


def main() -> int:
    df = pd.DataFrame(
        {
            "Event Date": pd.to_datetime(
                ["2020-01-15", "2020-01-15", "2021-06-20", "2022-12-01", "2022-12-02"]
                + ["2023-03-10"] * 15
            ),
            "cat-col": ["common"] * 16 + ["rare_a", "rare_b", "rare_c", "rare_d"],
            "x val": list(range(20)),
        }
    )
    cleaned = CleanColumnNames().fit_transform(df)
    cleaned["Event_Date"] = pd.to_datetime(cleaned["Event_Date"])
    dated = DateFeatureExtractor(
        features=("year", "month", "day", "dayofweek"), drop_original=True
    ).fit_transform(cleaned)
    grouped = RareCategoryGrouper(min_frequency=0.2, replacement="rare").fit_transform(
        dated
    )
    print(grouped.head())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
