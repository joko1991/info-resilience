import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame()
        df["n_words"] = X.str.split().str.len()
        df["has_you"] = X.str.contains(r"\byou\b|\byour\b", case=False, regex=True).astype(int)
        df["starts_with_number"] = X.str.match(r"^\d+\s").astype(int)
        return df

if __name__ == "__main__":
    przyklady = pd.Series([
        "13 Things You Need To Know About Cats",
        "Senate Passes New Budget Bill",
    ])
    fe = FeatureExtractor()
    print(fe.transform(przyklady))