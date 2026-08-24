from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.features.extractor import FeatureExtractor


def build_baseline():
    cechy = FeatureUnion([
        ('reczne', FeatureExtractor()),
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
    ])

    pipeline = Pipeline([
        ('cechy', cechy),
        ('model', LogisticRegression(class_weight='balanced', max_iter=1000)),
    ])

    return pipeline