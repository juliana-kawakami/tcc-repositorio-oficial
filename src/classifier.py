from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class Classifier:
    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)

    def train(self, features, labels):
        features_scaled = self.feature_scaler.fit_transform(features)
        features_pca = self.pca.fit_transform(features_scaled)
        self.classifier.fit(features_pca, labels)

    def predict(self, features):
        features_scaled = self.feature_scaler.transform(features)
        features_pca = self.pca.transform(features_scaled)
        return self.classifier.predict(features_pca)
