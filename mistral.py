import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import skops.io as skio

# Charger les données
df = pd.read_parquet('Training.parquet')

# Séparer les caractéristiques et la cible
X = df.drop(columns=["ClassLabel"], axis=1)
y = df["ClassLabel"]

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer un pipeline avec normalisation et classificateur
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])

# Entraîner le modèle
pipeline.fit(X_train, y_train)

# Évaluer le modèle
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Sauvegarder le modèle
with open("mistral.skio", "wb") as f:
    skio.dump(pipeline, f)
