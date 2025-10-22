import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import skops.io as skio
import warnings

# Ignorer les avertissements pour une sortie plus propre
warnings.filterwarnings('ignore', category=UserWarning, module='skops')
warnings.filterwarnings('ignore', category=FutureWarning)

def main():
    """
    Fonction principale pour charger les données, entraîner le modèle
    de classification de flux réseau et le sauvegarder.
    """
    # --- 1. Chargement et Nettoyage des Données ---
    print("Étape 1: Chargement et nettoyage des données...")
    try:
        # Charger le jeu de données d'entraînement
        # df = pd.read_csv('Training_export.csv')
        df = pd.read_parquet('Training.parquet')

        # Les noms de colonnes peuvent avoir des espaces superflus
        df.columns = df.columns.str.strip()

    except FileNotFoundError:
        print("Erreur: Le fichier 'Training_export.csv' est introuvable.")
        print("Assurez-vous que le fichier est dans le même répertoire que le script.")
        return

    # Remplacer les valeurs infinies (positives et négatives) par NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Supprimer les lignes contenant des valeurs NaN
    # Une autre stratégie pourrait être de les remplir (ex: df.fillna(0, inplace=True))
    df.dropna(inplace=True)

    if df.empty:
        print("Erreur: Le DataFrame est vide après le nettoyage. Le processus ne peut pas continuer.")
        return

    print(f"Données chargées avec succès. Forme du jeu de données après nettoyage: {df.shape}")
    
    # --- 2. Préparation des Données ---
    print("\nÉtape 2: Préparation des données (séparation features/cible)...")
    
    # Séparer les caractéristiques (X) de la variable cible (y)
    X = df.drop('ClassLabel', axis=1)
    y = df['ClassLabel']

    # Afficher la distribution des classes
    print("Distribution des classes dans le jeu de données :")
    print(y.value_counts())

    # --- 3. Création du Pipeline scikit-learn ---
    print("\nÉtape 3: Création du pipeline scikit-learn...")

    # Le pipeline va enchaîner deux étapes :
    # 1. StandardScaler : Mettre à l'échelle les features pour qu'elles aient une moyenne de 0 et un écart-type de 1.
    # 2. RandomForestClassifier : Un modèle robuste et performant pour les tâches de classification.
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    print("Pipeline créé avec succès :")
    print(pipeline)

    # --- 4. Entraînement et Évaluation Locale ---
    print("\nÉtape 4: Entraînement et évaluation locale du modèle...")
    
    # Diviser les données en un jeu d'entraînement et un jeu de test pour l'évaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Entraîner le pipeline sur le jeu d'entraînement
    print("Entraînement du pipeline...")
    pipeline.fit(X_train, y_train)
    
    # Évaluer le modèle sur le jeu de test local
    print("Évaluation du pipeline sur le jeu de test local...")
    y_pred = pipeline.predict(X_test)
    local_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"-> Précision (Accuracy) sur le jeu de test local : {local_accuracy:.4f}")

    # --- 5. Entraînement Final et Sauvegarde ---
    print("\nÉtape 5: Ré-entraînement sur toutes les données et sauvegarde du modèle final...")

    # Pour la soumission finale, il est préférable d'entraîner le modèle sur 100% des données disponibles
    # afin de maximiser ses performances sur le jeu de test inconnu.
    final_model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    print("Entraînement du modèle final sur l'ensemble des données...")
    final_model_pipeline.fit(X, y)

    # Sauvegarder le pipeline entraîné au format .skio
    try:
        skio.dump(final_model_pipeline, 'gemini.skio')
        print("\n-> Modèle final sauvegardé avec succès sous le nom 'gemini.skio'.")
    except Exception as e:
        print(f"Une erreur est survenue lors de la sauvegarde du modèle : {e}")

if __name__ == '__main__':
    main()
