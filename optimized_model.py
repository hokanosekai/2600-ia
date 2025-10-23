#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Modèles avancés
try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAVE_LGB = True
except ImportError:
    HAVE_LGB = False

# Pour l'équilibrage
try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAVE_IMBLEARN = True
except ImportError:
    HAVE_IMBLEARN = False

# Pour l'export
try:
    from skops import io as skio
    HAVE_SKOPS = True
except ImportError:
    HAVE_SKOPS = False
    import joblib


class NetworkTrafficClassifier:
    """Classifieur optimisé pour la détection de flux réseau."""
    
    def __init__(self, model_type='xgb', random_state=42):
        self.model_type = model_type
        self.random_state = random_state
        self.pipeline = None
        self.feature_names = None
        self.label_encoder = LabelEncoder()
        self.classes_ = None
        
    def create_features(self, df):
        """Crée des features dérivées robustes."""
        df = df.copy()
        
        # Features de base
        if all(col in df.columns for col in ['Total Fwd Packets', 'Total Backward Packets']):
            df['Total_Packets'] = df['Total Fwd Packets'] + df['Total Backward Packets']
            df['Fwd_Bwd_Ratio'] = np.where(
                df['Total Backward Packets'] > 0,
                df['Total Fwd Packets'] / df['Total Backward Packets'],
                0
            )
        
        if all(col in df.columns for col in ['Fwd Packets Length Total', 'Bwd Packets Length Total']):
            df['Total_Bytes'] = df['Fwd Packets Length Total'] + df['Bwd Packets Length Total']
            df['Bytes_per_Packet'] = np.where(
                df['Total_Packets'] > 0,
                df['Total_Bytes'] / df['Total_Packets'],
                0
            )
        
        # Features temporelles
        if 'Flow Duration' in df.columns:
            duration_safe = df['Flow Duration'].clip(lower=1)
            df['Bytes_per_Second'] = df['Total_Bytes'] / duration_safe
            df['Packets_per_Second'] = df['Total_Packets'] / duration_safe
        
        # Features de ratio
        if all(col in df.columns for col in ['Fwd Packet Length Mean', 'Bwd Packet Length Mean']):
            df['Packet_Size_Ratio'] = np.where(
                df['Bwd Packet Length Mean'] > 0,
                df['Fwd Packet Length Mean'] / df['Bwd Packet Length Mean'],
                0
            )
        
        # Features de flags
        flag_cols = [col for col in df.columns if any(x in col for x in ['Flag', 'URG', 'SYN', 'ACK', 'PSH', 'FIN', 'RST'])]
        if flag_cols:
            df['Total_Flags'] = df[flag_cols].sum(axis=1)
        
        return df
    
    def build_preprocessor(self, X):
        """Construit le préprocesseur avec gestion des numériques et catégorielles."""
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough',
            sparse_threshold=0
        )
        
        return preprocessor
    
    def get_model(self, params=None):
        """Retourne le modèle avec les paramètres optimisés."""
        if params is None:
            params = {}
            
        if self.model_type == 'xgb' and HAVE_XGB:
            default_params = {
                'n_estimators': 800,
                'max_depth': 10,
                'learning_rate': 0.1,
                'subsample': 0.9,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': self.random_state,
                'eval_metric': 'mlogloss',
                'use_label_encoder': False
            }
            default_params.update(params)
            return XGBClassifier(**default_params)
        
        elif self.model_type == 'hgb':
            default_params = {
                'max_iter': 1500,
                'learning_rate': 0.05,
                'max_depth': 12,
                'min_samples_leaf': 20,
                'l2_regularization': 1.0,
                'random_state': self.random_state
            }
            default_params.update(params)
            return HistGradientBoostingClassifier(**default_params)
        
        elif self.model_type == 'rf':
            default_params = {
                'n_estimators': 500,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'random_state': self.random_state,
                'n_jobs': -1,
                'class_weight': 'balanced'
            }
            default_params.update(params)
            return RandomForestClassifier(**default_params)
        
        elif self.model_type == 'lgb' and HAVE_LGB:
            default_params = {
                'n_estimators': 1000,
                'max_depth': 12,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'random_state': self.random_state,
                'n_jobs': -1
            }
            default_params.update(params)
            return LGBMClassifier(**default_params)
        
        else:
            # Fallback vers HGB
            print(f"Modèle {self.model_type} non disponible, utilisation de HGB")
            return HistGradientBoostingClassifier(
                max_iter=1000,
                random_state=self.random_state
            )
    
    def balance_dataset(self, X, y, samples_per_class=1500):
        """Équilibre le dataset avec combinaison d'over et under sampling."""
        if not HAVE_IMBLEARN:
            print("imbalanced-learn non disponible, retour des données originales")
            return X, y
            
        print(f"Équilibrage à {samples_per_class} échantillons par classe")
        print("Distribution avant:")
        print(y.value_counts().sort_index())
        
        # Séparer les classes minoritaires et majoritaires
        class_counts = y.value_counts()
        minority_classes = class_counts[class_counts < samples_per_class].index.tolist()
        majority_classes = class_counts[class_counts > samples_per_class].index.tolist()
        
        X_balanced, y_balanced = X.copy(), y.copy()
        
        # Over-sampling pour les classes minoritaires
        if minority_classes:
            minority_strategy = {cls: samples_per_class for cls in minority_classes}
            smote = SMOTE(sampling_strategy=minority_strategy, random_state=self.random_state)
            X_balanced, y_balanced = smote.fit_resample(X_balanced, y_balanced)
        
        # Under-sampling pour les classes majoritaires  
        if majority_classes:
            majority_strategy = {cls: samples_per_class for cls in majority_classes}
            under = RandomUnderSampler(sampling_strategy=majority_strategy, random_state=self.random_state)
            X_balanced, y_balanced = under.fit_resample(X_balanced, y_balanced)
        
        print("Distribution après:")
        print(y_balanced.value_counts().sort_index())
        
        return X_balanced, y_balanced
    
    def encode_labels(self, y):
        """Encode les labels strings en integers pour XGBoost."""
        if not hasattr(self.label_encoder, 'classes_'):
            self.label_encoder.fit(y)
            self.classes_ = self.label_encoder.classes_
        return self.label_encoder.transform(y)
    
    def decode_labels(self, y_encoded):
        """Décode les labels integers en strings."""
        return self.label_encoder.inverse_transform(y_encoded)
    
    def optimize_hyperparameters(self, X, y, cv=3, n_iter=20):
        """Optimise les hyperparamètres du modèle."""
        preprocessor = self.build_preprocessor(X)
        
        # Encoder les labels pour XGBoost
        if self.model_type in ['xgb', 'lgb']:
            y_encoded = self.encode_labels(y)
        else:
            y_encoded = y
        
        if self.model_type == 'xgb':
            param_dist = {
                'model__n_estimators': [500, 800, 1000, 1200],
                'model__max_depth': [8, 10, 12, 15],
                'model__learning_rate': [0.01, 0.05, 0.1, 0.15],
                'model__subsample': [0.8, 0.9, 1.0],
                'model__colsample_bytree': [0.7, 0.8, 0.9],
                'model__reg_alpha': [0, 0.1, 0.5, 1],
                'model__reg_lambda': [0, 0.1, 0.5, 1]
            }
        elif self.model_type == 'hgb':
            param_dist = {
                'model__max_iter': [1000, 1500, 2000],
                'model__learning_rate': [0.01, 0.05, 0.1],
                'model__max_depth': [8, 12, 15, 20],
                'model__min_samples_leaf': [10, 20, 30, 50],
                'model__l2_regularization': [0.1, 1.0, 10.0]
            }
        elif self.model_type == 'rf':
            param_dist = {
                'model__n_estimators': [200, 300, 500],
                'model__max_depth': [10, 15, 20, None],
                'model__min_samples_split': [2, 5, 10],
                'model__min_samples_leaf': [1, 2, 4],
                'model__max_features': ['sqrt', 'log2', None]
            }
        else:
            print(f"Optimisation non implémentée pour {self.model_type}")
            return None
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', self.get_model())
        ])
        
        search = RandomizedSearchCV(
            pipeline, param_dist, n_iter=n_iter, cv=cv,
            scoring='balanced_accuracy', n_jobs=-1, random_state=self.random_state,
            verbose=1
        )
        
        print(f"Optimisation des hyperparamètres pour {self.model_type}...")
        search.fit(X, y_encoded)
        
        print(f"Meilleur score: {search.best_score_:.4f}")
        print("Meilleurs paramètres:")
        for param, value in search.best_params_.items():
            print(f"  {param}: {value}")
        
        return search.best_estimator_
    
    def train(self, X, y, optimize=False, balance_classes=1500, test_size=0.2, cv_folds=5):
        """Entraîne le modèle avec les données fournies."""
        print("=== DÉBUT DE L'ENTRAÎNEMENT ===")
        
        # Feature engineering
        print("1. Feature engineering...")
        X_engineered = self.create_features(X)
        self.feature_names = X_engineered.columns.tolist()
        print(f"   {len(self.feature_names)} features créées")
        
        # Équilibrage des classes
        if balance_classes > 0:
            print("2. Équilibrage des classes...")
            X_processed, y_processed = self.balance_dataset(X_engineered, y, balance_classes)
        else:
            X_processed, y_processed = X_engineered, y
        
        # Encoder les labels pour les modèles qui en ont besoin
        if self.model_type in ['xgb', 'lgb']:
            y_encoded = self.encode_labels(y_processed)
        else:
            y_encoded = y_processed
        
        # Split train/test
        if test_size > 0:
            print(f"3. Split train/test ({test_size*100}% test)...")
            if self.model_type in ['xgb', 'lgb']:
                # Pour XGBoost et LightGBM, stratifier sur les labels encodés
                X_train, X_test, y_train, y_test = train_test_split(
                    X_processed, y_encoded, test_size=test_size, 
                    stratify=y_encoded, random_state=self.random_state
                )
            else:
                # Pour les autres modèles, stratifier sur les labels originaux
                X_train, X_test, y_train, y_test = train_test_split(
                    X_processed, y_encoded, test_size=test_size, 
                    stratify=y_processed, random_state=self.random_state
                )
        else:
            X_train, y_train = X_processed, y_encoded
            X_test, y_test = None, None
        
        # Optimisation ou entraînement direct
        if optimize:
            print("4. Optimisation des hyperparamètres...")
            self.pipeline = self.optimize_hyperparameters(X_train, y_train, cv=min(3, cv_folds))
        else:
            print("4. Construction du pipeline...")
            preprocessor = self.build_preprocessor(X_train)
            self.pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', self.get_model())
            ])
        
        if self.pipeline is None:
            raise ValueError("Échec de la création du pipeline")
        
        # Entraînement
        print("5. Entraînement du modèle...")
        self.pipeline.fit(X_train, y_train)
        
        # Évaluation
        if X_test is not None:
            print("6. Évaluation...")
            y_pred_encoded = self.pipeline.predict(X_test)
            
            # Décoder les prédictions pour l'évaluation
            if self.model_type in ['xgb', 'lgb']:
                y_pred = self.decode_labels(y_pred_encoded)
                y_test_decoded = self.decode_labels(y_test)
            else:
                y_pred = y_pred_encoded
                y_test_decoded = y_test
            
            accuracy = accuracy_score(y_test_decoded, y_pred)
            balanced_acc = balanced_accuracy_score(y_test_decoded, y_pred)
            f1 = f1_score(y_test_decoded, y_pred, average='macro')
            
            print(f"\n=== RÉSULTATS ===")
            print(f"Accuracy:       {accuracy:.4f}")
            print(f"Balanced Acc:   {balanced_acc:.4f}")
            print(f"F1 Macro:       {f1:.4f}")
            
            print("\nRapport détaillé:")
            print(classification_report(y_test_decoded, y_pred, digits=4))
            
            return accuracy, balanced_acc, f1
        
        return None, None, None
    
    def save_model(self, filename="student_model.skio"):
        """Sauvegarde le modèle entraîné."""
        if self.pipeline is None:
            raise ValueError("Le modèle n'est pas entraîné")
        
        # Créer un objet qui contient à la fois le pipeline et le label_encoder
        model_artifact = {
            'pipeline': self.pipeline,
            'label_encoder': self.label_encoder,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'classes': self.classes_
        }
        
        if HAVE_SKOPS:
            with open(filename, "wb") as f:
                skio.dump(model_artifact, f)
            print(f"✅ Modèle sauvegardé: {filename}")
        else:
            joblib.dump(model_artifact, filename.replace('.skio', '.pkl'))
            print(f"✅ Modèle sauvegardé: {filename.replace('.skio', '.pkl')}")


def load_data(file_path):
    """Charge les données depuis un fichier Parquet ou CSV."""
    path = Path(file_path)
    if path.suffix.lower() == '.parquet':
        return pd.read_parquet(path)
    else:
        # Essayer différents séparateurs pour CSV
        try:
            return pd.read_csv(path)
        except:
            return pd.read_csv(path, sep=';')


def main():
    parser = argparse.ArgumentParser(description='Classifieur optimisé pour flux réseau')
    parser.add_argument('--train-file', default='Training.parquet', help='Fichier d\'entraînement')
    parser.add_argument('--model', choices=['xgb', 'hgb', 'rf', 'lgb'], default='xgb', help='Type de modèle')
    parser.add_argument('--balance', type=int, default=1500, help='Échantillons par classe (0=désactivé)')
    parser.add_argument('--test-size', type=float, default=0.2, help='Taille du jeu de test')
    parser.add_argument('--cv', type=int, default=5, help='Folds pour validation croisée')
    parser.add_argument('--optimize', action='store_true', help='Optimiser les hyperparamètres')
    parser.add_argument('--seed', type=int, default=42, help='Seed aléatoire')
    parser.add_argument('--output', default='student_model.skio', help='Fichier de sortie')
    
    args = parser.parse_args()
    
    # Désactiver les warnings
    warnings.filterwarnings('ignore')
    
    print("🔧 CONFIGURATION:")
    print(f"   Modèle: {args.model}")
    print(f"   Équilibrage: {args.balance} par classe")
    print(f"   Test size: {args.test_size}")
    print(f"   Optimisation: {args.optimize}")
    print(f"   Seed: {args.seed}")
    
    # Chargement des données
    print("\n📊 CHARGEMENT DES DONNÉES...")
    df = load_data(args.train_file)
    df.columns = df.columns.str.strip()
    
    # Nettoyage
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Identification de la target
    target_col = 'ClassLabel'
    if target_col not in df.columns:
        # Chercher une colonne similaire
        for col in df.columns:
            if 'class' in col.lower() or 'label' in col.lower():
                target_col = col
                break
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    print(f"   Données: {len(X)} lignes, {len(X.columns)} features")
    print(f"   Target: {target_col}")
    print(f"   Classes: {y.nunique()}")
    print("   Distribution:", y.value_counts().to_dict())
    
    # Entraînement
    classifier = NetworkTrafficClassifier(model_type=args.model, random_state=args.seed)
    
    accuracy, balanced_acc, f1 = classifier.train(
        X, y,
        optimize=args.optimize,
        balance_classes=args.balance,
        test_size=args.test_size,
        cv_folds=args.cv
    )
    
    # Sauvegarde
    classifier.save_model(args.output)
    
    print("\n✅ TERMINÉ!")


if __name__ == "__main__":
    main()