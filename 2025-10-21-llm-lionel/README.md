
Gestion du kernel Jupyter avec uv et .venv
================================================

Ce projet utilise uv pour gérer l'environnement Python local dans `.venv` et l'enregistrement d'un kernel Jupyter dédié. Un `Makefile` est fourni pour automatiser les opérations courantes.

Prérequis
---------
- uv installé (voir https://docs.astral.sh/uv/)
- make disponible (généralement présent sur Linux)

Démarrage rapide
----------------
1) Créer le venv, installer Jupyter et enregistrer le kernel:

```sh
make setup
```

2) Lancer JupyterLab (ou Notebook) avec l'environnement `.venv`:

```sh
make jupyter
# ou
make notebook
```

Dans l'interface Jupyter, sélectionnez le kernel affiché comme: Python (LLM)

Commandes disponibles
---------------------

```sh
make help            # Affiche l'aide et les cibles
make venv            # Crée .venv avec uv (si nécessaire)
make install         # Installe ipykernel + jupyterlab (et requirements.txt si présent)
make kernel          # Enregistre le kernel Jupyter pointant vers .venv
make jupyter         # Lance JupyterLab depuis .venv
make notebook        # Lance Jupyter Notebook classique
make list-kernels    # Liste les kernels Jupyter installés
make uninstall-kernel# Supprime le kernel enregistré (.venv)
make clean           # Supprime le dossier .venv
make deep-clean      # Supprime .venv et désinstalle le kernel
```

Notes
-----
- Si un `pyproject.toml` est présent, vous pouvez utiliser `make sync` pour exécuter `uv sync` et installer les dépendances du projet dans `.venv`.
- Les variables peuvent être ajustées lors de l'appel à make, par exemple:

```sh
make kernel KERNEL_NAME=.venv DISPLAY_NAME="Python (LLM)"
```

Enregistrement manuel du kernel (optionnel)
-------------------------------------------
Si vous préférez la commande directe:

```sh
.venv/bin/python -m ipykernel install --user --name=.venv --display-name="Python (LLM)"
```

Dépannage
---------
- Vérifier l'installation:

```sh
make check
```

- Si Jupyter ne voit pas le kernel, réenregistrez-le:

```sh
make kernel
```

