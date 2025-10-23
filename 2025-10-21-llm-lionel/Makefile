# Makefile pour gérer un kernel Jupyter basé sur .venv avec uv
# Utilisation: `make <cible>`

# Variables
VENV ?= .venv
PYTHON := $(VENV)/bin/python
JUPYTER := $(VENV)/bin/jupyter
KERNEL_NAME ?= .venv
DISPLAY_NAME ?= Python (LLM)

.PHONY: help check uv-check venv sync install kernel jupyter notebook list-kernels uninstall-kernel setup clean deep-clean

help: ## Affiche cette aide
	@echo "Cibles disponibles:"
	@grep -E '^[a-zA-Z_-]+:.*?## ' Makefile | sed -E 's/:.*?## /\t- /'

check: ## Vérifie uv, Python et Jupyter
	@command -v uv >/dev/null 2>&1 || { echo "uv n'est pas installé. Voir: https://docs.astral.sh/uv/"; exit 1; }
	@echo "uv -> $$(uv --version)"
	@[ -x "$(PYTHON)" ] && echo "python -> $$($(PYTHON) --version)" || echo "python -> (pas encore créé)"
	@[ -x "$(JUPYTER)" ] && echo "jupyter -> $$($(JUPYTER) --version)" || echo "jupyter -> (pas encore installé)"

uv-check:
	@command -v uv >/dev/null 2>&1 || { echo "Erreur: uv n'est pas installé. Installez-le: https://docs.astral.sh/uv/"; exit 1; }

venv: uv-check ## Crée .venv avec uv si nécessaire
	@[ -d "$(VENV)" ] && echo "$(VENV) existe déjà" || uv venv "$(VENV)"

sync: uv-check ## (Optionnel) uv sync si un pyproject.toml est présent
	@if [ -f pyproject.toml ]; then \
		echo "pyproject.toml détecté -> uv sync"; \
		UV_PROJECT_ENVIRONMENT=$(VENV) uv sync; \
	else \
		echo "Aucun pyproject.toml trouvé -> skip"; \
	fi

install: venv ## Installe ipykernel et Jupyter dans .venv avec uv
	@echo "Installation des dépendances de base (ipykernel, jupyterlab)"
	uv pip install --python "$(PYTHON)" ipykernel jupyterlab
	@if [ -f requirements.txt ]; then \
		echo "requirements.txt détecté -> installation"; \
		uv pip install --python "$(PYTHON)" -r requirements.txt; \
	fi

kernel: install ## Enregistre le kernelspec Jupyter pointant vers .venv
	"$(PYTHON)" -m ipykernel install --user --name "$(KERNEL_NAME)" --display-name "$(DISPLAY_NAME)"
	@echo "Kernel installé: $(DISPLAY_NAME) (name=$(KERNEL_NAME))"

jupyter: ## Lance JupyterLab avec l'environnement .venv
	"$(JUPYTER)" lab

notebook: ## Lance Jupyter Notebook classique avec l'environnement .venv
	"$(JUPYTER)" notebook

list-kernels: ## Liste les kernels disponibles
	@"$(JUPYTER)" kernelspec list

uninstall-kernel: ## Désinstalle le kernel enregistré ($(KERNEL_NAME))
	@jupyter kernelspec remove -f "$(KERNEL_NAME)" || true

setup: ## Tout-en-un: venv + install + kernel
	$(MAKE) venv
	$(MAKE) install
	$(MAKE) kernel

clean: ## Supprime uniquement le venv (ne touche pas au kernel Jupyter)
	rm -rf "$(VENV)"

deep-clean: ## Supprime le venv et désinstalle le kernel Jupyter
	$(MAKE) uninstall-kernel
	rm -rf "$(VENV)"
