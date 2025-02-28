# Déclaration des variables
PYTHON=python3
ENV_NAME=venv
REQUIREMENTS=requirements.txt
SOURCE_DIR=model_pipeline.py
MAIN_SCRIPT=main.py
TEST_DIR=tests/

# Configuration de l'environnement virtuel
setup:
	@echo "Création de l'environnement virtuel et installation des dépendances..."
	@$(PYTHON) -m venv $(ENV_NAME)
	@./$(ENV_NAME)/bin/python3 -m pip install --upgrade pip
	@./$(ENV_NAME)/bin/python3 -m pip install -r $(REQUIREMENTS)
	@echo "Environnement configuré avec succès !"

# Vérification du code avec Black et Pylint
verify:
	@echo "Vérification de la qualité du code..."
	@. $(ENV_NAME)/bin/activate && $(PYTHON) -m black --exclude 'venv|mlops_env' .
	@. $(ENV_NAME)/bin/activate && $(PYTHON) -m pylint --disable=C,R $(SOURCE_DIR) || true
	@echo "Code vérifié avec succès !"

# Préparation des données
prepare:
	@echo "Préparation des données..."
	@./$(ENV_NAME)/bin/python3 $(MAIN_SCRIPT) --prepare
	@echo "Données préparées avec succès !"

# Entraînement du modèle
train:
	@echo "Entraînement du modèle..."
	@./$(ENV_NAME)/bin/python3 $(MAIN_SCRIPT) --train
	@echo "Modèle entraîné avec succès !"

# Évaluation du modèle
evaluate:
	@echo "Évaluation du modèle..."
	@./$(ENV_NAME)/bin/python3 $(MAIN_SCRIPT) --evaluate
	@echo "Évaluation terminée !"

# Exécution des tests
test:
	@echo "Exécution des tests unitaires..."
	@. $(ENV_NAME)/bin/activate && pytest $(TEST_DIR)
	@echo "Tests terminés avec succès !"

# Nettoyage des fichiers temporaires et environnement
clean:
	@echo "Suppression des fichiers temporaires..."
	rm -rf $(ENV_NAME)
	rm -f model.pkl scaler.pkl pca.pkl
	rm -rf __pycache__ .pytest_cache .pylint.d
	@echo "Nettoyage terminé !"

# Réinstallation complète de l'environnement
reinstall: clean setup

# Pipeline complet
all: setup verify prepare train test evaluate
	@echo "Pipeline MLOps exécuté avec succès !"
