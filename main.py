import argparse
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from model_pipeline import prepare_data, train_model, save_model, evaluate_model
import os

# Assurez-vous que le répertoire de MLflow existe et a les bonnes permissions
mlflow.set_tracking_uri("file:///tmp/mlruns")  # Utilisez un répertoire avec les bonnes permissions

if not os.path.exists("/tmp/mlruns"):
    os.makedirs("/tmp/mlruns", exist_ok=True)  # Créez le répertoire s'il n'existe pas
    os.chmod("/tmp/mlruns", 0o777)  # Donne les permissions d'écriture

# Configuration MLflow
mlflow.set_experiment("Prediction_Churn_ML")

class MLPipeline:
    def __init__(self):
        self.X_train_resampled = None
        self.X_test = None
        self.y_train_resampled = None
        self.y_test = None
        self.feature_scaler = None
        self.pca_transformer = None
        self.model = None
        self.run_id = None  # Ajouter un attribut pour stocker l'ID de la run

    def prepare_data(self):
        print("Préparation des données...")
        self.X_train_resampled, self.X_test, self.y_train_resampled, self.y_test, self.feature_scaler, self.pca_transformer = prepare_data()
        print("Données préparées avec succès !")

    def train_model(self):
        print("Entraînement du modèle avec MLflow...")

        with mlflow.start_run() as run:  # Utiliser le context manager pour capturer le run_id
            self.run_id = run.info.run_id  # Sauvegarder le run_id
            self.model = train_model(self.X_train_resampled, self.y_train_resampled)

            # Enregistrement des hyperparamètres du modèle
            mlflow.log_param("model_type", type(self.model).__name__)
            mlflow.log_param("random_state", 42)

            # Définition de la signature du modèle
            input_example = self.X_train_resampled[:5]
            signature = infer_signature(self.X_train_resampled, self.model.predict(self.X_train_resampled))

            # Enregistrement du modèle dans MLflow
            mlflow.sklearn.log_model(self.model, "model", signature=signature, input_example=input_example)

            # Enregistrement du modèle dans le Model Registry avec le run_id
            model_uri = f"runs:/{self.run_id}/model"
            model_name = "decision_tree_model"
            mlflow.register_model(model_uri, model_name)
            print(f"Le modèle a été mis à jour dans le Model Registry avec une nouvelle version.")

        print("Modèle entraîné avec succès !")

    def save_model(self):
        print("Sauvegarde du modèle et des artefacts...")
        joblib.dump(self.model, 'model.pkl')
        joblib.dump(self.feature_scaler, 'scaler.pkl')
        joblib.dump(self.pca_transformer, 'pca.pkl')
        print("Modèle, scaler et PCA sauvegardés avec succès !")

    def load_model(self):
        print("Chargement du modèle, scaler et PCA...")
        try:
            self.model = joblib.load('model.pkl')
            self.feature_scaler = joblib.load('scaler.pkl')
            self.pca_transformer = joblib.load('pca.pkl')
            print("Modèle, scaler et PCA chargés avec succès !")
        except FileNotFoundError:
            print("Aucun modèle sauvegardé trouvé. Veuillez entraîner et sauvegarder un modèle d'abord.")

    def evaluate_model(self):
        print("Évaluation du modèle...")

        if self.model is None:
            print("Modèle non formé. Entraînement du modèle avant l'évaluation.")
            if self.X_train_resampled is None:
                print("Données non préparées. Préparation des données avant l'entraînement.")
                self.prepare_data()
            self.train_model()

        metrics = evaluate_model(self.model, self.X_test, self.y_test)
        print("Évaluation terminée !")

        # Enregistrement des métriques dans MLflow
        with mlflow.start_run():
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            print("Métriques enregistrées dans MLflow.")

        return metrics


def main():
    parser = argparse.ArgumentParser(description="ML Pipeline CLI")
    parser.add_argument("--prepare", action="store_true", help="Préparer les données")
    parser.add_argument("--train", action="store_true", help="Entraîner le modèle")
    parser.add_argument("--evaluate", action="store_true", help="Évaluer le modèle")
    parser.add_argument("--save", action="store_true", help="Sauvegarder le modèle et les artefacts")
    parser.add_argument("--load", action="store_true", help="Charger le modèle sauvegardé")

    args = parser.parse_args()

    # Création de l'instance du pipeline
    pipeline = MLPipeline()

    if args.load:
        pipeline.load_model()

    if args.prepare:
        pipeline.prepare_data()

    if args.train:
        if pipeline.X_train_resampled is None:
            print("Données non préparées. Préparation des données avant l'entraînement.")
            pipeline.prepare_data()
        pipeline.train_model()

    if args.save:
        if pipeline.model is None:
            print("Modèle non formé. Entraînement du modèle avant la sauvegarde.")
            if pipeline.X_train_resampled is None:
                print("Données non préparées. Préparation des données avant l'entraînement.")
                pipeline.prepare_data()
            pipeline.train_model()
        pipeline.save_model()

    if args.evaluate:
        metrics = pipeline.evaluate_model()
        print("Métriques d'évaluation :", metrics)


if __name__ == "__main__":
    main()
