import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from model_pipeline import prepare_data, train_model, evaluate_model, save_model, load_model
from imblearn.over_sampling import SMOTE

class TestModelPipeline(unittest.TestCase):

    @patch("model_pipeline.pd.read_csv")
    def test_prepare_data(self, mock_read_csv):
        # Simuler un DataFrame retourné par read_csv
        mock_df = pd.DataFrame({
            'Churn': [1, 0, 1, 0],
            'State': ['CA', 'TX', 'NY', 'FL'],
            'International plan': ['Yes', 'No', 'Yes', 'No'],
            'Voice mail plan': ['Yes', 'No', 'Yes', 'No']
        })
        mock_read_csv.return_value = mock_df

        # Tester la fonction prepare_data
        X_train_resampled, X_test, y_train_resampled, y_test, feature_scaler, pca_transformer = prepare_data()

        # Vérifier que les données ont bien été préparées
        self.assertEqual(X_train_resampled.shape[0], 4)  # On devrait avoir 4 échantillons après SMOTE
        self.assertEqual(y_train_resampled.shape[0], 4)  # Vérifier que le nombre d'échantillons dans y_train correspond
        self.assertIsNotNone(feature_scaler)  # Vérifier que le scaler est retourné
        self.assertIsNotNone(pca_transformer)  # Vérifier que le PCA est retourné

    @patch("model_pipeline.DecisionTreeClassifier.fit")
    def test_train_model(self, mock_fit):
        # Simuler la formation du modèle
        mock_model = MagicMock()
        mock_fit.return_value = mock_model

        # Appeler la fonction pour entraîner le modèle
        model = train_model(None, None)

        # Vérifier que la méthode fit a été appelée
        mock_fit.assert_called_once()

    @patch("model_pipeline.joblib.dump")
    def test_save_model(self, mock_dump):
        # Simuler le modèle, le scaler et le PCA
        model = MagicMock()
        feature_scaler = MagicMock()
        pca_transformer = MagicMock()

        # Sauvegarder les artefacts
        save_model(model, feature_scaler, pca_transformer)

        # Vérifier que joblib.dump a été appelé 3 fois
        self.assertEqual(mock_dump.call_count, 3)

        # Vérifier que chaque appel de joblib.dump est fait avec le bon objet
        mock_dump.assert_any_call(model, 'model.pkl')  # Vérifiez que le modèle est bien sauvegardé
        mock_dump.assert_any_call(feature_scaler, 'scaler.pkl')  # Vérifiez le scaler
        mock_dump.assert_any_call(pca_transformer, 'pca.pkl')  # Vérifiez le PCA

    @patch("model_pipeline.joblib.load")
    def test_load_model(self, mock_load):
        # Simuler les objets chargés
        mock_model = MagicMock()
        mock_feature_scaler = MagicMock()
        mock_pca_transformer = MagicMock()

        mock_load.side_effect = [mock_model, mock_feature_scaler, mock_pca_transformer]

        # Charger le modèle
        model, feature_scaler, pca_transformer = load_model()

        # Vérifier que les objets sont bien chargés
        self.assertIsInstance(model, MagicMock)
        self.assertIsInstance(feature_scaler, MagicMock)
        self.assertIsInstance(pca_transformer, MagicMock)


if __name__ == "__main__":
    unittest.main()

