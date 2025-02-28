import unittest
from unittest.mock import patch
import sys
from main import main

class TestMain(unittest.TestCase):

    @patch("builtins.print")
    @patch("sys.argv", new=["main.py", "--prepare"])
    def test_main_prepare(self, mock_print):
        main()  # Appel à la fonction main qui exécute le script avec les arguments donnés
        mock_print.assert_called_with("Préparation des données...")  # Vérifier que le message est affiché

    @patch("builtins.print")
    @patch("sys.argv", new=["main.py", "--train"])
    def test_main_train(self, mock_print):
        main()
        mock_print.assert_called_with("Entraînement du modèle...")  # Vérifier que l'entraînement est lancé

    @patch("builtins.print")
    @patch("sys.argv", new=["main.py", "--save"])
    def test_main_save(self, mock_print):
        main()
        mock_print.assert_called_with("Sauvegarde du modèle et des artefacts...")

    @patch("builtins.print")
    @patch("sys.argv", new=["main.py", "--evaluate"])
    def test_main_evaluate(self, mock_print):
        main()
        mock_print.assert_called_with("Évaluation du modèle...")  # Vérifier que l'évaluation est lancée


if __name__ == "__main__":
    unittest.main()
