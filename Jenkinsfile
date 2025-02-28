pipeline {
    agent any

    environment {
        PYTHON = 'python3'
        ENV_NAME = 'venv'
        REQUIREMENTS = 'requirements.txt'
        SOURCE_DIR = 'model_pipeline.py'
        MAIN_SCRIPT = 'main.py'
    }

    stages {
        stage('Setup') {
            steps {
                script {
                    echo 'Création de l\'environnement virtuel et installation des dépendances...'
                    sh '''
                    ${PYTHON} -m venv ${ENV_NAME}
                    ./${ENV_NAME}/bin/python3 -m pip install --upgrade pip
                    ./${ENV_NAME}/bin/python3 -m pip install -r ${REQUIREMENTS}
                    '''
                    echo 'Environnement configuré avec succès !'
                }
            }
        }

        stage('Verify') {
            steps {
                script {
                    echo 'Vérification de la qualité du code...'
                    sh '''
                    ./${ENV_NAME}/bin/activate && ${PYTHON} -m black --exclude 'venv|mlops_env' .
                    ./${ENV_NAME}/bin/activate && ${PYTHON} -m pylint --disable=C,R ${SOURCE_DIR} || true
                    '''
                    echo 'Code vérifié avec succès !'
                }
            }
        }

        stage('Prepare Data') {
            steps {
                script {
                    echo 'Préparation des données...'
                    sh './${ENV_NAME}/bin/python3 ${MAIN_SCRIPT} --prepare'
                    echo 'Données préparées avec succès !'
                }
            }
        }

        stage('Train Model') {
            steps {
                script {
                    echo 'Entraînement du modèle...'
                    sh './${ENV_NAME}/bin/python3 ${MAIN_SCRIPT} --train'
                    echo 'Modèle entraîné avec succès !'
                }
            }
        }

        stage('Evaluate Model') {
            steps {
                script {
                    echo 'Évaluation du modèle...'
                    sh './${ENV_NAME}/bin/python3 ${MAIN_SCRIPT} --evaluate'
                    echo 'Évaluation terminée !'
                }
            }
        }

        stage('Clean') {
            steps {
                script {
                    echo 'Suppression des fichiers temporaires...'
                    sh '''
                    rm -rf ${ENV_NAME}
                    rm -f model.pkl scaler.pkl pca.pkl
                    rm -rf __pycache__ .pytest_cache .pylint.d
                    '''
                    echo 'Nettoyage terminé !'
                }
            }
        }
    }

    post {
        always {
            echo 'Pipeline terminé.'
        }
    }
}
