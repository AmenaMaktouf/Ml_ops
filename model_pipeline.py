import pandas as pd
import joblib
from imblearn.combine import SMOTEENN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def prepare_data():
    """Load, preprocess the dataset, split into train/test, and apply SMOTEENN."""
    print("Starting data preparation...")

    df = pd.read_csv("merged_churn1.csv")
    print("Dataset loaded successfully")

    # Checking for missing values
    print("Missing values per column:", df.isnull().sum())

    # Checking for duplicates
    print(f"Number of duplicate rows: {df.duplicated().sum()}")

    # Separating features and target
    target = df["Churn"]
    features = df.drop(columns=["Churn"])
    print("Target variable separated")

    # Frequency encoding for 'State'
    print("Applying frequency encoding for 'State'...")
    features["State"] = features["State"].map(
        features["State"].value_counts().to_dict()
    )
    print("Frequency encoding applied")

    # Convert categorical 'Yes'/'No' columns to numeric 0/1
    print("Converting categorical variables to numeric...")
    features["International plan"] = features["International plan"].map(
        {"Yes": 1, "No": 0}
    )
    features["Voice mail plan"] = features["Voice mail plan"].map({"Yes": 1, "No": 0})
    print("Categorical conversion completed")

    # Standardizing numerical features
    print("Applying standardization...")
    feature_scaler = StandardScaler()
    features_scaled = feature_scaler.fit_transform(features)
    print("Standardization completed")

    # Apply PCA for feature reduction (keeping 95% variance)
    print("Performing PCA for dimensionality reduction...")
    pca_transformer = PCA(n_components=0.95)
    features_pca = pca_transformer.fit_transform(features_scaled)
    print(f"PCA completed: Reduced to {features_pca.shape[1]} components")

    # Train-Test Split
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        features_pca, target, test_size=0.2, random_state=42, stratify=target
    )
    print("Train-test split completed")

    # Handling class imbalance with SMOTEENN
    print("Applying SMOTEENN for class balancing...")
    smote_enn = SMOTEENN(sampling_strategy="auto", random_state=100)
    X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train, y_train)
    print(f"SMOTEENN applied: {X_train_resampled.shape[0]} samples after resampling")

    print("Data preprocessing completed")
    return X_train_resampled, X_test, y_train_resampled, y_test, feature_scaler, pca_transformer


def train_model(train_features, train_labels):
    """Train a Decision Tree model."""
    print("Training the Decision Tree model...")
    dt_model = DecisionTreeClassifier(
        criterion="gini", random_state=100, max_depth=6, min_samples_leaf=8
    )
    dt_model.fit(train_features, train_labels)
    print("Model training completed")

    return dt_model


def evaluate_model(model_instance, test_features, test_labels):
    """Evaluate the model's performance."""
    print("Evaluating the model...")
    predictions = model_instance.predict(test_features)

    accuracy = accuracy_score(test_labels, predictions)
    report = classification_report(test_labels, predictions, output_dict=True)  # Convertir en dict
    conf_matrix = confusion_matrix(test_labels, predictions)

    print(f"Model Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", classification_report(test_labels, predictions))
    print("Confusion Matrix:\n", conf_matrix)
    # Retourner un dictionnaire des métriques pour éviter l'erreur
    return {
        "accuracy": accuracy,
        "precision_false": report["False"]["precision"],
        "recall_false": report["False"]["recall"],
        "f1_score_false": report["False"]["f1-score"],
        "precision_true": report["True"]["precision"],
        "recall_true": report["True"]["recall"],
        "f1_score_true": report["True"]["f1-score"]
    }


def save_model(model_instance, feature_scaler, pca_transformer):
    """Save model and preprocessing artifacts."""
    print("Saving model and preprocessing artifacts...")
    joblib.dump(model_instance, "model.pkl")
    joblib.dump(feature_scaler, "scaler.pkl")
    joblib.dump(pca_transformer, "pca.pkl")
    print("Model, scaler, and PCA saved successfully")


def load_model():
    """Load model and preprocessing artifacts if they exist."""
    try:
        model_instance= joblib.load("model.pkl")
        feature_scaler = joblib.load("scaler.pkl")
        pca_transformer = joblib.load("pca.pkl")
        print("Model and preprocessing artifacts loaded successfully")
        return model_instance, feature_scaler, pca_transformer
    except FileNotFoundError:
        print("Error: Model or preprocessing artifacts not found. Train the model first.")
        return None, None, None
