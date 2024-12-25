import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, classification_report, roc_curve, auc
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Title
st.title("Machine Learning Model Evaluation App")

# Sidebar for user interaction
st.sidebar.header("Upload Dataset")
file_upload = st.sidebar.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if file_upload is not None:
    if file_upload.name.endswith("csv"):
        data = pd.read_csv(file_upload)
    else:
        data = pd.read_excel(file_upload)

    st.write("### Dataset Preview")
    st.write(data)

    # Preprocessing Options
    st.sidebar.subheader("Preprocessing Options")
    target_column = st.sidebar.selectbox("Select Target Column", data.columns)

    # Feature Selection
    feature_columns = st.sidebar.multiselect("Select Features", [col for col in data.columns if col != target_column])
    
    if not feature_columns:
        st.error("Please select at least one feature for the model.")
    else:
        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

        # Standardize numerical features
        if st.sidebar.checkbox("Standardize numerical features", False):
            scaler = StandardScaler()
            data[feature_columns] = scaler.fit_transform(data[feature_columns])
            st.write("### Features standardized.")

        # Encode categorical variables if needed
        if data[target_column].dtype == 'object' or data.dtypes.isin(['object']).any():
            encoder = LabelEncoder()
            for col in data.select_dtypes(include='object').columns:
                data[col] = encoder.fit_transform(data[col])
            st.write("Dataset automatically encoded for categorical variables.")

        # Splitting data
        st.sidebar.subheader("Train/Test Split")
        test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)
        X = data[feature_columns]
        y = data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Model Selection
        st.sidebar.subheader("Choose Model")
        model_type = st.sidebar.selectbox("Model Type", ["Classification", "Regression"])

        if model_type == "Classification" and y.dtype != 'object':
            st.error("Selected model type 'Classification' but the target variable is continuous. Please choose 'Regression' instead.")
        elif model_type == "Regression" and y.dtype == 'object':
            st.error("Selected model type 'Regression' but the target variable is categorical. Please choose 'Classification' instead.")
        else:
            if model_type == "Classification":
                model_name = st.sidebar.selectbox(
                    "Classifier", ["Logistic Regression", "Decision Tree", "Random Forest", "K-Nearest Neighbors", "Support Vector Machine"])

                if model_name == "Logistic Regression":
                    model = LogisticRegression()
                elif model_name == "Decision Tree":
                    model = DecisionTreeClassifier()
                elif model_name == "Random Forest":
                    n_estimators = st.sidebar.slider("Number of Trees", 10, 100, 50)
                    model = RandomForestClassifier(n_estimators=n_estimators)
                elif model_name == "K-Nearest Neighbors":
                    n_neighbors = st.sidebar.slider("Number of Neighbors", 1, 20, 5)
                    model = KNeighborsClassifier(n_neighbors=n_neighbors)
                elif model_name == "Support Vector Machine":
                    kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly"])
                    model = SVC(kernel=kernel)

            elif model_type == "Regression":
                model_name = st.sidebar.selectbox(
                    "Regressor", ["Linear Regression", "Decision Tree", "Random Forest", "Gradient Boosting", "Support Vector Regressor"])

                if model_name == "Linear Regression":
                    model = LinearRegression()
                elif model_name == "Decision Tree":
                    model = DecisionTreeRegressor()
                elif model_name == "Random Forest":
                    n_estimators = st.sidebar.slider("Number of Trees", 10, 100, 50)
                    model = RandomForestRegressor(n_estimators=n_estimators)
                elif model_name == "Gradient Boosting":
                    n_estimators = st.sidebar.slider("Number of Boosting Stages", 10, 100, 50)
                    model = GradientBoostingRegressor(n_estimators=n_estimators)
                elif model_name == "Support Vector Regressor":
                    kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly"])
                    model = SVR(kernel=kernel)

            # Hyperparameter Tuning with GridSearchCV
            if st.sidebar.checkbox("Perform Hyperparameter Tuning", False):
                param_grid = {
                    "n_estimators": [10, 50, 100],
                    "max_depth": [None, 10, 20, 30]
                } if model_name == "Random Forest" else {}

                grid_search = GridSearchCV(model, param_grid, cv=5)
                model = grid_search.fit(X_train, y_train)
                st.write(f"Best Parameters: {model.best_params_}")

            # Train the Model
            if st.sidebar.button("Train Model"):
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                if model_type == "Classification":
                    acc = accuracy_score(y_test, predictions)
                    cm = confusion_matrix(y_test, predictions)
                    precision = precision_score(y_test, predictions, average='weighted')
                    recall = recall_score(y_test, predictions, average='weighted')
                    f1 = f1_score(y_test, predictions, average='weighted')

                    st.write(f"### {model_name} Performance")
                    st.write(f"Accuracy: {acc:.4f}")
                    st.write(f"Precision: {precision:.4f}")
                    st.write(f"Recall: {recall:.4f}")
                    st.write(f"F1 Score: {f1:.4f}")

                    st.write("#### Confusion Matrix (Heatmap)")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('True')
                    ax.set_title('Confusion Matrix')
                    st.pyplot(fig)

                    # Classification Report
                    st.write("#### Classification Report")
                    st.text(classification_report(y_test, predictions))

                    # ROC Curve (for binary classification)
                    if len(np.unique(y_test)) == 2:
                        fpr, tpr, _ = roc_curve(y_test, predictions)
                        roc_auc = auc(fpr, tpr)
                        st.write(f"ROC AUC: {roc_auc:.4f}")
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                        ax.set_xlim([0.0, 1.0])
                        ax.set_ylim([0.0, 1.05])
                        ax.set_xlabel('False Positive Rate')
                        ax.set_ylabel('True Positive Rate')
                        ax.set_title('Receiver Operating Characteristic')
                        ax.legend(loc="lower right")
                        st.pyplot(fig)

                elif model_type == "Regression":
                    mse = mean_squared_error(y_test, predictions)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, predictions)

                    st.write(f"### {model_name} Performance")
                    st.write(f"Mean Squared Error: {mse:.4f}")
                    st.write(f"Root Mean Squared Error: {rmse:.4f}")
                    st.write(f"R-squared: {r2:.4f}")

                    st.write("#### Actual vs Predicted (Scatter Plot)")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.scatter(y_test, predictions, alpha=0.6)
                    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal Fit')
                    ax.set_xlabel('Actual Values')
                    ax.set_ylabel('Predicted Values')
                    ax.set_title(f"{model_name} - Actual vs Predicted")
                    st.pyplot(fig)

                    # Feature Importance (for tree-based models)
                    if hasattr(model, 'feature_importances_'):
                        feature_importances = model.feature_importances_
                        sorted_idx = np.argsort(feature_importances)[::-1]
                        st.write("#### Feature Importance")
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.barh(np.array(feature_columns)[sorted_idx], feature_importances[sorted_idx])
                        ax.set_xlabel('Feature Importance')
                        ax.set_title('Top Features')
                        st.pyplot(fig)

# Instruction how to use
st.sidebar.markdown("---")
st.sidebar.markdown("### Documentation")
st.sidebar.markdown("- Upload a dataset to start.")
st.sidebar.markdown("- Choose preprocessing options.")
st.sidebar.markdown("- Select features and target variable.")
st.sidebar.markdown("- Select model type and specific model.")
st.sidebar.markdown("- Adjust parameters and train.")
