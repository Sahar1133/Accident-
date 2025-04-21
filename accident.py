
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Title
st.title("Feature Selection using Information Gain with Multiple Classifiers")

# File uploader
uploaded_file = st.file_uploader("Upload Excel dataset (.xlsx)", type=["xlsx"])
if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)
    st.subheader("Dataset Preview")
    st.write(data.head())

    # Handle missing values
    data.dropna(inplace=True)

    # Encode categorical variables
    label_encoders = {}
    for col in data.select_dtypes(include='object').columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    # Select target column
    target_col = st.selectbox("Select the Target Column", options=data.columns)
    X = data.drop(target_col, axis=1)
    y = data[target_col]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Information Gain using Decision Tree
    base_tree = DecisionTreeClassifier(criterion='entropy', random_state=42)
    base_tree.fit(X_train, y_train)
    info_gain = base_tree.feature_importances_

    feature_df = pd.DataFrame({'Feature': X.columns, 'Information_Gain': info_gain})
    feature_df.sort_values(by='Information_Gain', ascending=False, inplace=True)

    st.subheader("Feature Importances (Information Gain)")
    st.write(feature_df)

    thresholds = [0.0, 0.01, 0.05, 0.1]
    results = []

    for thresh in thresholds:
        selected = feature_df[feature_df["Information_Gain"] > thresh]["Feature"].tolist()
        st.markdown(f"### Threshold: {thresh} — Selected Features: {selected}")

        X_train_sel = X_train[selected]
        X_test_sel = X_test[selected]

        models = {
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "Naive Bayes": GaussianNB()
        }

        for model_name, model in models.items():
            model.fit(X_train_sel, y_train)
            y_pred = model.predict(X_test_sel)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='macro')
            rec = recall_score(y_test, y_pred, average='macro')

            results.append({
                "Model": model_name,
                "Threshold": thresh,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec
            })

    results_df = pd.DataFrame(results)

    st.subheader("Performance Summary")
    st.dataframe(results_df)

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    for metric in ['Accuracy', 'Precision', 'Recall']:
        for model in results_df['Model'].unique():
            subset = results_df[results_df['Model'] == model]
            ax.plot(subset['Threshold'], subset[metric], marker='o', label=f"{model} - {metric}")

    ax.set_title("Model Performance vs. Information Gain Threshold")
    ax.set_xlabel("Information Gain Threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)
    fig.tight_layout()
    st.pyplot(fig)

    st.subheader("Best Model by Metric")
    for metric in ["Accuracy", "Precision", "Recall"]:
        idx = results_df[metric].idxmax()
        st.markdown(f"**{metric}:** Best Model: `{results_df.loc[idx, 'Model']}`, "
                    f"Threshold: `{results_df.loc[idx, 'Threshold']}`, "
                    f"Score: `{results_df.loc[idx, metric]:.2f}`")