import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# --- PAGE CONFIG ---
st.set_page_config(page_title="Titanic Analysis", layout="wide")

st.title("🚢 Titanic Survival Analysis Dashboard")

# --- 1. LOAD DATA ---
@st.cache_data
def load_data():
    # Ensure Titanic-Dataset.csv is in the same folder
    return pd.read_csv('Titanic-Dataset.csv')

df = load_data()

# --- 2. DATA PREPARATION ---
# Selecting features (same logic as your original code)
X = df.drop(['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis=1)
y = df['Survived']

numeric_features = ['Age', 'SibSp', 'Parch', 'Fare']
categorical_features = ['Pclass', 'Sex', 'Embarked']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. BUILDING THE PIPELINE ---
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Creating the rf_pipeline variable BEFORE it is used in Section 5
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Training the model
rf_pipeline.fit(X_train, y_train)
rf_pred = rf_pipeline.predict(X_test)

# --- 4. DISPLAYING RESULTS IN STREAMLIT ---

col1, col2 = st.columns(2)

with col1:
    st.header("Feature Importance")
    # Section 5: Accessing the trained pipeline for importances
    cat_encoder = rf_pipeline.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot']
    encoded_cat_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
    all_feature_names = numeric_features + encoded_cat_names

    importances = rf_pipeline.named_steps['classifier'].feature_importances_
    feature_matrix = pd.DataFrame({'Feature': all_feature_names, 'Importance': importances})
    feature_matrix = feature_matrix.sort_values(by='Importance', ascending=False)
    
    # Show table
    st.dataframe(feature_matrix, use_container_width=True)
    
    # Simple Bar Chart
    st.bar_chart(feature_matrix.set_index('Feature'))

with col2:
    st.header("Confusion Matrix")
    conf_matrix = confusion_matrix(y_test, rf_pred)
    conf_df = pd.DataFrame(conf_matrix, 
                           index=['Actual Perished', 'Actual Survived'], 
                           columns=['Pred Perished', 'Pred Survived'])
    
    # Display accuracy score
    acc = accuracy_score(y_test, rf_pred)
    st.metric("Model Accuracy", f"{acc:.2%}")

    # Plotting Heatmap
    fig, ax = plt.subplots()
    sns.heatmap(conf_df, annot=True, fmt='d', cmap='RdBu', ax=ax)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    st.pyplot(fig)

st.divider()
if st.checkbox("Show Raw Data"):
    st.write(df)