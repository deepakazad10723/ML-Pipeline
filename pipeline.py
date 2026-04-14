import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression, f_classif

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

from sklearn.metrics import accuracy_score, mean_squared_error

st.set_page_config(layout="wide")
st.title("🚀 ML Pipeline Dashboard")

# Sidebar
st.sidebar.title("⚙️ Problem Setup")

problem_type = st.sidebar.selectbox(
    "Select Problem Type",
    ["Classification", "Regression", "Clustering"]
)

st.session_state.problem_type = problem_type

# Dataset
st.sidebar.title("📂 Dataset")

dataset_option = st.sidebar.selectbox(
    "Select Dataset",
    ["Upload CSV", "Iris (Classification)", "California Housing (Regression)"]
)

def load_dataset(option):
    if option == "Iris (Classification)":
        from sklearn.datasets import load_iris
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["target"] = data.target
        return df
    elif option == "California Housing (Regression)":
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["target"] = data.target
        return df
    return None

# Steps
st.sidebar.title("📌 Pipeline Steps")

step = st.sidebar.radio(
    "Go to Step",
    [
        "1. Input Data",
        "2. EDA",
        "3. Cleaning",
        "4. Feature Selection",
        "5. Data Split",
        "6. Model Selection",
        "7. Training",
        "8. Metrics"
    ]
)

# Session init
for key in ["df","X","y","X_train","X_test","y_train","y_test","model","model_name"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ================= STEP 1 =================
if step == "1. Input Data":
    if dataset_option == "Upload CSV":
        file = st.file_uploader("Upload CSV", type=["csv"])
        if file:
            df = pd.read_csv(file)
        else:
            st.stop()
    else:
        df = load_dataset(dataset_option)

    st.session_state.df = df
    st.write(df.head())

    if problem_type != "Clustering":
        target = st.selectbox("Target", df.columns, index=len(df.columns)-1)
        features = st.multiselect("Features", df.columns.drop(target), default=df.columns.drop(target))

        st.session_state.X = df[features]

        # ================= FINAL FIX FOR TARGET =================
        y = df[target]

        from sklearn.preprocessing import LabelEncoder

        if y.dtype == "object":
            le_y = LabelEncoder()
            y = pd.Series(le_y.fit_transform(y.astype(str)))

        st.session_state.y = y

        st.subheader("🎯 Target Preview")
        st.write(st.session_state.y.head())

        if problem_type == "Classification":
            st.bar_chart(st.session_state.y.value_counts())
        else:
            st.line_chart(st.session_state.y)

    else:
        st.session_state.X = df
        st.session_state.y = None

# ================= STEP 2 =================
elif step == "2. EDA":
    df = st.session_state.df

    if df is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Duplicates", df.duplicated().sum())

        st.write("Data Types", df.dtypes)
        st.write("Missing Values", df.isnull().sum())
        st.write("Unique Values", df.nunique())
        st.write("Summary", df.describe())

        # Correlation FIX
        df_encoded = pd.get_dummies(df, drop_first=True)
        numeric_df = df_encoded.select_dtypes(include=np.number)

        if numeric_df.shape[1] > 1:
            st.write("Correlation", numeric_df.corr())
        else:
            st.warning("Not enough numeric data")

# ================= STEP 3 =================
elif step == "3. Cleaning":
    df = st.session_state.df
    if df is not None:
        method = st.selectbox("Fill NA", ["Mean","Median","Mode"])

        if st.button("Apply"):
            for col in df.columns:
                if df[col].isnull().sum() > 0:
                    if method == "Mean":
                        df[col].fillna(df[col].mean(), inplace=True)
                    elif method == "Median":
                        df[col].fillna(df[col].median(), inplace=True)
                    else:
                        df[col].fillna(df[col].mode()[0], inplace=True)

            st.success("Cleaning Done")

# ================= STEP 4 =================
elif step == "4. Feature Selection":
    if st.session_state.X is not None:

        X = st.session_state.X.copy()
        y = st.session_state.y

        # ================= PREPROCESS =================
        from sklearn.preprocessing import LabelEncoder, StandardScaler

        # ✅ Separate numeric & categorical
        num_cols = X.select_dtypes(include=np.number).columns
        cat_cols = X.select_dtypes(exclude=np.number).columns

        # ✅ Encode categorical safely
        X_encoded = X.copy()
        for col in cat_cols:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))

        # ================= FEATURE SELECTION =================
        method = st.selectbox(
            "Method",
            ["Variance Threshold","Information Gain","Z-Score Filtering","ANOVA"]
        )

        # ---------------- VARIANCE ----------------
        if method == "Variance Threshold":
            threshold = st.slider("Threshold", 0.0, 1.0, 0.0, 0.01)
            sel = VarianceThreshold(threshold=threshold)
            sel.fit(X_encoded)

            selected_features = X_encoded.columns[sel.get_support()]
            st.write(selected_features)

            st.session_state.X = X_encoded[selected_features]

        # ---------------- INFO GAIN ----------------
       
        # ---------------- Z-SCORE (FULL FIX) ----------------
        elif method == "Z-Score Filtering":
            from scipy.stats import zscore

            # ✅ Only numeric after encoding
            X_numeric = X_encoded.astype(float)

            # ✅ Scaling (VERY IMPORTANT)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_numeric)

            # ✅ Z-score safe
            z_scores = np.abs(zscore(X_scaled, nan_policy='omit'))

            threshold = st.slider("Z-score Threshold", 1.0, 5.0, 3.0)

            # ✅ Feature selection
            mask = (z_scores < threshold).all(axis=0)
            selected_features = X_encoded.columns[mask]

            st.write("Selected Features:", selected_features)

            st.session_state.X = X_encoded[selected_features]

        # ---------------- ANOVA ----------------
        else:
            if problem_type != "Classification":
                st.warning("ANOVA only for classification")
                st.stop()

            f_scores, _ = f_classif(X_encoded, y)

            score_df = pd.DataFrame({
                "Feature": X_encoded.columns,
                "Score": f_scores
            }).sort_values(by="Score", ascending=False)

            st.write(score_df)

            top_k = st.slider("Top K", 1, len(score_df), 5)
            st.session_state.X = X_encoded[score_df.head(top_k)["Feature"]]

# ================= STEP 5 =================
elif step == "5. Data Split":

    X = st.session_state.X.copy()

    # ✅ CLUSTERING FIX (NO SPLIT)
    if problem_type == "Clustering":

        # ✅ Ensure numeric + encoding
        X = pd.get_dummies(X, drop_first=True)

        # ✅ Safety: empty check
        if X.shape[1] == 0:
            st.error("No usable features for clustering")
            st.stop()

        st.session_state.X_train = X

        st.success("Clustering Data Ready ✅ (No Split Required)")

    else:
        # ✅ Classification & Regression (same as before)
        X = pd.get_dummies(X, drop_first=True)
        y = st.session_state.y

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test

        st.success("Split Done ✅")

# ================= STEP 6 =================
elif step == "6. Model Selection":
    if st.session_state.X_train is None:
        st.warning("Complete Step 5 first")
        st.stop()

    if problem_type == "Regression":
        model_name = st.selectbox("Model", ["Linear Regression","SVR","Random Forest"])
    elif problem_type == "Classification":
        model_name = st.selectbox("Model", ["SVM","Random Forest"])
    else:
        model_name = st.selectbox("Model", ["KMeans"])

    st.session_state.model_name = model_name
    st.success(f"Selected: {model_name}")

# ================= STEP 7 =================
# ================= STEP 7 =================
elif step == "7. Training":

    if st.session_state.X_train is None:
        st.warning("Run Step 5 first")
        st.stop()

    X_train = st.session_state.X_train

    # ================= CLUSTERING =================
    if problem_type == "Clustering":

        if st.session_state.model_name == "KMeans":
            model = KMeans(n_clusters=3)
        else:
            from sklearn.cluster import DBSCAN
            model = DBSCAN()

        model.fit(X_train)

        st.session_state.model = model
        st.success("Clustering Completed ✅")
        st.stop()

    # ================= TARGET =================
    y_train = st.session_state.y_train

    if y_train is None:
        st.error("Target missing. Please re-run Step 1 & Step 5")
        st.stop()

    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    # ✅ FORCE CLEAN COPY
    y_train = pd.Series(y_train).copy()

    # ================= SAFE ENCODING =================
    # convert only if string/object
    if str(y_train.dtype) == "object" or "string" in str(y_train.dtype):
        le_y = LabelEncoder()
        y_train = le_y.fit_transform(y_train.astype(str))

    # ================= MODEL SELECTION =================
    if problem_type == "Regression":
        if st.session_state.model_name == "Linear Regression":
            model = LinearRegression()
        elif st.session_state.model_name == "SVR":
            model = SVR()
        else:
            model = RandomForestRegressor()

    elif problem_type == "Classification":
        if st.session_state.model_name == "SVM":
            model = SVC()
        else:
            model = RandomForestClassifier()

    # ================= TRAIN =================
    model.fit(X_train, y_train)

    st.session_state.model = model
    st.success("Model Trained Successfully ✅")
# ================= STEP 8 =================
# ================= STEP 8 =================
elif step == "8. Metrics":
    model = st.session_state.model

    if model is None:
        st.warning("Train model first")
        st.stop()

    from sklearn.preprocessing import LabelEncoder

    # ================= CLASSIFICATION =================
    if problem_type == "Classification":

        X_test = st.session_state.X_test
        y_test = st.session_state.y_test

        # ✅ FIX: encode y_test if string
        if isinstance(y_test, pd.Series) and y_test.dtype == "object":
            le_y = LabelEncoder()
            y_test = le_y.fit_transform(y_test.astype(str))

        preds = model.predict(X_test)

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average='weighted', zero_division=0)
        rec = recall_score(y_test, preds, average='weighted', zero_division=0)
        f1 = f1_score(y_test, preds, average='weighted', zero_division=0)

        st.subheader("📊 Classification Metrics")
        st.write(f"Accuracy: {acc:.4f}")
        st.write(f"Precision: {prec:.4f}")
        st.write(f"Recall: {rec:.4f}")
        st.write(f"F1 Score: {f1:.4f}")

    # ================= REGRESSION =================
    elif problem_type == "Regression":

        X_test = st.session_state.X_test
        y_test = st.session_state.y_test

        # ✅ FIX: encode y_test if string
        if isinstance(y_test, pd.Series) and y_test.dtype == "object":
            le_y = LabelEncoder()
            y_test = le_y.fit_transform(y_test.astype(str))

        preds = model.predict(X_test)

        from sklearn.metrics import mean_squared_error
        import numpy as np

        rmse = np.sqrt(mean_squared_error(y_test, preds))

        st.subheader("📊 Regression Metrics")
        st.write(f"RMSE: {rmse:.4f}")

    # ================= CLUSTERING =================
    else:
        st.success("Clustering Completed")