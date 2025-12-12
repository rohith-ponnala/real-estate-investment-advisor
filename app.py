import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ===================== STREAMLIT PAGE CONFIG (MUST BE FIRST) ===================== #

st.set_page_config(
    page_title="Real Estate Investment Advisor",
    layout="wide"
)

# ===================== PATHS ===================== #

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "india_housing_prices.csv")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
CLASSIFIER_PATH = os.path.join(SAVED_MODELS_DIR, "best_classifier.pkl")
REGRESSOR_PATH = os.path.join(SAVED_MODELS_DIR, "best_regressor.pkl")


# ===================== LOAD MODELS & DATA (CACHED) ===================== #

@st.cache_resource
def load_models():
    try:
        clf = joblib.load(CLASSIFIER_PATH)
        reg = joblib.load(REGRESSOR_PATH)
        return clf, reg
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.stop()


@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_PATH)

        CURRENT_YEAR = 2025

        # Ensure Age_of_Property exists (same idea as in training)
        if "Age_of_Property" not in df.columns and "Year_Built" in df.columns:
            df["Age_of_Property"] = CURRENT_YEAR - df["Year_Built"]

        # Ensure PTA is numeric for medians
        if "Public_Transport_Accessibility" in df.columns:
            df["Public_Transport_Accessibility"] = pd.to_numeric(
                df["Public_Transport_Accessibility"], errors="coerce"
            )

        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()


# ===================== HELPER: BUILD INPUT DF ===================== #

def build_input_df(
    state, city, locality, property_type,
    bhk, size_sqft, price_lakhs,
    furnished_status, floor_no, total_floors,
    year_built,
    nearby_schools, nearby_hospitals,
    public_transport, parking_space,
    security, amenities,
    facing, owner_type, availability_status
):
    # Compute derived features
    price_per_sqft = (price_lakhs * 100000) / size_sqft
    current_year = 2025
    age_of_property = current_year - year_built

    input_dict = {
        "State": [state],
        "City": [city],
        "Locality": [locality],
        "Property_Type": [property_type],
        "BHK": [bhk],
        "Size_in_SqFt": [size_sqft],
        "Price_in_Lakhs": [price_lakhs],
        "Price_per_SqFt": [price_per_sqft],
        "Year_Built": [year_built],
        "Furnished_Status": [furnished_status],
        "Floor_No": [floor_no],
        "Total_Floors": [total_floors],
        "Age_of_Property": [age_of_property],
        "Nearby_Schools": [nearby_schools],
        "Nearby_Hospitals": [nearby_hospitals],
        "Public_Transport_Accessibility": [public_transport],
        "Parking_Space": [parking_space],
        "Security": [security],
        "Amenities": [amenities],
        "Facing": [facing],
        "Owner_Type": [owner_type],
        "Availability_Status": [availability_status],
    }

    return pd.DataFrame(input_dict)


# ===================== HELPER: ADD INVESTMENT_SCORE ===================== #

def add_investment_score(input_df: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    """
    Recreate the Investment_Score feature for the single input row,
    using the same kind of logic as in the training notebook:
      - +1 if Parking_Space > 0
      - +1 if PTA >= median PTA
      - +1 if Age_of_Property <= median Age_of_Property
    """
    df_in = input_df.copy()

    # Medians from training data
    median_age = None
    if "Age_of_Property" in data.columns:
        median_age = data["Age_of_Property"].median()
    elif "Year_Built" in data.columns:
        median_age = (2025 - data["Year_Built"]).median()

    median_pta = None
    if "Public_Transport_Accessibility" in data.columns:
        median_pta = data["Public_Transport_Accessibility"].median()

    score = 0

    # Parking space rule
    if "Parking_Space" in df_in.columns:
        try:
            score += int(df_in.loc[0, "Parking_Space"] > 0)
        except Exception:
            pass

    # Public transport rule
    if (
        "Public_Transport_Accessibility" in df_in.columns
        and median_pta is not None
        and not np.isnan(median_pta)
    ):
        try:
            score += int(df_in.loc[0, "Public_Transport_Accessibility"] >= median_pta)
        except Exception:
            pass

    # Age rule (newer than median)
    if (
        "Age_of_Property" in df_in.columns
        and median_age is not None
        and not np.isnan(median_age)
    ):
        try:
            score += int(df_in.loc[0, "Age_of_Property"] <= median_age)
        except Exception:
            pass

    df_in["Investment_Score"] = score
    return df_in


# ===================== FEATURE IMPORTANCE HELPER ===================== #

def get_feature_importance_from_pipeline(pipeline):
    """
    Try to extract feature importance or coefficients from a sklearn Pipeline
    that has steps: preprocessor + model.
    Returns a dataframe (feature, importance) or None if not possible.
    """
    try:
        pre = pipeline.named_steps["preprocessor"]
        model = pipeline.named_steps["model"]

        # Get transformed feature names
        feature_names = pre.get_feature_names_out()

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            coefs = model.coef_
            if coefs.ndim > 1:
                coefs = coefs[0]
            importances = np.abs(coefs)
        else:
            return None

        fi = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False)

        return fi.head(15)
    except Exception:
        return None


# ===================== MAIN APP ===================== #

def main():
    # Load models & data lazily (after page config)
    clf_model, reg_model = load_models()
    data = load_data()

    st.title("🏠 Real Estate Investment Advisor")

    st.write(
        """
        This app helps investors analyze whether a property is a **Good Investment** and 
        predicts its **estimated price after 5 years**, using machine learning models trained on real estate data.
        """
    )

    # Sidebar navigation
    mode = st.sidebar.radio(
        "Choose Mode",
        ["🔮 Investment Prediction", "📊 Data Explorer & Insights"]
    )

    # ---------------- MODE 1: PREDICTION ---------------- #
    if mode == "🔮 Investment Prediction":
        st.header("🔮 Property Investment Prediction")

        col1, col2, col3 = st.columns(3)

        with col1:
            state = st.text_input("State", value="Karnataka")
            city = st.text_input("City", value="Bengaluru")
            locality = st.text_input("Locality", value="Sample Locality")

            property_type = st.selectbox(
                "Property Type",
                ["Apartment", "Villa", "Independent House", "Other"]
            )

            bhk = st.number_input("BHK", min_value=1, max_value=10, value=2, step=1)
            size_sqft = st.number_input("Size (SqFt)", min_value=200, max_value=10000, value=1000, step=50)

        with col2:
            price_lakhs = st.number_input("Current Price (Lakhs)", min_value=5.0, max_value=10000.0, value=75.0, step=1.0)
            year_built = st.number_input("Year Built", min_value=1950, max_value=2025, value=2018, step=1)

            furnished_status = st.selectbox(
                "Furnished Status",
                ["Unfurnished", "Semi-Furnished", "Fully-Furnished"]
            )

            floor_no = st.number_input("Floor No", min_value=0, max_value=200, value=2, step=1)
            total_floors = st.number_input("Total Floors", min_value=1, max_value=200, value=10, step=1)

            nearby_schools = st.number_input("Nearby Schools (count)", min_value=0, max_value=50, value=3, step=1)
            nearby_hospitals = st.number_input("Nearby Hospitals (count)", min_value=0, max_value=50, value=2, step=1)

        with col3:
            parking_space = st.number_input("Parking Space (slots)", min_value=0, max_value=20, value=1, step=1)
            public_transport = st.slider("Public Transport Accessibility (1–5)", min_value=1, max_value=5, value=4)

            security = st.text_input("Security (e.g., Gated, CCTV)", value="Gated")
            amenities = st.text_input("Amenities (e.g., Gym, Pool)", value="Gym, Pool")

            facing = st.selectbox("Facing", ["North", "South", "East", "West", "Other"])
            owner_type = st.selectbox("Owner Type", ["Individual", "Builder", "Agent"])
            availability_status = st.selectbox("Availability Status", ["Available", "Under Construction", "Sold"])

        if st.button("🔍 Predict Investment & Future Price"):
            # Base input
            input_df = build_input_df(
                state, city, locality, property_type,
                bhk, size_sqft, price_lakhs,
                furnished_status, floor_no, total_floors,
                year_built,
                nearby_schools, nearby_hospitals,
                public_transport, parking_space,
                security, amenities,
                facing, owner_type, availability_status
            )

            # Add Investment_Score so that it matches training features
            input_df = add_investment_score(input_df, data)

            st.subheader("📋 Input Summary")
            st.dataframe(input_df)

            # --- Classification prediction ---
            good_invest_pred = clf_model.predict(input_df)[0]
            if hasattr(clf_model, "predict_proba"):
                good_invest_proba = clf_model.predict_proba(input_df)[0][1]
            else:
                good_invest_proba = None

            # --- Regression prediction ---
            future_price_pred = reg_model.predict(input_df)[0]

            st.subheader("🧠 Model Outputs")

            colA, colB = st.columns(2)

            with colA:
                st.markdown("### Investment Decision")
                if good_invest_pred == 1:
                    st.success("✅ This property is predicted to be a **Good Investment**.")
                else:
                    st.error("⚠️ This property may **not be a great investment**.")

                if good_invest_proba is not None:
                    st.write(f"**Model Confidence (Good Investment):** {good_invest_proba * 100:.2f}%")
                    st.progress(float(min(max(good_invest_proba, 0.0), 1.0)))

            with colB:
                st.markdown("### Future Price Forecast (5 Years)")
                st.write(f"**Estimated Price after 5 years:** ~ ₹ {future_price_pred:.2f} Lakhs")

                curr_price = price_lakhs
                appreciation = ((future_price_pred - curr_price) / curr_price) * 100
                st.write(f"**Expected Appreciation:** {appreciation:.2f}%")

            st.markdown("---")
            st.subheader("📌 Feature Importance (Global, Model-based)")

            fi_clf = get_feature_importance_from_pipeline(clf_model)
            if fi_clf is not None:
                st.markdown("**Top features impacting Good Investment prediction:**")
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(data=fi_clf, x="importance", y="feature", ax=ax)
                ax.set_xlabel("Importance")
                ax.set_ylabel("Feature")
                st.pyplot(fig)
            else:
                st.info("Feature importance could not be extracted for the classifier model.")

    # ---------------- MODE 2: DATA EXPLORER ---------------- #
    elif mode == "📊 Data Explorer & Insights":
        st.header("📊 Data Explorer & Visual Insights")

        st.markdown("Use filters in the sidebar to explore properties and trends.")

        # Sidebar filters
        st.sidebar.subheader("Filters")

        df_filtered = data.copy()

        # City filter
        if "City" in df_filtered.columns:
            cities = sorted(df_filtered["City"].dropna().unique().tolist())
            selected_cities = st.sidebar.multiselect(
                "Filter by City",
                options=cities,
                default=cities[:3] if len(cities) >= 3 else cities
            )
            if selected_cities:
                df_filtered = df_filtered[df_filtered["City"].isin(selected_cities)]

        # BHK filter
        if "BHK" in df_filtered.columns:
            min_bhk = int(df_filtered["BHK"].min())
            max_bhk = int(df_filtered["BHK"].max())
            bhk_range = st.sidebar.slider(
                "BHK Range", min_value=min_bhk, max_value=max_bhk,
                value=(min_bhk, max_bhk)
            )
            df_filtered = df_filtered[
                (df_filtered["BHK"] >= bhk_range[0]) &
                (df_filtered["BHK"] <= bhk_range[1])
            ]

        # Price filter
        if "Price_in_Lakhs" in df_filtered.columns:
            min_price = float(df_filtered["Price_in_Lakhs"].min())
            max_price = float(df_filtered["Price_in_Lakhs"].max())
            price_range = st.sidebar.slider(
                "Price Range (Lakhs)",
                min_value=min_price,
                max_value=max_price,
                value=(min_price, max_price)
            )
            df_filtered = df_filtered[
                (df_filtered["Price_in_Lakhs"] >= price_range[0]) &
                (df_filtered["Price_in_Lakhs"] <= price_range[1])
            ]

        st.subheader("📄 Filtered Properties")
        st.write(f"Showing **{len(df_filtered)}** properties after filters.")
        st.dataframe(df_filtered.head(200))

        # Visual 1: Avg Price per SqFt by City
        if "City" in df_filtered.columns and "Price_per_SqFt" in df_filtered.columns:
            st.markdown("### 💸 Average Price per SqFt by City")
            city_pps = df_filtered.groupby("City")["Price_per_SqFt"].mean().sort_values(ascending=False).head(15)
            fig1, ax1 = plt.subplots(figsize=(10, 4))
            sns.barplot(x=city_pps.values, y=city_pps.index, ax=ax1)
            ax1.set_xlabel("Avg Price per SqFt")
            ax1.set_ylabel("City")
            st.pyplot(fig1)

        # Visual 2: Size vs Price
        if "Size_in_SqFt" in df_filtered.columns and "Price_in_Lakhs" in df_filtered.columns:
            st.markdown("### 📐 Size vs Price")
            sample_df = df_filtered.copy()
            if len(sample_df) > 500:
                sample_df = sample_df.sample(500, random_state=42)

            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.scatterplot(
                data=sample_df,
                x="Size_in_SqFt",
                y="Price_in_Lakhs",
                hue="City" if "City" in sample_df.columns else None,
                ax=ax2
            )
            ax2.set_xlabel("Size (SqFt)")
            ax2.set_ylabel("Price (Lakhs)")
            st.pyplot(fig2)

        # Visual 3: Heatmap City × Property Type
        if {"City", "Property_Type", "Price_per_SqFt"}.issubset(df_filtered.columns):
            st.markdown("### 🌍 Location-wise Heatmap (City × Property Type)")
            pivot = df_filtered.pivot_table(
                index="City",
                columns="Property_Type",
                values="Price_per_SqFt",
                aggfunc="mean"
            )
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            sns.heatmap(pivot, ax=ax3)
            ax3.set_xlabel("Property Type")
            ax3.set_ylabel("City")
            st.pyplot(fig3)

        st.markdown("---")
        st.info("These insights help investors compare locations, property types, and price trends for better decision making.")


if __name__ == "__main__":
    main()
