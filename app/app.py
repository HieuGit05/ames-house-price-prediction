"""Streamlit application for the Ames House Price Prediction project.

Run from the repository root:
    streamlit run app/app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "house_price_model.pkl"
DATA_PATH = ROOT_DIR / "data" / "train.csv"
METRICS_PATH = ROOT_DIR / "models" / "metrics.json"

st.set_page_config(
    page_title="Ames House Price Prediction",
    page_icon="🏠",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_model_package() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy model tại: {MODEL_PATH}")

    package = joblib.load(MODEL_PATH)
    if not isinstance(package, dict) or "model" not in package:
        raise ValueError("File model không đúng định dạng package mong đợi.")
    return package


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu tại: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def safe_mode(series: pd.Series, fallback: Any = "None") -> Any:
    values = series.dropna().mode()
    return values.iloc[0] if not values.empty else fallback


@st.cache_data(show_spinner=False)
def create_default_raw_row() -> dict[str, Any]:
    """Create one complete raw input row from train-set medians and modes."""
    df = load_data().drop(columns=["SalePrice", "Id"], errors="ignore")
    defaults: dict[str, Any] = {}

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            value = df[column].median()
            defaults[column] = 0.0 if pd.isna(value) else value
        else:
            defaults[column] = safe_mode(df[column])
    return defaults


def apply_feature_engineering(
    raw_row: pd.DataFrame,
    feature_config: dict[str, Any],
) -> pd.DataFrame:
    """Apply exactly the deterministic transformations used by the notebook."""
    result = raw_row.copy()

    result = result.drop(
        columns=feature_config.get("drop_cols", []),
        errors="ignore",
    )

    for column in feature_config.get("none_cols", []):
        if column in result.columns:
            result[column] = result[column].fillna("None")

    for column in feature_config.get("zero_cols", []):
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column], errors="coerce"
            ).fillna(0)

    quality_map = feature_config.get("quality_map", {})
    for column in feature_config.get("quality_cols", []):
        if column in result.columns:
            result[column] = (
                result[column]
                .fillna("None")
                .map(quality_map)
                .fillna(0)
                .astype(float)
            )

    result["HouseAge"] = (
        pd.to_numeric(result["YrSold"], errors="coerce")
        - pd.to_numeric(result["YearBuilt"], errors="coerce")
    ).clip(lower=0)

    result["RemodAge"] = (
        pd.to_numeric(result["YrSold"], errors="coerce")
        - pd.to_numeric(result["YearRemodAdd"], errors="coerce")
    ).clip(lower=0)

    result["TotalSF"] = (
        pd.to_numeric(result["TotalBsmtSF"], errors="coerce").fillna(0)
        + pd.to_numeric(result["1stFlrSF"], errors="coerce").fillna(0)
        + pd.to_numeric(result["2ndFlrSF"], errors="coerce").fillna(0)
    )

    result["TotalBath"] = (
        pd.to_numeric(result["FullBath"], errors="coerce").fillna(0)
        + 0.5 * pd.to_numeric(result["HalfBath"], errors="coerce").fillna(0)
        + pd.to_numeric(result["BsmtFullBath"], errors="coerce").fillna(0)
        + 0.5 * pd.to_numeric(result["BsmtHalfBath"], errors="coerce").fillna(0)
    )

    porch_columns = [
        "WoodDeckSF",
        "OpenPorchSF",
        "EnclosedPorch",
        "3SsnPorch",
        "ScreenPorch",
    ]
    result["TotalPorchSF"] = sum(
        pd.to_numeric(result[column], errors="coerce").fillna(0)
        for column in porch_columns
    )

    result["HasGarage"] = (
        pd.to_numeric(result["GarageArea"], errors="coerce").fillna(0) > 0
    ).astype(int)
    result["HasBsmt"] = (
        pd.to_numeric(result["TotalBsmtSF"], errors="coerce").fillna(0) > 0
    ).astype(int)
    result["HasFireplace"] = (
        pd.to_numeric(result["Fireplaces"], errors="coerce").fillna(0) > 0
    ).astype(int)

    return result


def build_model_input(
    user_values: dict[str, Any],
    package: dict[str, Any],
) -> pd.DataFrame:
    defaults = create_default_raw_row()
    defaults.update(user_values)

    raw_columns = package.get("raw_input_columns", list(defaults.keys()))
    raw_row = pd.DataFrame(
        [{column: defaults.get(column, np.nan) for column in raw_columns}]
    )

    engineered = apply_feature_engineering(
        raw_row,
        package.get("feature_config", {}),
    )

    required_columns = package.get("engineered_input_columns")
    if required_columns:
        missing = [c for c in required_columns if c not in engineered.columns]
        if missing:
            raise ValueError(f"Thiếu feature sau engineering: {missing}")
        engineered = engineered[required_columns]

    return engineered


def predict_price(
    user_values: dict[str, Any],
    package: dict[str, Any],
) -> float:
    model_input = build_model_input(user_values, package)
    prediction = float(package["model"].predict(model_input)[0])
    if not np.isfinite(prediction) or prediction <= 0:
        raise ValueError("Model trả về giá trị dự đoán không hợp lệ.")
    return prediction


def main() -> None:
    st.title("🏠 Ames House Price Prediction")
    st.caption(
        "Ứng dụng minh họa mô hình Machine Learning dự đoán giá nhà tại Ames, Iowa."
    )

    try:
        package = load_model_package()
        data = load_data()
    except Exception as exc:
        st.error(f"Không thể tải tài nguyên của ứng dụng: {exc}")
        st.stop()

    metrics = package.get("test_metrics", {})
    model_name = package.get("model_name", "Unknown")

    with st.sidebar:
        st.header("Thông tin mô hình")
        st.write(f"**Model:** {model_name} Regression")
        if metrics:
            st.metric("Test RMSE", f"${metrics.get('rmse', 0):,.0f}")
            st.metric("Test MAE", f"${metrics.get('mae', 0):,.0f}")
            st.metric("Test R²", f"{metrics.get('r2', 0):.4f}")
        st.info(
            "Các feature không xuất hiện trên form được điền bằng median hoặc mode "
            "từ tập huấn luyện. Đây là demo học tập, không phải công cụ định giá thực tế."
        )

    neighborhoods = sorted(data["Neighborhood"].dropna().unique().tolist())
    quality_options = ["Po", "Fa", "TA", "Gd", "Ex"]
    basement_quality_options = ["None", *quality_options]

    with st.form("prediction_form"):
        st.subheader("Nhập thông tin căn nhà")

        col1, col2, col3 = st.columns(3)
        with col1:
            neighborhood = st.selectbox(
                "Khu vực",
                neighborhoods,
                index=neighborhoods.index("CollgCr") if "CollgCr" in neighborhoods else 0,
            )
            overall_qual = st.slider("Chất lượng tổng thể", 1, 10, 6)
            year_built = st.number_input(
                "Năm xây dựng", min_value=1870, max_value=2010, value=2000
            )
            lot_area = st.number_input(
                "Diện tích đất (sq ft)", min_value=1000, max_value=200000, value=9000, step=100
            )

        with col2:
            gr_liv_area = st.number_input(
                "Diện tích sinh hoạt (sq ft)", min_value=300, max_value=6000, value=1500, step=50
            )
            first_floor = st.number_input(
                "Diện tích tầng 1 (sq ft)", min_value=300, max_value=5000, value=1000, step=50
            )
            second_floor = st.number_input(
                "Diện tích tầng 2 (sq ft)", min_value=0, max_value=2500, value=500, step=50
            )
            total_bsmt = st.number_input(
                "Diện tích tầng hầm (sq ft)", min_value=0, max_value=4000, value=900, step=50
            )

        with col3:
            garage_cars = st.number_input("Sức chứa gara", 0, 5, 2)
            garage_area = st.number_input(
                "Diện tích gara (sq ft)", min_value=0, max_value=1600, value=480, step=20
            )
            full_bath = st.number_input("Phòng tắm đầy đủ", 0, 5, 2)
            bedroom = st.number_input("Phòng ngủ", 0, 8, 3)
            fireplaces = st.number_input("Số lò sưởi", 0, 4, 1)

        st.markdown("#### Chất lượng chi tiết")
        q1, q2, q3, q4 = st.columns(4)
        with q1:
            exter_qual = st.selectbox("Ngoại thất", quality_options, index=2)
        with q2:
            kitchen_qual = st.selectbox("Nhà bếp", quality_options, index=2)
        with q3:
            heating_qc = st.selectbox("Hệ thống sưởi", quality_options, index=3)
        with q4:
            bsmt_qual = st.selectbox("Tầng hầm", basement_quality_options, index=3)

        submitted = st.form_submit_button(
            "Dự đoán giá",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        user_values = {
            "Neighborhood": neighborhood,
            "OverallQual": overall_qual,
            "YearBuilt": year_built,
            "YearRemodAdd": year_built,
            "YrSold": 2010,
            "LotArea": lot_area,
            "GrLivArea": gr_liv_area,
            "1stFlrSF": first_floor,
            "2ndFlrSF": second_floor,
            "TotalBsmtSF": total_bsmt,
            "GarageCars": garage_cars,
            "GarageArea": garage_area,
            "FullBath": full_bath,
            "BedroomAbvGr": bedroom,
            "Fireplaces": fireplaces,
            "ExterQual": exter_qual,
            "KitchenQual": kitchen_qual,
            "HeatingQC": heating_qc,
            "BsmtQual": bsmt_qual,
        }

        try:
            with st.spinner("Đang dự đoán..."):
                price = predict_price(user_values, package)
        except Exception as exc:
            st.error(f"Dự đoán thất bại: {exc}")
            st.stop()

        rmse = float(metrics.get("rmse", 0))
        lower = max(0.0, price - rmse)
        upper = price + rmse
        neighborhood_mean = float(
            data.loc[data["Neighborhood"] == neighborhood, "SalePrice"].mean()
        )

        st.success("Dự đoán thành công")
        c1, c2, c3 = st.columns(3)
        c1.metric("Giá dự đoán", f"${price:,.0f}")
        c2.metric("Trung bình khu vực", f"${neighborhood_mean:,.0f}")
        c3.metric("Chênh lệch", f"${price - neighborhood_mean:+,.0f}")

        if rmse > 0:
            st.write(
                f"Khoảng tham khảo theo test RMSE: **${lower:,.0f} – ${upper:,.0f}**."
            )

        comparison = pd.DataFrame(
            {
                "Mức giá": ["Dự đoán", f"Trung bình {neighborhood}"],
                "USD": [price, neighborhood_mean],
            }
        ).set_index("Mức giá")
        st.bar_chart(comparison)

    with st.expander("Giới hạn của ứng dụng"):
        st.markdown(
            """
- Model được huấn luyện trên dữ liệu Ames Housing tại Iowa, Hoa Kỳ.
- Form chỉ cho nhập một nhóm feature quan trọng; các feature còn lại dùng giá trị đại diện từ tập train.
- Giá dự đoán chỉ phục vụ mục đích học tập và minh họa quy trình Machine Learning.
            """
        )


if __name__ == "__main__":
    main()
