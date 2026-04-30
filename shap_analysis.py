import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_shap_analysis(data_dir: Path, out_dir: Path) -> None:
    scores_path = next(data_dir.glob("*_xh_scores.csv"), data_dir / "xh_scores.csv")

    if not scores_path.exists():
        raise FileNotFoundError(f"xh_scores.csv를 찾을 수 없습니다: {scores_path}")

    print(f"분석 중인 파일: {scores_path}")
    merged = pd.read_csv(scores_path)

    if merged.empty:
        raise ValueError("데이터가 비어 있습니다. 입력 데이터를 확인하세요.")

    # [V3.1] New Scoring Logic Features
    features_list = [
        "f_ball_speed", "f_ball_accel", "player_density", 
        "f_motion", "f_audio", "inv_dist_centroid_masked"
    ]
    
    # Check if columns exist
    for col in features_list:
        if col not in merged.columns:
            print(f"경고: {col} 컬럼이 없습니다. 0으로 채웁니다.")
            merged[col] = 0.0

    X = merged[features_list]
    y = merged["smoothed_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"RMSE: {rmse:.4f}")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    shap.summary_plot(shap_values, X_train, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary.png", dpi=200)
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_bar.png", dpi=200)
    plt.close()

    shap_df = pd.DataFrame(shap_values, columns=X.columns)
    shap_df.to_csv(out_dir / "shap_values.csv", index=False)

    print(f"SHAP 결과 저장 위치: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="XGBoost + SHAP 검증")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="highlights",
        help="detections.csv, xh_scores.csv가 있는 폴더",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="shap_outputs",
        help="SHAP 결과 저장 폴더",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    run_shap_analysis(data_dir, out_dir)


if __name__ == "__main__":
    main()
