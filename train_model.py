"""GroupKFold 기반 하이라이트 이진 분류 학습.

변경 요점
- frame_id 에서 match_id(경기 단위)를 추출해 `GroupKFold(5)` 로 학습/검증 분할
- 같은 경기의 anchor 들이 train/test 에 섞이지 않도록 해 실제 일반화 성능 측정
- 폴드별 정확도/ROC-AUC/Precision/Recall 출력
- 최종 모델은 전체 데이터로 재학습해 저장
- 에러 분석 CSV는 전체 OOF(out-of-fold) 예측 기준으로 생성
"""
import os
import re
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FILE = os.path.join(BASE_DIR, 'ai_training_dataset.csv')
MODEL_OUT = os.path.join(BASE_DIR, 'highlight_model.xgb')
ERROR_OUT = os.path.join(BASE_DIR, 'error_analysis.csv')

# extract.py / make_dataset.py의 ML_FEATURES와 동기화
FEATURES = [
    "inv_dist_centroid_masked_max",
    "inv_dist_centroid_masked_mean",
    "f_ball_speed_max",
    "f_ball_speed_anchor",
    "f_ball_accel_std",
    "f_goalpost_visible_mean",
    "f_ball_dir_change_max",
    "f_ball_dir_change_mean",
    "f_possession_switches_mean",
    "f_sprint_count_max",
    "f_ball_to_goal_width_ratio_mean",
    "f_goal_bbox_width_norm_mean",
    "f_players_near_ball_max",
]

N_SPLITS = 5
RANDOM_STATE = 42


def extract_match_id(frame_id: str) -> str:
    """frame_id 어디에 있든 hex(uuid, 16자리 이상) 덩어리를 match_id로 추출.

    예시:
      - '7956cde2...16.mp4_40663'  -> '7956cde2...'
      - 'hard_neg_06584d67...ef_5000' -> '06584d67...ef'
    """
    if not isinstance(frame_id, str):
        return 'unknown'
    # 문자열 내 가장 긴 hex 토큰을 match_id 로 사용 (위치 무관)
    tokens = re.findall(r'[0-9a-fA-F]{16,}', frame_id)
    if tokens:
        return max(tokens, key=len)
    return frame_id.split('_', 1)[0]


def new_model() -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.08,
        max_depth=4,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        reg_alpha=0.1,
        tree_method='hist',
        eval_metric='logloss',
    )


def main():
    if not os.path.exists(DATASET_FILE):
        print(f"데이터셋이 없습니다: {DATASET_FILE} (먼저 make_dataset.py 를 실행하세요)")
        return

    df = pd.read_csv(DATASET_FILE)
    df = df.fillna(0)

    # 누락 피처 0 채움
    for col in FEATURES:
        if col not in df.columns:
            print(f"경고: 누락된 피처 {col} 를 0으로 채웁니다.")
            df[col] = 0.0

    # match_id 추출
    df['match_id'] = df['frame_id'].apply(extract_match_id)
    n_matches = df['match_id'].nunique()
    print(f"총 샘플: {len(df)} | 경기(match_id) 수: {n_matches}")
    print(df['match_id'].value_counts().head(8))

    X = df[FEATURES].values
    y = df['is_highlight'].astype(int).values
    groups = df['match_id'].values

    if n_matches < 2:
        print("경기가 1개뿐이라 GroupKFold를 사용할 수 없습니다. 일반 split으로 폴백합니다.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y,
        )
        model = new_model(); model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        if len(np.unique(y_test)) > 1:
            print(f"ROC-AUC : {roc_auc_score(y_test, y_prob):.4f}")
        model.save_model(MODEL_OUT)
        print(f"모델 저장: {MODEL_OUT}")
        return

    # 실제 가능한 split 수는 그룹 수를 초과할 수 없음
    n_splits = min(N_SPLITS, n_matches)
    if n_splits != N_SPLITS:
        print(f"경기 수가 적어 n_splits={n_splits}로 조정")

    gkf = GroupKFold(n_splits=n_splits)

    # OOF 예측 저장 (에러 분석 용)
    oof_pred = np.zeros(len(df), dtype=int)
    oof_prob = np.zeros(len(df), dtype=float)

    fold_metrics = []
    for fold_idx, (tr, te) in enumerate(gkf.split(X, y, groups=groups), start=1):
        test_matches = np.unique(groups[te])
        train_matches = np.unique(groups[tr])
        overlap = set(test_matches) & set(train_matches)
        assert not overlap, f"Fold {fold_idx}: match 누수 감지 {overlap}"

        # 학습 데이터 내 positive/negative 균형 확인
        n_pos_tr = int((y[tr] == 1).sum())
        n_neg_tr = int((y[tr] == 0).sum())

        model = new_model()
        # 불균형 시 scale_pos_weight
        if n_pos_tr > 0:
            model.set_params(scale_pos_weight=max(1.0, n_neg_tr / max(1, n_pos_tr)))
        model.fit(X[tr], y[tr])

        y_pred = model.predict(X[te])
        y_prob = model.predict_proba(X[te])[:, 1]
        oof_pred[te] = y_pred
        oof_prob[te] = y_prob

        acc = accuracy_score(y[te], y_pred)
        auc = roc_auc_score(y[te], y_prob) if len(np.unique(y[te])) > 1 else float('nan')
        pr = precision_score(y[te], y_pred, zero_division=0)
        rc = recall_score(y[te], y_pred, zero_division=0)
        f1 = f1_score(y[te], y_pred, zero_division=0)

        fold_metrics.append({
            'fold': fold_idx,
            'n_train': len(tr), 'n_test': len(te),
            'test_matches': len(test_matches),
            'accuracy': acc, 'roc_auc': auc,
            'precision': pr, 'recall': rc, 'f1': f1,
        })
        print(f"[Fold {fold_idx}] n_test={len(te):3d} (match {len(test_matches):2d}) | "
              f"Acc={acc:.3f} AUC={auc:.3f} P={pr:.3f} R={rc:.3f} F1={f1:.3f}")

    # 전체 요약
    fm_df = pd.DataFrame(fold_metrics)
    print("\n--- GroupKFold 교차검증 요약 ---")
    print(f"Accuracy: {fm_df['accuracy'].mean():.4f} ± {fm_df['accuracy'].std():.4f}")
    if not fm_df['roc_auc'].isna().all():
        print(f"ROC-AUC : {fm_df['roc_auc'].mean():.4f} ± {fm_df['roc_auc'].std():.4f}")
    print(f"Precision: {fm_df['precision'].mean():.4f} ± {fm_df['precision'].std():.4f}")
    print(f"Recall   : {fm_df['recall'].mean():.4f} ± {fm_df['recall'].std():.4f}")
    print(f"F1       : {fm_df['f1'].mean():.4f} ± {fm_df['f1'].std():.4f}")

    # OOF 기준 confusion matrix
    cm = confusion_matrix(y, oof_pred, labels=[0, 1])
    print("\nOOF Confusion Matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=['true_0', 'true_1'], columns=['pred_0', 'pred_1']))

    # --- OOF 기준 에러 분석 ---
    error_mask = oof_pred != y
    error_df = pd.DataFrame({
        'frame_id': df.loc[error_mask, 'frame_id'].values,
        'match_id': df.loc[error_mask, 'match_id'].values,
        'clip_name': df.loc[error_mask, 'clip_name'].values if 'clip_name' in df.columns else '',
        'anchor_source': df.loc[error_mask, 'anchor_source'].values if 'anchor_source' in df.columns else '',
        'sample_offset_sec': df.loc[error_mask, 'sample_offset_sec'].values if 'sample_offset_sec' in df.columns else 0.0,
        'true_label': y[error_mask],
        'predicted_label': oof_pred[error_mask],
        'predicted_prob': oof_prob[error_mask],
    })
    if not error_df.empty:
        error_df.sort_values('predicted_prob', inplace=True)
        error_df.to_csv(ERROR_OUT, index=False)
        print(f"\n에러 분석 저장: {ERROR_OUT} ({len(error_df)}행 / 전체 {len(df)})")
    else:
        print("\n모든 OOF 예측이 정확합니다.")

    # --- 최종 모델: 전체 데이터로 재학습 후 저장 ---
    n_pos = int((y == 1).sum()); n_neg = int((y == 0).sum())
    final_model = new_model()
    if n_pos > 0:
        final_model.set_params(scale_pos_weight=max(1.0, n_neg / max(1, n_pos)))
    final_model.fit(X, y)
    final_model.save_model(MODEL_OUT)
    print(f"\n최종 모델(전체 데이터 학습) 저장: {MODEL_OUT}")

    # 피처 중요도
    print("\n--- 피처 중요도 ---")
    importances = final_model.feature_importances_
    for name, val in sorted(zip(FEATURES, importances), key=lambda x: x[1], reverse=True):
        print(f"  [{name:35}] {val:.4f}")


if __name__ == '__main__':
    main()
