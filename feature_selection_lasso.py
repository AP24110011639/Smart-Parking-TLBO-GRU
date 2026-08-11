import numpy as np
from sklearn.linear_model import LassoCV


def lasso_select_features(X_flat, y_flat, feature_names=None, min_features=2):
    model = LassoCV(cv=5, random_state=0, max_iter=5000).fit(X_flat, y_flat)

    coefs = model.coef_
    selected_idx = list(np.where(np.abs(coefs) > 1e-6)[0])

    if len(selected_idx) < min_features:
        selected_idx = list(np.argsort(-np.abs(coefs))[:min_features])

    if feature_names is not None:
        kept = [feature_names[i] for i in selected_idx]
        print(f"[LASSO] kept features: {kept}")

    return selected_idx, coefs