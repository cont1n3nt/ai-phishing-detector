from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

def get_models():
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
        "XGBoost": XGBClassifier( n_estimators=200,tree_method='hist',n_jobs=-1,use_label_encoder=False,eval_metric='logloss',random_state=42 )
    }
    return models