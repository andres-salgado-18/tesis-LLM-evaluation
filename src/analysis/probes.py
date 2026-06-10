import numpy as np
import pandas as pd
from src.utils.data_utils import create_eval_deploy_label, get_embedding_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.feature_extraction.text import TfidfVectorizer
def run_scenario_probe_final(df: pd.DataFrame, scenario_name: str) -> tuple:
    """
    Trains a linear probe to classify 'eval' vs 'deploy' context within a scenario.

    Args:
        df: DataFrame with embedding_response and scenario data.
        scenario_name: Name of the scenario to probe.

    Returns:
        tuple: (dict of metrics, pd.DataFrame of test predictions).
    """
    subset = df[df["scenario"] == scenario_name].copy()
    if subset.empty:
        return None, None

    y_raw = create_eval_deploy_label(subset)
    y = np.where(y_raw == "eval", 1, 0)

    if len(np.unique(y)) < 2:
        return None, None

    X = get_embedding_matrix(subset)

 
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, subset.index, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

 
    y_prob = model.predict_proba(X_test)[:, 1]
    
  
    fixed_threshold = 0.5
    final_preds = (y_prob >= fixed_threshold).astype(int)


    test_df = subset.loc[idx_test].copy()
    test_df["gt"] = y_test
    test_df["prediction"] = final_preds
    test_df["prob_eval"] = y_prob
    test_df["scenario"] = scenario_name

    metrics = {
        "scenario": scenario_name,
        "auc": roc_auc_score(y_test, y_prob),
        "threshold": fixed_threshold,
        "precision_eval": precision_score(y_test, final_preds, zero_division=0),
        "recall_eval": recall_score(y_test, final_preds, zero_division=0),
        "f1_eval": f1_score(y_test, final_preds, zero_division=0),
        "eval_in_test": int(sum(y_test)),
        "deploy_in_test": int(len(y_test) - sum(y_test)),
        "steering_vector": model.coef_[0]
    }

    return metrics, test_df




def run_probe_pipeline(df: pd.DataFrame, scenario_list: list = None) -> tuple:
    """
    Executes the probing pipeline across all specified scenarios.

    Args:
        df: Consolidated dataset.
        scenario_list: List of scenarios to process. If None, uses all.

    Returns:
        tuple: (summary_df with metrics, predictions_df with test results).
    """
    if scenario_list is None:
        scenario_list = df["scenario"].unique()
        
    results_list = []
    all_test_dfs = []


    for scenario_name in scenario_list:
     
        metrics, test_subset_df = run_scenario_probe_final(df, scenario_name)
        
        if metrics:
            results_list.append(metrics)
        if test_subset_df is not None:
            all_test_dfs.append(test_subset_df)

   
    summary_df = pd.DataFrame(results_list)
    predictions_df = pd.concat(all_test_dfs, ignore_index=True)
    
    return summary_df, predictions_df




def probe_held_out_scenario(df: pd.DataFrame) -> tuple:
    """
    Evaluates model generalization by training on all scenarios except one.

    Args:
        df: Dataset containing embeddings and scenarios.

    Returns:
        tuple: (pd.DataFrame of AUC metrics per held-out scenario, dict of steering vectors).
    """
    raw_results_list = []
    raw_steering_vectors = {}
    
    scenarios = df["scenario"].unique()
    
    for held_out_scenario in scenarios:
      
        train_subset = df[df["scenario"] != held_out_scenario].copy()
        test_subset = df[df["scenario"] == held_out_scenario].copy()

        y_train_raw = create_eval_deploy_label(train_subset)
        y_test_raw = create_eval_deploy_label(test_subset)
        
        encoder = LabelEncoder()
        y_train = encoder.fit_transform(y_train_raw)
        
        try:
            y_test = encoder.transform(y_test_raw)
        except ValueError:
            raw_results_list.append({
                "held_out_scenario": held_out_scenario,
                "auc": np.nan,
                "status": "Unseen Test Label"
            })
            continue

     
        X_train = get_embedding_matrix(train_subset)
        X_test = get_embedding_matrix(test_subset)

        
        model = LogisticRegression(max_iter=5000, random_state=42)
        model.fit(X_train, y_train)

      
        raw_steering_vectors[held_out_scenario] = model.coef_[0]

    
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)

        raw_results_list.append({
            "held_out_scenario": held_out_scenario,
            "auc": auc,
            "train_size": len(train_subset),
            "test_size": len(test_subset),
            "status": "Success"
        })

    return pd.DataFrame(raw_results_list), raw_steering_vectors


      
def run_tfidf_probe(df: pd.DataFrame, scenario_name: str) -> dict:
    """
    Trains a TF-IDF baseline probe to detect keyword-based classification.

    Args:
        df: Dataset containing response texts.
        scenario_name: Scenario identifier.

    Returns:
        dict: Performance metrics (AUC) and top features contributing to eval/deploy prediction.
    """
    subset = df[df["scenario"] == scenario_name].copy()

    if subset.empty:
        return None

    y_raw = create_eval_deploy_label(subset)

    y = np.where(y_raw == "eval", 1, 0)


    texts = subset["response"].astype(str).values

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        texts,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=None,
        max_features=10000,
        ngram_range=(1, 2)
    )

    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    model = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]

  
    auc = roc_auc_score(y_test, y_prob)


    feature_names = np.array(
        vectorizer.get_feature_names_out()
    )

    coefficients = model.coef_[0]

    top_eval_idx = np.argsort(coefficients)[-20:][::-1]

    top_eval_features = pd.DataFrame({
        "feature": feature_names[top_eval_idx],
        "weight": coefficients[top_eval_idx]
    })

 
    top_deploy_idx = np.argsort(coefficients)[:20]

    top_deploy_features = pd.DataFrame({
        "feature": feature_names[top_deploy_idx],
        "weight": coefficients[top_deploy_idx]
    })

    return {
        "scenario": scenario_name,
        "tfidf_auc": round(auc, 4),
        "gt": y_test,        
        "prob_eval": y_prob,  
        "top_eval_features": top_eval_features,
        "top_deploy_features": top_deploy_features
    }



def run_tfidf_pipeline(df: pd.DataFrame, scenario_list: list = None) -> tuple:
    """
    Executes the TF-IDF probing baseline across requested scenarios.

    Args:
        df: Consolidated dataset.
        scenario_list: Scenarios to process. If None, uses all.

    Returns:
        tuple: (pd.DataFrame summary of AUCs, list of detailed audit results).
    """
  
    if scenario_list is None:
        scenario_list = df["scenario"].unique()
        
    results_list = []
    

    
    for scenario_name in scenario_list:
       
        result = run_tfidf_probe(df, scenario_name)
        
        if result:
            results_list.append(result)
            

    summary_df = pd.DataFrame([
        {"scenario": r["scenario"], "tfidf_auc": r["tfidf_auc"]} 
        for r in results_list
    ])
    
    return summary_df, results_list
def print_tfidf_report(audit_details_list: list) -> None:
    """
    Prints top lexical features identified by the TF-IDF probes for audit.

    Args:
        audit_details_list: List of results returned by run_tfidf_pipeline.
    """
   
    for r in audit_details_list:
        print("\n" + "=" * 80)
        print(f"SCENARIO: {r['scenario']}")
        print("=" * 80)
        print("TOP EVAL FEATURES:", r["top_eval_features"])
        print("TOP DEPLOY FEATURES:", r["top_deploy_features"])