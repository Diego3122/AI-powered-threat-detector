#!/usr/bin/env python
import json
import numpy as np
from typing import Dict, List, Any

try:
    import shap
    _HAVE_SHAP = True
except ImportError:
    _HAVE_SHAP = False


class TFIDFExplainer:
    def __init__(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer
        self.feature_names = vectorizer.get_feature_names_out()
        self.coefficients = model.coef_[0] if hasattr(model, 'coef_') else None
    
    def explain(self, text: str, top_k: int = 5) -> dict:
        try:
            X = self.vectorizer.transform([text])
            X_dense = X.toarray()[0]
            
            nonzero_indices = X_dense.nonzero()[0]
            nonzero_features = {}
            
            for idx in nonzero_indices:
                feature_name = self.feature_names[idx]
                tfidf_value = X_dense[idx]
                
                if self.coefficients is not None and idx < len(self.coefficients):
                    coef = self.coefficients[idx]
                    importance = abs(tfidf_value * coef)
                else:
                    importance = abs(tfidf_value)
                
                nonzero_features[feature_name] = {
                    "tfidf_value": tfidf_value,
                    "coefficient": coef if self.coefficients is not None else 0,
                    "importance": importance,
                }
            
            top_features_list = sorted(
                nonzero_features.items(),
                key=lambda x: abs(x[1]["importance"]),
                reverse=True
            )[:top_k]
            
            top_features = [
                {
                    "feature": name,
                    "importance_score": float(data["importance"]),
                    "contribution": "increases_threat" if data.get("coefficient", 0) > 0 else "decreases_threat",
                    "tfidf_value": float(data["tfidf_value"]),
                }
                for name, data in top_features_list
            ]
            
            return {
                "model": "tfidf_fallback",
                "top_features": top_features,
                "explanation_method": "Feature Importance (TF-IDF + Coefficients)",
            }
        except Exception as e:
            return {
                "model": "tfidf_fallback",
                "error": str(e),
                "top_features": [],
            }


class DistilBertExplainer:
    def __init__(self, model, tokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.explainer = None
        
        if _HAVE_SHAP:
            def predict_fn(texts):
                import torch
                scores = []
                for text in texts:
                    inputs = self.tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="pt")
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                        proba = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()
                        score = float(proba[1]) if len(proba) >= 2 else float(proba[0]) if len(proba) == 1 else 0.0
                        scores.append(score)
                
                return np.array(scores)
            
            self.predict_fn = predict_fn
    
    def explain(self, text: str, top_k: int = 5) -> Dict[str, Any]:
        if not _HAVE_SHAP or self.predict_fn is None:
            return {
                "model": "distilbert",
                "error": "SHAP not available",
                "top_words": [],
            }
        
        try:
            tokens = self.tokenizer.tokenize(text)
            token_scores = {}
            
            base_score = self.predict_fn([text])[0]
            
            # Estimate token impact by removing each token and rescoring the text.
            for token in set(tokens):
                modified_text = text.replace(token, "")
                modified_score = self.predict_fn([modified_text])[0]
                token_scores[token] = base_score - modified_score
            
            top_tokens = sorted(token_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
            
            top_words = [
                {
                    "token": token,
                    "impact_score": float(score),
                    "contribution": "increases_threat" if score > 0 else "decreases_threat",
                }
                for token, score in top_tokens
            ]
            
            return {
                "model": "distilbert",
                "top_words": top_words,
                "base_prediction": float(base_score),
            }
        except Exception as e:
            return {
                "model": "distilbert",
                "error": str(e),
                "top_words": [],
            }


def format_explanation(explanation: Dict[str, Any], model_type: str) -> Dict[str, Any]:
    if model_type == "tfidf_fallback":
        return {
            "model": "tfidf_fallback",
            "explanation_type": "SHAP Linear",
            "top_features": explanation.get("top_features", []),
            "base_value": explanation.get("base_value"),
            "error": explanation.get("error"),
        }
    elif model_type == "distilbert":
        return {
            "model": "distilbert",
            "explanation_type": "Token Impact",
            "top_words": explanation.get("top_words", []),
            "base_prediction": explanation.get("base_prediction"),
            "error": explanation.get("error"),
        }
    else:
        return {"error": f"Unknown model type: {model_type}"}
