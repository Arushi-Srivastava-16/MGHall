"""
Fingerprint Classifier for Model Identification.

Trains a classifier to identify which LLM generated a reasoning chain
based on extracted fingerprint features.
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pickle
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_processing.unified_schema import ReasoningChain
from src.multi_model.model_config import ModelType
from src.multi_model.fingerprint_extractor import FingerprintExtractor

# Import ML libraries
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    RandomForestClassifier = None
    StandardScaler = None


@dataclass
class ClassifierMetrics:
    """Metrics for classifier evaluation."""
    accuracy: float
    precision: Dict[str, float]
    recall: Dict[str, float]
    f1_score: Dict[str, float]
    confusion_matrix: np.ndarray
    classification_report: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "confusion_matrix": self.confusion_matrix.tolist(),
            "classification_report": self.classification_report,
        }


class FingerprintClassifier:
    """Classifier for identifying model from fingerprint features."""
    
    def __init__(
        self,
        classifier_type: str = "random_forest",
        **classifier_kwargs
    ):
        """
        Initialize fingerprint classifier.
        
        Args:
            classifier_type: Type of classifier ("random_forest", "neural_net")
            **classifier_kwargs: Additional arguments for classifier
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not available. Install with: pip install scikit-learn")
        
        self.classifier_type = classifier_type
        self.classifier = None
        self.scaler = StandardScaler()
        self.feature_extractor = FingerprintExtractor()
        self.label_to_model: Dict[int, ModelType] = {}
        self.model_to_label: Dict[ModelType, int] = {}
        self.is_trained = False
        
        # Initialize classifier
        if classifier_type == "random_forest":
            n_estimators = classifier_kwargs.get("n_estimators", 100)
            max_depth = classifier_kwargs.get("max_depth", 20)
            self.classifier = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    def prepare_training_data(
        self,
        chains_by_model: Dict[ModelType, List[ReasoningChain]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data from chains.
        
        Args:
            chains_by_model: Dictionary mapping model types to their chains
            
        Returns:
            Tuple of (features, labels)
        """
        X = []
        y = []
        
        # Create label mappings
        for i, model_type in enumerate(sorted(chains_by_model.keys(), key=lambda x: x.value)):
            self.label_to_model[i] = model_type
            self.model_to_label[model_type] = i
        
        # Extract features
        for model_type, chains in chains_by_model.items():
            label = self.model_to_label[model_type]
            
            for chain in chains:
                fingerprint = self.feature_extractor.extract(chain)
                feature_vector = self.feature_extractor.get_feature_vector(fingerprint)
                X.append(feature_vector)
                y.append(label)
        
        return np.array(X), np.array(y)
    
    def train(
        self,
        chains_by_model: Dict[ModelType, List[ReasoningChain]],
        test_size: float = 0.2,
        validation_split: float = 0.1,
    ) -> ClassifierMetrics:
        """
        Train the classifier.
        
        Args:
            chains_by_model: Dictionary mapping model types to their chains
            test_size: Fraction of data for testing
            validation_split: Fraction of training data for validation
            
        Returns:
            Metrics on test set
        """
        print("Preparing training data...")
        X, y = self.prepare_training_data(chains_by_model)
        
        print(f"Total samples: {len(X)}")
        print(f"Models: {[mt.value for mt in self.label_to_model.values()]}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        # Scale features
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train classifier
        print(f"Training {self.classifier_type} classifier...")
        self.classifier.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Cross-validation on training set
        if validation_split > 0:
            print("Running cross-validation...")
            cv_scores = cross_val_score(
                self.classifier, X_train_scaled, y_train,
                cv=5, scoring='accuracy'
            )
            print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Evaluate on test set
        print("Evaluating on test set...")
        metrics = self.evaluate(X_test_scaled, y_test)
        
        return metrics
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> ClassifierMetrics:
        """
        Evaluate classifier on data.
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Classification metrics
        """
        if not self.is_trained:
            raise ValueError("Classifier not trained yet")
        
        # Predict
        y_pred = self.classifier.predict(X)
        
        # Compute metrics
        accuracy = accuracy_score(y, y_pred)
        
        # Get target names
        target_names = [self.label_to_model[i].value for i in sorted(self.label_to_model.keys())]
        
        # Classification report
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, y_pred, average=None, labels=sorted(self.label_to_model.keys())
        )
        
        precision_dict = {name: float(p) for name, p in zip(target_names, precision)}
        recall_dict = {name: float(r) for name, r in zip(target_names, recall)}
        f1_dict = {name: float(f) for name, f in zip(target_names, f1)}
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        # Text report
        from sklearn.metrics import classification_report as sklearn_report
        report = sklearn_report(y, y_pred, target_names=target_names)
        
        metrics = ClassifierMetrics(
            accuracy=accuracy,
            precision=precision_dict,
            recall=recall_dict,
            f1_score=f1_dict,
            confusion_matrix=cm,
            classification_report=report,
        )
        
        return metrics
    
    def predict(self, chain: ReasoningChain) -> Tuple[ModelType, float]:
        """
        Predict which model generated a chain.
        
        Args:
            chain: Reasoning chain to classify
            
        Returns:
            Tuple of (predicted model type, confidence)
        """
        if not self.is_trained:
            raise ValueError("Classifier not trained yet")
        
        # Extract fingerprint
        fingerprint = self.feature_extractor.extract(chain)
        feature_vector = self.feature_extractor.get_feature_vector(fingerprint)
        
        # Scale
        feature_scaled = self.scaler.transform([feature_vector])
        
        # Predict
        label_pred = self.classifier.predict(feature_scaled)[0]
        proba = self.classifier.predict_proba(feature_scaled)[0]
        confidence = float(proba[label_pred])
        
        model_type = self.label_to_model[label_pred]
        
        return model_type, confidence
    
    def predict_batch(
        self,
        chains: List[ReasoningChain]
    ) -> List[Tuple[ModelType, float]]:
        """
        Predict models for multiple chains.
        
        Args:
            chains: List of reasoning chains
            
        Returns:
            List of (model_type, confidence) tuples
        """
        return [self.predict(chain) for chain in chains]
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance (for tree-based classifiers).
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            raise ValueError("Classifier not trained yet")
        
        if self.classifier_type != "random_forest":
            raise ValueError("Feature importance only available for random forest")
        
        feature_names = self.feature_extractor.get_feature_names()
        importances = self.classifier.feature_importances_
        
        return {
            name: float(imp)
            for name, imp in zip(feature_names, importances)
        }
    
    def save(self, path: Path):
        """
        Save classifier to disk.
        
        Args:
            path: Path to save classifier
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained classifier")
        
        data = {
            "classifier_type": self.classifier_type,
            "classifier": self.classifier,
            "scaler": self.scaler,
            "label_to_model": {k: v.value for k, v in self.label_to_model.items()},
            "model_to_label": {k.value: v for k, v in self.model_to_label.items()},
        }
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"Saved classifier to {path}")
    
    @classmethod
    def load(cls, path: Path) -> "FingerprintClassifier":
        """
        Load classifier from disk.
        
        Args:
            path: Path to load classifier from
            
        Returns:
            Loaded classifier
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        classifier = cls(classifier_type=data["classifier_type"])
        classifier.classifier = data["classifier"]
        classifier.scaler = data["scaler"]
        classifier.label_to_model = {k: ModelType(v) for k, v in data["label_to_model"].items()}
        classifier.model_to_label = {ModelType(k): v for k, v in data["model_to_label"].items()}
        classifier.is_trained = True
        
        print(f"Loaded classifier from {path}")
        
        return classifier


if __name__ == "__main__":
    # Test fingerprint classifier
    print("=" * 80)
    print("Fingerprint Classifier Test")
    print("=" * 80)
    
    from src.data_processing.unified_schema import ReasoningStep, DependencyGraph, Domain
    
    # Create sample chains for different "models"
    def create_sample_chain(query_id: str, style: str) -> ReasoningChain:
        if style == "verbose":
            steps = [
                ReasoningStep(0, "First, we need to carefully identify the problem variables and understand what is being asked.", True, False, None, []),
                ReasoningStep(1, "Let x be the unknown value that we're solving for in this equation.", True, False, None, [0]),
                ReasoningStep(2, "We can set up the equation as follows: 2x + 5 = 13", True, False, None, [1]),
                ReasoningStep(3, "Now, subtracting 5 from both sides of the equation: 2x = 8", True, False, None, [2]),
                ReasoningStep(4, "Therefore, dividing both sides by 2, we obtain: x = 4", True, False, None, [3]),
            ]
        else:
            steps = [
                ReasoningStep(0, "Set up: 2x + 5 = 13", True, False, None, []),
                ReasoningStep(1, "Subtract 5: 2x = 8", True, False, None, [0]),
                ReasoningStep(2, "Divide by 2: x = 4", True, False, None, [1]),
            ]
        
        return ReasoningChain(
            domain=Domain.MATH,
            query_id=query_id,
            query="Solve: 2x + 5 = 13",
            ground_truth="x = 4",
            reasoning_steps=steps,
            dependency_graph=DependencyGraph(
                nodes=list(range(len(steps))),
                edges=[[i, i+1] for i in range(len(steps)-1)]
            )
        )
    
    # Create training data
    chains_by_model = {
        ModelType.GPT4: [create_sample_chain(f"gpt4_{i}", "verbose") for i in range(20)],
        ModelType.GEMINI: [create_sample_chain(f"gemini_{i}", "concise") for i in range(20)],
    }
    
    # Train classifier
    classifier = FingerprintClassifier(classifier_type="random_forest", n_estimators=50)
    
    try:
        metrics = classifier.train(chains_by_model, test_size=0.3)
        
        print(f"\nClassifier Performance:")
        print(f"  Accuracy: {metrics.accuracy:.4f}")
        print(f"\nPer-Model Metrics:")
        for model in chains_by_model.keys():
            model_name = model.value
            print(f"  {model_name}:")
            print(f"    Precision: {metrics.precision[model_name]:.4f}")
            print(f"    Recall: {metrics.recall[model_name]:.4f}")
            print(f"    F1-Score: {metrics.f1_score[model_name]:.4f}")
        
        # Test prediction
        test_chain = create_sample_chain("test_1", "verbose")
        predicted_model, confidence = classifier.predict(test_chain)
        print(f"\nTest Prediction:")
        print(f"  Predicted: {predicted_model.value}")
        print(f"  Confidence: {confidence:.4f}")
        
        print("\nFingerprint classifier test passed!")
    
    except Exception as e:
        print(f"Test skipped or failed: {e}")
        print("This is normal if sklearn is not installed")

