"""
Main Pipeline for Credit Card Fraud Detection Project
Run this file to execute the complete ML pipeline
"""

import argparse
import warnings
warnings.filterwarnings('ignore')

from src.data_preprocessing import DataPreprocessor
from src.model import FraudDetectionModel, ModelComparison
from src.evaluate import ModelEvaluator, compare_models


def main(args):
    """Main execution function"""
    
    print("\n" + "="*70)
    print("CREDIT CARD FRAUD DETECTION - ML PIPELINE")
    print("="*70)
    
    # ========================
    # STEP 1: DATA LOADING
    # ========================
    print("\n[STEP 1/5] Loading and exploring data...")
    preprocessor = DataPreprocessor(data_path=args.data_path)
    df = preprocessor.load_data()
    df = preprocessor.explore_data(df)
    
    # ========================
    # STEP 2: PREPROCESSING
    # ========================
    print("\n[STEP 2/5] Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocessor.preprocess_data(
        df, test_size=args.test_size
    )
    
    # Handle imbalance if requested
    if args.use_smote:
        X_train, y_train = preprocessor.handle_imbalance(method='smote')
    
    # ========================
    # STEP 3: MODEL TRAINING
    # ========================
    print("\n[STEP 3/5] Training models...")
    
    if args.compare_models:
        # Train and compare multiple models
        comparison = ModelComparison()
        
        model_types = ['logistic', 'random_forest', 'xgboost']
        for model_type in model_types:
            model = FraudDetectionModel(model_type=model_type)
            comparison.add_model(model_type, model)
        
        comparison.train_all(X_train, y_train)
        
        # Save all models
        if args.save_models:
            comparison.save_all_models()
    
    else:
        # Train single model
        model = FraudDetectionModel(model_type=args.model_type)
        model.train(X_train, y_train)
        
        if args.save_models:
            model.save_model()
    
    # ========================
    # STEP 4: EVALUATION
    # ========================
    print("\n[STEP 4/5] Evaluating models...")
    
    results_dict = {}
    
    if args.compare_models:
        # Evaluate all models
        predictions = comparison.get_predictions(X_test)
        
        for model_name, model_obj in comparison.models.items():
            evaluator = ModelEvaluator(model_name=model_name)
            y_pred = model_obj.predict(X_test)
            y_pred_proba = model_obj.predict_proba(X_test)
            
            results = evaluator.evaluate(y_test, y_pred, y_pred_proba)
            results_dict[model_name] = results
            
            # Generate plots
            evaluator.plot_confusion_matrix(y_test, y_pred)
            evaluator.plot_roc_curve(y_test, y_pred_proba)
            evaluator.plot_precision_recall_curve(y_test, y_pred_proba)
            
            # Feature importance (for tree-based models)
            feature_importance = model_obj.get_feature_importance(X_train.columns)
            if feature_importance:
                evaluator.plot_feature_importance(feature_importance)
        
        # Compare models
        compare_models(results_dict)
    
    else:
        # Evaluate single model
        evaluator = ModelEvaluator(model_name=args.model_type)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        results = evaluator.evaluate(y_test, y_pred, y_pred_proba)
        
        # Generate plots
        evaluator.plot_confusion_matrix(y_test, y_pred)
        evaluator.plot_roc_curve(y_test, y_pred_proba)
        evaluator.plot_precision_recall_curve(y_test, y_pred_proba)
        
        # Feature importance
        feature_importance = model.get_feature_importance(X_train.columns)
        if feature_importance:
            evaluator.plot_feature_importance(feature_importance)
    
    # ========================
    # STEP 5: SUMMARY
    # ========================
    print("\n[STEP 5/5] Pipeline completed!")
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"✓ Data loaded: {df.shape[0]:,} transactions")
    print(f"✓ Models trained: {len(results_dict) if args.compare_models else 1}")
    print(f"✓ Evaluation complete")
    print(f"✓ Results saved to 'results/' directory")
    print(f"✓ Models saved to 'models/' directory" if args.save_models else "")
    print("\n" + "="*70)
    print("Pipeline execution finished successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Credit Card Fraud Detection Pipeline')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, 
                       default='data/raw/creditcard.csv',
                       help='Path to the dataset')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='xgboost',
                       choices=['logistic', 'random_forest', 'xgboost', 'decision_tree'],
                       help='Type of model to train')
    parser.add_argument('--compare_models', action='store_true',
                       help='Train and compare multiple models')
    
    # Preprocessing arguments
    parser.add_argument('--use_smote', action='store_true',
                       help='Use SMOTE for handling imbalanced data')
    
    # Output arguments
    parser.add_argument('--save_models', action='store_true',
                       help='Save trained models')
    
    args = parser.parse_args()
    
    # Run pipeline
    main(args)


# Quick usage examples:
"""
# Train single XGBoost model:
python main.py --save_models

# Train with SMOTE:
python main.py --use_smote --save_models

# Compare multiple models:
python main.py --compare_models --save_models

# Train specific model:
python main.py --model_type random_forest --use_smote --save_models
"""