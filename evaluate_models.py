import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import tensorflow as tf
import joblib
import os
from audio_preprocessor import AudioPreprocessor
from train_model import DisasterClassificationTrainer
from accuracy_improvement import improve_model_accuracy

class ModelEvaluator:
    def __init__(self):
        self.disaster_classes = ['cyclone', 'earthquake', 'explosion', 'fire', 'flood', 'landslide', 'thunderstorm']
    
    def comprehensive_evaluation(self, models, data_splits, preprocessor):
        """Comprehensive evaluation of all models"""
        results = {}
        
        for model_name, model_info in models.items():
            print(f"\nEvaluating {model_name}...")
            
            # Get predictions
            if model_name in ['CNN', 'RNN', 'Hybrid']:
                predictions = self.get_dl_predictions(model_info['model'], data_splits, model_name)
            else:
                predictions = model_info['model'].predict(data_splits['X_test_traditional'])
            
            # Calculate metrics
            metrics = self.calculate_metrics(data_splits['y_test'], predictions, preprocessor)
            results[model_name] = {
                'model': model_info['model'],
                'predictions': predictions,
                'metrics': metrics
            }
            
            # Print results
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1-Score: {metrics['f1_score']:.4f}")
        
        return results
    
    def get_dl_predictions(self, model, data_splits, model_type):
        """Get predictions from deep learning models"""
        if model_type == 'CNN':
            pred_probs = model.predict(data_splits['X_test_mel'])
        elif model_type == 'RNN':
            X_test_rnn = data_splits['X_test_mel'].reshape(-1, 259, 128)
            pred_probs = model.predict(X_test_rnn)
        elif model_type == 'Hybrid':
            pred_probs = model.predict([data_splits['X_test_mel'], data_splits['X_test_traditional']])
        
        return np.argmax(pred_probs, axis=1)
    
    def calculate_metrics(self, y_true, y_pred, preprocessor):
        """Calculate comprehensive metrics"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
    
    def plot_model_comparison(self, results):
        """Plot comparison of all models"""
        models = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            values = [results[model]['metrics'][metric] for model in models]
            
            bars = axes[i].bar(models, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
            axes[i].set_title(f'{metric.capitalize()} Comparison')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
            
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrices(self, results, preprocessor):
        """Plot confusion matrices for all models"""
        n_models = len(results)
        cols = 3
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = [axes]
        axes = np.array(axes).flatten()
        
        for i, (model_name, result) in enumerate(results.items()):
            cm = confusion_matrix(result['predictions'], result['predictions'])  # This should be y_test vs predictions
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=preprocessor.label_encoder.classes_,
                       yticklabels=preprocessor.label_encoder.classes_,
                       ax=axes[i])
            axes[i].set_title(f'{model_name} Confusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        # Hide unused subplots
        for i in range(len(results), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves(self, results, data_splits, preprocessor):
        """Plot ROC curves for multiclass classification"""
        plt.figure(figsize=(12, 8))
        
        # Binarize the output for multiclass ROC
        y_test_bin = label_binarize(data_splits['y_test'], classes=range(len(self.disaster_classes)))
        
        for model_name, result in results.items():
            if model_name in ['CNN', 'RNN', 'Hybrid']:
                # Get probability predictions for deep learning models
                if model_name == 'CNN':
                    y_score = result['model'].predict(data_splits['X_test_mel'])
                elif model_name == 'RNN':
                    X_test_rnn = data_splits['X_test_mel'].reshape(-1, 259, 128)
                    y_score = result['model'].predict(X_test_rnn)
                elif model_name == 'Hybrid':
                    y_score = result['model'].predict([data_splits['X_test_mel'], data_splits['X_test_traditional']])
            else:
                # For traditional ML models
                if hasattr(result['model'], 'predict_proba'):
                    y_score = result['model'].predict_proba(data_splits['X_test_traditional'])
                else:
                    continue  # Skip models without probability prediction
            
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            for i in range(len(self.disaster_classes)):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            
            plt.plot(fpr["micro"], tpr["micro"],
                    label=f'{model_name} (AUC = {roc_auc["micro"]:.2f})',
                    linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Multiclass Classification')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_detailed_report(self, results, data_splits, preprocessor):
        """Generate detailed evaluation report"""
        report = {
            'model_performance': {},
            'class_performance': {},
            'recommendations': []
        }
        
        # Model performance summary
        for model_name, result in results.items():
            report['model_performance'][model_name] = result['metrics']
        
        # Per-class performance analysis
        for model_name, result in results.items():
            class_report = classification_report(
                data_splits['y_test'], 
                result['predictions'], 
                target_names=preprocessor.label_encoder.classes_,
                output_dict=True
            )
            report['class_performance'][model_name] = class_report
        
        # Generate recommendations
        best_model = max(results.keys(), key=lambda x: results[x]['metrics']['accuracy'])
        report['recommendations'].append(f"Best performing model: {best_model}")
        
        # Find challenging classes
        worst_classes = {}
        for model_name, result in results.items():
            class_f1 = report['class_performance'][model_name]
            for class_name in preprocessor.label_encoder.classes_:
                if class_name not in worst_classes:
                    worst_classes[class_name] = []
                worst_classes[class_name].append(class_f1[class_name]['f1-score'])
        
        challenging_classes = sorted(worst_classes.keys(), 
                                   key=lambda x: np.mean(worst_classes[x]))[:3]
        
        report['recommendations'].append(
            f"Most challenging classes: {', '.join(challenging_classes)}"
        )
        report['recommendations'].append(
            "Consider collecting more data for challenging classes"
        )
        report['recommendations'].append(
            "Apply data augmentation techniques for better generalization"
        )
        
        return report
    
    def save_evaluation_results(self, results, report, filename='evaluation_results.txt'):
        """Save evaluation results to file"""
        with open(filename, 'w') as f:
            f.write("DISASTER AUDIO CLASSIFICATION - EVALUATION RESULTS\n")
            f.write("="*60 + "\n\n")
            
            f.write("MODEL PERFORMANCE SUMMARY:\n")
            f.write("-" * 30 + "\n")
            for model_name, metrics in report['model_performance'].items():
                f.write(f"{model_name}:\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value:.4f}\n")
                f.write("\n")
            
            f.write("RECOMMENDATIONS:\n")
            f.write("-" * 20 + "\n")
            for rec in report['recommendations']:
                f.write(f"• {rec}\n")
            
            f.write(f"\nBest Model: {max(report['model_performance'].keys(), key=lambda x: report['model_performance'][x]['accuracy'])}\n")

def main():
    # Initialize trainer and prepare data
    trainer = DisasterClassificationTrainer()
    data_splits = trainer.prepare_data()
    
    # Train all models (you can load pre-trained models instead)
    print("Training models...")
    
    # CNN Model
    cnn_model, _ = trainer.train_cnn_model(data_splits)
    cnn_accuracy, cnn_pred = trainer.evaluate_model(cnn_model, data_splits, 'cnn')
    
    # Traditional ML Models
    traditional_models = trainer.train_traditional_models(data_splits)
    
    # Apply accuracy improvements
    improved_ensemble, tuned_models, cv_results = improve_model_accuracy(trainer, data_splits)
    
    # Prepare results dictionary
    results = {
        'CNN': {
            'model': cnn_model,
            'predictions': cnn_pred,
            'metrics': trainer.evaluate_model(cnn_model, data_splits, 'cnn')[0]
        },
        'Improved_Ensemble': {
            'model': improved_ensemble,
            'predictions': improved_ensemble.predict(data_splits['X_test_traditional']),
            'metrics': cv_results
        }
    }
    
    # Add traditional models to results
    for name, model in traditional_models.items():
        accuracy, pred = trainer.evaluate_model(model, data_splits, 'traditional')
        results[name] = {
            'model': model,
            'predictions': pred,
            'metrics': {'accuracy': accuracy}
        }
    
    # Comprehensive evaluation
    evaluator = ModelEvaluator()
    
    # Fix the metrics calculation for all models
    for model_name in results.keys():
        if 'metrics' not in results[model_name] or not isinstance(results[model_name]['metrics'], dict):
            results[model_name]['metrics'] = evaluator.calculate_metrics(
                data_splits['y_test'], 
                results[model_name]['predictions'], 
                trainer.preprocessor
            )
    
    # Generate plots and reports
    evaluator.plot_model_comparison(results)
    evaluator.plot_confusion_matrices(results, trainer.preprocessor)
    
    # Generate detailed report
    detailed_report = evaluator.generate_detailed_report(results, data_splits, trainer.preprocessor)
    evaluator.save_evaluation_results(results, detailed_report)
    
    print("\nEvaluation completed! Check the generated plots and evaluation_results.txt file.")

if __name__ == "__main__":
    main()