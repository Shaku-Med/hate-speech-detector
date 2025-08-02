import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from model import HateSpeechDetector
import os
import json

class ModelEvaluator:
    def __init__(self, model_path, test_data_path=None):
        self.detector = HateSpeechDetector()
        self.detector.load_model(model_path)
        self.test_data_path = test_data_path
        self.class_names = ['Hate Speech', 'Offensive Language', 'Neither']
        
    def load_test_data(self):
        if self.test_data_path and os.path.exists(self.test_data_path):
            df = pd.read_csv(self.test_data_path)
            texts = df['tweet'].values
            labels = df['class'].values
            return texts, labels
        else:
            print("No test data provided or file not found")
            return None, None
    
    def evaluate_model(self, X_test, y_test):
        print("Evaluating model performance...")
        
        predictions = self.detector.predict(X_test)
        y_pred_classes = np.argmax(predictions, axis=1)
        
        accuracy = accuracy_score(y_test, y_pred_classes)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_classes, average='weighted')
        
        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"Weighted Precision: {precision:.4f}")
        print(f"Weighted Recall: {recall:.4f}")
        print(f"Weighted F1-Score: {f1:.4f}")
        
        report = classification_report(y_test, y_pred_classes, 
                                     target_names=self.class_names, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': predictions,
            'y_pred_classes': y_pred_classes,
            'classification_report': report
        }
    
    def plot_confusion_matrix(self, y_test, y_pred_classes, save_path='confusion_matrix.png'):
        cm = confusion_matrix(y_test, y_pred_classes)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm
    
    def plot_roc_curves(self, y_test, predictions, save_path='roc_curves.png'):
        plt.figure(figsize=(12, 8))
        
        for i, class_name in enumerate(self.class_names):
            y_binary = (y_test == i).astype(int)
            fpr, tpr, _ = roc_curve(y_binary, predictions[:, i])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for All Classes')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_precision_recall_curves(self, y_test, predictions, save_path='pr_curves.png'):
        plt.figure(figsize=(12, 8))
        
        for i, class_name in enumerate(self.class_names):
            y_binary = (y_test == i).astype(int)
            precision, recall, _ = precision_recall_curve(y_binary, predictions[:, i])
            pr_auc = auc(recall, precision)
            
            plt.plot(recall, precision, label=f'{class_name} (AUC = {pr_auc:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves for All Classes')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_class_performance(self, y_test, predictions, y_pred_classes):
        class_analysis = {}
        
        for i, class_name in enumerate(self.class_names):
            class_indices = np.where(y_test == i)[0]
            
            if len(class_indices) > 0:
                class_predictions = predictions[class_indices]
                class_confidences = np.max(class_predictions, axis=1)
                class_pred_classes = y_pred_classes[class_indices]
                
                class_accuracy = np.mean(class_pred_classes == i)
                class_precision = np.mean(class_pred_classes == i)
                
                class_analysis[class_name] = {
                    'sample_count': len(class_indices),
                    'accuracy': class_accuracy,
                    'avg_confidence': np.mean(class_confidences),
                    'confidence_std': np.std(class_confidences),
                    'min_confidence': np.min(class_confidences),
                    'max_confidence': np.max(class_confidences)
                }
        
        return class_analysis
    
    def plot_confidence_distributions(self, y_test, predictions, save_dir='confidence_plots'):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        for i, class_name in enumerate(self.class_names):
            class_indices = np.where(y_test == i)[0]
            
            if len(class_indices) > 0:
                class_predictions = predictions[class_indices]
                class_confidences = np.max(class_predictions, axis=1)
                
                plt.figure(figsize=(10, 6))
                plt.hist(class_confidences, bins=20, alpha=0.7, edgecolor='black')
                plt.title(f'Prediction Confidence Distribution - {class_name}')
                plt.xlabel('Confidence')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
                plt.savefig(f'{save_dir}/confidence_{class_name.lower().replace(" ", "_")}.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
    
    def generate_error_analysis(self, X_test, y_test, y_pred_classes, top_n=10):
        errors = []
        
        for i, (true_label, pred_label) in enumerate(zip(y_test, y_pred_classes)):
            if true_label != pred_label:
                errors.append({
                    'text': X_test[i],
                    'true_class': self.class_names[true_label],
                    'predicted_class': self.class_names[pred_label],
                    'confidence': np.max(self.detector.predict([X_test[i]])[0])
                })
        
        errors.sort(key=lambda x: x['confidence'], reverse=True)
        
        print(f"\nTop {min(top_n, len(errors))} High-Confidence Errors:")
        print("-" * 80)
        
        for i, error in enumerate(errors[:top_n]):
            print(f"{i+1}. Text: {error['text'][:100]}...")
            print(f"   True: {error['true_class']}, Predicted: {error['predicted_class']}")
            print(f"   Confidence: {error['confidence']:.4f}")
            print()
        
        return errors
    
    def save_evaluation_report(self, results, save_path='evaluation_report.json'):
        report = {
            'overall_metrics': {
                'accuracy': float(results['accuracy']),
                'precision': float(results['precision']),
                'recall': float(results['recall']),
                'f1_score': float(results['f1_score'])
            },
            'classification_report': results['classification_report'],
            'model_info': {
                'vocabulary_size': len(self.detector.vocabulary),
                'max_sequence_length': self.detector.max_sequence_length,
                'embedding_dim': self.detector.embedding_dim
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Evaluation report saved to {save_path}")
    
    def run_complete_evaluation(self, X_test=None, y_test=None):
        if X_test is None or y_test is None:
            X_test, y_test = self.load_test_data()
            if X_test is None:
                print("No test data available for evaluation")
                return
        
        print("Running complete model evaluation...")
        print("=" * 60)
        
        results = self.evaluate_model(X_test, y_test)
        
        print("\nGenerating visualizations...")
        self.plot_confusion_matrix(y_test, results['y_pred_classes'])
        self.plot_roc_curves(y_test, results['predictions'])
        self.plot_precision_recall_curves(y_test, results['predictions'])
        self.plot_confidence_distributions(y_test, results['predictions'])
        
        print("\nAnalyzing class-specific performance...")
        class_analysis = self.analyze_class_performance(y_test, results['predictions'], results['y_pred_classes'])
        
        for class_name, metrics in class_analysis.items():
            print(f"\n{class_name}:")
            print(f"  Sample count: {metrics['sample_count']}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Average confidence: {metrics['avg_confidence']:.4f}")
            print(f"  Confidence std: {metrics['confidence_std']:.4f}")
        
        print("\nGenerating error analysis...")
        errors = self.generate_error_analysis(X_test, y_test, results['y_pred_classes'])
        
        print(f"\nTotal errors: {len(errors)}")
        print(f"Error rate: {len(errors) / len(y_test):.4f}")
        
        self.save_evaluation_report(results)
        
        return results, class_analysis, errors

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Model Evaluation Script')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--test_data', type=str, help='Path to test data CSV file')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Output directory for results')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    evaluator = ModelEvaluator(args.model, args.test_data)
    
    if args.test_data:
        X_test, y_test = evaluator.load_test_data()
        if X_test is not None:
            evaluator.run_complete_evaluation(X_test, y_test)
    else:
        print("No test data provided. Please provide test data using --test_data argument.")

if __name__ == "__main__":
    main() 