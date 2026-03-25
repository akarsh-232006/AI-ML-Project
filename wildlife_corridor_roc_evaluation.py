"""
AI-Enhanced Wildlife Corridor Identification and Prediction
Project Task: Evaluate models with ROC curve

Student: [Akarsh sahu]
Course: CSE-UCS-1002 - Introduction to AI-ML
Course Coordinator: Dr. Saurabh Shanu

This project evaluates different machine learning models for wildlife corridor identification
using ROC (Receiver Operating Characteristic) curves and related metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class WildlifeCorridorROCEvaluator:
    """
    A comprehensive class for evaluating wildlife corridor prediction models using ROC analysis.
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        
    def generate_wildlife_data(self, n_samples=2000, random_state=42):
        """
        Generate synthetic wildlife corridor data with realistic features.
        
        Features represent:
        - Vegetation density
        - Water source proximity
        - Human disturbance level
        - Elevation gradient
        - Habitat connectivity index
        - Species richness
        - Forest cover percentage
        - Road density (negative impact)
        - Protected area proximity
        - Climate suitability index
        """
        print("Generating synthetic wildlife corridor dataset...")
        
        # Generate base features
        X, y = make_classification(
            n_samples=n_samples,
            n_features=10,
            n_informative=8,
            n_redundant=2,
            n_clusters_per_class=2,
            class_sep=0.8,
            random_state=random_state
        )
        
        # Create meaningful feature names
        feature_names = [
            'vegetation_density',
            'water_proximity',
            'human_disturbance',
            'elevation_gradient',
            'habitat_connectivity',
            'species_richness',
            'forest_cover_pct',
            'road_density',
            'protected_area_proximity',
            'climate_suitability'
        ]
        
        # Convert to DataFrame for better handling
        df = pd.DataFrame(X, columns=feature_names)
        
        # Normalize features to realistic ranges
        df['vegetation_density'] = (df['vegetation_density'] - df['vegetation_density'].min()) / (df['vegetation_density'].max() - df['vegetation_density'].min()) * 100
        df['water_proximity'] = np.abs(df['water_proximity']) * 10  # Distance in km
        df['human_disturbance'] = (df['human_disturbance'] - df['human_disturbance'].min()) / (df['human_disturbance'].max() - df['human_disturbance'].min()) * 10
        df['elevation_gradient'] = np.abs(df['elevation_gradient']) * 500  # Elevation in meters
        df['habitat_connectivity'] = (df['habitat_connectivity'] - df['habitat_connectivity'].min()) / (df['habitat_connectivity'].max() - df['habitat_connectivity'].min())
        df['species_richness'] = np.abs(df['species_richness']) * 50
        df['forest_cover_pct'] = (df['forest_cover_pct'] - df['forest_cover_pct'].min()) / (df['forest_cover_pct'].max() - df['forest_cover_pct'].min()) * 100
        df['road_density'] = np.abs(df['road_density']) * 5  # Roads per sq km
        df['protected_area_proximity'] = np.abs(df['protected_area_proximity']) * 20  # Distance in km
        df['climate_suitability'] = (df['climate_suitability'] - df['climate_suitability'].min()) / (df['climate_suitability'].max() - df['climate_suitability'].min())
        
        print(f"Dataset created with {n_samples} samples and {len(feature_names)} features")
        print(f"Target distribution: {np.bincount(y)}")
        
        return df, y, feature_names
    
    def prepare_models(self):
        """
        Initialize different classification models for comparison.
        """
        print("Initializing machine learning models...")
        
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ),
            'Support Vector Machine': SVC(
                kernel='rbf',
                probability=True,
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42,
                max_iter=1000
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                random_state=42
            ),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42
            )
        }
        
        print(f"Initialized {len(self.models)} models for evaluation")
    
    def train_and_evaluate_models(self, X_train, X_test, y_train, y_test):
        """
        Train all models and collect predictions for ROC analysis.
        """
        print("Training and evaluating models...")
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Get predictions and probabilities
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            # Store results
            self.results[name] = {
                'model': model,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds,
                'auc': roc_auc,
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            print(f"{name} - AUC: {roc_auc:.4f}")
    
    def plot_roc_curves(self, save_path=None):
        """
        Create comprehensive ROC curve visualization.
        """
        plt.figure(figsize=(12, 8))
        
        # Plot ROC curves for all models
        for name, result in self.results.items():
            plt.plot(
                result['fpr'],
                result['tpr'],
                linewidth=2,
                label=f"{name} (AUC = {result['auc']:.4f})"
            )
        
        # Plot diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier (AUC = 0.5000)')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
        plt.title('ROC Curves Comparison for Wildlife Corridor Prediction Models', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_auc_comparison(self, save_path=None):
        """
        Create bar chart comparing AUC scores.
        """
        model_names = list(self.results.keys())
        auc_scores = [self.results[name]['auc'] for name in model_names]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, auc_scores, color=sns.color_palette("husl", len(model_names)))
        
        # Add value labels on bars
        for bar, score in zip(bars, auc_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.ylim([0, 1.1])
        plt.ylabel('AUC Score', fontsize=12)
        plt.xlabel('Models', fontsize=12)
        plt.title('AUC Score Comparison for Wildlife Corridor Prediction Models', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrices(self, y_test, save_path=None):
        """
        Create confusion matrices for all models.
        """
        n_models = len(self.results)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, (name, result) in enumerate(self.results.items()):
            cm = confusion_matrix(y_test, result['y_pred'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{name}\nAUC: {result["auc"]:.4f}', fontweight='bold')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        # Hide the last subplot if we have 5 models
        if n_models < 6:
            axes[-1].set_visible(False)
        
        plt.suptitle('Confusion Matrices for Wildlife Corridor Prediction Models', fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        plt.show()
    
    def generate_detailed_report(self, feature_names, save_path=None):
        """
        Generate a comprehensive evaluation report.
        """
        report = []
        report.append("="*80)
        report.append("WILDLIFE CORRIDOR PREDICTION MODEL EVALUATION REPORT")
        report.append("="*80)
        report.append("")
        
        # Model Performance Summary
        report.append("MODEL PERFORMANCE SUMMARY")
        report.append("-" * 40)
        
        # Sort models by AUC score
        sorted_results = sorted(self.results.items(), key=lambda x: x[1]['auc'], reverse=True)
        
        for rank, (name, result) in enumerate(sorted_results, 1):
            report.append(f"{rank}. {name}")
            report.append(f"   AUC Score: {result['auc']:.4f}")
            report.append(f"   Accuracy: {result['classification_report']['accuracy']:.4f}")
            report.append(f"   Precision: {result['classification_report']['1']['precision']:.4f}")
            report.append(f"   Recall: {result['classification_report']['1']['recall']:.4f}")
            report.append(f"   F1-Score: {result['classification_report']['1']['f1-score']:.4f}")
            report.append("")
        
        # Feature Information
        report.append("DATASET FEATURES")
        report.append("-" * 40)
        for i, feature in enumerate(feature_names, 1):
            report.append(f"{i:2d}. {feature}")
        report.append("")
        
        # ROC Analysis Interpretation
        report.append("ROC CURVE ANALYSIS")
        report.append("-" * 40)
        best_model = sorted_results[0]
        report.append(f"Best performing model: {best_model[0]} (AUC: {best_model[1]['auc']:.4f})")
        report.append("")
        
        if best_model[1]['auc'] >= 0.9:
            interpretation = "Excellent discrimination ability"
        elif best_model[1]['auc'] >= 0.8:
            interpretation = "Good discrimination ability"
        elif best_model[1]['auc'] >= 0.7:
            interpretation = "Fair discrimination ability"
        elif best_model[1]['auc'] >= 0.6:
            interpretation = "Poor discrimination ability"
        else:
            interpretation = "Very poor discrimination ability"
        
        report.append(f"Model Performance Interpretation: {interpretation}")
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)
        report.append("1. Deploy the best performing model for wildlife corridor prediction")
        report.append("2. Consider ensemble methods to combine multiple models")
        report.append("3. Collect more real-world data to improve model accuracy")
        report.append("4. Perform feature importance analysis for better insights")
        report.append("5. Regular model retraining with new data")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text

def main():
    """
    Main function to execute the complete ROC evaluation pipeline.
    """
    print("AI-Enhanced Wildlife Corridor Identification - ROC Model Evaluation")
    print("="*70)
    
    # Initialize evaluator
    evaluator = WildlifeCorridorROCEvaluator()
    
    # Generate wildlife corridor data
    X, y, feature_names = evaluator.generate_wildlife_data(n_samples=2000)
    
    # Display dataset information
    print("\nDataset Information:")
    print(f"Shape: {X.shape}")
    print(f"Features: {feature_names}")
    print(f"Target classes: Corridor (1), Non-corridor (0)")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale features
    X_train_scaled = evaluator.scaler.fit_transform(X_train)
    X_test_scaled = evaluator.scaler.transform(X_test)
    
    # Initialize and train models
    evaluator.prepare_models()
    evaluator.train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Generate visualizations
    print("\nGenerating ROC curve visualization...")
    evaluator.plot_roc_curves(save_path='/home/user/roc_curves_comparison.png')
    
    print("Generating AUC comparison chart...")
    evaluator.plot_auc_comparison(save_path='/home/user/auc_comparison.png')
    
    print("Generating confusion matrices...")
    evaluator.plot_confusion_matrices(y_test, save_path='/home/user/confusion_matrices.png')
    
    # Generate report
    print("Generating detailed evaluation report...")
    report = evaluator.generate_detailed_report(feature_names, save_path='/home/user/evaluation_report.txt')
    print("\n" + report)
    
    print("\nProject completed successfully!")
    print("Generated files:")
    print("- roc_curves_comparison.png")
    print("- auc_comparison.png") 
    print("- confusion_matrices.png")
    print("- evaluation_report.txt")

if __name__ == "__main__":
    main()
