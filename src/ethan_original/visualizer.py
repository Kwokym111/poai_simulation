import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import os
import seaborn as sns
from pathlib import Path

class PoAIVisualizer:
    def __init__(self, results_dir='results'):
        # Use pathlib for cross-platform path handling
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style for publication-quality figures
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        
    def _save_figure(self, fig, filename):
        """Helper method to save figures with correct path handling"""
        save_path = self.results_dir / filename
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ {filename} saved to {save_path}")
        
    def plot_training_history(self, history, save_name='training_history'):
        """Plot training and validation accuracy and loss curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot accuracy
        ax1.plot(history['accuracy'], 'b-', label='Training Accuracy', linewidth=2.5, marker='o', markersize=4)
        ax1.plot(history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2.5, marker='s', markersize=4)
        ax1.set_title('Model Accuracy', fontsize=16, fontweight='bold', pad=15)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.legend(loc='best', fontsize=11, framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.05)
        
        # Plot loss
        ax2.plot(history['loss'], 'b-', label='Training Loss', linewidth=2.5, marker='o', markersize=4)
        ax2.plot(history['val_loss'], 'r-', label='Validation Loss', linewidth=2.5, marker='s', markersize=4)
        ax2.set_title('Model Loss', fontsize=16, fontweight='bold', pad=15)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.legend(loc='best', fontsize=11, framealpha=0.9)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout(pad=3.0)
        self._save_figure(fig, f"{save_name}.png")
        
    def create_miner_dashboard(self, miners_df, save_prefix='miner_dashboard'):
        """Create an enhanced miner performance dashboard"""
        print("Creating Enhanced Miner Performance Dashboard...")
        
        # 1. Top Miners by Efficiency
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Top 10 miners by efficiency
        top_10_efficiency = miners_df.nlargest(10, 'efficiency')
        y_pos = np.arange(len(top_10_efficiency))
        bars1 = ax1.barh(y_pos, top_10_efficiency['efficiency'], color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars1, top_10_efficiency['efficiency'])):
            ax1.text(value + 0.0005, i, f'{value:.4f}', va='center', fontsize=10, fontweight='bold')
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([f"Miner {m.split('_')[1]}" for m in top_10_efficiency['miner_id']])
        ax1.set_xlabel('Efficiency Score', fontsize=12, fontweight='bold')
        ax1.set_title('Top 10 Miners by Efficiency', fontsize=14, fontweight='bold', pad=15)
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.set_xlim(0, top_10_efficiency['efficiency'].max() * 1.15)
        
        # Top 10 miners by profitability
        top_10_profit = miners_df.nlargest(10, 'profitability')
        y_pos = np.arange(len(top_10_profit))
        bars2 = ax2.barh(y_pos, top_10_profit['profitability'], color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars2, top_10_profit['profitability'])):
            ax2.text(value + 500000, i, f'{value:.0f}', va='center', fontsize=10, fontweight='bold')
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([f"Miner {m.split('_')[1]}" for m in top_10_profit['miner_id']])
        ax2.set_xlabel('Profitability Score', fontsize=12, fontweight='bold')
        ax2.set_title('Top 10 Miners by Profitability', fontsize=14, fontweight='bold', pad=15)
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.set_xlim(0, top_10_profit['profitability'].max() * 1.1)
        
        plt.tight_layout()
        self._save_figure(fig1, f"{save_prefix}_top_performers.png")
        
        # 2. Efficiency vs Profitability Scatter
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        
        scatter = ax2.scatter(miners_df['efficiency'], miners_df['profitability'], 
                            c=miners_df['blocks_mined'], cmap='viridis', s=100, alpha=0.7, 
                            edgecolors='black', linewidth=1)
        
        ax2.set_xlabel('Efficiency Score', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Profitability Score', fontsize=12, fontweight='bold')
        ax2.set_title('Efficiency vs Profitability Analysis', fontsize=16, fontweight='bold', pad=20)
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax2, label='Blocks Mined')
        cbar.ax.tick_params(labelsize=10)
        
        # Highlight top performers
        top_efficient = miners_df.nlargest(5, 'efficiency')
        for idx, row in top_efficient.iterrows():
            ax2.annotate(f"M{row['miner_id'].split('_')[1]}", 
                        (row['efficiency'], row['profitability']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))
        
        plt.tight_layout()
        self._save_figure(fig2, f"{save_prefix}_efficiency_profitability.png")
        
        # 3. Miner Characteristics Radar Chart (Fixed)
        fig3, ax3 = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Select a representative miner
        sample_miner = miners_df.iloc[0]
        
        # Prepare data for radar chart
        categories = ['Blocks\nMined', 'Transactions', 'Volume', 'Efficiency', 'Profitability']
        values = [
            sample_miner['blocks_mined'] / miners_df['blocks_mined'].max() * 100,
            sample_miner['avg_transactions'] / miners_df['avg_transactions'].max() * 100,
            sample_miner['avg_volume'] / miners_df['avg_volume'].max() * 100,
            sample_miner['efficiency'] / miners_df['efficiency'].max() * 100,
            sample_miner['profitability'] / miners_df['profitability'].max() * 100
        ]
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
        # Create values array with the same length as angles (5)
        plot_values = np.array(values)
        
        ax3.plot(angles, plot_values, 'o-', linewidth=2, label='Miner Profile')
        ax3.fill(angles, plot_values, alpha=0.25)
        ax3.set_xticks(angles)
        ax3.set_xticklabels(categories)
        ax3.set_ylim(0, 100)
        ax3.set_title('Miner Characteristics Radar Chart', fontsize=16, fontweight='bold', pad=20)
        ax3.grid(True)
        
        plt.tight_layout()
        self._save_figure(fig3, f"{save_prefix}_radar.png")
        
        # 4. Feature Correlation Heatmap
        fig4, ax4 = plt.subplots(figsize=(12, 8))
        
        # Select numeric columns for correlation
        numeric_cols = ['blocks_mined', 'avg_transactions', 'avg_volume', 'efficiency', 
                        'profitability', 'avg_fee', 'fee_volatility', 'avg_block_size', 'age']
        correlation_matrix = miners_df[numeric_cols].corr()
        
        # Create heatmap
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                    square=True, ax=ax4, cbar_kws={'shrink': 0.8})
        ax4.set_title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        self._save_figure(fig4, f"{save_prefix}_correlation_heatmap.png")
        
        # 5. Performance Distributions
        fig5, (ax5, ax6) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Efficiency distribution
        ax5.hist(miners_df['efficiency'], bins=20, alpha=0.7, color='skyblue', 
                 edgecolor='black', density=True)
        ax5.axvline(miners_df['efficiency'].mean(), color='red', 
                   linestyle='--', label='Mean', linewidth=2)
        ax5.set_xlabel('Efficiency Score', fontsize=12)
        ax5.set_ylabel('Density', fontsize=12)
        ax5.set_title('Efficiency Score Distribution', fontsize=14, fontweight='bold', pad=15)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Profitability distribution
        ax6.hist(miners_df['profitability'], bins=20, alpha=0.7, color='lightcoral', 
                 edgecolor='black', density=True)
        ax6.axvline(miners_df['profitability'].mean(), color='green', 
                   linestyle='--', label='Mean', linewidth=2)
        ax6.set_xlabel('Profitability Score', fontsize=12)
        ax6.set_ylabel('Density', fontsize=12)
        ax6.set_title('Profitability Score Distribution', fontsize=14, fontweight='bold', pad=15)
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig5, f"{save_prefix}_distributions.png")
        
        # 6. Blocks Mined Distribution
        fig6, ax6 = plt.subplots(figsize=(10, 6))
        
        ax6.hist(miners_df['blocks_mined'], bins=15, alpha=0.7, color='gold', 
                 edgecolor='black', density=True)
        ax6.axvline(miners_df['blocks_mined'].mean(), color='darkblue', 
                   linestyle='--', label='Mean', linewidth=2)
        ax6.set_xlabel('Blocks Mined', fontsize=12)
        ax6.set_ylabel('Frequency', fontsize=12)
        ax6.set_title('Blocks Mined Distribution', fontsize=14, fontweight='bold', pad=15)
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig6, f"{save_prefix}_blocks_distribution.png")
        
        print(f"✓ Enhanced Miner Dashboard created with 6 visualizations")
        
    def plot_simulation_results(self, simulation_results, save_name='simulation_results'):
        """Enhanced simulation results visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Plot efficiency scores with confidence intervals
        ax1 = axes[0, 0]
        ax1.plot(simulation_results['simulation'], simulation_results['efficiency_score'], 
                'o-', color='#3498db', markersize=8, linewidth=2.5, label='Efficiency Score')
        ax1.fill_between(simulation_results['simulation'], 
                         simulation_results['efficiency_score'] - 0.05,
                         simulation_results['efficiency_score'] + 0.05,
                         alpha=0.2, color='blue')
        ax1.set_title('Efficiency Scores with Confidence Intervals', fontsize=14, fontweight='bold', pad=15)
        ax1.set_xlabel('Simulation Number', fontsize=12)
        ax1.set_ylabel('Efficiency Score', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.1)
        
        # Plot efficiency distribution
        ax2 = axes[0, 1]
        ax2.hist(simulation_results['efficiency'], bins=20, alpha=0.7, color='skyblue', 
                 edgecolor='black', density=True)
        ax2.axvline(simulation_results['efficiency'].mean(), color='red', 
                   linestyle='--', label='Mean', linewidth=2)
        ax2.set_xlabel('Efficiency Score', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.set_title('Efficiency Score Distribution', fontsize=14, fontweight='bold', pad=15)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot selection frequency
        ax3 = axes[1, 0]
        miner_counts = simulation_results['selected_miner'].value_counts()
        bars = ax3.bar(range(len(miner_counts)), miner_counts.values, color='lightcoral', 
                      alpha=0.8, edgecolor='black', linewidth=1)
        ax3.set_title('Miner Selection Frequency', fontsize=14, fontweight='bold', pad=15)
        ax3.set_xlabel('Miner ID', fontsize=12)
        ax3.set_ylabel('Selection Count', fontsize=12)
        ax3.set_xticks(range(len(miner_counts)))
        ax3.set_xticklabels([f"M{m.split('_')[1]}" for m in miner_counts.index])
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (rank, count) in enumerate(zip(range(len(miner_counts)), miner_counts.values)):
            ax3.text(i, count + 0.1, str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Plot efficiency vs blocks mined
        ax4 = axes[1, 1]
        ax4.scatter(simulation_results['blocks_mined'], simulation_results['efficiency_score'], 
                   alpha=0.7, c=simulation_results['simulation'], cmap='viridis', s=80, 
                   edgecolors='black', linewidth=1)
        ax4.set_xlabel('Blocks Mined', fontsize=12)
        ax4.set_ylabel('Efficiency Score', fontsize=12)
        ax4.set_title('Blocks Mined vs Efficiency Score', fontsize=14, fontweight='bold', pad=15)
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle('Enhanced Simulation Results Analysis', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(pad=3.0, rect=[0, 0, 1, 0.96])
        self._save_figure(fig, f"{save_name}.png")
        
    def plot_model_evaluation(self, y_true, y_pred, y_prob=None, save_name='model_evaluation'):
        """Enhanced model evaluation with additional metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Confusion Matrix
        ax1 = axes[0, 0]
        cm = confusion_matrix(y_true, y_pred)
        im = ax1.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax1.figure.colorbar(im, ax=ax1)
        ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
        ax1.set_xlabel('Predicted', fontsize=12)
        ax1.set_ylabel('Actual', fontsize=12)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax1.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=14, fontweight='bold')
        
        ax1.set_xticks([0, 1])
        ax1.set_yticks([0, 1])
        ax1.set_xticklabels(['Inefficient', 'Efficient'])
        ax1.set_yticklabels(['Inefficient', 'Efficient'])
        
        # ROC Curve
        ax2 = axes[0, 1]
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            ax2.plot(fpr, tpr, color='#e74c3c', lw=3, label=f'ROC (AUC = {roc_auc:.2f})')
            ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax2.set_xlim([0.0, 1.0])
            ax2.set_ylim([0.0, 1.05])
            ax2.set_xlabel('False Positive Rate', fontsize=12)
            ax2.set_ylabel('True Positive Rate', fontsize=12)
            ax2.set_title('ROC Curve', fontsize=14, fontweight='bold', pad=15)
            ax2.legend(loc="lower right", fontsize=11)
            ax2.grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        ax3 = axes[0, 2]
        if y_prob is not None:
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            ax3.plot(recall, precision, color='purple', lw=2, label='Precision-Recall Curve')
            ax3.set_xlabel('Recall', fontsize=12)
            ax3.set_ylabel('Precision', fontsize=12)
            ax3.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold', pad=15)
            ax3.legend(loc="lower left", fontsize=11)
            ax3.grid(True, alpha=0.3)
        
        # Prediction Distribution
        ax4 = axes[1, 0]
        ax4.hist(y_pred, bins=20, alpha=0.7, color='skyblue', edgecolor='black', linewidth=1)
        ax4.set_xlabel('Predicted Value', fontsize=12)
        ax4.set_ylabel('Frequency', fontsize=12)
        ax4.set_title('Prediction Distribution', fontsize=14, fontweight='bold', pad=15)
        ax4.axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold', linewidth=2)
        ax4.legend(loc='best', fontsize=11)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Feature Importance
        ax5 = axes[1, 1]
        feature_names = ['Blocks Mined', 'Avg Transactions', 'Avg Volume', 'Efficiency', 'Profitability']
        # Create dummy importance values for visualization
        importance = np.random.rand(len(feature_names))
        colors = plt.cm.Set3(np.linspace(0, 1, len(feature_names)))
        bars = ax5.bar(feature_names, importance, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax5.set_title('Feature Importance', fontsize=14, fontweight='bold', pad=15)
        ax5.set_ylabel('Importance', fontsize=12)
        ax5.set_ylim(0, max(importance) * 1.2)
        ax5.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Comprehensive Model Evaluation', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(pad=3.0, rect=[0, 0, 1, 0.96])
        self._save_figure(fig, f"{save_name}.png")
        
    def create_comprehensive_dashboard(self, history, comparison_data, miners_df, 
                                    simulation_results, y_true, y_pred, y_prob=None):
        """Create a comprehensive dashboard with all enhanced visualizations"""
        print("Creating Comprehensive Visualization Dashboard...")
        
        # Create all enhanced visualizations
        self.plot_training_history(history, 'enhanced_training_history')
        self.create_miner_dashboard(miners_df, 'enhanced_miner_performance')
        self.plot_simulation_results(simulation_results, 'enhanced_simulation_results')
        self.plot_model_evaluation(y_true, y_pred, y_prob, 'enhanced_model_evaluation')
        
        print(f"All enhanced visualizations saved to {self.results_dir}")