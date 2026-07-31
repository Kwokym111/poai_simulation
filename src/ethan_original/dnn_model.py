import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve
from sklearn.preprocessing import StandardScaler
import joblib

class DNNModel:
    def __init__(self, input_shape):
        self.model = MLPClassifier(
            hidden_layer_sizes=(64, 32, 16, 8),  # More complex architecture
            activation='relu',
            solver='adam',
            alpha=0.001,  # Reduced regularization
            max_iter=1,  # We'll control iterations manually
            random_state=42,
            learning_rate_init=0.001,
            batch_size=16,
            warm_start=True  # Allows incremental learning
        )
        self.history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
        
    def train(self, X_train, y_train, epochs=100, batch_size=16, validation_split=0.2):
        """Train the DNN model with custom history tracking"""
        print(f"Training model with {X_train.shape[0]} samples...")
        
        # Convert pandas Series to numpy arrays if needed
        if hasattr(y_train, 'values'):
            y_train = y_train.values
        
        # Split data for validation
        n_samples = X_train.shape[0]
        n_val = int(n_samples * validation_split)
        
        if n_val < 1:
            n_val = 1
        
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        
        train_idx = indices[:-n_val]
        val_idx = indices[-n_val:]
        
        X_val = X_train[val_idx]
        y_val = y_train[val_idx]
        X_train_sub = X_train[train_idx]
        y_train_sub = y_train[train_idx]
        
        # Initial fit
        self.model.fit(X_train_sub, y_train_sub)
        
        # Custom training loop with history tracking
        best_val_acc = 0
        patience = 20
        patience_counter = 0
        
        for epoch in range(epochs):
            # Shuffle training data
            idx = np.random.permutation(len(X_train_sub))
            X_train_sub = X_train_sub[idx]
            y_train_sub = y_train_sub[idx]
            
            # Train on batches
            for i in range(0, len(X_train_sub), batch_size):
                X_batch = X_train_sub[i:i+batch_size]
                y_batch = y_train_sub[i:i+batch_size]
                
                # Use partial_fit for incremental learning
                self.model.partial_fit(X_batch, y_batch)
            
            # Calculate training accuracy
            train_pred = self.model.predict(X_train_sub)
            train_acc = accuracy_score(y_train_sub, train_pred)
            self.history['accuracy'].append(train_acc)
            
            # Calculate validation accuracy
            val_pred = self.model.predict(X_val)
            val_acc = accuracy_score(y_val, val_pred)
            self.history['val_accuracy'].append(val_acc)
            
            # Calculate loss (using negative score as loss)
            train_loss = self.model.score(X_train_sub, y_train_sub)
            val_loss = self.model.score(X_val, y_val)
            self.history['loss'].append(-train_loss)
            self.history['val_loss'].append(-val_loss)
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
                
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs} - train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}")
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model with detailed metrics"""
        # Convert pandas Series to numpy arrays if needed
        if hasattr(y_test, 'values'):
            y_test = y_test.values
            
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nTest Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Print confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        return accuracy
    
    def predict(self, X):
        """Make predictions with confidence scores"""
        probabilities = self.model.predict_proba(X)
        predictions = probabilities[:, 1].reshape(-1, 1)
        confidence = np.max(probabilities, axis=1).reshape(-1, 1)
        
        return predictions, confidence
    
    def cross_validate(self, X, y, cv=5):
        """Perform stratified cross-validation"""
        # Convert to numpy arrays if needed
        if hasattr(y, 'values'):
            y = y.values
        
        # Use stratified k-fold cross-validation
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        # Perform cross-validation
        scores = cross_val_score(self.model, X, y, cv=skf, scoring='accuracy')
        
        print(f"\nStratified Cross-Validation Scores: {scores}")
        print(f"Mean CV Accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return scores
    
    def plot_training_history(self, save_path=None):
        """Plot training and validation accuracy and loss curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot accuracy
        ax1.plot(self.history['accuracy'], 'b-', label='Training Accuracy', linewidth=2.5, marker='o', markersize=4)
        ax1.plot(self.history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2.5, marker='s', markersize=4)
        ax1.set_title('Model Accuracy', fontsize=16, fontweight='bold', pad=15)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.legend(loc='best', fontsize=11, framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.05)
        
        # Plot loss
        ax2.plot(self.history['loss'], 'b-', label='Training Loss', linewidth=2.5, marker='o', markersize=4)
        ax2.plot(self.history['val_loss'], 'r-', label='Validation Loss', linewidth=2.5, marker='s', markersize=4)
        ax2.set_title('Model Loss', fontsize=16, fontweight='bold', pad=15)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.legend(loc='best', fontsize=11, framealpha=0.9)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout(pad=3.0)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_learning_curve(self, X, y, save_path=None):
        """Plot learning curve to analyze model performance"""
        train_sizes, train_scores, val_scores = learning_curve(
            self.model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),
            random_state=42
        )
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color='blue', label='Training score')
        plt.plot(train_sizes, np.mean(val_scores, axis=1), 's-', color='red', label='Cross-validation score')
        
        plt.title('Learning Curve', fontsize=16, fontweight='bold')
        plt.xlabel('Training Set Size', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Learning curve saved to {save_path}")
        else:
            plt.show()
    
    def save_model(self, path):
        """Save the trained model"""
        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load a trained model"""
        self.model = joblib.load(path)
        print(f"Model loaded from {path}")