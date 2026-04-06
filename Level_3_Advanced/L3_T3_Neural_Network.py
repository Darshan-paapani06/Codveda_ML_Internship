import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

BASE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(BASE, '..', 'outputs')
os.makedirs(OUT, exist_ok=True)

print("TensorFlow version :", tf.__version__)

digits = load_digits()
X = digits.data
y = digits.target

print("\nDataset loaded!")
print("Shape         :", X.shape)
print("Classes       :", digits.target_names)
print("Total samples :", len(X))
print("Features      : 64 pixel values from 8x8 digit images")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain : {len(X_train)} | Test : {len(X_test)}")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

y_train_cat = to_categorical(y_train, num_classes=10)
y_test_cat  = to_categorical(y_test,  num_classes=10)

print("\n--- Neural Network Architecture ---")
model = Sequential([
    Input(shape=(64,)),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64,  activation='relu'),
    Dropout(0.2),
    Dense(32,  activation='relu'),
    Dense(10,  activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

print("\n--- Training the Model ---")
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=0
)

history = model.fit(
    X_train_sc, y_train_cat,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

print(f"\nTraining stopped at epoch : {len(history.history['loss'])}")

print("\n--- Evaluating the Model ---")
test_loss, test_acc = model.evaluate(X_test_sc, y_test_cat, verbose=0)
print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_acc:.4f}")

y_pred_proba = model.predict(X_test_sc, verbose=0)
y_pred       = np.argmax(y_pred_proba, axis=1)

print("\nClassification Report :")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix :")
print(cm)

train_acc_final = history.history['accuracy'][-1]
val_acc_final   = history.history['val_accuracy'][-1]
print(f"\nFinal Training   Accuracy : {train_acc_final:.4f}")
print(f"Final Validation Accuracy : {val_acc_final:.4f}")
print(f"Final Test       Accuracy : {test_acc:.4f}")

print("\n--- Sample Predictions ---")
print(f"{'#':<5} {'Actual':>8} {'Predicted':>11} {'Correct':>10}")
print(f"{'─'*5} {'─'*8} {'─'*11} {'─'*10}")
for i in range(15):
    correct = '✅' if y_test[i] == y_pred[i] else '❌'
    print(f"{i+1:<5} {y_test[i]:>8} {y_pred[i]:>11} {correct:>10}")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Neural Network — Digit Classification (TensorFlow/Keras)', fontsize=14)

axes[0, 0].plot(history.history['accuracy'],     color='steelblue', lw=2, label='Train Accuracy')
axes[0, 0].plot(history.history['val_accuracy'], color='coral',     lw=2, label='Validation Accuracy')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].set_title('Training vs Validation Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

axes[0, 1].plot(history.history['loss'],     color='steelblue', lw=2, label='Train Loss')
axes[0, 1].plot(history.history['val_loss'], color='coral',     lw=2, label='Validation Loss')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].set_title('Training vs Validation Loss')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

im = axes[1, 0].imshow(cm, cmap='Blues')
axes[1, 0].set_xticks(range(10))
axes[1, 0].set_yticks(range(10))
axes[1, 0].set_xlabel('Predicted Digit')
axes[1, 0].set_ylabel('Actual Digit')
axes[1, 0].set_title(f'Confusion Matrix\nTest Accuracy = {test_acc:.4f}')
plt.colorbar(im, ax=axes[1, 0])
for i in range(10):
    for j in range(10):
        col = 'white' if cm[i, j] > cm.max() / 2 else 'black'
        axes[1, 0].text(j, i, str(cm[i, j]), ha='center', va='center',
                        fontsize=8, fontweight='bold', color=col)

sample_images = X_test[:16].reshape(-1, 8, 8)
for idx in range(16):
    ax_sub = axes[1, 1].inset_axes(
        [(idx % 4) * 0.25, (3 - idx // 4) * 0.25, 0.23, 0.23]
    )
    ax_sub.imshow(sample_images[idx], cmap='gray_r')
    color = 'green' if y_test[idx] == y_pred[idx] else 'red'
    ax_sub.set_title(f'A:{y_test[idx]} P:{y_pred[idx]}',
                     fontsize=7, color=color, pad=1)
    ax_sub.axis('off')

axes[1, 1].axis('off')
axes[1, 1].set_title('Sample Predictions\n(green=correct  red=wrong)', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'L3_T3_neural_network_results.png'), dpi=150, bbox_inches='tight')
plt.close()
print("\nPlot saved!")

results_df = pd.DataFrame({
    'Actual'   : y_test,
    'Predicted': y_pred,
    'Correct'  : (y_test == y_pred)
})
results_df.to_csv(os.path.join(OUT, 'L3_T3_predictions.csv'), index=False)
print("Predictions CSV saved!")

print("\n--- Summary ---")
print(f"Dataset       : Sklearn Digits (1797 samples, 64 features, 10 classes)")
print(f"Architecture  : Input(64) → Dense(128) → Dropout → Dense(64) → Dropout → Dense(32) → Output(10)")
print(f"Optimizer     : Adam (lr=0.001)")
print(f"Loss          : Categorical Crossentropy")
print(f"Epochs run    : {len(history.history['loss'])}")
print(f"Test Accuracy : {test_acc:.4f}  ({test_acc*100:.1f}%)")
print(f"Test Loss     : {test_loss:.4f}")
print("\nAll 3 tasks of Level 3 completed!")
print("Internship tasks DONE! 🎉")