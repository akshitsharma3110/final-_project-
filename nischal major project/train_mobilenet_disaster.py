import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import json

DATA_DIR = r"b:\nischal major project\disasters"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4
MODEL_SAVE = r"b:\nischal major project\disaster_mobilenet.h5"

class_names = ['biological and chemical pandemic', 'cyclone', 'drought', 'earthquake', 'flood', 'landslide', 'tsunami', 'wildfire']
NUM_CLASSES = len(class_names)
print("Detected classes:", class_names)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.08,
    height_shift_range=0.08,
    shear_range=0.08,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=(0.8, 1.2),
    validation_split=0.15,
    fill_mode='nearest'
)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    classes=class_names,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    classes=class_names,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

y_train_labels = train_gen.classes
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_labels),
    y=y_train_labels
)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

base_model = MobileNetV2(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), include_top=False, weights='imagenet')
base_model.trainable = False

inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = inputs
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

checkpoint = keras.callbacks.ModelCheckpoint(r"b:\nischal major project\best_model.h5", monitor='val_accuracy', save_best_only=True, verbose=1)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)

print("\n=== Initial Training ===")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[checkpoint, reduce_lr, early_stop]
)

print("\n=== Fine-tuning ===")
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE * 0.1),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

fine_tune_epochs = 15
history_ft = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=fine_tune_epochs,
    class_weight=class_weights,
    callbacks=[checkpoint, reduce_lr, early_stop]
)

model.save(MODEL_SAVE)
print(f"Saved model to {MODEL_SAVE}")

val_loss, val_acc = model.evaluate(val_gen, verbose=1)
print(f"\nValidation loss={val_loss:.4f}, acc={val_acc:.4f}")

with open(r"b:\nischal major project\class_names.json", 'w') as f:
    json.dump(class_names, f)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history.get('loss', []) + history_ft.history.get('loss', []))
plt.plot(history.history.get('val_loss', []) + history_ft.history.get('val_loss', []))
plt.title('Loss')
plt.legend(['train', 'val'])
plt.subplot(1, 2, 2)
plt.plot(history.history.get('accuracy', []) + history_ft.history.get('accuracy', []))
plt.plot(history.history.get('val_accuracy', []) + history_ft.history.get('val_accuracy', []))
plt.title('Accuracy')
plt.legend(['train', 'val'])
plt.tight_layout()
plt.savefig(r"b:\nischal major project\training_curves.png")
print("Training curves saved")
