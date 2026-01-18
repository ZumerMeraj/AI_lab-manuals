# ============================================
# Cats vs Dogs CNN Classification - LAB
# ============================================

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing import image

# -----------------------------
# 1. Dataset Path
# -----------------------------
DATASET_DIR = "dataset"

IMG_SIZE = 150
BATCH_SIZE = 32

# -----------------------------
# 2. Load & Preprocess Images
# -----------------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

test_data = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# -----------------------------
# 3. Build CNN Model
# -----------------------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -----------------------------
# 4. Train Model
# -----------------------------
EPOCHS = 15

history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=test_data
)

# -----------------------------
# 5. Plot Accuracy & Loss
# -----------------------------
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss")
plt.legend()

plt.show()

# -----------------------------
# 6. Confusion Matrix
# -----------------------------
test_data.reset()
preds = model.predict(test_data)
preds = (preds > 0.5).astype(int).flatten()

true_labels = test_data.classes

cm = confusion_matrix(true_labels, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Cat", "Dog"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# -----------------------------
# 7. Predict New Image
# -----------------------------
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(150,150))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction < 0.5:
        print("✅ This image is a CAT")
    else:
        print("✅ This image is a DOG")

    plt.imshow(img)
    plt.axis("off")
    plt.show()

# -----------------------------
# 8. Test With Sample Image
# -----------------------------
TEST_IMAGE = "dataset/cats/cat1.jpg"   # change image name if needed
predict_image(TEST_IMAGE)
