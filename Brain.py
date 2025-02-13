import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization, Dropout, GlobalAveragePooling2D, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

np.random.seed(42)
tf.random.set_seed(42)

train_glioma_dir = 'C:/Users/USER/Downloads/archive/Train/Glioma/images'
train_meningioma_dir = 'C:/Users/USER/Downloads/archive/Train/Meningioma/images'
train_normal_dir = 'C:/Users/USER/Downloads/archive/Train/No_Tumor/images'
train_pituitary_dir = 'C:/Users/USER/Downloads/archive/Train/Pituitary/images'

val_glioma_dir = 'C:/Users/USER/Downloads/archive/Val/Glioma/images'
val_meningioma_dir = 'C:/Users/USER/Downloads/archive/Val/Meningioma/images'
val_normal_dir = 'C:/Users/USER/Downloads/archive/Val/No_Tumor/images'
val_pituitary_dir = 'C:/Users/USER/Downloads/archive/Val/Pituitary/images'

# Список всех изображений
train_glioma = [os.path.join(train_glioma_dir, i) for i in os.listdir(train_glioma_dir)]
train_meningioma = [os.path.join(train_meningioma_dir, i) for i in os.listdir(train_meningioma_dir)]
train_normal = [os.path.join(train_normal_dir, i) for i in os.listdir(train_normal_dir)]
train_pituitary = [os.path.join(train_pituitary_dir, i) for i in os.listdir(train_pituitary_dir)]

val_glioma = [os.path.join(val_glioma_dir, i) for i in os.listdir(val_glioma_dir)]
val_meningioma = [os.path.join(val_meningioma_dir, i) for i in os.listdir(val_meningioma_dir)]
val_normal = [os.path.join(val_normal_dir, i) for i in os.listdir(val_normal_dir)]
val_pituitary = [os.path.join(val_pituitary_dir, i) for i in os.listdir(val_pituitary_dir)]

total_images = len(
    train_glioma + train_meningioma + train_normal + train_pituitary + val_glioma + val_meningioma + val_normal + val_pituitary)
total_glioma = len(train_glioma + val_glioma)
total_meningioma = len(train_meningioma + val_meningioma)
total_normal = len(train_normal + val_normal)
total_pituitary = len(train_pituitary + val_pituitary)

print('The number of total images is', total_images)
print('The total number of glioma images is', total_glioma)
print('The total number of meningioma images is', total_meningioma)
print('The total number of normal images is', total_normal)
print('The total number of pituitary images is', total_pituitary)

train_set = train_glioma + train_meningioma + train_normal + train_pituitary
val_set = val_glioma + val_meningioma + val_normal + val_pituitary


def preprocess_image(image_list, new_size):
    x = []
    y = []

    for image in image_list:
        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        img_resize = cv2.resize(img, new_size)
        img_norm = img_resize.astype(np.float32) / 255.0
        img_norm = np.stack((img_norm,) * 3, axis=-1)
        x.append(img_norm)

        # Получаем базовое имя файла
        base_name = os.path.basename(image).lower()
        # Присваиваем метки на основе имени файла
        if 'gg' in base_name or 'gl' in base_name:
            y.append(0)
        elif 'no' in base_name or 'image' in base_name:
            y.append(2)
        elif 'me' in base_name or 'm' in base_name:
            y.append(1)
        elif 'p' in base_name:
            y.append(3)
        else:
            print(f"Error {image}")
    return np.array(x), np.array(y)


# Предобработка данных
new_size = (224, 224)
X_train, y_train = preprocess_image(train_set, new_size)
X_val, y_val = preprocess_image(val_set, new_size)

# Преобразование меток в one-hot encoding для categorical_crossentropy
y_train = to_categorical(y_train, num_classes=4)
y_val = to_categorical(y_val, num_classes=4)


# Создание TensorFlow датасетов
def create_tf_dataset(images, labels, batch_size=32, is_training=True):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if is_training:
        dataset = dataset.shuffle(buffer_size=len(images))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


batch_size = 32
train_dataset = create_tf_dataset(X_train, y_train, batch_size=batch_size, is_training=True)
val_dataset = create_tf_dataset(X_val, y_val, batch_size=batch_size, is_training=False)

# Загрузка предобучнной модели
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False

# Создание новой модели и добавление VGG16 в качестве основы
# Активация - Softmax. Averege Pooling(пулинг по среднему)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)

# Обучение модели
history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=val_dataset,
    callbacks=[early_stopping, model_checkpoint]
)

# Диаграммы точности и потерь
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.tight_layout()
plt.show()

# Прогнозирование на валидационном наборе
preds = model.predict(val_dataset)

# Преобразование вероятностей в классы
preds_classes = np.argmax(preds, axis=1)
y_val_classes = np.argmax(y_val, axis=1)

acc = accuracy_score(y_val_classes, preds_classes) * 100

# Матрица ошибок
cm = confusion_matrix(y_val_classes, preds_classes)

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'],
            yticklabels=['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

print('\n============TEST METRICS=============')
print('Accuracy: {}%'.format(acc))

# Полный отчет о классификации
report = classification_report(y_val_classes, preds_classes,
                               target_names=['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'])
print(report)