from keras import layers, models
from keras.applications import MobileNet
from keras.optimizers import Adam
from keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt


# Define paths to the files
train_data_dir = "train"
test_data_dir = 'test'
# set up the datasets to the size of 256 by 256
train_ds = image_dataset_from_directory(
    train_data_dir,
    labels='inferred',
    label_mode='binary',
    image_size=(256, 256),
    batch_size=32
)
test_ds = image_dataset_from_directory(
    test_data_dir,
    labels='inferred',
    label_mode='binary',
    image_size=(256, 256),
    batch_size=32
)
class_names = train_ds.class_names
plt.figure(figsize=(10, 10))

for images, labels in train_ds.take(1):  # Take one batch of images
    for i in range(9):  # Display 9 images
        ax = plt.subplot(3, 3, i + 1)
        plt.suptitle("dataset samples")
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[int(labels[i])])
        plt.axis("off")

plt.show()
# Define the pretrained model based on mobilenet(faster to train) using the imagenet data set
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
for layer in base_model.layers: # ensure model used is not changed during training
    layer.trainable = False
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)  # Reduce the dimensions from 2d to 1d
x = layers.Dropout(0.5)(x)  # drops connections to prevent overfitting
# (https://medium.com/analytics-vidhya/a-simple-introduction-to-dropout-regularization-with-code-5279489dda1e)
predictions = layers.Dense(1, activation='sigmoid')(x)  # Final output classification
# Setting up the model to have the whole structure from the base model of mobile net to the outputs of predictions
model = models.Model(inputs=base_model.input, outputs=predictions)
# binary crossentropy with Adam
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
# Train Model
history = model.fit(
    train_ds,
    epochs=20,
    validation_data=test_ds
)
# print results
test_loss, test_acc = model.evaluate(test_ds)
print('Test accuracy:', test_acc)


plt.figure(figsize=(10, 10))
# get a batch in the test dataset with the images and labels
for images, labels in test_ds.take(1):
    # make a prediction based on the trained model
    predictions = model.predict(images)

    for i in range(9):  # Display 9 images and place the predicted label as well as the actual label on a graph
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        # Binary sigmoid so prediction based on if greater than 0.5 or not
        predicted_label = class_names[int(predictions[i] > 0.5)]
        actual_label = class_names[int(labels[i])]
        # Attach the labels and stack them
        plt.title(f'Actual: {actual_label}\nPredicted: {predicted_label}')
        plt.axis("off")
plt.show()

