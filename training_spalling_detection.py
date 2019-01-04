import numpy, os
from keras import models
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, Conv2D, Dense, Flatten, MaxPooling2D
from keras.optimizers import SGD
from skimage import data
 
training_image_files_crack = [f for f in os.listdir('/home/labuser/sid/research/spalling/data/train/spalling/') if '.jpg' in f]
training_image_files_no_crack = [f for f in os.listdir('/home/labuser/sid/research/spalling/data/train/non_spalling/') if '.jpg' in f]
validation_image_files_crack = [f for f in os.listdir('/home/labuser/sid/research/spalling/data/validate/spalling/') if '.jpg' in f]
validation_image_files_no_crack = [f for f in os.listdir('/home/labuser/sid/research/spalling/data/validate/non_spalling/') if '.jpg' in f]
test_image_files = [f for f in os.listdir('/home/labuser/sid/research/spalling/data/test/') if '.jpg' in f]

print("training (crack):",len(training_image_files_crack))
print("training (no-crack):",len(training_image_files_no_crack))
print("validation (crack):",len(validation_image_files_crack))
print("validation (no-crack):",len(validation_image_files_no_crack))
print("test:",len(test_image_files))

training_images = []
training_labels = []
validation_images = []
validation_labels = []
test_images = []

for i in range(0, len(training_image_files_crack)):
    image = data.imread('/home/labuser/sid/research/spalling/data/train/spalling/'+training_image_files_crack[i])
    training_images.append(image)
    training_labels.append(0)

for i in range(0, len(training_image_files_no_crack)):
    image = data.imread('/home/labuser/sid/research/spalling/data/train/non_spalling/'+training_image_files_no_crack[i])
    training_images.append(image)
    training_labels.append(1)

for i in range(0, len(validation_image_files_crack)):
    image = data.imread('/home/labuser/sid/research/spalling/data/validate/spalling/'+validation_image_files_crack[i])
    validation_images.append(image)
    validation_labels.append(0)

for i in range(0, len(validation_image_files_no_crack)):
    image = data.imread('/home/labuser/sid/research/spalling/data/validate/non_spalling/'+validation_image_files_no_crack[i])
    validation_images.append(image)
    validation_labels.append(1)

for i in range(0, len(test_image_files)):
    image = data.imread('/home/labuser/sid/research/spalling/data/test/'+test_image_files[i])
    test_images.append(image)

training_images = numpy.asarray(training_images)

training_images = training_images.astype('float32')
training_images /= 255.0

training_labels = numpy.asarray(training_labels)

temp_labels = numpy.ndarray(shape=(len(training_labels), 2), dtype=float)
for i in range(0, len(training_labels)):
    if training_labels[i]:
        temp_labels[i][1] = 1.
        temp_labels[i][0] = 0.
    else:
        temp_labels[i][0] = 1.
        temp_labels[i][1] = 0.

training_labels = temp_labels

validation_images = numpy.asarray(validation_images)

validation_images = validation_images.astype('float32')
validation_images /= 255.0

validation_labels = numpy.asarray(validation_labels)

temp_labels = numpy.ndarray(shape=(len(validation_labels), 2), dtype=float)
for i in range(0, len(validation_labels)):
    if validation_labels[i]:
        temp_labels[i][1] = 1
        temp_labels[i][0] = 0
    else:
        temp_labels[i][0] = 1
        temp_labels[i][1] = 0

validation_labels = temp_labels
##############################
test_images = numpy.asarray(test_images)

test_images = test_images.astype('float32')
test_images /= 255.0

#################################
model = models.Sequential()
model.add(Conv2D(24, kernel_size=20, strides=2, input_shape=(256, 256, 3)))
model.add(MaxPooling2D(pool_size=(7, 7), strides=2))
model.add(Conv2D(48, kernel_size=15, strides=2))
model.add(MaxPooling2D(pool_size=(4, 4), strides=2))
model.add(Conv2D(96, kernel_size=10, strides=2))
model.add(Activation('relu'))
model.add(Conv2D(2, kernel_size=1, strides=1))
model.add(Activation('softmax'))
model.add(Flatten())

opt = SGD(lr=0.001)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

model.summary()

checkpoint_callback = ModelCheckpoint(filepath='model.hdf5', monitor='val_acc', save_best_only=True, mode='auto', period=1)

model.fit(training_images, training_labels, epochs=200, batch_size=16, verbose=1,
            validation_data=(validation_images, validation_labels), callbacks=[checkpoint_callback])
