import numpy
import os
import random
import csv

from deap import algorithms, base, creator, tools
from keras import models
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from skimage import data

global evaluation_count
global population_size

evaluation_count = 0
population_size = 5


""" length of binary string (change for more parameters/filters/etc.)
using 64 + 64 + 64 + 64 + 64 + 64
filter size 1 (6 bits) +
filter count 1 (6 bits) +
filter size 2 (6 bits) +
filter count 2 (6 bits) +
filter size 3 (6 bits) +
filter count 3 (6 bits) 

36 bits total
"""
LENGTH = 36

# load images for training and testing
training_image_files_crack = [
    f for f in os.listdir('/home/labuser/sid/research/spalling/data/ready_data_256_1/train/spalling/') if '.jpg' in f]
training_image_files_no_crack = [
    f for f in os.listdir('/home/labuser/sid/research/spalling/data/ready_data_256_1/train/non_spalling/') if '.jpg' in f]
validation_image_files_crack = [
    f for f in os.listdir('/home/labuser/sid/research/spalling/data/ready_data_256_1/validate/spalling/') if '.jpg' in f]
validation_image_files_no_crack = [
    f for f in os.listdir('/home/labuser/sid/research/spalling/data/ready_data_256_1/validate/non_spalling/') if '.jpg' in f]

training_images = []
training_labels = []
validation_images = []
validation_labels = []
validation_file_names = []

for i in range(0, len(training_image_files_crack)):
    image = data.imread('/home/labuser/sid/research/spalling/data/ready_data_256_1/train/spalling/' + training_image_files_crack[i])
    training_images.append(image)
    training_labels.append(0)

for i in range(0, len(training_image_files_no_crack)):
    image = data.imread('/home/labuser/sid/research/spalling/data/ready_data_256_1/train/non_spalling/' + training_image_files_no_crack[i])
    training_images.append(image)
    training_labels.append(1)

for i in range(0, len(validation_image_files_crack)):
    image = data.imread('/home/labuser/sid/research/spalling/data/ready_data_256_1/validate/spalling/' + validation_image_files_crack[i])
    validation_images.append(image)
    validation_labels.append(0)

for i in range(0, len(validation_image_files_no_crack)):
    image = data.imread('/home/labuser/sid/research/spalling/data/ready_data_256_1/validate/non_spalling/' + validation_image_files_no_crack[i])
    validation_images.append(image)
    validation_labels.append(1)

validation_labels_copy = validation_labels

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

# maximization problem
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# initialize toolbox
toolbox = base.Toolbox()

# using binary representation
toolbox.register("attr_bool", random.randint, 0, 1)

# individual will consist of LENGTH bits
toolbox.register("individual", tools.initRepeat,
                 creator.Individual, toolbox.attr_bool, LENGTH)

# population consists of individuals
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evaluate(individual):
    global evaluation_count
    global population_size
    evaluation_count += 1

    fsize1_bin = individual[0:6]
    fcount1_bin = individual[6:12]
    fsize2_bin = individual[12:18]
    fcount2_bin = individual[18:24]
    fsize3_bin = individual[24:30]
    fcount3_bin = individual[30:]

    fsize1 = 1 + to_decimal(fsize1_bin)
    fcount1 = 1 + to_decimal(fcount1_bin)
    fsize2 = 1 + to_decimal(fsize2_bin)
    fcount2 = 1 + to_decimal(fcount2_bin)
    fsize3 = 1 + to_decimal(fsize3_bin)
    fcount3 = 1 + to_decimal(fcount3_bin)

    model = models.Sequential()
    model.add(Conv2D(fcount1, kernel_size=fsize1,
                     activation='relu', padding='same', input_shape=(256,256,3)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(fcount2, kernel_size=fsize2,
                     activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(fcount3, kernel_size=fsize3,
                     activation='relu', padding='same'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    opt = SGD(lr=0.001)

    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    gen_number = int(evaluation_count/population_size)

    model_name = str(gen_number) + '_____' + str(fsize1) + '-' + \
	str(fcount1) + '-' + str(fsize2) + '-' + str(fcount2) + '-' + \
        str(fsize3) + '-' + str(fcount3)

    checkpoint_callback = ModelCheckpoint(
        filepath="/home/labuser/sid/research/spalling/data/ready_data_256_1/models_ga/" + model_name + ".hdf5", monitor='val_acc', save_best_only=True, mode='auto', period=1)

    history = model.fit(training_images, training_labels, epochs=100, batch_size=16, verbose=1,
              validation_data=(validation_images, validation_labels), callbacks=[checkpoint_callback])

    print(history.history.keys())

    with open('/home/labuser/sid/research/spalling/data/ready_data_256_1/graph/accuracy'+ model_name +'.csv', 'w') as writeFile:
      writer = csv.writer(writeFile)
      writeFile.write(str(history.history['val_acc']))

    with open('/home/labuser/sid/research/spalling/data/ready_data_256_1/graph/loss'+ model_name +'.csv', 'w') as writeFile:
      writer = csv.writer(writeFile)
      writeFile.write(str(history.history['val_loss']))


    predictions = model.predict_classes(validation_images, batch_size=16)

    binary_predictions = []

    crack = predictions[:300]
    no_crack = predictions[300:]

    crack = [value for value in crack if value == 0]
    no_crack = [value for value in no_crack if value == 1]

    accuracy = (len(crack)+len(no_crack))/len(validation_images)

    output_file = open("/home/labuser/sid/research/spalling/data/ready_data_256_1/models_ga/"+model_name+".txt", 'w')
    output_file.write(str(accuracy))
    output_file.close()

    return accuracy,

toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selRoulette)
toolbox.register("evaluate", evaluate)


def to_decimal(array):
    string = ""

    for index in range(len(array)):
        string += str(array[index])

    return int(string, 2)


def main():
    random.seed(1)

    pop = toolbox.population(n=6)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    algorithms.eaSimple(pop, toolbox, 0.67, 0.01, 10, stats=stats,
                        halloffame=hof)

    return pop, stats, hof

if __name__ == "__main__":
    main()



