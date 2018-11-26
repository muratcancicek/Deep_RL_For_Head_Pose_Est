# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/
import os
#Dirty importing that allows the main author to switch environments easily
if '.' in __name__:
    from ClassificationWithCNN.NeighborFolderimporter import *
else:
    from NeighborFolderimporter import *

from DatasetHandler.BiwiBrowser import *

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import keras

def getData():
    biwi = readBIWIDataset(dataFolder = BIWI_SnippedData_folder, 
                                labelsTarFile = BIWI_Lebels_file_Local, 
                                subjectList = [1])
    frames = []
    for inputMatrix, labels in biwi:
        frames.append(inputMatrix)
    return frames[0]


#################### Testing ####################
def main():
    image = getData()
    # load the model
    modelVGG16 = VGG16(weights='imagenet', include_top=False, input_shape= BIWI_Frame_Shape)
    modelVGG16.trainable = False

    x = keras.layers.Flatten(name='flatten')(modelVGG16)
    x = keras.layers.Dense(512, activation='relu', name='fc1')(x)
    x = keras.layers.Dense(512, activation='relu', name='fc2')(x)
    x = keras.layers.Dense(3, activation='softmax', name='predictions')(x)
    model = keras.models.Model(inputs=inp, outputs=x)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

from keras.layers import *
from keras import Model
    
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape= BIWI_Frame_Shape)

# Inspect the model
vgg16.summary()

# This shape has to match the last layer in VGG16 (without top)
dense_input  = Input(shape=(7, 7, 512))
dense_output = Flatten(name='flatten')(dense_input)
dense_output = Dense(dense_layer_1, activation='relu', name='fc1')(dense_output)
dense_output = Dense(dense_layer_2, activation='relu', name='fc2')(dense_output)
dense_output = Dense(num_classes, activation='softmax', name='predictions')(dense_output)

top_model = Model(inputs=dense_input, outputs=dense_output, name='top_model')

# from: https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
top_model.load_weights(top_model_weights_path)

block5_pool = vgg16.get_layer('block5_pool').output

# Now combine the two models
full_output = top_model(block5_pool)
full_model  = Model(inputs=vgg16.input, outputs=full_output)

# set the first 15 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
# WARNING: this may not be applicable for Inception V3
for layer in full_model.layers[:15]:
    layer.trainable = False

# Verify things look as expected
full_model.summary()

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
full_model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.SGD(lr=5e-5, momentum=0.9),
    metrics=['accuracy'])

# Train the 
if __name__ == "__main__":
    main()
    print('Done')

