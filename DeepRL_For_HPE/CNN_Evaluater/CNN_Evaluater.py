from NeighborFolderimporter import *
from FC_RNN_Evaluater.FC_RNN_Evaluater import *
from DatasetHandler.BiwiBrowser import *
from CNN_Configuration import *
from keras.preprocessing.image import ImageDataGenerator

def trainCNN(full_model, modelID, in_epochs, subjectList, output_begin, num_outputs, 
                  batch_size, exp = -1, record = False, preprocess_input = None):
    samples_per_epoch, biwiGenerators = getGeneratorsForBIWIDataset(in_epochs, subjectList = subjectList, preprocess_input = preprocess_input)
    #print(samples_per_epoch, train_batch_size, samples_per_epoch//train_batch_size)
    gen = batchGeneratorFromBIWIGenerators(biwiGenerators, train_batch_size, output_begin, num_outputs)
    samples_per_epoch -= 1
    try:
        full_model.fit_generator(gen, steps_per_epoch=samples_per_epoch//train_batch_size, epochs=in_epochs, verbose=1) 
    except KeyboardInterrupt:
        interruptSmoothly(full_model, modelID, record = record)
    return full_model

######### Evaluation Methods ############
def evaluateSubject(full_model, subject, inputMatrix, test_labels, output_begin, num_outputs, angles, record = False, preprocess_input = None):
    if num_outputs == 1: angles = ['Yaw']
    printLog('For the Subject %d (%s):' % (subject, BIWI_Subject_IDs[subject]), record = record)
    inputGenerator = (frame.reshape(1, 224, 224, 3) for frame in inputMatrix)
    predictions = full_model.predict_generator(inputGenerator, steps = len(test_labels), verbose = 1)
    test_labels = test_labels[:, output_begin:output_begin+num_outputs]
    test_labels, predictions = unscaleEstimations(test_labels, predictions, BIWI_Lebel_Scalers, output_begin, num_outputs)
    outputs = []
    for i in range(num_outputs):
        matrix = numpy.concatenate((test_labels[:, i:i+1], predictions[:, i:i+1]), axis=1)
        differences = (test_labels[:, i:i+1] - predictions[:, i:i+1])
        absolute_mean_error = np.abs(differences).mean()
        printLog("\tThe absolute mean error on %s angle estimation: %.2f Degree" % (angles[i], absolute_mean_error), record = record)
        outputs.append((matrix, absolute_mean_error))
    return full_model, outputs

def evaluateCNN(full_model, label_rescaling_factor, testSubjects, output_begin, 
                     num_outputs, batch_size, angles, record = False, preprocess_input = None):
    if num_outputs == 1: angles = ['Yaw']
    results = []
    biwi = readBIWIDataset(subjectList = testSubjects, preprocess_input = preprocess_input)
    for subject, (inputMatrix, test_labels) in zip(testSubjects, biwi):
        full_model, outputs = evaluateSubject(full_model, subject, inputMatrix, test_labels, output_begin, num_outputs, angles, record = record, preprocess_input = preprocess_input)
        results.append((subject, outputs))
    means = evaluateAverage(results, angles, num_outputs, record = record)
    return full_model, means, results 

    
if __name__ == "__main__":
    main()
    print('Done')