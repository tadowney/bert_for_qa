from bert2 import runBERT

trainfile0 = 'train_0-10_record'
record00_size = 61477 

predictfile0 = 'predictions_sample.json'

runBERT(trainfile0, record00_size, predictfile0, False, True)


