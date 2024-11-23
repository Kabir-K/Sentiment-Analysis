import csv
import util
import numpy as np
import perceptron
import pegasos
import kernal 
import kernal_final
import ffnn_final

def load_data(path_data, extras=False):
    basic_fields = {'sentiment', 'text'}
    numeric_fields = {'sentiment', 'helpfulY', 'helpfulN'}
    
    with open(path_data, encoding="latin1") as f_data:
        reader = csv.DictReader(f_data, delimiter='\t')
        
        data = [
            {key: (int(value) if key in numeric_fields and value else value)
             for key, value in datum.items()
             if extras or key in basic_fields}
            for datum in reader
        ]

    return data

train_data = load_data('reviews_train.tsv')
val_data = load_data('reviews_val.tsv')
test_data = load_data('reviews_test.tsv')


train_texts, train_labels = zip(*((sample['text'], sample['sentiment']) for sample in train_data))
val_texts, val_labels = zip(*((sample['text'], sample['sentiment']) for sample in val_data))
test_texts, test_labels = zip(*((sample['text'], sample['sentiment']) for sample in test_data))

dictionary = util.bag_of_words(train_texts)

train_features = util.extract_feature_vectors(train_texts, dictionary)
val_features = util.extract_feature_vectors(val_texts, dictionary)
test_features = util.extract_feature_vectors(test_texts, dictionary)

data = (train_features, train_labels, val_features, val_labels)
Ts = [75]

accuracy=[]
precision=[]
recall=[]
specificity=[]
fpr=[]
f1score=[]

##Perceptron###

pct_tune_results,roc = util.tune_perceptron(Ts, *data)
T_best=Ts[np.argmax(pct_tune_results[1])]
util.plot_roc_curve(roc,"Perceptron")
print('Best Accuracy on Validation for Perceptron = {:.4f}, T={:.4f}'.format(np.max(pct_tune_results[1]), T_best))
util.plot_tune_results('Perceptron', 'T', Ts, *pct_tune_results)


theta,theta_0=perceptron.perceptron(train_features,train_labels,T_best)

test_result=util.classify(test_features,theta,theta_0)
train_result=util.classify(train_features,theta,theta_0)

train_accuracy=util.accuracy(train_result,np.array(train_labels))
test_accuracy=util.accuracy(test_result,np.array(test_labels))

print("Train Accuracy for Perceptron : ",train_accuracy)
print("Test Accuracy for Perceptron : ",test_accuracy)

accuracy.append(test_accuracy)
precision.append(util.precision(test_result,np.array(test_labels)))
recall.append(util.recall(test_result,np.array(test_labels)))
specificity.append(util.specificity(test_result,np.array(test_labels)))
f1score.append(util.f1_score(test_result,np.array(test_labels)))
fpr.append(1-util.specificity(test_result,np.array(test_labels)))





###################
###################


###Average Perceptron###

avg_pct_tune_results,roc = util.tune_avg_perceptron(Ts, *data)
T_best=Ts[np.argmax(avg_pct_tune_results[1])]
util.plot_roc_curve(roc,"Average Perceptron")
print('Best Accuracy on Validation for Average Perceptron = {:.4f}, T={:.4f}'.format(np.max(avg_pct_tune_results[1]), T_best))
util.plot_tune_results('Avg Perceptron', 'T', Ts, *avg_pct_tune_results)

theta,theta_0=perceptron.average_perceptron(train_features,train_labels,T_best)

test_result=util.classify(test_features,theta,theta_0)
train_result=util.classify(train_features,theta,theta_0)

train_accuracy=util.accuracy(train_result,np.array(train_labels))
test_accuracy=util.accuracy(test_result,np.array(test_labels))

print("Train Accuracy Average Perceptron : ",train_accuracy)
print("Test Accuracy Average Perceptron : ",test_accuracy)

accuracy.append(test_accuracy)
precision.append(util.precision(test_result,np.array(test_labels)))
recall.append(util.recall(test_result,np.array(test_labels)))
specificity.append(util.specificity(test_result,np.array(test_labels)))
f1score.append(util.f1_score(test_result,np.array(test_labels)))
fpr.append(1-util.specificity(test_result,np.array(test_labels)))

###################
###################


###Pegasos###
Ts = [75]
Ls = [0.1, 0.01]
peg_tune_results_TL,roc = util.tune_pegasos_TL(Ts, Ls, *data)
util.plot_roc_curve(roc,"Pegasos")
best_idx = np.unravel_index(np.argmax(peg_tune_results_TL[1]), peg_tune_results_TL[1].shape)
T_best = Ts[best_idx[0]]
L_best = Ls[best_idx[1]]
print('Best Accuracy on Validation for Pegasos = {:.4f}, T={:.4f}, L={:.4f}'.format(np.max(peg_tune_results_TL[1]), T_best, L_best))
util.plot_tune_results_3d(Ts, Ls, peg_tune_results_TL[0], peg_tune_results_TL[1])

theta,theta_0=pegasos.pegasos(train_features,train_labels,T_best,L_best)

test_result=util.classify(test_features,theta,theta_0)
train_result=util.classify(train_features,theta,theta_0)

train_accuracy=util.accuracy(train_result,np.array(train_labels))
test_accuracy=util.accuracy(test_result,np.array(test_labels))

print("Train Accuracy Pegasos : ",train_accuracy)
print("Test Accuracy Pegasos : ",test_accuracy)

accuracy.append(test_accuracy)
precision.append(util.precision(test_result,np.array(test_labels)))
recall.append(util.recall(test_result,np.array(test_labels)))
specificity.append(util.specificity(test_result,np.array(test_labels)))
f1score.append(util.f1_score(test_result,np.array(test_labels)))
fpr.append(1-util.specificity(test_result,np.array(test_labels)))

###################
###################


###########################
######KERNAL_PERCEPTRON####

test_result,train_result=kernal_final.kernal_perceptron(train_features,train_labels,test_features,test_labels,val_features,val_labels)
train_accuracy=util.accuracy(train_result,np.array(train_labels))
test_accuracy=util.accuracy(test_result,np.array(test_labels))

print("Train Accuracy Kernal Perceptron : ",train_accuracy)
print("Test Accuracy Kernal Perceptron : ",test_accuracy)

accuracy.append(test_accuracy)
precision.append(util.precision(test_result,np.array(test_labels)))
recall.append(util.recall(test_result,np.array(test_labels)))
specificity.append(util.specificity(test_result,np.array(test_labels)))
f1score.append(util.f1_score(test_result,np.array(test_labels)))
fpr.append(1-util.specificity(test_result,np.array(test_labels)))

###########################
###########################



################
######FFNN######


train_result,test_result=ffnn_final.ffnn(train_features,train_labels,test_features,test_labels)
train_accuracy=util.accuracy(train_result,np.array(train_labels))
test_accuracy=util.accuracy(test_result,np.array(test_labels))

print("Train Accuracy FFNN : ",train_accuracy)
print("Test Accuracy FFNN : ",test_accuracy)

accuracy.append(test_accuracy)
precision.append(util.precision(test_result,np.array(test_labels)))
recall.append(util.recall(test_result,np.array(test_labels)))
specificity.append(util.specificity(test_result,np.array(test_labels)))
f1score.append(util.f1_score(test_result,np.array(test_labels)))
fpr.append(1-util.specificity(test_result,np.array(test_labels)))

#########################
#########################






table_data= {
    'Classifier': ['Perceptron', 'Average Perceptron', 'Pegasos','FFNN','Kernal Perceptron'],
    'Precision': precision,
    'Accuracy': accuracy,
    'Sensitivity (Recall)': recall,
    'Specificity (TNR)': specificity,
    'F1 Score': f1score,
    'FPR': fpr
}


util.create_table(table_data)