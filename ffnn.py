import csv
import util
import numpy as np

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

train_features = train_features.T
test_features=test_features.T

train_labels = np.array(train_labels)
train_labels = train_labels.reshape(1, 4000)


w1 = np.random.randn(1024, train_features.shape[0]) * np.sqrt(2. / train_features.shape[0])
b1 = np.zeros((1024, 1))
w2 = np.random.randn(512, 1024) * np.sqrt(2. / 1024)
b2 = np.zeros((512, 1))
w3 = np.random.randn(124, 512) * np.sqrt(2. / 512)
b3 = np.zeros((124, 1))
w4 = np.random.randn(64, 124) * np.sqrt(2. / 124)
b4 = np.zeros((64, 1))
w5 = np.random.randn(1, 64) * np.sqrt(2. / 64)
b5 = np.zeros((1, 1))

def relu(z):
    return np.maximum(0,z)

m=train_labels.shape[1]

def relu_derivative(z):
    return np.where(z > 0, 1, 0)


learning_rate=0.005


for i in range(1):

    z1=np.dot(w1,train_features)+b1
    a1=relu(z1)
    z2=np.dot(w2,a1)+b2
    a2=relu(z2)
    z3=np.dot(w3,a2)+b3
    a3=relu(z3)
    z4=np.dot(w4,a3)+b4
    a4=relu(z4)
    z5=np.dot(w5,a4)+b5
    a5=np.tanh(z5)



    loss=np.mean(((a5 - train_labels)**2)/2)

    
    
    loss_derivative=(a5-train_labels)/m
    
    da5_dz5=1-np.tanh(z5)**2
    term1=loss_derivative*da5_dz5
    dw5 = (1 / m) * np.dot(term1, a4.T)
    db5 = (1 / m) * np.sum(term1, axis=1, keepdims=True)


    term2=np.dot((loss_derivative*da5_dz5).T,w5)*relu_derivative(z4.T)
    dw4 = (1 / m) * np.dot(term2.T, a3.T)
    db4 = (1 / m) * np.sum(term2.T, axis=1, keepdims=True)


    term3=np.dot(term2,w4)*relu_derivative(z3.T)
    dw3 = (1 / m) * np.dot(term3.T,a2.T)
    db3 = (1 / m) * np.sum(term3.T, axis=1, keepdims=True)
    

    term4=np.dot(term3,w3)*relu_derivative(z2.T)
    dw2 = (1 / m) * np.dot(term4.T,a1.T)
    db2 = (1 / m) * np.sum(term4.T, axis=1, keepdims=True)

    term5=np.dot(term4,w2)*relu_derivative(z1.T)
    dw1 = (1 / m) * np.dot(term5.T,train_features.T)
    db1 = (1 / m) * np.sum(term5.T, axis=1, keepdims=True)

    w1 -= learning_rate * dw1
    b1 -= learning_rate * db1
    w2 -= learning_rate * dw2
    b2 -= learning_rate * db2
    w3 -= learning_rate * dw3
    b3 -= learning_rate * db3
    w4 -= learning_rate * dw4
    b4 -= learning_rate * db4
    w5 -= learning_rate * dw5
    b5 -= learning_rate * db5
result=np.sign(a5)
accuracy= (result == train_labels).astype(int)
print(a5.shape)