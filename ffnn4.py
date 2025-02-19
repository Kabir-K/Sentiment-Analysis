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
test_features = test_features.T
train_labels = np.array(train_labels).reshape(1, -1)
test_labels = np.array(test_labels).reshape(1, -1)


d = train_features.shape[0]
initial_lr = 0.001
decay_rate = 0.95
batch_size = 20 
num_batches = train_features.shape[1] // batch_size
l2_lambda = 0

w1 = np.random.randn(20, d) * np.sqrt(2. / d)
b1 = np.zeros((20, 1))
w2 = np.random.randn(1, 20) * np.sqrt(2. / 20)
b2 = np.zeros((1, 1))

def relu(z):
    return np.maximum(0, z)


for t in range(100):  
    print(f"Epoch {t+1}")
    learning_rate = initial_lr * (decay_rate ** t)
    for i in range(num_batches):
        batch_indices = slice(i * batch_size, (i + 1) * batch_size)
        feature_batch = train_features[:, batch_indices] 
        label_batch = train_labels[:, batch_indices]  



        z1 = np.dot(w1, feature_batch) + b1
        a1 = relu(z1)
        z2 = np.dot(w2, a1) + b2
        a2 = np.tanh(z2)

        dl_da2 = -label_batch

        term1=(1 - np.tanh(z2)**2)* dl_da2
        dl_dw2=np.dot(term1,a1.T)
        dl_db2=np.sum(term1, axis=1, keepdims=True)




        term=dl_da2*(1-(np.tanh(z2)**2))
        term=w2.T @ term
        term*= np.where(z1 > 0, 1, 0)
        dl_dw1 = np.dot(term, feature_batch.T)
        dl_db1 = np.sum(term, axis=1, keepdims=True)



        w1 -= learning_rate * (dl_dw1 + l2_lambda * w1)
        b1 -= learning_rate * dl_db1
        w2 -= learning_rate * (dl_dw2 + l2_lambda * w2)
        b2 -= learning_rate * dl_db2


    z1 = np.dot(w1, test_features) + b1
    a1 = relu(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = np.tanh(z2)

    print("Accuracy Test :", util.accuracy(np.sign(a2), test_labels))

z1 = np.dot(w1, train_features) + b1
a1 = relu(z1)
z2 = np.dot(w2, a1) + b2
a2 = np.tanh(z2)


print("Accuracy:", util.accuracy(np.sign(a2), train_labels))

