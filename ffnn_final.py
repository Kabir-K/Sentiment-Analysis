import numpy as np
import util 

def ffnn(train_features,train_labels,test_features,test_labels):
    
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


    for t in range(90):  
        
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


    z1 = np.dot(w1, train_features) + b1
    a1 = relu(z1)
    z2 = np.dot(w2, a1) + b2
    a2_train = np.tanh(z2)

    z1 = np.dot(w1, test_features) + b1
    a1 = relu(z1)
    z2 = np.dot(w2, a1) + b2
    a2_test = np.tanh(z2)

    return np.sign(a2_train),np.sign(a2_test)

