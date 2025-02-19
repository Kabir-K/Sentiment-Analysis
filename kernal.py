import numpy as np

def kernal_perceptron(train_features,train_labels,test_features,test_labels):
    train_features = train_features.T
    test_features = test_features.T
    train_labels = np.array(train_labels).reshape(1, -1)
    test_labels = np.array(test_labels).reshape(1, -1)
    theta_0=0
    k=(train_features.T @ train_features)
    k+=(k**2+(k**3))  
    errors = np.zeros((1, 4000))
    for t in range(12):
        for i in range(4000):
            ki=k[:,i].reshape(1,-1)
            term=(np.sum(ki*errors*train_labels)+theta_0)*train_labels[0,i]
            if term<=0:
                errors[0,i]+=1
                theta_0+=train_labels[0,i]
    r=(test_features.T @ train_features)
    r+=(r**2+(r**3))  
    r=r*train_labels*errors
    r=np.sum(r,axis=1)
    r=r.reshape(1,500)+theta_0
    k=k*train_labels*errors
    k=np.sum(k,axis=1)
    k=k.reshape(1,4000)+theta_0
    return np.sign(r),np.sign(k)
