import numpy as np
import util

def kernal_perceptron(train_features,train_labels,test_features,test_labels,val_features,val_labels):
    train_features = train_features.T
    test_features = test_features.T
    val_features = val_features.T
    train_labels = np.array(train_labels).reshape(1, -1)
    test_labels = np.array(test_labels).reshape(1, -1)
    val_labels = np.array(val_labels).reshape(1, -1)

    val_acc=[]
    train_acc=[]
    roc_value=[]
    
    theta_0=0

    k=(train_features.T @ train_features)
    k+=(k**2+(k**3))  
    v=(val_features.T@train_features)
    v+=(v**2+(v**3))

    errors = np.zeros((1, 4000))

    for t in range(12):
        for i in range(4000):
            ki=k[:,i].reshape(1,-1)
            term=(np.sum(ki*errors*train_labels)+theta_0)*train_labels[0,i]
            if term<=0:
                errors[0,i]+=1
                theta_0+=train_labels[0,i]
        vt=v*train_labels*errors
        vt=np.sum(vt,axis=1)
        vt=vt.reshape(1,500)+theta_0
        val_acc.append(util.accuracy(np.sign(vt),val_labels))
        roc_value.append((1-util.specificity(np.sign(vt), val_labels),util.recall(np.sign(vt), val_labels)))
        kt=k*train_labels*errors
        kt=np.sum(kt,axis=1)
        kt=kt.reshape(1,4000)+theta_0
        train_acc.append(util.accuracy(np.sign(kt),train_labels))
    util.plot_tune_results("Kernal Perceptron","T",[1,2,3,4,5,6,7,8,9,10,11,12],train_acc,val_acc)
    util.plot_roc_curve(roc_value,"Kernal Perceptron")
    T=val_acc.index(max(val_acc))+1
    print('Best Accuracy on Validation for Kernal Perceptron = {:.4f}, T={:.4f}'.format(val_acc[T-1],T))
    errors = np.zeros((1, 4000))
    for t in range(T):
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
