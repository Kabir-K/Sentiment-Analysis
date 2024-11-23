from string import punctuation, digits
import numpy as np
from perceptron import perceptron,average_perceptron
from pegasos import pegasos
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pandas as pd

def extract_words(text):
    for c in punctuation + digits:
        text = text.replace(c, ' ' + c + ' ')
    return text.lower().split()



def bag_of_words(texts, remove_stopword=False):
    with open("stopwords.txt") as f:
        a=f.read()
        f.close()
    indices_by_word = {} 
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word in indices_by_word: continue
            if word in a and remove_stopword: continue
            indices_by_word[word] = len(indices_by_word)

    return indices_by_word



def extract_feature_vectors(reviews, indices_by_word, binarize=True):
    feature_matrix = np.zeros([len(reviews), len(indices_by_word)], dtype=np.float64)
    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word not in indices_by_word: continue
            feature_matrix[i, indices_by_word[word]] += 1
    if binarize:
        feature_matrix[feature_matrix > 0] = 1
    return feature_matrix



def confusion_matrix(preds, targets):
    TP = np.sum((preds == 1) & (targets == 1))
    TN = np.sum((preds == -1) & (targets == -1)) 
    FP = np.sum((preds == 1) & (targets == -1))  
    FN = np.sum((preds == -1) & (targets == 1))   
    return TP, TN, FP, FN

def accuracy(preds, targets):
    return (preds == targets).mean()

def precision(preds, targets):
    TP = np.sum((preds == 1) & (targets == 1))
    FP = np.sum((preds == 1) & (targets == -1))
    return TP / (TP + FP) if (TP + FP) > 0 else 0

def recall(preds, targets):
    TP = np.sum((preds == 1) & (targets == 1))
    FN = np.sum((preds == -1) & (targets == 1))
    return TP / (TP + FN) if (TP + FN) > 0 else 0

def f1_score(preds, targets):
    p = precision(preds, targets)
    r = recall(preds, targets)
    return 2 * (p * r) / (p + r) if (p + r) > 0 else 0

def specificity(preds, targets):
    TN = np.sum((preds == -1) & (targets == -1))
    FP = np.sum((preds == 1) & (targets == -1))
    return TN / (TN + FP) if (TN + FP) > 0 else 0


def classify(feature_matrix, theta, theta_0):
    y=(feature_matrix @ theta)+theta_0
    a=[]
    for i in range(len(y)):
        if y[i]>1e-7:
            a.append(1)
        else:
            a.append(-1)
    return np.array(a)





def pca_manual(X, n_components):

    X_meaned = X - np.mean(X, axis=0)

    covariance_matrix = np.cov(X_meaned, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    sorted_index = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_index]
    sorted_eigenvectors = eigenvectors[:, sorted_index]

    eigenvector_subset = sorted_eigenvectors[:, 0:n_components]

    X_reduced = np.dot(X_meaned, eigenvector_subset)

    return X_reduced

def plot_tune_results(algo_name, param_name, param_vals, acc_train, acc_val):

    plt.subplots()
    plt.plot(param_vals, acc_train, '-o')
    plt.plot(param_vals, acc_val, '-o')

    algo_name = ' '.join((word.capitalize() for word in algo_name.split(' ')))
    param_name = param_name.capitalize()
    plt.suptitle('Classification Accuracy vs {} ({})'.format(param_name, algo_name))
    plt.legend(['train','val'], loc='upper right', title='Partition')
    plt.xlabel(param_name)
    plt.ylabel('Accuracy (%)')
    plt.show()




def plot_tune_results_3d(T_vals, L_vals, acc_train, acc_val, highlight_T=None, highlight_L=None, highlight_acc=None):

    T_vals, L_vals = np.meshgrid(T_vals, L_vals)

    fig = plt.figure(figsize=(12, 8))

    ax = fig.add_subplot(111, projection='3d')
    

    train_surface = ax.plot_surface(T_vals, L_vals, acc_train.T, cmap='Blues', alpha=0.7, rstride=1, cstride=1, 
                                    edgecolor='none', antialiased=True)
    ax.plot_wireframe(T_vals, L_vals, acc_train.T, color='white', linewidth=0.5)
    

    val_surface = ax.plot_surface(T_vals, L_vals, acc_val.T, cmap='Oranges', alpha=0.7, rstride=1, cstride=1, 
                                  edgecolor='none', antialiased=True)
    ax.plot_wireframe(T_vals, L_vals, acc_val.T, color='gray', linewidth=0.5)  


    fig.colorbar(train_surface, ax=ax, shrink=0.5, aspect=5, pad=0.1, label='Train Accuracy')
    fig.colorbar(val_surface, ax=ax, shrink=0.5, aspect=5, pad=0.1, label='Validation Accuracy')

    ax.scatter(T_vals, L_vals, acc_train.T, color='green', s=20 + 50 * acc_train.T, edgecolor='black', linewidth=0.5)

    ax.scatter(T_vals, L_vals, acc_val.T, color='red', s=20 + 50 * acc_val.T, edgecolor='black', linewidth=0.5)


    if highlight_T is not None and highlight_L is not None and highlight_acc is not None:
        ax.scatter(highlight_T, highlight_L, highlight_acc, color='blue', s=150, edgecolor='yellow', linewidth=2, marker='o')
        
        ax.text(highlight_T, highlight_L, highlight_acc, 
                f'Best (T={highlight_T}, L={highlight_L}, Acc={highlight_acc:.4f})', 
                color='black', fontsize=12, weight='bold', bbox=dict(facecolor='white', edgecolor='black', alpha=0.7))
        ax.plot([highlight_T], [highlight_L], [highlight_acc], color='blue', marker='o', markersize=8)


    ax.set_xlabel('T (Iterations)', fontsize=14, weight='bold')
    ax.set_ylabel('L (Lambda)', fontsize=14, weight='bold')
    ax.set_zlabel('Accuracy', fontsize=14, weight='bold')
    ax.set_title('Enhanced 3D Plot of Accuracy vs T and L', fontsize=16, weight='bold')

    ax.grid(True)
    ax.view_init(elev=30, azim=45)  
    plt.show()

def tune(train_fn, param_vals, train_feats, train_labels, val_feats, val_labels):
    train_accs = np.ndarray(len(param_vals))
    val_accs = np.ndarray(len(param_vals))

    for i, val in enumerate(param_vals):
        theta, theta_0 = train_fn(train_feats, train_labels, val)

        train_preds = classify(train_feats, theta, theta_0)
        train_accs[i] = accuracy(train_preds, train_labels)

        val_preds = classify(val_feats, theta, theta_0)
        val_accs[i] = accuracy(val_preds, val_labels)

    return train_accs, val_accs


def tune_perceptron(*args):
    return tune2(perceptron, *args)

def tune_avg_perceptron(*args):
    return tune2(average_perceptron, *args)

def tune_pegasos_T(best_L, *args):
    def train_fn(features, labels, T):
        return pegasos(features, labels, T, best_L)
    return tune(train_fn, *args)


def tune_pegasos_L(best_T, *args):
    def train_fn(features, labels, L):
        return pegasos(features, labels, best_T, L)
    return tune(train_fn, *args)

def tune_pegasos_TL(T_vals, L_vals, train_feats, train_labels, val_feats, val_labels):
    train_accs = np.zeros((len(T_vals), len(L_vals)))
    val_accs = np.zeros((len(T_vals), len(L_vals)))
    train_labels=np.array(train_labels)
    val_labels=np.array(val_labels)
    roc_value=[]

    for i, T in enumerate(T_vals):
        for j, L in enumerate(L_vals):
            theta, theta_0 = pegasos(train_feats, train_labels, T, L)

            train_preds = classify(train_feats, theta, theta_0)
            train_accs[i, j] = accuracy(train_preds, train_labels)

            val_preds = classify(val_feats, theta, theta_0)
            val_accs[i, j] = accuracy(val_preds, val_labels)

            roc_value.append((1-specificity(val_preds, val_labels),recall(val_preds, val_labels)))

    return (train_accs, val_accs),roc_value





def tune2(train_fn, param_vals, train_feats, train_labels, val_feats, val_labels):
    train_accs = np.ndarray(len(param_vals))
    val_accs = np.ndarray(len(param_vals))
    roc_value=[]
    train_labels=np.array(train_labels)
    val_labels=np.array(val_labels)


    for i, val in enumerate(param_vals):
        theta, theta_0 = train_fn(train_feats, train_labels, val)

        train_preds = classify(train_feats, theta, theta_0)
        train_accs[i] = accuracy(train_preds, train_labels)

        val_preds = classify(val_feats, theta, theta_0)
        val_accs[i] = accuracy(val_preds, val_labels)

        roc_value.append((1-specificity(val_preds, val_labels),recall(val_preds, val_labels)))

    return (train_accs, val_accs),roc_value




def plot_roc_curve(roc, algo_name):
    fpr, tpr = zip(*roc)

    fpr = np.array(fpr)
    tpr = np.array(tpr)

    sorted_indices = np.argsort(fpr)
    fpr_sorted = fpr[sorted_indices]
    tpr_sorted = tpr[sorted_indices]

    area_real=auc(fpr_sorted,tpr_sorted)


    extended_fpr = np.concatenate(([0], fpr_sorted, [1]))
    extended_tpr = np.concatenate(([0], tpr_sorted, [1]))

    area = auc(extended_fpr, extended_tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_sorted, tpr_sorted, marker='.', label=f'{algo_name} (AUC = {area:.2f}) (AUC REAL = {area_real:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Random Guess')

    plt.fill_between(extended_fpr, extended_tpr, color='lightgray', alpha=0.5)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {algo_name}')
    plt.xlim(0, 1) 
    plt.ylim(0, 1) 
    plt.legend()
    plt.grid()
    plt.show()

def auc(fpr, tpr):
    return np.trapz(tpr, fpr)


def create_table(data):
    df = pd.DataFrame(data)


    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('tight')
    ax.axis('off')


    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')


    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.auto_set_column_width(col=list(range(len(df.columns))))  


    for i in range(len(df.columns)):
        table[0, i].set_text_props(weight='bold', color='white')
        table[0, i].set_facecolor('#4C72B0')  

    for i in range(1, len(df) + 1):
        color = '#E5E5E5' if i % 2 == 0 else '#FFFFFF'
        for j in range(len(df.columns)):
            table[i, j].set_facecolor(color)

    plt.show()