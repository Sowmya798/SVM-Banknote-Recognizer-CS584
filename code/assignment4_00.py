import numpy as np
import matplotlib.pyplot as plt
import cvxopt
from cvxopt import solvers
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import urllib

solvers.options['show_progress'] = False

def gen_sep_data(size):
    c0 = np.array(zip(np.random.randint(1, size+1, size), np.random.randint(1, size+1, size)))
    c1 = np.array(zip(np.random.randint(size+2, 2*(size+2), size), np.random.randint(size+2, 2*(size+2), size)))
    return c0, c1


def get_svm(x, y, c=None):
    g = None
    h = None
    m = x.shape[0]
    if c is not None:
        g_0 = np.zeros([m, m])
        g_1 = np.zeros([m, m])
        h = np.zeros(2 * m)
        np.fill_diagonal(g_0, -1)
        np.fill_diagonal(g_1, 1)
        h[m:] = c
        g = np.append(g_0, g_1, axis=0)
    else:
        g = np.zeros([m , m])
        np.fill_diagonal(g, -1)
        h = np.zeros(m)

    y = y.reshape(-1, 1)

    p = np.dot(y, y.T) * np.dot(x, x.T)
    q = np.array([-1] * x.shape[0])
    A = y.T

    res = solvers.qp(cvxopt.matrix(p) , cvxopt.matrix(q, tc='d'), cvxopt.matrix(g), cvxopt.matrix(h), cvxopt.matrix(A), cvxopt.matrix(0, tc='d'))
    alpha = np.array(res['x'])

    w = np.sum(alpha * y * x, axis=0)
    y = y.reshape(-1, 1)[0]
    w0 = np.mean(y - np.dot(x, w.T))

    alpha_tf = alpha.reshape(1, -1)[0] > alpha.mean()
    supp_vect_idx = np.array([i for i in xrange(alpha.shape[0]) if alpha_tf[i] == True])

    return supp_vect_idx, w, w0

def get_svm_poly(x, y, pow=2, c=5):
    g = None
    h = None
    m = x.shape[0]
    if c is not None:
        g_0 = np.zeros([m, m])
        g_1 = np.zeros([m, m])
        h = np.zeros(2 * m)
        np.fill_diagonal(g_0, -1)
        np.fill_diagonal(g_1, 1)
        h[m:] = c
        g = np.append(g_0, g_1, axis=0)
    else:
        g = np.zeros([m , m])
        np.fill_diagonal(g, -1)
        h = np.zeros(m)

    y = y.reshape(-1, 1)

    p = np.dot(y, y.T) * (np.dot(x, x.T) + 1)**pow
    q = np.array([-1] * x.shape[0])
    A = y.T

    res = solvers.qp(cvxopt.matrix(p) , cvxopt.matrix(q, tc='d'), cvxopt.matrix(g), cvxopt.matrix(h), cvxopt.matrix(A, tc='d'), cvxopt.matrix(0, tc='d'))
    alpha = np.array(res['x'])

    #w = np.sum(alpha * y * x, axis=0)
    alpha = alpha.reshape(1, -1)[0]
    alpha_tf = alpha > np.mean(alpha)
    supp_vect_idx = np.array([i for i in xrange(alpha.shape[0]) if alpha_tf[i] == True])

    k = (np.dot(x, x.T) + 1)**pow
    y = y.reshape(1, -1)[0]
    w0 = np.mean(y[supp_vect_idx] - np.array([np.sum(alpha * y * k_elem) for k_elem in k[supp_vect_idx]]))

    return supp_vect_idx, alpha, w0


def get_svm_gauss(x, y, sigma=2, c=5):
    g = None
    h = None
    m = x.shape[0]
    if c is not None:
        g_0 = np.zeros([m, m])
        g_1 = np.zeros([m, m])
        h = np.zeros(2 * m)
        np.fill_diagonal(g_0, -1)
        np.fill_diagonal(g_1, 1)
        h[m:] = c
        g = np.append(g_0, g_1, axis=0)
    else:
        g = np.zeros([m , m])
        np.fill_diagonal(g, -1)
        h = np.zeros(m)

    y = y.reshape(-1, 1)

    k = np.exp([-np.array([np.dot(diff0_elem.T, diff0_elem) for diff0_elem in np.array([x_r - x_elem for x_elem in x])]) / sigma for x_r in x])

    p = np.dot(y, y.T) * k ##########
    q = np.array([-1] * x.shape[0])
    A = y.T

    res = solvers.qp(cvxopt.matrix(p) , cvxopt.matrix(q, tc='d'), cvxopt.matrix(g), cvxopt.matrix(h), cvxopt.matrix(A, tc='d'), cvxopt.matrix(0, tc='d'))
    alpha = np.array(res['x'])

    alpha = alpha.reshape(1, -1)[0]
    alpha_tf = alpha > np.mean(alpha)
    supp_vect_idx = np.array([i for i in xrange(alpha.shape[0]) if alpha_tf[i] == True])

    #k = (np.dot(x, x.T) + 1)**pow #########
    y = y.reshape(1, -1)[0]
    w0 = np.mean(y[supp_vect_idx] - np.array([np.sum(alpha * y * k_elem) for k_elem in k[supp_vect_idx]]))

    return supp_vect_idx, alpha, w0


def svm_gauss_predict(supp_vect_idx, alpha, w0, x_org, y_org, x, sigma=2):
    k = np.exp([-np.array([np.dot(diff0_elem.T, diff0_elem) for diff0_elem in np.array([x_r - x_elem for x_elem in x_org])]) / sigma for x_r in x])
    ans = np.array([np.sum(alpha * y_org * k_elem) for k_elem in k]) + w0
    ans[ans>0] = 1
    ans[ans<0] = -1
    return ans

def svm_poly_predict(supp_vect_idx, alpha, w0, x_org, y_org, x, pow=2):
    k = (np.dot(x, x_org.T) + 1)**pow
    ans = np.array([np.sum(alpha * y_org * k_elem) for k_elem in k]) + w0
    ans[ans>0] = 1
    ans[ans<0] = -1
    return ans


def svm_predict(w, w0, x):
    ans = np.dot(w, x.T) + w0
    ans[ans < 0] = -1
    ans[ans > 0] = 1
    return ans


def gen_non_sep_data(size):
    size0 = 0
    size1 = 0
    c0 = []
    c1 = []
    while size0 < size or size1 < size:
        c = np.array([np.random.randint(1, size+1), np.random.randint(1, size+1)])
        if c[1] > (c[0]**2) / 4 - 50 and size0 < size:
            c0.append(c * np.random.rand())
            size0 += 1
        elif c[1] < (c[0]**2) / 4 + 50 and size1 < size:
            c1.append(c * np.random.rand())
            size1 += 1
    return np.array(c0), np.array(c1)

def parse_data(data):
    """Parses raw bank note dataset into feature vectors and labels"""
    labels = np.array([int(d.strip().split(',')[-1:][0]) for d in data])
    feature_vects = np.array([np.array([float(e) for e in d.strip().split(',')[:-1]]) for d in data])
    return feature_vects, labels

def parse_data_bal(data):
    """Parses raw balance dataset into feature vectors and labels"""
    labels = np.array([d.strip().split(',')[0] for d in data])
    feature_vects = np.array([np.array([float(e) for e in d.strip().split(',')[1:]]) for d in data])
    feature_vects = feature_vects[labels!='B']
    labels = labels[labels!='B']
    num_labels = np.ones(labels.shape[0])
    num_labels[labels=='L'] = int(-1)
    idx = np.arange(labels.shape[0])
    np.random.shuffle(idx)
    num_labels = num_labels[idx]
    feature_vects = feature_vects[idx]
    return feature_vects, num_labels
    

if __name__ == "__main__":
    c0_sep, c1_sep = gen_sep_data(50)
    c0_non_sep, c1_non_sep = gen_non_sep_data(50)

    c0_sep = c0_sep / 100.
    c1_sep = c1_sep / 100.
    c0_non_sep = c0_non_sep / 100.
    c1_non_sep = c1_non_sep / 100.

    f1 = urllib.urlopen('http://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt')
    data_bn = f1.readlines() #Bank Note Dataset

    x_bn, y_bn = parse_data(data_bn)
    y_bn[y_bn==0]=-1

    idx = np.arange(x_bn.shape[0])
    np.random.shuffle(idx)
    x_bn = x_bn[idx]
    y_bn = y_bn[idx]


    f2 = urllib.urlopen('http://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data')
    data_bal = f2.readlines() #Balance Dataset
    x_bal, y_bal = parse_data_bal(data_bal)

    prec0, = plt.plot(c0_sep[:,0], c0_sep[:,1], 'o')
    prec1, = plt.plot(c1_sep[:,0], c1_sep[:,1], 'o')
    plt.title('Linearly Separable Dataset')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend([prec0, prec1], ['Class 0', 'Class 1'])
    plt.show()

    prec0, = plt.plot(c0_non_sep[:,0], c0_non_sep[:,1], 'o', color='y')
    prec1, = plt.plot(c1_non_sep[:,0], c1_non_sep[:,1], 'o', color='g')
    plt.title('Non-Separable dataset')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend([prec0, prec1], ['Class 0', 'Class 1'])
    plt.show()

    sep_data = np.append(c0_sep, c1_sep, axis=0) 
    sep_labels = np.append(-np.ones(50), np.ones(50))

    non_sep_data = np.append(c0_non_sep, c1_non_sep, axis=0) 
    non_sep_labels = np.append(-np.ones(50), np.ones(50))

    nfolds = 4

    #Hard Margins

    sv_idx_sep, _, _ = get_svm(sep_data, sep_labels)

    prec0, = plt.plot(c0_sep[:,0], c0_sep[:,1], 'o')
    prec1, = plt.plot(c1_sep[:,0], c1_sep[:,1], 'o')

    prec_sv, = plt.plot(sep_data[sv_idx_sep, 0], sep_data[sv_idx_sep, 1], '^', color='y')

    plt.title('Linearly Separable Dataset with Support Vectors (Hard Margins)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend([prec0, prec1, prec_sv], ['Class 0', 'Class 1', 'Support Vectors'])
    plt.show()

    sv_idx_nsep, _, _ = get_svm(non_sep_data, non_sep_labels)

    prec0, = plt.plot(c0_non_sep[:,0], c0_non_sep[:,1], 'o', color='y')
    prec1, = plt.plot(c1_non_sep[:,0], c1_non_sep[:,1], 'o', color='g')

    prec_sv_0, = plt.plot(non_sep_data[sv_idx_nsep[sv_idx_nsep<50], 0], non_sep_data[sv_idx_nsep[sv_idx_nsep<50], 1], '^', color='b')
    prec_sv_1, = plt.plot(non_sep_data[sv_idx_nsep[sv_idx_nsep>50], 0], non_sep_data[sv_idx_nsep[sv_idx_nsep>50], 1], '^', color=(0,0,0))

    plt.title('Linearly Non-Separable Dataset with Support Vectors (Hard Margins)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend([prec0, prec1, prec_sv_0, prec_sv_1], ['Class 0', 'Class 1', 'Support Vectors, Class 0', 'Support Vectors, Class 1'])
    plt.show()

    kf = KFold(100, n_folds=nfolds, shuffle=True)

    avg_accuracy =  0.; avg_precision =  np.array([0., 0.]); avg_recall =  np.array([0., 0.]); avg_fscore =  np.array([0., 0.]); avg_conf_mat = np.zeros([2, 2])
    
    for train_ids, test_ids in kf:
        sv_idx_sep, w, w0 = get_svm(sep_data[train_ids], sep_labels[train_ids])
        preds_sep = svm_predict(w, w0, sep_data[test_ids])
        avg_accuracy += accuracy_score(sep_labels[test_ids], preds_sep)
        precision1, recall1, fscore1, supp1 = precision_recall_fscore_support(sep_labels[test_ids], preds_sep)
        conf_mat = confusion_matrix(sep_labels[test_ids], preds_sep)
        avg_precision += precision1; avg_recall += recall1; avg_fscore += fscore1; avg_conf_mat += conf_mat

    avg_accuracy /= nfolds
    avg_precision /= nfolds
    avg_recall /= nfolds
    avg_fscore /= nfolds
    avg_conf_mat /= nfolds

    print 'Separable Dataset Metrics (Hard Margins)-'
    print 'Accuracy =', avg_accuracy
    print 'Precision =', avg_precision
    print 'Recall =', avg_recall
    print 'F-Score =', avg_fscore
    print 'Confusion Matrix -'
    print avg_conf_mat


    avg_accuracy =  0.; avg_precision =  np.array([0., 0.]); avg_recall =  np.array([0., 0.]); avg_fscore =  np.array([0., 0.]); avg_conf_mat = np.zeros([2, 2])
    
    for train_ids, test_ids in kf:
        sv_idx_nsep, w, w0 = get_svm(non_sep_data[train_ids], non_sep_labels[train_ids])
        preds_nsep = svm_predict(w, w0, non_sep_data[test_ids])
        avg_accuracy += accuracy_score(non_sep_labels[test_ids], preds_nsep)
        precision1, recall1, fscore1, supp1 = precision_recall_fscore_support(non_sep_labels[test_ids], preds_nsep)
        conf_mat = confusion_matrix(non_sep_labels[test_ids], preds_nsep)
        avg_precision += precision1; avg_recall += recall1; avg_fscore += fscore1; avg_conf_mat += conf_mat

    avg_accuracy /= nfolds
    avg_precision /= nfolds
    avg_recall /= nfolds
    avg_fscore /= nfolds
    avg_conf_mat /= nfolds

    print '\nNon-Separable Dataset Metrics (Hard Margins)-'
    print 'Accuracy =', avg_accuracy
    print 'Precision =', avg_precision
    print 'Recall =', avg_recall
    print 'F-Score =', avg_fscore
    print 'Confusion Matrix -'
    print avg_conf_mat




    #Soft Margins

    sv_idx_sep, _, _ = get_svm(sep_data, sep_labels, c=30)

    prec0, = plt.plot(c0_sep[:,0], c0_sep[:,1], 'o')
    prec1, = plt.plot(c1_sep[:,0], c1_sep[:,1], 'o')

    prec_sv, = plt.plot(sep_data[sv_idx_sep, 0], sep_data[sv_idx_sep, 1], '^', color='y')

    plt.title('Linearly Separable Dataset with Support Vectors (Soft Margins with c = 30)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend([prec0, prec1, prec_sv], ['Class 0', 'Class 1', 'Support Vectors'])
    plt.show()
    
    sv_idx_nsep, _, _ = get_svm(non_sep_data, non_sep_labels, c=30)

    prec0, = plt.plot(c0_non_sep[:,0], c0_non_sep[:,1], 'o', color='y')
    prec1, = plt.plot(c1_non_sep[:,0], c1_non_sep[:,1], 'o', color='g')

    prec_sv_0, = plt.plot(non_sep_data[sv_idx_nsep[sv_idx_nsep<50], 0], non_sep_data[sv_idx_nsep[sv_idx_nsep<50], 1], '^', color='b')
    prec_sv_1, = plt.plot(non_sep_data[sv_idx_nsep[sv_idx_nsep>50], 0], non_sep_data[sv_idx_nsep[sv_idx_nsep>50], 1], '^', color=(0,0,0))

    plt.title('Linearly Non-Separable Dataset with Support Vectors (Soft Margins with c = 30)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend([prec0, prec1, prec_sv_0, prec_sv_1], ['Class 0', 'Class 1', 'Support Vectors, Class 0', 'Support Vectors, Class 1'])
    plt.show()
    
    #Testing Performance
    nfolds = 4
    kf = KFold(100, n_folds=nfolds, shuffle=True)

    slack = 30
    avg_accuracy =  0.; avg_precision =  np.array([0., 0.]); avg_recall =  np.array([0., 0.]); avg_fscore =  np.array([0., 0.]); avg_conf_mat = np.zeros([2, 2])
    
    for train_ids, test_ids in kf:
        sv_idx_sep, w, w0 = get_svm(sep_data[train_ids], sep_labels[train_ids], c=slack)
        preds_sep = svm_predict(w, w0, sep_data[test_ids])
        avg_accuracy += accuracy_score(sep_labels[test_ids], preds_sep)
        precision1, recall1, fscore1, supp1 = precision_recall_fscore_support(sep_labels[test_ids], preds_sep)
        conf_mat = confusion_matrix(sep_labels[test_ids], preds_sep)
        avg_precision += precision1; avg_recall += recall1; avg_fscore += fscore1; avg_conf_mat += conf_mat

    avg_accuracy /= nfolds
    avg_precision /= nfolds
    avg_recall /= nfolds
    avg_fscore /= nfolds
    avg_conf_mat /= nfolds

    print '\nSeparable Dataset Metrics (Soft Margin with c =', slack, ')-'
    print 'Accuracy =', avg_accuracy
    print 'Precision =', avg_precision
    print 'Recall =', avg_recall
    print 'F-Score =', avg_fscore
    print 'Confusion Matrix -'
    print avg_conf_mat

    avg_accuracy =  0.; avg_precision =  np.array([0., 0.]); avg_recall =  np.array([0., 0.]); avg_fscore =  np.array([0., 0.]); avg_conf_mat = np.zeros([2, 2])
    
    for train_ids, test_ids in kf:
        sv_idx_nsep, w, w0 = get_svm(non_sep_data[train_ids], non_sep_labels[train_ids], c=slack)
        preds_nsep = svm_predict(w, w0, non_sep_data[test_ids])
        avg_accuracy += accuracy_score(non_sep_labels[test_ids], preds_nsep)
        precision1, recall1, fscore1, supp1 = precision_recall_fscore_support(non_sep_labels[test_ids], preds_nsep)
        conf_mat = confusion_matrix(non_sep_labels[test_ids], preds_nsep)
        avg_precision += precision1; avg_recall += recall1; avg_fscore += fscore1; avg_conf_mat += conf_mat

    avg_accuracy /= nfolds
    avg_precision /= nfolds
    avg_recall /= nfolds
    avg_fscore /= nfolds
    avg_conf_mat /= nfolds

    print '\nNon-Separable Dataset Metrics (Soft Margin with c =', slack, ')-'

    print 'Accuracy =', avg_accuracy
    print 'Precision =', avg_precision
    print 'Recall =', avg_recall
    print 'F-Score =', avg_fscore
    print 'Confusion Matrix -'
    print avg_conf_mat


    #Polynomial Kernel Function

    sv_idx_sep, _, _ = get_svm_poly(sep_data, sep_labels) #Separable Data

    prec0, = plt.plot(c0_sep[:,0], c0_sep[:,1], 'o')
    prec1, = plt.plot(c1_sep[:,0], c1_sep[:,1], 'o')

    prec_sv, = plt.plot(sep_data[sv_idx_sep, 0], sep_data[sv_idx_sep, 1], '^', color='y')

    plt.title('Linearly Separable Dataset with Support Vectors (Polynomial Kernel)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend([prec0, prec1, prec_sv], ['Class 0', 'Class 1', 'Support Vectors'])
    plt.show()

    sv_idx_nsep, _, _ = get_svm_poly(non_sep_data, non_sep_labels)

    prec0, = plt.plot(c0_non_sep[:,0], c0_non_sep[:,1], 'o', color='y')
    prec1, = plt.plot(c1_non_sep[:,0], c1_non_sep[:,1], 'o', color='g')

    prec_sv_0, = plt.plot(non_sep_data[sv_idx_nsep[sv_idx_nsep<50], 0], non_sep_data[sv_idx_nsep[sv_idx_nsep<50], 1], '^', color='b')
    prec_sv_1, = plt.plot(non_sep_data[sv_idx_nsep[sv_idx_nsep>50], 0], non_sep_data[sv_idx_nsep[sv_idx_nsep>50], 1], '^', color=(0,0,0))

    plt.title('Linearly Non-Separable Dataset with Support Vectors (Polynomial Kernel)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend([prec0, prec1, prec_sv_0, prec_sv_1], ['Class 0', 'Class 1', 'Support Vectors, Class 0', 'Support Vectors, Class 1'])
    plt.show()

    acc_sep = []
    acc_nsep = []
    acc_bn = []
    acc_bal = []
    max_deg = 20
    for d in np.arange(1, max_deg):

        kf = KFold(100, n_folds=nfolds, shuffle=True)
        avg_accuracy =  0.; avg_precision =  np.array([0., 0.]); avg_recall =  np.array([0., 0.]); avg_fscore =  np.array([0., 0.]); avg_conf_mat = np.zeros([2, 2])
        for train_ids, test_ids in kf:
            sv_idx_sep, alpha, w0 = get_svm_poly(sep_data[train_ids], sep_labels[train_ids], pow=d)
            preds_sep = svm_poly_predict(sv_idx_sep, alpha, w0, sep_data[train_ids], sep_labels[train_ids], sep_data[test_ids], pow=d)
            avg_accuracy += accuracy_score(sep_labels[test_ids], preds_sep)
            precision1, recall1, fscore1, supp1 = precision_recall_fscore_support(sep_labels[test_ids], preds_sep)
            conf_mat = confusion_matrix(sep_labels[test_ids], preds_sep)
            avg_precision += precision1; avg_recall += recall1; avg_fscore += fscore1; avg_conf_mat += conf_mat

        avg_accuracy /= nfolds
        avg_precision /= nfolds
        avg_recall /= nfolds
        avg_fscore /= nfolds
        avg_conf_mat /= nfolds

        acc_sep.append(avg_accuracy)

        print '\nSeparable Dataset Metrics (Polynomial Kernel with degree =', d, ')-'
        print 'Accuracy =', avg_accuracy
        print 'Precision =', avg_precision
        print 'Recall =', avg_recall
        print 'F-Score =', avg_fscore
        print 'Confusion Matrix -'
        print avg_conf_mat


        kf = KFold(100, n_folds=nfolds, shuffle=True)
        avg_accuracy =  0.; avg_precision =  np.array([0., 0.]); avg_recall =  np.array([0., 0.]); avg_fscore =  np.array([0., 0.]); avg_conf_mat = np.zeros([2, 2])
        for train_ids, test_ids in kf:
            sv_idx_nsep, alpha, w0 = get_svm_poly(non_sep_data[train_ids], non_sep_labels[train_ids], pow=d)
            preds_nsep = svm_poly_predict(sv_idx_nsep, alpha, w0, non_sep_data[train_ids], non_sep_labels[train_ids], non_sep_data[test_ids], pow=d)
            avg_accuracy += accuracy_score(non_sep_labels[test_ids], preds_nsep)
            precision1, recall1, fscore1, supp1 = precision_recall_fscore_support(non_sep_labels[test_ids], preds_nsep)
            conf_mat = confusion_matrix(non_sep_labels[test_ids], preds_nsep)
            avg_precision += precision1; avg_recall += recall1; avg_fscore += fscore1; avg_conf_mat += conf_mat

        avg_accuracy /= nfolds
        avg_precision /= nfolds
        avg_recall /= nfolds
        avg_fscore /= nfolds
        avg_conf_mat /= nfolds

        acc_nsep.append(avg_accuracy)

        print '\nNon-Separable Dataset Metrics (Polynomial Kernel with degree =', d, ')-'
        print 'Accuracy =', avg_accuracy
        print 'Precision =', avg_precision
        print 'Recall =', avg_recall
        print 'F-Score =', avg_fscore
        print 'Confusion Matrix -'
        print avg_conf_mat



        
        x_bn = x_bn[:500] / x_bn.max(axis = 0)
        y_bn = y_bn[:500]
        kf = KFold(x_bn.shape[0], n_folds=nfolds, shuffle=True)
        avg_accuracy =  0.; avg_precision =  np.array([0., 0.]); avg_recall =  np.array([0., 0.]); avg_fscore =  np.array([0., 0.]); avg_conf_mat = np.zeros([2, 2])
        for train_ids, test_ids in kf:
            sv_idx_bn, alpha, w0 = get_svm_poly(x_bn[train_ids], y_bn[train_ids], pow=d)
            preds_bn = svm_poly_predict(sv_idx_bn, alpha, w0, x_bn[train_ids], y_bn[train_ids], x_bn[test_ids], pow=d)
            avg_accuracy += accuracy_score(y_bn[test_ids], preds_bn)
            precision1, recall1, fscore1, supp1 = precision_recall_fscore_support(y_bn[test_ids], preds_bn)
            conf_mat = confusion_matrix(y_bn[test_ids], preds_bn)
            avg_precision += precision1; avg_recall += recall1; avg_fscore += fscore1; avg_conf_mat += conf_mat

        avg_accuracy /= nfolds
        avg_precision /= nfolds
        avg_recall /= nfolds
        avg_fscore /= nfolds
        avg_conf_mat /= nfolds

        acc_bn.append(avg_accuracy)

        print '\nBanknote Dataset Metrics (Polynomial Kernel with degree =', d, ')-'
        print 'Accuracy =', avg_accuracy
        print 'Precision =', avg_precision
        print 'Recall =', avg_recall
        print 'F-Score =', avg_fscore
        print 'Confusion Matrix -'
        print avg_conf_mat


        x_bal = x_bal / x_bal.max(axis = 0)
        kf = KFold(x_bal.shape[0], n_folds=nfolds, shuffle=True)
        avg_accuracy =  0.; avg_precision =  np.array([0., 0.]); avg_recall =  np.array([0., 0.]); avg_fscore =  np.array([0., 0.]); avg_conf_mat = np.zeros([2, 2])
        for train_ids, test_ids in kf:
            sv_idx_bal, alpha, w0 = get_svm_poly(x_bal[train_ids], y_bal[train_ids], pow=d)
            preds_bal = svm_poly_predict(sv_idx_bal, alpha, w0, x_bal[train_ids], y_bal[train_ids], x_bal[test_ids], pow=d)
            avg_accuracy += accuracy_score(y_bal[test_ids], preds_bal)
            precision1, recall1, fscore1, supp1 = precision_recall_fscore_support(y_bal[test_ids], preds_bal)
            conf_mat = confusion_matrix(y_bal[test_ids], preds_bal)
            avg_precision += precision1; avg_recall += recall1; avg_fscore += fscore1; avg_conf_mat += conf_mat

        avg_accuracy /= nfolds
        avg_precision /= nfolds
        avg_recall /= nfolds
        avg_fscore /= nfolds
        avg_conf_mat /= nfolds

        acc_bal.append(avg_accuracy)

        print '\nBalance Dataset Metrics (Polynomial Kernel with degree =', d, ')-'
        print 'Accuracy =', avg_accuracy
        print 'Precision =', avg_precision
        print 'Recall =', avg_recall
        print 'F-Score =', avg_fscore
        print 'Confusion Matrix -'
        print avg_conf_mat

    prec_sv_0, = plt.plot(np.arange(1, max_deg), acc_sep, '-o')
    prec_sv_1, = plt.plot(np.arange(1, max_deg), acc_nsep, '-o')
    prec_sv_2, = plt.plot(np.arange(1, max_deg), acc_bn, '-o')
    prec_sv_3, = plt.plot(np.arange(1, max_deg), acc_bal, '-o')
    plt.title('Accuracy vs Sigma in Polynomial Kernel SVM')
    plt.xlabel('Degree')
    plt.ylabel('Accuracy')
    plt.xticks(range(1, max_deg))
    plt.legend([prec_sv_0, prec_sv_1, prec_sv_2, prec_sv_3], ['Separable Data', 'Non-Separable Data', 'Bank-note Data', 'Balance Data'])
    #plt.legend([prec_sv_0, prec_sv_1], ['Separable Data', 'Non-Separable Data'])
    plt.show()


    #Gaussian Kernel Function

    sv_idx_sep, _, _ = get_svm_gauss(sep_data, sep_labels) #Separable Data

    prec0, = plt.plot(c0_sep[:,0], c0_sep[:,1], 'o')
    prec1, = plt.plot(c1_sep[:,0], c1_sep[:,1], 'o')

    prec_sv, = plt.plot(sep_data[sv_idx_sep, 0], sep_data[sv_idx_sep, 1], '^', color='y')

    plt.title('Linearly Separable Dataset with Support Vectors (Gaussian Kernel)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend([prec0, prec1, prec_sv], ['Class 0', 'Class 1', 'Support Vectors'])
    plt.show()

    sv_idx_nsep, _, _ = get_svm_gauss(non_sep_data, non_sep_labels)

    prec0, = plt.plot(c0_non_sep[:,0], c0_non_sep[:,1], 'o', color='y')
    prec1, = plt.plot(c1_non_sep[:,0], c1_non_sep[:,1], 'o', color='g')

    prec_sv_0, = plt.plot(non_sep_data[sv_idx_nsep[sv_idx_nsep<50], 0], non_sep_data[sv_idx_nsep[sv_idx_nsep<50], 1], '^', color='b')
    prec_sv_1, = plt.plot(non_sep_data[sv_idx_nsep[sv_idx_nsep>50], 0], non_sep_data[sv_idx_nsep[sv_idx_nsep>50], 1], '^', color=(0,0,0))

    plt.title('Linearly Non-Separable Dataset with Support Vectors (Gaussian Kernel)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend([prec0, prec1, prec_sv_0, prec_sv_1], ['Class 0', 'Class 1', 'Support Vectors, Class 0', 'Support Vectors, Class 1'])
    plt.show()


    acc_sep = []
    acc_nsep = []
    acc_bn = []
    acc_bal = []
    for s in np.arange(.1, 5, .1):

        kf = KFold(100, n_folds=nfolds, shuffle=True)
        avg_accuracy =  0.; avg_precision =  np.array([0., 0.]); avg_recall =  np.array([0., 0.]); avg_fscore =  np.array([0., 0.]); avg_conf_mat = np.zeros([2, 2])
        for train_ids, test_ids in kf:
            sv_idx_sep, alpha, w0 = get_svm_gauss(sep_data[train_ids], sep_labels[train_ids], sigma=s)
            preds_sep = svm_gauss_predict(sv_idx_sep, alpha, w0, sep_data[train_ids], sep_labels[train_ids], sep_data[test_ids], sigma=s)
            avg_accuracy += accuracy_score(sep_labels[test_ids], preds_sep)
            precision1, recall1, fscore1, supp1 = precision_recall_fscore_support(sep_labels[test_ids], preds_sep)
            conf_mat = confusion_matrix(sep_labels[test_ids], preds_sep)
            avg_precision += precision1; avg_recall += recall1; avg_fscore += fscore1; avg_conf_mat += conf_mat

        avg_accuracy /= nfolds
        avg_precision /= nfolds
        avg_recall /= nfolds
        avg_fscore /= nfolds
        avg_conf_mat /= nfolds

        acc_sep.append(avg_accuracy)

        print '\nSeparable Dataset Metrics (Gaussian Kernel with sigma =', s, ')-'
        print 'Accuracy =', avg_accuracy
        print 'Precision =', avg_precision
        print 'Recall =', avg_recall
        print 'F-Score =', avg_fscore
        print 'Confusion Matrix -'
        print avg_conf_mat


        kf = KFold(100, n_folds=nfolds, shuffle=True)
        avg_accuracy =  0.; avg_precision =  np.array([0., 0.]); avg_recall =  np.array([0., 0.]); avg_fscore =  np.array([0., 0.]); avg_conf_mat = np.zeros([2, 2])
        for train_ids, test_ids in kf:
            sv_idx_nsep, alpha, w0 = get_svm_gauss(non_sep_data[train_ids], non_sep_labels[train_ids], sigma=s)
            preds_nsep = svm_gauss_predict(sv_idx_nsep, alpha, w0, non_sep_data[train_ids], non_sep_labels[train_ids], non_sep_data[test_ids], sigma=s)
            avg_accuracy += accuracy_score(non_sep_labels[test_ids], preds_nsep)
            precision1, recall1, fscore1, supp1 = precision_recall_fscore_support(non_sep_labels[test_ids], preds_nsep)
            conf_mat = confusion_matrix(non_sep_labels[test_ids], preds_nsep)
            avg_precision += precision1; avg_recall += recall1; avg_fscore += fscore1; avg_conf_mat += conf_mat

        avg_accuracy /= nfolds
        avg_precision /= nfolds
        avg_recall /= nfolds
        avg_fscore /= nfolds
        avg_conf_mat /= nfolds

        acc_nsep.append(avg_accuracy)

        print '\nNon-Separable Dataset Metrics (Gaussian Kernel with sigma =', s, ')-'
        print 'Accuracy =', avg_accuracy
        print 'Precision =', avg_precision
        print 'Recall =', avg_recall
        print 'F-Score =', avg_fscore
        print 'Confusion Matrix -'
        print avg_conf_mat



        x_bn = x_bn[:500] / x_bn.max(axis = 0)
        y_bn = y_bn[:500]
        kf = KFold(x_bn.shape[0], n_folds=nfolds, shuffle=True)
        avg_accuracy =  0.; avg_precision =  np.array([0., 0.]); avg_recall =  np.array([0., 0.]); avg_fscore =  np.array([0., 0.]); avg_conf_mat = np.zeros([2, 2])
        for train_ids, test_ids in kf:
            sv_idx_bn, alpha, w0 = get_svm_gauss(x_bn[train_ids], y_bn[train_ids], sigma=s)
            preds_bn = svm_gauss_predict(sv_idx_bn, alpha, w0, x_bn[train_ids], y_bn[train_ids], x_bn[test_ids], sigma=s)
            avg_accuracy += accuracy_score(y_bn[test_ids], preds_bn)
            precision1, recall1, fscore1, supp1 = precision_recall_fscore_support(y_bn[test_ids], preds_bn)
            conf_mat = confusion_matrix(y_bn[test_ids], preds_bn)
            avg_precision += precision1; avg_recall += recall1; avg_fscore += fscore1; avg_conf_mat += conf_mat

        avg_accuracy /= nfolds
        avg_precision /= nfolds
        avg_recall /= nfolds
        avg_fscore /= nfolds
        avg_conf_mat /= nfolds

        acc_bn.append(avg_accuracy)

        print '\nBanknote Dataset Metrics (Gaussian Kernel with sigma =', s, ')-'
        print 'Accuracy =', avg_accuracy
        print 'Precision =', avg_precision
        print 'Recall =', avg_recall
        print 'F-Score =', avg_fscore
        print 'Confusion Matrix -'
        print avg_conf_mat


        x_bal = x_bal / x_bal.max(axis = 0)
        kf = KFold(x_bal.shape[0], n_folds=nfolds, shuffle=True)
        avg_accuracy =  0.; avg_precision =  np.array([0., 0.]); avg_recall =  np.array([0., 0.]); avg_fscore =  np.array([0., 0.]); avg_conf_mat = np.zeros([2, 2])
        for train_ids, test_ids in kf:
            sv_idx_bal, alpha, w0 = get_svm_gauss(x_bal[train_ids], y_bal[train_ids], sigma=s)
            preds_bal = svm_gauss_predict(sv_idx_bal, alpha, w0, x_bal[train_ids], y_bal[train_ids], x_bal[test_ids], sigma=s)
            avg_accuracy += accuracy_score(y_bal[test_ids], preds_bal)
            precision1, recall1, fscore1, supp1 = precision_recall_fscore_support(y_bal[test_ids], preds_bal)
            conf_mat = confusion_matrix(y_bal[test_ids], preds_bal)
            avg_precision += precision1; avg_recall += recall1; avg_fscore += fscore1; avg_conf_mat += conf_mat

        avg_accuracy /= nfolds
        avg_precision /= nfolds
        avg_recall /= nfolds
        avg_fscore /= nfolds
        avg_conf_mat /= nfolds

        acc_bal.append(avg_accuracy)

        print '\nBalance Dataset Metrics (Gaussian Kernel with sigma =', s, ')-'
        print 'Accuracy =', avg_accuracy
        print 'Precision =', avg_precision
        print 'Recall =', avg_recall
        print 'F-Score =', avg_fscore
        print 'Confusion Matrix -'
        print avg_conf_mat


    prec_sv_0, = plt.plot(np.arange(.1, 5, .1), acc_sep, '-o')
    prec_sv_1, = plt.plot(np.arange(.1, 5, .1), acc_nsep, '-o')
    prec_sv_2, = plt.plot(np.arange(.1, 5, .1), acc_bn, '-o')
    prec_sv_3, = plt.plot(np.arange(.1, 5, .1), acc_bal, '-o')
    plt.title('Accuracy vs Sigma in Gaussian Kernel SVM')
    plt.xlabel('Sigma')
    plt.ylabel('Accuracy')
    plt.legend([prec_sv_0, prec_sv_1, prec_sv_2, prec_sv_3], ['Separable Data', 'Non-Separable Data', 'Bank-note Data', 'Balance Data'])
    plt.show()


    non_sep_data = np.append(c0_non_sep[:10], c1_non_sep, axis=0) 
    non_sep_labels = np.append(-np.ones(10), np.ones(50))

    sv_idx_nsep, _, _ = get_svm_gauss(non_sep_data, non_sep_labels)

    prec0, = plt.plot(c0_non_sep[:10,0], c0_non_sep[:10,1], 'o', color='y')
    prec1, = plt.plot(c1_non_sep[:,0], c1_non_sep[:,1], 'o', color='g')

    prec_sv_0, = plt.plot(non_sep_data[sv_idx_nsep[sv_idx_nsep<10], 0], non_sep_data[sv_idx_nsep[sv_idx_nsep<10], 1], '^', color='b')
    prec_sv_1, = plt.plot(non_sep_data[sv_idx_nsep[sv_idx_nsep>10], 0], non_sep_data[sv_idx_nsep[sv_idx_nsep>10], 1], '^', color=(0,0,0))

    plt.title('Linearly Non-Separable Dataset with Support Vectors (Gaussian Kernel)\nClass 0 contains 1/5th the #training samples as Class 1')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend([prec0, prec1, prec_sv_0, prec_sv_1], ['Class 0', 'Class 1', 'Support Vectors, Class 0', 'Support Vectors, Class 1'])
    plt.show()

    s=.5
    kf = KFold(non_sep_data.shape[0], n_folds=nfolds, shuffle=True)
    avg_accuracy =  0.; avg_precision =  np.array([0., 0.]); avg_recall =  np.array([0., 0.]); avg_fscore =  np.array([0., 0.]); avg_conf_mat = np.zeros([2, 2])
    for train_ids, test_ids in kf:
        sv_idx_nsep, alpha, w0 = get_svm_gauss(non_sep_data[train_ids], non_sep_labels[train_ids], sigma=s)
        preds_nsep = svm_gauss_predict(sv_idx_nsep, alpha, w0, non_sep_data[train_ids], non_sep_labels[train_ids], non_sep_data[test_ids], sigma=s)
        avg_accuracy += accuracy_score(non_sep_labels[test_ids], preds_nsep)
        precision1, recall1, fscore1, supp1 = precision_recall_fscore_support(non_sep_labels[test_ids], preds_nsep)
        conf_mat = confusion_matrix(non_sep_labels[test_ids], preds_nsep)
        avg_precision += precision1; avg_recall += recall1; avg_fscore += fscore1; avg_conf_mat += conf_mat

    avg_accuracy /= nfolds
    avg_precision /= nfolds
    avg_recall /= nfolds
    avg_fscore /= nfolds
    avg_conf_mat /= nfolds

    print '\nNon-Separable Dataset Metrics (Gaussian Kernel with sigma =', s, ')-'
    print 'Class 0 contains 1/5th the #training samples as Class 1'
    print 'Accuracy =', avg_accuracy
    print 'Precision =', avg_precision
    print 'Recall =', avg_recall
    print 'F-Score =', avg_fscore
    print 'Confusion Matrix -'
    print avg_conf_mat
