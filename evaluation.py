import pickle
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix


NLP_model_preds = []
NLP_model_ytests = []
with open("data_file_BATS0.8.pkl", "rb") as f:
    while True:
        try:
            labels_list, predictions_list = pickle.load(f)
            NLP_model_preds.extend(predictions_list)
            NLP_model_ytests.extend(labels_list)
        except EOFError:
            break


print("len(NLP_model_preds)",len(NLP_model_preds))
print("len(NLP_model_tests)",len(NLP_model_ytests))
def evaluation_metrics(y_trues, y_pred_probs):
    y_trues = [(label + 1) / 2 for label in y_trues]
    #print("y_trues",y_trues)
    #print("y_pred_probs",y_pred_probs)
    fpr, tpr, thresholds = roc_curve(y_true=y_trues, y_score=y_pred_probs, pos_label=1)
    auc_ = auc(fpr, tpr)
    #print("thresholdss:",thresholdss)
    y_preds = [1 if p >= 0.5
               else 0 for p in y_pred_probs]

    acc = accuracy_score(y_true=y_trues, y_pred=y_preds)
    prc = precision_score(y_true=y_trues, y_pred=y_preds)
    rc = recall_score(y_true=y_trues, y_pred=y_preds)
    f1 = 2 * prc * rc / (prc + rc)


    print('***------------***')
    # print('Evaluating AUC, F1, +Recall, -Recall')
    print('Test data size: {}, Incorrect: {}, Correct: {}'.format(len(y_trues), y_trues.count(0), y_trues.count(1)))
    print('AUC: %f -- F1: %f  -- Accuracy: %f -- Precision: %f ' % (auc_, f1, acc, prc,))
    #print('AUC: %f -- F1: %f ' % (auc_, f1,))

    if y_trues == y_preds:
        tn, fp, fn, tp = 1, 0, 0, 1
    else:
        tn, fp, fn, tp = confusion_matrix(y_trues, y_preds).ravel()
    recall_p = tp / (tp + fn)
    recall_n = tn / (tn + fp)
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    print("tp, tn, fp, fn ", tp ,tn, fp, fn)
    print('AUC: {:.3f}, +Recall: {:.3f}, -Recall: {:.3f}'.format(auc_, recall_p, recall_n))
    print('MCC: %f' % (mcc))
    # return , auc_
    # print('AP: {}'.format(average_precision_score(y_trues, y_pred_probs)))
    return recall_p, recall_n, acc, prc, rc, f1, auc_

_, _, _, _, _, f1_quatrain, auc_quatrain = evaluation_metrics(NLP_model_ytests, NLP_model_preds)