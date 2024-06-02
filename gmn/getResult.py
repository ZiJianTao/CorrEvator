import tensorflow as tf
import sys
import pickle as pkl
from utils import *
from models import *
from layers import *
import collections
import time
import random
import copy
from tqdm import tqdm
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix

from gmn.layers import GraphEncoder
from repo_name import *

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

GraphData = collections.namedtuple(
    "GraphData",
    ["from_idx", "to_idx", "node_features", "edge_features", "graph_idx", "n_graphs"],
)

f_result = open("result.txt", "w", encoding="UTF-8")

fileName = ""
modelName = ""
REPO_ID = 0
test_graphs = []

def exact_hamming_similarity(x, y):
    """Compute the binary Hamming similarity."""
    match = tf.cast(tf.equal(x > 0, y > 0), dtype=tf.float32)
    return tf.reduce_mean(match, axis=1)

def compute_similarity(config, x, y):
    """Compute the distance between x and y vectors.

  The distance will be computed based on the training loss type.

  Args:
    config: a config dict.
    x: [n_examples, feature_dim] float tensor.
    y: [n_examples, feature_dim] float tensor.

  Returns:
    dist: [n_examples] float tensor.

  Raises:
    ValueError: if loss type is not supported.
  """
    if config["training"]["loss"] == "margin":
        # similarity is negative distance
        return -euclidean_distance(x, y)
    elif config["training"]["loss"] == "hamming":
        return exact_hamming_similarity(x, y)
    else:
        raise ValueError("Unknown loss type %s" % config["training"]["loss"])
'''
def auc(scores, labels, **auc_args):
    """Compute the AUC for pair classification.

  See `tf.metrics.auc` for more details about this metric.

  Args:
    scores: [n_examples] float.  Higher scores mean higher preference of being
      assigned the label of +1.
    labels: [n_examples] int.  Labels are either +1 or -1.
    **auc_args: other arguments that can be used by `tf.metrics.auc`.

  Returns:
    auc: the area under the ROC curve.
  """
    scores_max = tf.reduce_max(scores)
    scores_min = tf.reduce_min(scores)
    # normalize scores to [0, 1] and add a small epislon for safety
    scores = (scores - scores_min) / (scores_max - scores_min + 1e-8)

    labels = (labels + 1) / 2
    # The following code should be used according to the tensorflow official
    # documentation:
    # value, _ = tf.metrics.auc(labels, scores, **auc_args)

    # However `tf.metrics.auc` is currently (as of July 23, 2019) buggy so we have
    # to use the following:
    _, value = tf.metrics.auc(labels, scores, **auc_args)
    return value
'''
def evaluation_metrics(y_trues, y_pred_probs):
    y_trues = [(label + 1) / 2 for label in y_trues]
    #print("y_trues",y_trues)
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

"""Build the model"""
def reshape_and_split_tensor(tensor, n_splits):
    """Reshape and split a 2D tensor along the last dimension.

  Args:
    tensor: a [num_examples, feature_dim] tensor.  num_examples must be a
      multiple of `n_splits`.
    n_splits: int, number of splits to split the tensor into.

  Returns:
    splits: a list of `n_splits` tensors.  The first split is [tensor[0],
      tensor[n_splits], tensor[n_splits * 2], ...], the second split is
      [tensor[1], tensor[n_splits + 1], tensor[n_splits * 2 + 1], ...], etc..
  """
    feature_dim = tensor.shape.as_list()[-1]
    # feature dim must be known, otherwise you can provide that as an input
    assert isinstance(feature_dim, int)
    tensor = tf.reshape(tensor, [-1, feature_dim * n_splits])
    return tf.split(tensor, n_splits, axis=-1)

def build_placeholders(node_feature_dim, edge_feature_dim):
    """Build the placeholders needed for the model.

  Args:
    node_feature_dim: int.
    edge_feature_dim: int.

  Returns:
    placeholders: a placeholder name -> placeholder tensor dict.
  """
    # `n_graphs` must be specified as an integer, as `tf.dynamic_partition`
    # requires so.
    return {
        "node_features": tf.placeholder(tf.float32, [None, node_feature_dim]),
        "edge_features": tf.placeholder(tf.float32, [None, edge_feature_dim]),
        "from_idx": tf.placeholder(tf.int32, [None]),
        "to_idx": tf.placeholder(tf.int32, [None]),
        "graph_idx": tf.placeholder(tf.int32, [None]),
        # only used for pairwise training and evaluation
        "labels": tf.placeholder(tf.int32, [None]),
    }

def fill_feed_dict(placeholders, batch):
    """Create a feed dict for the given batch of data.

  Args:
    placeholders: a dict of placeholders.
    batch: a batch of data, should be either a single `GraphData` instance for
      triplet training, or a tuple of (graphs, labels) for pairwise training.

  Returns:
    feed_dict: a feed_dict that can be used in a session run call.
  """
    if isinstance(batch, GraphData):
        graphs = batch
        labels = None
    else:
        graphs, labels = batch
    #labels = [-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1,-1,1]
    # print(graphs)
    # print(labels)

    feed_dict = {
        placeholders["node_features"]: graphs.node_features,
        placeholders["edge_features"]: graphs.edge_features,
        placeholders["from_idx"]: graphs.from_idx,
        placeholders["to_idx"]: graphs.to_idx,
        placeholders["graph_idx"]: graphs.graph_idx,
    }
    if labels is not None:
        feed_dict[placeholders["labels"]] = labels
    return feed_dict


def build_model(config, node_feature_dim, edge_feature_dim):
    encoder = GraphEncoder(**config["encoder"])
    aggregator = GraphAggregator(**config["aggregator"])
    if config["model_type"] == "embedding":
        model = GraphEmbeddingNet(encoder, aggregator, **config["graph_embedding_net"])
    elif config["model_type"] == "matching":
        model = GraphMatchingNet(encoder, aggregator, **config["graph_matching_net"])
    else:
        raise ValueError("Unknown model type: %s" % config["model_type"])

    training_n_graphs_in_batch = config["training"]["batch_size"]
    if config["training"]["mode"] == "pair":
        training_n_graphs_in_batch *= 2
    elif config["training"]["mode"] == "triplet":
        training_n_graphs_in_batch *= 4
    else:
        raise ValueError("Unknown training mode: %s" % config["training"]["mode"])

    placeholders = build_placeholders(node_feature_dim, edge_feature_dim)

    # training
    model_inputs = placeholders.copy()
    del model_inputs["labels"]
    model_inputs["n_graphs"] = training_n_graphs_in_batch
    graph_vectors = model(**model_inputs)

    if config["training"]["mode"] == "pair":
        x, y = reshape_and_split_tensor(graph_vectors, 2)
        labels = placeholders["labels"]
        sim = compute_similarity(config, x, y)

    return (
        {
            "metrics": {
                "training": {
                    "x": x,
                    "y": y,
                    "sim": sim,
                    "label": labels,
                }
            }
        },
        placeholders,
        model,
    )

def get_test_graphs():

    #prefix = "data/test-dataset/" + repo_br[i][0] + "/"
    prefix = "DATA/RQ1/BATS0.8/"
    filename = prefix + fileName + ".test_graphs"
    test_graphs.append(filename)
    #print(test_graphs)
    #print(len(test_graphs))

dup_in_repo = [166]

"""Main run process"""
if __name__ == "__main__":

    #if len(sys.argv) == 4:
    #    fileName = sys.argv[1]
    #    modelName = sys.argv[2]
    #    REPO_ID = int(sys.argv[3])

    if len(sys.argv) == 3:
        fileName = sys.argv[1]
        modelName = sys.argv[2]
        #thresholdss = float(sys.argv[3])
    get_test_graphs()
    config = get_default_config()
    config["training"]["n_training_steps"] = 2
    tf.reset_default_graph()

    # Set random seeds
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed + 1)
    tf.set_random_seed(seed + 2)

    batch_size = config["training"]["batch_size"]
    print("batch_size", batch_size)
    tensors, placeholders, model = build_model(config, 300, 1)
    accumulated_metrics = collections.defaultdict(list)
    t_start = time.time()
    init_ops = (tf.global_variables_initializer(), tf.local_variables_initializer())

    # If we already have a session instance, close it and start a new one
    if "sess" in globals():
        sess.close()
    saver = tf.train.Saver()
    cfg = tf.ConfigProto()
    cfg.gpu_options.per_process_gpu_memory_fraction = 0.1
    model_path = "model_tzj_1_WS20/" + modelName + "/"
    with tf.Session(config=cfg) as sess:
        if os.path.exists("gmn/" + model_path + 'checkpoint'):
            print("yes")
            saver.restore(sess,"gmn/"+ model_path + 'lyl.GMN-9')
        else:
            print("no")
            init = tf.global_variables_initializer()
            sess.run(init)
        test_graph = []

        aa = 0

        total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0
        total_predictions = []  # 用于计算AUC
        predictions_list = []  # 存储预测值
        labels_list = []  # 存储标签


        for graph in test_graphs:
            #print("aa:", aa)
            aa += 1
            print("aa:", aa)
            with open(graph, 'rb') as f:
                test_graph = pkl.load(f)

            size = len(test_graph)
            #rate1, rate2, rate3, rate4, rate5, = int(size * 0.01), int(size * 0.02), int(size * 0.03), int(size * 0.04), int(size * 0.05)
            rate1, rate2, rate3, rate4, rate5, rate100 = int(size * 0.01), int(size * 0.02), int(size * 0.03), int(size * 0.04), int(size * 0.05), int(size)
            dup_all = dup_in_repo[aa - 1]
            #print("size", size, "dup_all", dup_all,
            #      "rate1", rate1, "rate2", rate2, "rate3", rate3, "rate4", rate4, "rate5", rate5)
            print("size", size, "dup_all", dup_all,
                  "rate1", rate1, "rate2", rate2, "rate3", rate3, "rate4", rate4, "rate5", rate5, "rate100", rate100)

            sim_list = {}
            label_list = {}
            for i in tqdm(range(len(test_graph))):
                batch = test_graph[i]
                #print(batch)
                sim, label = sess.run(
                    [tensors["metrics"]["training"]["sim"], tensors["metrics"]["training"]["label"]],
                    feed_dict=fill_feed_dict(placeholders, batch)
                )

                sim_list[i] = 1 / (1 + ((-sim[0]) ** 0.5))
                label_list[i] = label

                # if sim_list[i] > 0.7 and label == -1:
                #     print("find one in ", graph, ", index = ", i, ", sim = ", sim_list[i])
                # elif sim_list[i] < 0.3 and label == 1:
                #     print("find two in ", graph, ", index = ", i, ", sim = ", sim_list[i])
            print("len(sim_list)", len(sim_list))

            #sim_list_sorted = [(x, y) for x, y in sorted(sim_list.items(), key=lambda x: x[1], reverse=True)][:rate5]
            sim_list_sorted = [(x, y) for x, y in sorted(sim_list.items(), key=lambda x: x[1], reverse=True)][:rate100]

           # print("sim_list_sorted", sim_list_sorted)

            label_list_sorted = []
            for x, y in sim_list_sorted:
                # print(label_list[x])
                label_list_sorted.extend(label_list[x])
            #print("label_list_sorted", label_list_sorted)


            threshold = 0.5 * len(label_list_sorted)
            for i in range(0, len(label_list_sorted)):
                if label_list_sorted[i] == 1:
                    # 计算 TP, FP, Recall
                    if i < threshold:
                        total_tp += 1
                    else:
                        total_fn += 1
                else:
                    # 计算 TN, FP
                    if i < threshold:
                        total_fp += 1
                    else:
                        total_tn += 1
                predictions_list.append(sim_list_sorted[i][1])
                labels_list.append(label_list_sorted[i])

        #print("predictions_list", predictions_list)
        #print("labels_list", labels_list)
        
        '''
        # 在每一轮结束后关闭文件
        f.close()
        # 在下一轮开始前重新打开文件
        with open("data_file_BATS0.8.pkl", "ab") as f:
        # 将文件指针重置到文件的开头
          f.seek(0)

        # 写入当前轮次的结果
          pickle.dump((list(labels_list), list(predictions_list)), f)
        '''


        _, _, _, _, _, f1_quatrain, auc_quatrain = evaluation_metrics(labels_list, predictions_list)
        print("f1_quatrain:", f1_quatrain)
        print("auc_quatrain:", auc_quatrain)