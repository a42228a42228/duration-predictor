import evaluate
import numpy as np
import evaluate
from sklearn.metrics import precision_score, recall_score, f1_score

# seqeval = evaluate.load("seqeval")
# precision_metric = evaluate.load("precision")
label_list = ['0', '1']

def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    # print(predictions)
    # print(labels)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    true_predictions = flatten(true_predictions)
    true_labels = flatten(true_labels)
    # print(true_labels)
    # print(true_predictions)
    # results = seqeval.compute(predictions=true_predictions, references=true_labels)
    precision = precision_score(true_labels, true_predictions, average='macro')
    recall = recall_score(true_labels, true_predictions, average='macro')
    f1 = f1_score(true_labels, true_predictions, average='macro')
    accuracy = (np.array(true_labels) == np.array(true_predictions)).mean()
    
    return {
                "eval_precision": precision,
                "eval_recall": recall,
                "eval_f1": f1,
                "eval_accuracy": accuracy,
            }
    
    
# regression problem
# metric = evaluate.load('mse')

# def compute_metrics(eval_preds):
#     logits, labels = eval_preds
#     predictions = logits.squeeze()

#     flatten = lambda x: [z for y in x for z in (flatten(y) if hasattr(y, '__iter__') and not isinstance(y, str) else (y,))]
#     true_labels = flatten([[l for l in label if l != -100] for label in labels])
#     true_predictions = flatten([
#         [p for (p, l) in zip(prediction, label) if l != -100]
#         for prediction, label in zip(predictions, labels)
#     ])

#     all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
#     print(true_labels)
#     print(true_predictions)
#     return {"mse": all_metrics["mse"]}