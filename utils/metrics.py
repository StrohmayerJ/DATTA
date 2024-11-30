import torch

# Recall
def recallMetric(confusion_matrix):
    recall = confusion_matrix.diag() / confusion_matrix.sum(1)
    recall = recall[~torch.isnan(recall)].mean()
    return recall.item()

# Precision
def precisionMetric(confusion_matrix):
    precision = confusion_matrix.diag() / confusion_matrix.sum(0)
    precision = precision[~torch.isnan(precision)].mean()
    return precision.item()

# Confusion matrix
def create_confusion_matrix(pC, tC, num_classes):
    predicted_classes = torch.argmax(pC, dim=1)
    confusion_matrix = torch.zeros(num_classes, num_classes, device=pC.device)
    for t, p in zip(tC, predicted_classes):
        confusion_matrix[t.item(), p.item()] += 1
    return confusion_matrix.cpu()