import numpy as np
import torch

class IoUMetric:
    def __init__(self, num_classes=3):
        self.results = []
        self.num_classes = num_classes

    def process(self, pred_label, label):
        
        intersect = pred_label[pred_label == label]
        # Calculate the histogram of intersected values between the predicted labels and ground truth labels for each class
        area_intersect = np.histogram(intersect, bins=self.num_classes, range=(0, self.num_classes - 1))[0]
        area_pred_label = np.histogram(pred_label, bins=self.num_classes, range=(0, self.num_classes - 1))[0]
        area_label = np.histogram(label, bins=self.num_classes, range=(0, self.num_classes - 1))[0]
        area_union = area_pred_label + area_label - area_intersect

        self.results.append((area_intersect, area_union, area_pred_label, area_label))

    def compute_metrics(self):
        # shape: [num_classes, 4, k_samples]
        results = np.array(self.results).T 
        metrics = []
        for i in range(self.num_classes):
            i_metrics = self.compute_metrics_oneclass(results[i])
            print(f'IoU:{i_metrics[0]:.4f} Acc:{i_metrics[1]:.4f} Dice:{i_metrics[2]:.4f} Prec:{i_metrics[3]:.4f} Rec:{i_metrics[4]:.4f}')
            metrics.append(i_metrics) 
        return np.mean([x[0] for x in metrics][1:]), np.mean([x[2] for x in metrics][1:])
        
        
    def compute_metrics_oneclass(self, results):
    
        
        total_area_intersect = results[0].sum()
        total_area_union = results[1].sum()
        total_area_pred_label = results[2].sum()
        total_area_label = results[3].sum()

        iou = total_area_intersect / total_area_union
        acc = total_area_intersect / total_area_label
        dice = 2 * total_area_intersect / (total_area_pred_label + total_area_label)
        precision = total_area_intersect / total_area_pred_label
        recall = total_area_intersect / total_area_label

        metrics = [iou, acc, dice, precision, recall]

        return metrics
