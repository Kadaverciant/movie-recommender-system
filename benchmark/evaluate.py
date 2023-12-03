import pandas as pd
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import argparse

DEVICE = "cuda"  if torch.cuda.is_available() else 'cpu'

Ks = [5, 10, 20, 50]
DEFAULT_MODEL_PATH = "./models/checkpoint_190_model"
DEFAULT_DATASET_PATH = "./benchmark/data/test_dataset.csv"
DEFAULT_TARGETS_PATH = "./benchmark/data/test_targets.csv"
DEFAULT_MASKS_PATH = "./benchmark/data/test_masks.csv"

def load_data(dataset_path, targets_path, masks_path):
    """
    Loads data from files to pandas data frames
    """
    test_dataset = pd.read_csv(dataset_path)
    test_targets = pd.read_csv(targets_path)
    test_masks = pd.read_csv(masks_path)["movie_ids"]
    return test_dataset, test_targets, test_masks


def convert_mask(train_masks):
    """
    Converts string to list of ints.
    We need that function since pandas stores lists of ints as strings in csv.
    """
    ans = []
    for elem in train_masks:
        ans.append([int(x) for x in elem[1:-1].split(", ")])
    return ans


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


class RecSys(nn.Module):
    def __init__(self):
        super(RecSys, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1705, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
        )
        self.head = nn.Linear(2048, 1682)

    def forward(self, x):
        deep_logits = self.linear_relu_stack(x)
        total_logits = self.head(deep_logits)
        return F.sigmoid(total_logits)


def build_mask(size, offset, indexes):
    """
    Builds boolean mask from indexes
    """
    mask = [True for _ in range(size)]
    for elem in indexes:
        mask[elem + offset - 1] = False
    return torch.tensor(mask)


def mask_target(target, masks):
    """
    Applies mask on whole list of tensors
    """
    masked = []
    for i in range(len(target)):
        elem = target[i]
        mask = build_mask(len(elem), 0, masks[i]).to(DEVICE)
        masked_elem = torch.masked_select(elem, mask)
        masked.append(masked_elem)
    return torch.stack(masked)


def mask_one_row(target, mask):
    """
    Applies mask on only one tensor
    """
    torch_mask = build_mask(len(target), 0, mask).to(DEVICE)
    return torch.masked_select(target, torch_mask)


def evaluate_model(model, X_test, Y_test_masked, test_masks, ks):
    """
    Calculates metrics for given model
    """
    actual = []
    predictions = []
    for i in range(len(Y_test_masked)):
        probs = model(X_test[i])
        probs = mask_one_row(probs, test_masks[i])
        temp = probs.detach().cpu().numpy()
        act = Y_test_masked[i].detach().cpu().numpy()
        actual.append(np.argsort(act)[len(act) - int(sum(act)) :])
        predictions.append(np.argsort(temp)[::-1])

    ans = []
    for k in ks:
        elem = mapk(actual, predictions, k)
        ans.append(elem)
    return ans


def show_metrics(metrics, ks):
    """
    Prints given metrics
    """
    for i, k in enumerate(ks):
        print(f"K={k}\tMAP@K: {metrics[i]}")


def evaluate():
    """Evaluate model"""
    parser = argparse.ArgumentParser(description="Evaluates given model")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        dest="model",
        default=DEFAULT_MODEL_PATH,
        help="path to pytorch model",
    )
    parser.add_argument(
        "-K",
        "--Ks",
        type=int,
        nargs="+",
        dest="Ks",
        default=Ks,
        help=f"k values for MAP@K (default: {Ks})",
    )
    parser.add_argument(
        "-d",
        "--dataset-path",
        type=str,
        dest="dataset_path",
        default=DEFAULT_DATASET_PATH,
        help=f"path to load test dataset from (default: {DEFAULT_DATASET_PATH})",
    )
    parser.add_argument(
        "-t",
        "--targets-path",
        type=str,
        dest="targets_path",
        default=DEFAULT_TARGETS_PATH,
        help=f"path to load targets from (default: {DEFAULT_TARGETS_PATH})",
    )
    parser.add_argument(
        "-masks",
        "--masks-path",
        type=str,
        dest="masks_path",
        default=DEFAULT_MASKS_PATH,
        help=f"path to load masks from (default: {DEFAULT_MASKS_PATH})",
    )

    namespace = parser.parse_args()
    (
        model_path,
        ks,
        dataset_path,
        targets_path,
        masks_path,
    ) = (
        namespace.model,
        namespace.Ks,
        namespace.dataset_path,
        namespace.targets_path,
        namespace.masks_path,
    )
    test_dataset, test_targets, test_masks = load_data(dataset_path, targets_path, masks_path)
    test_masks = convert_mask(test_masks.tolist())
    X_test = torch.Tensor(test_dataset.values).to(DEVICE)
    Y_test = torch.Tensor(test_targets.values).to(DEVICE)
    Y_test_masked  = mask_target(Y_test, test_masks)
    model = RecSys().to(DEVICE)
    model = torch.load(model_path).to(DEVICE)
    best_model_metrics = evaluate_model(model, X_test, Y_test_masked, test_masks, ks)
    show_metrics(best_model_metrics, ks)     


if __name__ == "__main__":
    evaluate()
