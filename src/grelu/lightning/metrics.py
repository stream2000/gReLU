"""
`grelu.lightning.metrics` contains custom metrics to measure the performance of
sequence-to-function models. These metrics are used in grelu.lightning.

All metrics inherit from the `torchmetrics.Metric` class and have __init__,
update and compute functions defined. All metrics produce an output value
per task, which can optionally be averaged across tasks by setting average=True.
"""

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve
from torchmetrics import Metric
from torchmetrics.utilities.checks import _check_same_shape


class BestF1(Metric):
    """
    Metric class to calculate the best F1 score for each task.

    Args:
        num_labels: Number of tasks
        average: If true, return the average metric across tasks.
            Otherwise, return a separate value for each task

    As input to forward and update the metric accepts the following input:
        preds: Probabilities of shape (N, n_tasks, L)
        target: Ground truth labels of shape (N, n_tasks, L)

    As output of forward and compute the metric returns the following output:
        output: A tensor with the best F1 score
    """

    def __init__(self, num_labels: int = 1, average: bool = True) -> None:
        super().__init__()
        self.add_state("preds", default=torch.empty(0, num_labels), dist_reduce_fx=None)
        self.add_state(
            "target", default=torch.empty(0, num_labels), dist_reduce_fx=None
        )
        self.average = average

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        _check_same_shape(preds, target)
        preds = (
            preds.swapaxes(1, 2).flatten(start_dim=0, end_dim=1).to(torch.float32)
        )  # NxL, n_tasks
        target = (
            target.swapaxes(1, 2).flatten(start_dim=0, end_dim=1).to(torch.float32)
        )  # NxL, n_tasks
        self.preds = torch.vstack([self.preds, preds])
        self.target = torch.vstack([self.target, target])

    def compute(self) -> torch.Tensor:
        best_f1_list = []

        # Compute best F1 per task
        for task in range(self.preds.shape[1]):
            y_pred = self.preds[:, task].detach().cpu().numpy()
            y_true = self.target[:, task].detach().cpu().numpy().astype(int)
            prec, rec, thre = precision_recall_curve(y_true, y_pred)
            f1_scores = 2 * rec * prec / (rec + prec + 1e-20)
            best_f1 = np.nanmax(f1_scores)
            best_f1_list.append(best_f1)

        # Convert to tensor for consistency
        output = torch.tensor(best_f1_list).type(torch.float)

        # Average over tasks if required
        if self.average:
            return output.mean()
        else:
            return output


def _log1p_transform(x: torch.Tensor) -> torch.Tensor:
    """Compress coverage-scale values before a metric consumes them.

    Track targets here are CPM-style counts whose dynamic range spans several
    orders of magnitude, so raw-scale metrics are decided almost entirely by a
    handful of peak bins. Negative values are clamped away because both the
    softplus-activated predictions and the coverage targets are non-negative.
    """
    return torch.log1p(x.clamp(min=0.0))


class MSE(Metric):
    """
    Metric class to calculate the MSE for each task.

    Args:
        num_outputs: Number of tasks
        average: If true, return the average metric across tasks.
            Otherwise, return a separate value for each task
        log_transform: If True, apply log1p to predictions and targets before
            computing the error. Use this for coverage-scale targets, where the
            raw-scale MSE is dominated by a few peak bins.

    As input to forward and update the metric accepts the following input:
        preds: Predictions of shape (N, n_tasks, L)
        target: Ground truth labels (N, n_tasks, L)

    As output of forward and compute the metric returns the following output:
        output: A tensor with the MSE
    """

    def __init__(
        self,
        num_outputs: int = 1,
        average: bool = True,
        log_transform: bool = False,
    ) -> None:
        super().__init__()
        self.add_state(
            "sum_squared_error", default=torch.zeros(num_outputs), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.average = average
        self.log_transform = log_transform

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        _check_same_shape(preds, target)
        if target.dim() > 1:
            self.total += target.shape[0]
        else:
            self.total += len(target)

        preds = preds.to(torch.float32)
        target = target.to(torch.float32)
        if self.log_transform:
            preds = _log1p_transform(preds)
            target = _log1p_transform(target)

        diff = preds - target  # (N, n_tasks, L)
        self.sum_squared_error += diff.square().sum(axis=0).mean(axis=-1)

    def compute(self) -> torch.Tensor:
        # Compute the mean squared error
        output = self.sum_squared_error / self.total

        # Average across tasks if needed
        if self.average:
            return output.mean()
        else:
            return output


class PearsonCorrCoef(Metric):
    """
    Metric class to calculate the Pearson correlation coefficient for each task.

    Accumulation uses float64 sufficient statistics. Coverage targets reach
    ~4e4 while most bins are zero, and a float32 running correlation loses
    enough precision on that range to report a NaN variance for the whole
    validation set.

    Args:
        num_outputs: Number of tasks
        average: If true, return the average metric across tasks.
            Otherwise, return a separate value for each task
        log_transform: If True, apply log1p to predictions and targets before
            correlating. On coverage-scale targets the raw-scale coefficient
            saturates near 1 for any model that merely places the peaks, so
            the log-scale variant is the one that tracks training progress.

    As input to forward and update the metric accepts the following input:
        preds: Predictions of shape (N, n_tasks, L)
        target: Ground truth labels of shape (N, n_tasks, L)

    As output of forward and compute the metric returns the following output:
        output: A tensor with the Pearson coefficient.
    """

    def __init__(
        self,
        num_outputs: int = 1,
        average: bool = True,
        log_transform: bool = False,
    ) -> None:
        super().__init__()
        zeros = torch.zeros(num_outputs, dtype=torch.float64)
        for name in ("sum_x", "sum_y", "sum_xx", "sum_yy", "sum_xy"):
            self.add_state(name, default=zeros.clone(), dist_reduce_fx="sum")
        self.add_state(
            "n_obs", default=torch.zeros(1, dtype=torch.float64), dist_reduce_fx="sum"
        )
        self.num_outputs = num_outputs
        self.average = average
        self.log_transform = log_transform

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        _check_same_shape(preds, target)
        preds = (
            preds.swapaxes(1, 2).flatten(start_dim=0, end_dim=1).to(torch.float64)
        )  # N*L, n_tasks
        target = (
            target.swapaxes(1, 2).flatten(start_dim=0, end_dim=1).to(torch.float64)
        )  # N*L, n_tasks
        if self.log_transform:
            preds = _log1p_transform(preds)
            target = _log1p_transform(target)

        self.sum_x += preds.sum(dim=0)
        self.sum_y += target.sum(dim=0)
        self.sum_xx += preds.square().sum(dim=0)
        self.sum_yy += target.square().sum(dim=0)
        self.sum_xy += (preds * target).sum(dim=0)
        self.n_obs += preds.shape[0]

    def compute(self) -> torch.Tensor:
        n = self.n_obs
        if float(n) < 2:
            output = torch.full(
                (self.num_outputs,), float("nan"), dtype=torch.float64
            )
        else:
            cov = self.sum_xy / n - (self.sum_x / n) * (self.sum_y / n)
            var_x = self.sum_xx / n - (self.sum_x / n).square()
            var_y = self.sum_yy / n - (self.sum_y / n).square()
            denominator = (var_x.clamp(min=0.0) * var_y.clamp(min=0.0)).sqrt()
            # A constant prediction or a constant target leaves the coefficient
            # undefined; report NaN for that task rather than a divide-by-zero.
            output = torch.where(
                denominator > 0,
                cov / denominator,
                torch.full_like(cov, float("nan")),
            ).clamp(-1.0, 1.0)

        output = output.to(torch.float32)
        if self.average:
            return output.mean()
        else:
            return output
