"""
Focal Loss for XGBoost

Focal Loss down-weights easy examples and focuses on hard, misclassified cases.
This is particularly useful for class imbalance (148 TDEs vs 2895 non-TDEs).

Formula: FL(p_t) = -(1 - p_t)^γ * log(p_t)
where:
- p_t is the predicted probability for the true class
- γ (gamma) is the focusing parameter (typically 2)
- Higher γ = more focus on hard examples

Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
Used in: Object detection, fraud detection, medical diagnosis
"""

import numpy as np


def focal_loss_objective(gamma=2.0):
    """
    Create a focal loss objective function for XGBoost.

    Args:
        gamma: Focusing parameter (default=2.0)

    Returns:
        Objective function that returns (gradient, hessian)
    """
    def objective(preds, dtrain):
        """
        Focal loss objective function.

        Args:
            preds: Predicted values (raw logits before sigmoid)
            dtrain: DMatrix with labels

        Returns:
            grad: Gradient
            hess: Hessian
        """
        labels = dtrain.get_label()

        # Convert logits to probabilities
        preds_prob = 1.0 / (1.0 + np.exp(-preds))

        # Focal loss components
        # For positive class (y=1): FL = -(1-p)^γ * log(p)
        # For negative class (y=0): FL = -p^γ * log(1-p)

        # Compute p_t (probability of true class)
        p_t = labels * preds_prob + (1 - labels) * (1 - preds_prob)

        # Compute modulating factor: (1 - p_t)^γ
        modulating_factor = (1 - p_t) ** gamma

        # Gradient computation
        # d/dx FL = (1-p_t)^γ * (γ * p_t * log(p_t) + p_t - y)
        # Simplified for binary case:
        grad = modulating_factor * (preds_prob - labels)

        # Add gamma term for better gradient
        grad = grad * (1 + gamma * p_t * np.log(p_t + 1e-7))

        # Hessian (second derivative)
        # Approximate as: modulating_factor * p * (1-p)
        hess = modulating_factor * preds_prob * (1 - preds_prob)
        hess = np.maximum(hess, 1e-16)  # Numerical stability

        return grad, hess

    return objective


def focal_loss_eval(gamma=2.0):
    """
    Create a focal loss evaluation metric for XGBoost.

    Args:
        gamma: Focusing parameter (default=2.0)

    Returns:
        Evaluation function that returns (name, value)
    """
    def eval_metric(preds, dtrain):
        """
        Focal loss evaluation metric.

        Args:
            preds: Predicted values (raw logits)
            dtrain: DMatrix with labels

        Returns:
            name: Metric name
            value: Focal loss value
        """
        labels = dtrain.get_label()

        # Convert logits to probabilities
        preds_prob = 1.0 / (1.0 + np.exp(-preds))

        # Clip probabilities for numerical stability
        preds_prob = np.clip(preds_prob, 1e-7, 1 - 1e-7)

        # Compute p_t
        p_t = labels * preds_prob + (1 - labels) * (1 - preds_prob)

        # Focal loss: -(1-p_t)^γ * log(p_t)
        focal_loss = -(1 - p_t) ** gamma * np.log(p_t)

        return 'focal_loss', float(np.mean(focal_loss))

    return eval_metric


def balanced_focal_loss_objective(gamma=2.0, alpha=0.25):
    """
    Create a balanced focal loss objective (with alpha weighting).

    Alpha balances importance between classes:
    - alpha for positive class (TDEs)
    - (1-alpha) for negative class (non-TDEs)

    Args:
        gamma: Focusing parameter (default=2.0)
        alpha: Class balance weight for positive class (default=0.25)

    Returns:
        Objective function
    """
    def objective(preds, dtrain):
        labels = dtrain.get_label()
        preds_prob = 1.0 / (1.0 + np.exp(-preds))

        # Compute alpha_t (alpha for true class)
        alpha_t = labels * alpha + (1 - labels) * (1 - alpha)

        # Compute p_t
        p_t = labels * preds_prob + (1 - labels) * (1 - preds_prob)

        # Focal loss with alpha
        modulating_factor = alpha_t * (1 - p_t) ** gamma

        # Gradient
        grad = modulating_factor * (preds_prob - labels)
        grad = grad * (1 + gamma * p_t * np.log(p_t + 1e-7))

        # Hessian
        hess = modulating_factor * preds_prob * (1 - preds_prob)
        hess = np.maximum(hess, 1e-16)

        return grad, hess

    return objective
