import numpy as np


def one_step_error(y, y_hat, lambda_=0):
    """
    Returns L(t-1) = ||theta(t) - theta_hat(t)|| + lambda_ ||theta'(t) - theta_hat'(t)||
    Agrees with (3.13) when lambda_=1.  Note the shift in index.

    y : True values of theta (first two columns) and theta' (last two columns)
    y_hat : predicted values of y
    lambda : regularization parameter
    """
    theta, theta_hat = y[:, :2], y_hat[:, :2]
    theta_prime, theta_prime_hat = y[:, 2:-1], y_hat[:, 2:-1]

    return (np.linalg.norm(theta - theta_hat, axis=1) +
            lambda_ * np.linalg.norm(theta_prime - theta_prime_hat, axis=1))

def time_to_divergence(y, y_hat, lambda_=0, tol=1e-1):
    """
    Returns min {t : (||theta(t) - theta_hat(t)|| + lambda_ ||theta'(t) - theta_hat'(t)||) >= tol}

    y : True values of theta (first two columns) and theta' (last two columns)
    y_hat : predicted values of y
    t_idx : The index such that y[t_idx] = y(T)
    lambda : regularization parameter
    tol : maximum error
    """
    idx = np.where(one_step_error(y, y_hat, lambda_) >= tol)
    return np.min(idx[0]) if len(idx[0]) != 0 else len(y_hat)

def global_error(t_idx, y, y_hat, lambda_=0):
    """
    Returns max_{t<T} (||theta(t) - theta_hat(t)|| + lambda_ ||theta'(t) - theta_hat'(t)||)
    Agrees with spherical error (3.11) when lambda_=1 and t_idx is as large as possible

    y : True values of theta (first two columns) and theta' (last two columns)
    y_hat : predicted values of y
    t_idx : The index such that y[t_idx] = y(T)
    lambda : regularization parameter
    """
    return max(one_step_error(y, y_hat, lambda_)[:t_idx])