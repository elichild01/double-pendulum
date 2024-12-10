import numpy as np

def time_to_divergence(y, y_hat, lambda_=0, tol=1e-5):
    """
    Returns min {t : (||theta(t) - theta_hat(t)|| + lambda_ ||theta'(t) - theta_hat'(t)||) >= tol}

    y : True values of theta (first two columns) and theta' (last two columns)
    y_hat : predicted values of y
    t_idx : The index such that y[t_idx] = y(T)
    lambda : regularization parameter
    tol : maximum error
    """
    return min(np.where(one_step_error(y, y_hat, lambda_) >= tol))


def total_divergence_at_time(t_idx, y, y_hat, lambda_=0):
    """
    Returns ||theta - theta_hat|| + lambda_ ||theta' - theta_hat'||

    y : True values of theta (first two columns) and theta' (last two columns)
    y_hat : predicted values of y
    t_idx : The index such that y[t_idx] = y(T)
    lambda : regularization parameter
    """
    theta, theta_hat = y[:t_idx, :2], y_hat[:t_idx, :2]
    theta_prime, theta_prime_hat = y[:t_idx, 2:], y_hat[:t_idx, 2:]
    
    return max(np.linalg.norm(theta - theta_hat) + 
               lambda_ * np.linalg.norm(theta_prime - theta_prime_hat)) 
    

def max_divergence_at_time(t_idx, y, y_hat, lambda_=0):
    """
    Returns max_{t<T} (||theta(t) - theta_hat(t)|| + lambda_ ||theta'(t) - theta_hat'(t)||)
    Agrees with global error (3.11) when lambda_=1 and t_idx is as large as possible

    y : True values of theta (first two columns) and theta' (last two columns)
    y_hat : predicted values of y
    t_idx : The index such that y[t_idx] = y(T)
    lambda : regularization parameter
    """
    return max(one_step_error(y, y_hat, lambda_)[:t_idx]) 
    

def one_step_error(y, y_hat, lambda_=0):
    """
    Returns L(t-1) = ||theta(t) - theta_hat(t)|| + lambda_ ||theta'(t) - theta_hat'(t)||
    Agrees with (3.13) when lambda_=1.  Note the shift in index.

    y : True values of theta (first two columns) and theta' (last two columns)
    y_hat : predicted values of y
    lambda : regularization parameter
    """
    theta, theta_hat = y[:, :2], y_hat[:, :2]
    theta_prime, theta_prime_hat = y[:, 2:], y_hat[:, 2:]
    
    return (np.linalg.norm(theta - theta_hat, axis=1) + 
            lambda_ * np.linalg.norm(theta_prime - theta_prime_hat, axis=1))
