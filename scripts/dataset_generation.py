from itertools import product
import numpy as np
from scipy.integrate import solve_ivp
from tqdm import tqdm

# Set constants/parameters
G = 9.81  # acceleration due to gravity (m/s^2)
T_FINAL = 10
NUM_TSTEPS = 2000
RANDOM_PARAMS = True
DATASET_NAME = 'simulations-random_parameters-gitsized'
# L1 = 1.0  # length of the first rod (m)
# L2 = 1.0  # length of the second rod (m)
# m1 = 1.0  # mass of the first bob (kg)
# m2 = 1.0  # mass of the second bob (kg)
if RANDOM_PARAMS:
    n_simulations = 1000
else:
    length1_vals = [1, 2]
    length2_vals = [1]
    mass1_vals = [1, 2]
    mass2_vals = [1]
    v1_init_vals = [0]
    v2_init_vals = [0]
    num_thetas = 50
    # Initial conditions sampling
    theta1_vals = np.linspace(0, 2*np.pi, num_thetas)
    theta2_vals = np.linspace(0, 2*np.pi, num_thetas)
    n_simulations = len(theta1_vals) * len(theta2_vals) * len(length1_vals) * len(length2_vals) * len(mass1_vals) * len(mass2_vals)


# Equations of motion
def derivatives(t, state, params):
    L1, L2, m1, m2 = params
    theta1, z1, theta2, z2 = state
    delta = theta2 - theta1

    denominator1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta) ** 2
    denominator2 = (L2 / L1) * denominator1

    dtheta1_dt = z1
    dz1_dt = (
        (m2 * L1 * z1 ** 2 * np.sin(delta) * np.cos(delta)
         + m2 * G * np.sin(theta2) * np.cos(delta)
         + m2 * L2 * z2 ** 2 * np.sin(delta)
         - (m1 + m2) * G * np.sin(theta1))
        / denominator1
    )
    dtheta2_dt = z2
    dz2_dt = (
        (-m2 * L2 * z2 ** 2 * np.sin(delta) * np.cos(delta)
         + (m1 + m2) * G * np.sin(theta1) * np.cos(delta)
         - (m1 + m2) * L1 * z1 ** 2 * np.sin(delta)
         - (m1 + m2) * G * np.sin(theta2))
        / denominator2
    )

    return np.array([dtheta1_dt, dz1_dt, dtheta2_dt, dz2_dt])

# Simulation parameters
t_eval = np.linspace(0, T_FINAL, NUM_TSTEPS)


# Store results in a list of dictionaries
print(f'Running simulations...')

def run_simulation(theta1_init, theta2_init, l1, l2, m1, m2, v1, v2):
    state_0 = [theta1_init, v1, theta2_init, v2]
    # Solve the system
    params = [[l1, l2, m1, m2]]
    solution = solve_ivp(
        derivatives, (0, T_FINAL), state_0, t_eval=t_eval, args=params,
    )
    # Return data as dictionary
    return {
        "length1": l1,
        "length2": l2,
        "mass1": m1,
        "mass2": m2,
        "duration": T_FINAL,
        "num_tsteps": NUM_TSTEPS,
        "theta1_init": theta1_init,
        "theta2_init": theta2_init,
        "theta1": solution.y[0].tolist(),
        "theta2": solution.y[2].tolist(),
        "theta1prime": solution.y[1].tolist(),
        "theta2prime": solution.y[3].tolist(),
    }
    
results = []
if RANDOM_PARAMS:
    for _ in tqdm(range(n_simulations)):
        theta1_init, theta2_init = np.random.uniform(-np.pi, np.pi, 2)
        l1, l2 = np.clip(np.random.normal(1, .5, 2), 0.1, 3)
        m1, m2 = np.clip(np.random.normal(1, .5, 2), 0.1, 3)
        v1, v2 = np.random.normal(size=2)
        results.append(run_simulation(theta1_init=theta1_init, theta2_init=theta2_init, l1=l1, l2=l2,
                                      m1=m1, m2=m2, v1=v1, v2=v2))
else:
    for theta1_init, theta2_init, l1, l2, m1, m2, v1, v2 in tqdm(product(theta1_vals, theta2_vals, length1_vals,
                                                                length2_vals, mass1_vals, mass2_vals,
                                                                v1_init_vals, v2_init_vals), total=n_simulations):
        results.append(run_simulation(theta1_init=theta1_init, theta2_init=theta2_init, l1=l1, l2=l2,
                                      m1=m1, m2=m2, v1=v1, v2=v2))
        

# Save to JSON (Git-friendly format)
import json
with open(f"../data/{DATASET_NAME}.json", "w") as f:
    json.dump(results, f, indent=4)