
from torch.utils.data.dataloader import Dataset
import numpy as np
from scipy.integrate import solve_ivp


class Pendulum_Data(Dataset):
    def __init__(self, min_length=1, max_length=1, G=9.81, delta_t=0.005):
        self.__dict__.update(locals())

    @staticmethod
    def derivatives(t, state, params, G=9.81):
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

    def run_simulation(self, theta1_init, theta2_init, l1, l2, m1, m2, v1, v2, t_eval):
        state_0 = [theta1_init, v1, theta2_init, v2]
        # Solve the system
        params = [[l1, l2, m1, m2]]
        solution = solve_ivp(
            Pendulum_Data.derivatives, (0, t_eval[-1]), state_0, t_eval=t_eval, args=params,
        )
        # Return data as dictionary
        return np.array(
            [[l1] * len(t_eval), [l2] * len(t_eval), [m1] * len(t_eval), [m2] * len(t_eval), *solution.y, t_eval]).T

    def __getitem__(self, i):
        t_final = np.random.randint(self.min_length, self.max_length + 1) * self.delta_t
        theta1_init, theta2_init = np.random.uniform(-np.pi, np.pi, 2)
        l1, l2 = np.clip(np.random.normal(1, .5, 2), 0.1, 3)
        m1, m2 = np.clip(np.random.normal(1, .5, 2), 0.1, 3)
        v1, v2 = np.random.normal(size=2)
        return self.run_simulation(theta1_init=theta1_init, theta2_init=theta2_init, l1=l1, l2=l2, m1=m1, m2=m2, v1=v1,
                                   v2=v2, t_eval=np.arange(0, t_final, self.delta_t))

    def __len__(self):
        return 2 ** 15