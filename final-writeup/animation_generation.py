from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

G = 9.81  # acceleration due to gravity (m/s^2)

def generate_animation_three_double_pendulum_systems(l1=1.0, l2=1.0, m1=1.0, m2=1.0, t_final=20.0,
                        animation_name='double_pendulum_simulation', initial_offset=1e-6, num_tsteps=2000,
                        theta1_init=np.pi - 1e-1, theta2_init=np.pi + 1e-1, z1_init=0.0, z2_init=0.0):
    # Initial conditions
    theta1_init2 = theta1_init + initial_offset  # initial angle of the first pendulum in the second set of pendulums (rad)
    theta2_init2 = theta2_init + initial_offset  # initial angle of the second pendulum in the second set of pendulums (rad)
    theta1_init3 = theta1_init - initial_offset  # initial angle of the first pendulum in the third set of pendulums (rad)
    theta2_init3 = theta2_init - initial_offset  # initial angle of the second pendulum in the third set of pendulums (rad)
    z1_init = 0.0            # initial angular velocity of the first pendulum in each set (rad/s)
    z2_init = 0.0            # initial angular velocity of the second pendulum in each set (rad/s)

    # Equations of motion
    def derivatives(t, state):
        theta1, z1, theta2, z2 = state
        delta = theta2 - theta1

        denominator1 = (m1 + m2) * l1 - m2 * l1 * np.cos(delta) ** 2
        denominator2 = (l2 / l1) * denominator1

        dtheta1_dt = z1
        dz1_dt = (
            (m2 * l1 * z1 ** 2 * np.sin(delta) * np.cos(delta)
            + m2 * G * np.sin(theta2) * np.cos(delta)
            + m2 * l2 * z2 ** 2 * np.sin(delta)
            - (m1 + m2) * G * np.sin(theta1))
            / denominator1
        )
        dtheta2_dt = z2
        dz2_dt = (
            (-m2 * l2 * z2 ** 2 * np.sin(delta) * np.cos(delta)
            + (m1 + m2) * G * np.sin(theta1) * np.cos(delta)
            - (m1 + m2) * l1 * z1 ** 2 * np.sin(delta)
            - (m1 + m2) * G * np.sin(theta2))
            / denominator2
        )

        return np.array([dtheta1_dt, dz1_dt, dtheta2_dt, dz2_dt])


    state_0 = np.array([theta1_init, z1_init, theta2_init, z2_init])
    state_02 = np.array([theta1_init2, z1_init, theta2_init2, z2_init])
    state_03 = np.array([theta1_init3, z1_init, theta2_init3, z2_init])
    t = np.linspace(0, t_final, num_tsteps)  # time array

    # Solve the system
    solution = solve_ivp(derivatives, (0, t_final), state_0, t_eval=t)
    solution2 = solve_ivp(derivatives, (0, t_final), state_02, t_eval=t)
    solution3 = solve_ivp(derivatives, (0, t_final), state_03, t_eval=t)
    theta1, theta2 = solution.y[0], solution.y[2]
    theta1_2, theta2_2 = solution2.y[0], solution2.y[2]
    theta1_3, theta2_3 = solution3.y[0], solution3.y[2]

    # Convert to Cartesian coordinates
    x1 = l1 * np.sin(theta1)
    y1 = -l1 * np.cos(theta1)
    x2 = x1 + l2 * np.sin(theta2)
    y2 = y1 - l2 * np.cos(theta2)
    x1_2 = l1 * np.sin(theta1_2)
    y1_2 = -l1 * np.cos(theta1_2)
    x2_2 = x1_2 + l2 * np.sin(theta2_2)
    y2_2 = y1_2 - l2 * np.cos(theta2_2)
    x1_3 = l1 * np.sin(theta1_3)
    y1_3 = -l1 * np.cos(theta1_3)
    x2_3 = x1_3 + l2 * np.sin(theta2_3)
    y2_3 = y1_3 - l2 * np.cos(theta2_3)

    # Animation
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-(l1+l2)*1.1, (l1+l2)*1.1)
    ax.set_ylim(-(l1+l2)*1.1, (l1+l2)*1.1)
    ax.set_aspect('equal')
    line, = ax.plot([], [], '-', lw=2, color='blue')
    line2, = ax.plot([], [], '-', lw=2, color='red')
    line3, = ax.plot([], [], '-', lw=2, color='black')
    m1_pt, = ax.plot([], [], 'o', color='blue', markersize=m1*10)
    m2_pt, = ax.plot([], [], 'o', color='blue', markersize=m2*10)
    m1_2_pt, = ax.plot([], [], 'o', color='red', markersize=m1*10)
    m2_2_pt, = ax.plot([], [], 'o', color='red', markersize=m2*10)
    m1_3_pt, = ax.plot([], [], 'o', color='black', markersize=m1*10)
    m2_3_pt, = ax.plot([], [], 'o', color='black', markersize=m2*10)

    def update(frame):
        line.set_data([0, x1[frame], x2[frame]], [0, y1[frame], y2[frame]])
        m1_pt.set_data([x1[frame]], [y1[frame]])
        m2_pt.set_data([x2[frame]], [y2[frame]])
        line2.set_data([0, x1_2[frame], x2_2[frame]], [0, y1_2[frame], y2_2[frame]])
        m1_2_pt.set_data([x1_2[frame]], [y1_2[frame]])
        m2_2_pt.set_data([x2_2[frame]], [y2_2[frame]])
        line3.set_data([0, x1_3[frame], x2_3[frame]], [0, y1_3[frame], y2_3[frame]])
        m1_3_pt.set_data([x1_3[frame]], [y1_3[frame]])
        m2_3_pt.set_data([x2_3[frame]], [y2_3[frame]])
        return line,

    ani = FuncAnimation(fig, update, frames=len(t), interval=(t_final / 2), blit=True)
    ani.save(f'{animation_name}.mp4')
    plt.close(fig)