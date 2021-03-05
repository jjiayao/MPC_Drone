import casadi as ca
import casadi.tools as ca_tools
import numpy as np
from time import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


def shift_movement(T, t0, x0, u, x_f, f):
    f_value = f(x0, u[:, 0])
    st = x0 + T * f_value.full()
    t = t0 + T
    u_end = np.concatenate((u[:, 1:], u[:, -1:]), axis=1)
    x_f = np.concatenate((x_f[:, 1:], x_f[:, -1:]), axis=1)
    return t, st, u_end, x_f


if __name__ == '__main__':
    T = 0.03
    N = 50
    rob_r = 0.5  # size of robot
    m = 1.535
    g1 = 9.8066
    I1 = 0.029125
    I2 = 0.029125
    I3 = 0.055225
    alpha = 1

    # states 13
    q0 = ca.SX.sym('q0')
    q1 = ca.SX.sym('q1')
    q2 = ca.SX.sym('q2')
    q3 = ca.SX.sym('q3')
    o1 = ca.SX.sym('o1')
    o2 = ca.SX.sym('o2')
    o3 = ca.SX.sym('o3')
    p1 = ca.SX.sym('p1')
    p2 = ca.SX.sym('p2')
    p3 = ca.SX.sym('p3')
    v1 = ca.SX.sym('v1')
    v2 = ca.SX.sym('v2')
    v3 = ca.SX.sym('v3')
    states = ca.vertcat(q0, q1, q2, q3, o1, o2, o3, p1, p2, p3, v1, v2, v3)
    n_states = states.size()[0]

    # controls 4
    t1 = ca.SX.sym('t1')
    t2 = ca.SX.sym('t2')
    t3 = ca.SX.sym('t3')
    th = ca.SX.sym('th')
    controls = ca.vertcat(t1, t2, t3, th)
    n_controls = controls.size()[0]

    # right hand side of the system dynamics
    rhs = ca.vertcat((-q1 * o1 - q2 * o2 - q3 * o3) / 2 - alpha * (q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2 - 1) * q0,
                     (q0 * o1 - q3 * o2 + q2 * o3) / 2 - alpha * (q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2 - 1) * q1,
                     (q3 * o1 + q0 * o2 - q1 * o3) / 2 - alpha * (q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2 - 1) * q2,
                     (-q2 * o1 + q1 * o2 + q0 * o3) / 2 - alpha * (q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2 - 1) * q3,
                     ((I2 - I3) * o2 * o3 + t1) / I1,
                     ((I3 - I1) * o1 * o3 + t2) / I2,
                     ((I1 - I2) * o1 * o2 + t3) / I3,
                     v1,
                     v2,
                     v3,
                     2 * (q0 * q2 + q1 * q3) * th / m,
                     2 * (q2 * q3 - q0 * q1) * th / m,
                     (q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2) * th / m - g1)

    # mapping function
    f = ca.Function('f', [states, controls], [rhs], ['input_state', 'control_input'], ['rhs'])
    U = ca.SX.sym('U', n_controls, N)
    X = ca.SX.sym('X', n_states, N + 1)
    P = ca.SX.sym('P', n_states + n_states)
    con_ref = [0, 0, 0, m * g1]
    Q = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1000, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1000, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1000, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
    R = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 10]])
    obj = 0
    g = [X[:, 0] - P[:13]]
    for i in range(N):
        obj = obj \
              + ca.mtimes([(X[:, i] - P[13:]).T, Q, X[:, i] - P[13:]]) \
              + ca.mtimes([(U[:, i] - con_ref).T, R, (U[:, i] - con_ref)])
        x_next_ = f(X[:, i], U[:, i]) * T + X[:, i]
        g.append(X[:, i + 1] - x_next_)

    # constraints
    obs_x = -8
    obs_y = 1
    obs_z = 0
    obs_x2 = 5
    obs_y2 = -2
    obs_z2 = 0
    obs_x3 = -3
    obs_y3 = -11
    obs_z3 = -1

    obs_r_1 = 5
    obs_r_2 = 5
    obs_r_3 = 5
    for i in range(N + 1):
        g.append(ca.sqrt((X[7, i] - obs_x) ** 2 + (X[8, i] - obs_y) ** 2 + (X[9, i] - obs_z) ** 2) - (obs_r_1 + rob_r))
        g.append(
            ca.sqrt((X[7, i] - obs_x2) ** 2 + (X[8, i] - obs_y2) ** 2 + (X[9, i] - obs_z2) ** 2) - (obs_r_2 + rob_r))
        g.append(
            ca.sqrt((X[7, i] - obs_x3) ** 2 + (X[8, i] - obs_y3) ** 2 + (X[9, i] - obs_z3) ** 2) - (obs_r_3 + rob_r))
    opt_variables = ca.vertcat(ca.reshape(U, -1, 1), ca.reshape(X, -1, 1))

    nlp_prob = {'f': obj, 'x': opt_variables, 'p': P, 'g': ca.vertcat(*g)}
    opts_setting = {
                    'ipopt.print_level': 0,
                    'print_time': 0,
                    'ipopt.acceptable_tol': 1e-8,
                    'ipopt.acceptable_obj_change_tol': 1e-6}

    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)

    lbg = []
    ubg = []
    lbx = []
    ubx = []
    for _ in range(N + 1):
        # dynamics constraints
        lbg.append(0.0)
        lbg.append(0.0)
        lbg.append(0.0)
        lbg.append(0.0)
        lbg.append(0.0)
        lbg.append(0.0)
        lbg.append(0.0)
        lbg.append(0.0)
        lbg.append(0.0)
        lbg.append(0.0)
        lbg.append(0.0)
        lbg.append(0.0)
        lbg.append(0.0)

        ubg.append(0.0)
        ubg.append(0.0)
        ubg.append(0.0)
        ubg.append(0.0)
        ubg.append(0.0)
        ubg.append(0.0)
        ubg.append(0.0)
        ubg.append(0.0)
        ubg.append(0.0)
        ubg.append(0.0)
        ubg.append(0.0)
        ubg.append(0.0)
        ubg.append(0.0)
    for _ in range(N + 1):
        # obstacle constraints
        lbg.append(0.0)
        ubg.append(np.inf)
        lbg.append(0.0)
        ubg.append(np.inf)
        lbg.append(0.0)
        ubg.append(np.inf)

    for _ in range(N):
        # controls 4
        lbx.append(-20)
        lbx.append(-20)
        lbx.append(-20)
        lbx.append(0)

        ubx.append(20)
        ubx.append(20)
        ubx.append(20)
        ubx.append(20)
    for _ in range(N + 1):
        # states 13
        # q0, q1, q2, q3
        lbx.append(1)
        lbx.append(-np.inf)
        lbx.append(-np.inf)
        lbx.append(-np.inf)
        # o1, o2, o3
        lbx.append(-np.inf)
        lbx.append(-np.inf)
        lbx.append(-np.inf)
        # p1, p2, p3
        lbx.append(-20)
        lbx.append(-20)
        lbx.append(-20)
        # v1, v2, v3
        lbx.append(-np.inf)
        lbx.append(-np.inf)
        lbx.append(-np.inf)

        ubx.append(1)
        ubx.append(np.inf)
        ubx.append(np.inf)
        ubx.append(np.inf)

        ubx.append(np.inf)
        ubx.append(np.inf)
        ubx.append(np.inf)

        ubx.append(20)
        ubx.append(20)
        ubx.append(20)

        ubx.append(np.inf)
        ubx.append(np.inf)
        ubx.append(np.inf)

    # Simulation
    t0 = 0.0
    x0 = np.array([1, 0, 0, 0, 0, 0, 0, -15, 0, 0, 0, 0, 0]).reshape(-1, 1)
    x0_ = x0.copy()
    x_m = np.zeros((n_states, N + 1))
    next_states = x_m.copy()
    xs = np.array([1, 0, 0, 0, 0, 0, 0, 11, -2, 0, 0, 0, 0]).reshape(-1, 1)
    u0 = np.array([0, 0, 0, 0] * N).reshape(-1, 4).T  # np.ones((N, 2)) # controls
    x_c = []  # contains for the history of the state
    u_c = []
    t_c = [t0]  # for the time
    xx = []
    sim_time = 20.0
    p1 = []
    p2 = []
    p3 = []
    # start MPC
    mpciter = 0
    main_loop = time()
    # while np.linalg.norm(x0[7:9] - xs[7:9]) > 0.01:
    for _ in range(1):
        c_p = np.concatenate((x0, xs))
        init_control = np.concatenate((u0.T.reshape(-1, 1), next_states.T.reshape(-1, 1)))

        res = solver(x0=init_control, p=c_p, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx)

        estimated_opt = res['x'].full()
        u0 = estimated_opt[:N * n_controls].reshape(N, n_controls).T  # (n_controls, N)
        x_m = estimated_opt[N * n_controls:].reshape(N + 1, n_states).T  # [n_states, N]
        x_c.append(x_m.T)
        u_c.append(u0[:, 0])
        print('T', T)
        print('t0', t0)
        print('x0', x0)
        print('u0', u0)
        print('x_m', x_m)
        print('f', f)
        t0, x0, u0, next_states = shift_movement(T, t0, x0, u0, x_m, f)
        x0 = ca.reshape(x0, -1, 1)
        x0 = x0.full()
        xx.append(x0)
        p1.append(float(xx[mpciter][7]))
        p2.append(float(xx[mpciter][8]))
        p3.append(float(xx[mpciter][9]))
        print('p1:', xx[mpciter][7])
        print('p2:', xx[mpciter][8])
        print('p3:', xx[mpciter][9])
        mpciter = mpciter + 1
        print('mpc_iter:', mpciter)
    main_loop_time = time()
    print('\n\n')
    print('Total time: ', main_loop_time - main_loop)

    # # dataset = [p1, p2, p3]
    # # name = ['p1', 'p2', 'p3']
    # # test = pd.DataFrame(index=name, data=dataset)
    # # test.to_csv('/Users/barry_mac/Desktop/testcsv.csv', encoding='gbk')
