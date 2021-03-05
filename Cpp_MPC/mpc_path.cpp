#include "casadi/casadi.hpp"
#include "vector"
using namespace casadi;
using namespace std;

int main() {

    clock_t  start, finish;

    float T = 0.1; // time step
    int N = 50; // time horizon
    float rob_r = 0.5; // size of robot
    float m = 1.535; // weight of robot
    float g1 = 9.8066; // gravity
    float I1 = 0.029125;
    float I2 = 0.029125;
    float I3 = 0.055225;
    // system states 13
    SX q0 = SX::sym("q0", 1);
    SX q1 = SX::sym("q1", 1);
    SX q2 = SX::sym("q2", 1);
    SX q3 = SX::sym("q3", 1);
    SX o1 = SX::sym("o1", 1);
    SX o2 = SX::sym("o2", 1);
    SX o3 = SX::sym("o3", 1);
    SX p1 = SX::sym("p1", 1);
    SX p2 = SX::sym("p2", 1);
    SX p3 = SX::sym("p3", 1);
    SX v1 = SX::sym("v1", 1);
    SX v2 = SX::sym("v2", 1);
    SX v3 = SX::sym("v3", 1);
    SX q = vertcat(q0, q1, q2, q3);
    SX o = vertcat(o1, o2, o3);
    SX p = vertcat(p1, p2, p3);
    SX v = vertcat(v1, v2, v3);
    SX states = vertcat(q, o, p, v);
    int n_states = states.numel();
    // controls
    SX t1 = SX::sym("t1", 1);
    SX t2 = SX::sym("t2", 1);
    SX t3 = SX::sym("t3", 1);
    SX th = SX::sym("th", 1);
    SX controls = vertcat(t1, t2, t3, th);
    int n_controls = controls.numel();
    // system dynamics
    SX dot_q0 = (-q1 * o1 - q2 * o2 - q3 * o3) / 2 - (pow(q0, 2) + pow(q1, 2) + pow(q2, 2) + pow(q3, 2) - 1) * q0;
    SX dot_q1 = (q0 * o1 - q3 * o2 + q2 * o3) / 2 - (pow(q0, 2) + pow(q1, 2) + pow(q2, 2) + pow(q3, 2) - 1) * q1;
    SX dot_q2 = (q3 * o1 + q0 * o2 - q1 * o3) / 2 - (pow(q0, 2) + pow(q1, 2) + pow(q2, 2) + pow(q3, 2) - 1) * q2;
    SX dot_q3 = (-q2 * o1 + q1 * o2 + q0 * o3) / 2 - (pow(q0, 2) + pow(q1, 2) + pow(q2, 2) + pow(q3, 2) - 1) * q3;
    SX dot_q = vertcat(dot_q0, dot_q1, dot_q2, dot_q3);
    SX dot_o1 = ((I2 - I3) * o2 * o3 + t1) / I1;
    SX dot_o2 = ((I3 - I1) * o1 * o3 + t2) / I2;
    SX dot_o3 = ((I1 - I2) * o1 * o2 + t3) / I3;
    SX dot_o = vertcat(dot_o1, dot_o2, dot_o3);
    SX dot_p = vertcat(v1, v2, v3);
    SX dot_v1 = 2 * (q0 * q2 + q1 * q3) * th / m;
    SX dot_v2 = 2 * (q2 * q3 - q0 * q1) * th / m;
    SX dot_v3 = (pow(q0, 2) - pow(q1, 2) - pow(q2, 2) + pow(q3, 2)) * th / m - g1;
    SX dot_v = vertcat(dot_v1, dot_v2, dot_v3);
    SX rhs = vertcat(dot_q, dot_o, dot_p, dot_v);
    // mapping function
    Function f = Function("f", {states, controls}, {rhs}, {"input_state", "control_input"}, {"rhs"});
    SX U = SX::sym("U", n_controls, N);
    SX X = SX::sym("X", n_states, N + 1);
    SX P = SX::sym("P", n_states + n_states);
    // float con_ref[4] = {0, 0, 0, m * g1};
    SX obj = SX::sym("obj", 1);
    obj = 0;
    // float obj = 0;
    SX g = SX::sym("g", n_states, N + 1); //13*1
    SX conf = SX::sym("conf", n_states);
    conf = {0, 0, 0, m * g1};
    // first column value of g
    for (int i = 0; i < 13; i++) { g(i, 0) = X(i, 0) - P(i); }

    //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> obj function
    //obj for sates part
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < 13; j++) {
            if (j == 7 || j == 8 || j == 9) { obj = obj + 100 * pow(X(j, i) - P(j + 13), 2); }
            else { obj = obj + pow(X(j, i) - P(j + 13), 2); }
        }
    }
    //obj for controls part
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < 4; j++) {
            if (j == 3) { obj = obj + 10 * pow(U(j, i) - conf(j), 2); }
            else { obj = obj + pow(U(j, i) - conf(j), 2); }
        }
    }
    //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> obj function

    //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> g
    SX next_x = SX::sym("next_x", n_states, 1);
    SX X_1 = SX::sym("X_1", n_states, 1);
    SX X_1_n = SX::sym("X_1_n", n_states, 1);
    SX U_1 = SX::sym("U_1", n_controls, 1);
    SX x_next = SX::sym("x_next", n_states, 1);
    // x_next
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < 13; i++) {
            X_1(i) = X(i, j);
        } // X_1 columns
        for (int i = 0; i < 4; i++) {
            U_1(i) = U(i, j);
        }// U_1 columns
        for (int i = 0; i < 13; i++) {
            X_1_n(i) = X(i, j + 1);
        } // X_1_next columns

        x_next = f(std::vector<SX>{X_1, U_1}).at(0) * T + X_1; // [13, 1]
        for (int i = 0; i < 13; i++) {
            g(i, j + 1) = X_1_n(i) - x_next(i);
        }
    }
    // g is 13,51 equality constraints
    // opt_g, one column of all the equality constraints 13*51,1  663*1
    // opt_g finally go to solver
    SX opt_g = SX::sym("opt_g", n_states * (N + 1));
    int l = 0;
    for (int i = 0; i < N + 1; i++) {
        for (int j = 0; j < n_states; j++) {
            opt_g(l) = g(j, i);
            l++;
        }
    }
    //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> g

    //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> opt_variables
    SX opt_var = SX::sym("opt_var", n_controls * N + n_states * (N + 1), 1);
    int k = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < n_controls; j++) {
            opt_var(k) = U(j, i);
            k++;
        };
    }
    for (int i = 0; i < N + 1; i++) {
        for (int j = 0; j < n_states; j++) {
            opt_var(k) = X(j, i);
            k++;
        }
    }
    //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> opt_variables

    //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> solver setting
    SXDict nlp_prob;
    nlp_prob = {{"f", obj},
                {"x", opt_var},
                {"p", P},
                {"g", opt_g}};
    Dict opts_dict = Dict();
    opts_dict["ipopt.print_level"] = 0;
    opts_dict["print_time"] = 0;
    opts_dict["ipopt.acceptable_tol"] = 100;
    opts_dict["ipopt.acceptable_obj_change_tol"] = 100;
    Function solver;
    solver = nlpsol("solver", "ipopt", nlp_prob, opts_dict);

    //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> solver setting

    //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> constraints
    std::vector<double> lb_g;
    std::vector<double> ub_g;
    std::vector<double> lb_x;
    std::vector<double> ub_x;
    // constraints on g
    for (int i = 0; i < N + 1; i++) {
        int j = 0;
        while (j < 13) {
            lb_g.push_back(0);
            ub_g.push_back(0);
            j++;
        }
    }
    // controls
    for (int i = 0; i < N; i++) {
        // 4 controls
        lb_x.push_back(-20);
        lb_x.push_back(-20);
        lb_x.push_back(-20);
        lb_x.push_back(0);
        ub_x.push_back(20);
        ub_x.push_back(20);
        ub_x.push_back(20);
        ub_x.push_back(20);
    }
    // states
    for (int i = 0; i < N + 1; i++) {
        // q0, q1, q2, q3
        lb_x.push_back(1);
        lb_x.push_back(-inf);
        lb_x.push_back(-inf);
        lb_x.push_back(-inf);
        // o1, o2, o3
        lb_x.push_back(-inf);
        lb_x.push_back(-inf);
        lb_x.push_back(-inf);
        // p1, p2, p3
        lb_x.push_back(-20);
        lb_x.push_back(-20);
        lb_x.push_back(-20);
        // v1, v2, v3
        lb_x.push_back(-inf);
        lb_x.push_back(-inf);
        lb_x.push_back(-inf);

        // q0, q1, q2, q3
        ub_x.push_back(1);
        ub_x.push_back(inf);
        ub_x.push_back(inf);
        ub_x.push_back(inf);
        // o1, o2, o3
        ub_x.push_back(inf);
        ub_x.push_back(inf);
        ub_x.push_back(inf);
        // p1, p2, p3
        ub_x.push_back(20);
        ub_x.push_back(20);
        ub_x.push_back(20);
        // v1, v2, v3
        ub_x.push_back(inf);
        ub_x.push_back(inf);
        ub_x.push_back(inf);
    }
    //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> constraints

    //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> simulation initialization
    float x0_t[13] = {1, 0, 0, 0, 0, 0, 0, -15, 0, 0, 0, 0, 0};
    SX x0 = SX::sym("x0", 13, 1);
    for (int i = 0; i < 13; i++) {
        x0(i) = x0_t[i];
    }
    SX x_m = SX::sym("x_m");
    x_m = casadi::SX::zeros(13, N + 1);
    SX u0 = SX::sym("u0");
    u0 = casadi::SX::zeros(4, N);
    float xs_t[13] = {1, 0, 0, 0, 0, 0, 0, 11, -2, 0, 0, 0, 0};
    SX xs = SX::sym("xs", 13, 1);
    for (int i = 0; i < 13; ++i) {
        xs(i) = xs_t[i];
    }
    SX next_states = x_m;
    std::vector<float> temp;
    SX c_pp = SX::sym("c_pp");
    SX init_control = SX::sym("init_control");
    std::map<std::string, DM> arg;
    std::map<std::string, DM> result;
    SX estimated_opt;
    SX tmp1 = SX::sym("tmp1", n_controls * N, 1);
    SX tmp2 = SX::sym("tmp2", n_states * (N + 1), 1);
    SX u0_0 = SX::sym("u0_0", n_controls, 1);
    SX u0_1TL = SX::sym("u0_1TL", n_controls, N - 1);
    SX u0_L = SX::sym("u0_1TL", n_controls, 1);
    SX next_states_1TL = SX::sym("next_states_1TL", n_states, N);
    SX next_states_L = SX::sym("next_states_L", n_states, 1);
    std::vector<double> judge;
    //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> simulation initialization

    //>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> simulation start
    judge = std::vector<double>(norm_2(x0(7)-xs(7))+norm_2(x0(8)-xs(8))+norm_2(x0(9)-xs(9)));
    int mpciter = 0;


    arg["lbg"] = lb_g; // vector
    arg["lbx"] = lb_x; // vector
    arg["ubx"] = ub_x; // vector
    arg["ubg"] = ub_g; // vector
    start = clock();
    while (judge[0] > 0.01) {
    //for (int jj=0; jj<3; jj++){
        judge = std::vector<double>(norm_2(x0(7) - xs(7)) + norm_2(x0(8) - xs(8)) + norm_2(x0(9) - xs(9)));
        c_pp = vertcat(x0, xs);
        init_control = vertcat(reshape(u0.T(), -1, 1), reshape(next_states.T(), -1, 1));
        arg["x0"] = init_control; //SX
        arg["p"] = c_pp; // SX
        result = solver(arg);
        estimated_opt = result["x"]; // 863 results
        for (int i = 0; i < n_controls * N; i++) {
            tmp1(i) = estimated_opt(i);
        }
        u0 = reshape(tmp1, n_controls, N);
        for (int i = 0; i < n_controls; i++) {
            u0_0(i) = u0(i, 0);
        }
        for (int i = 0; i < n_states * (N + 1); i++) {
            tmp2(i) = estimated_opt(i + n_controls * N);
        }
        x_m = reshape(tmp2, n_states, N + 1);

        ///////// shift movement
        x0 = x0 + f(std::vector<SX>{x0, u0_0}).at(0) * T;  // 13*1
        // u0_1TL
        for (int i = 0; i < n_controls; i++) {
            for (int j = 0; j < N - 1; j++) {
                u0_1TL(i, j) = u0(i, j + 1);
            }
        }
        // u0_L
        for (int i = 0; i < n_controls; i++) {
            u0_L(i) = u0(i, -1);
        }
        u0 = vertcat(u0_1TL.T(), u0_L.T()).T(); // 4*N

        // next_states_1TL
        for (int i = 0; i < n_states; i++) {
            for (int j = 0; j < N; j++) {
                next_states_1TL(i, j) = x_m(i, j + 1);
            }
        }
        // next_states_L
        for (int i = 0; i < n_states; i++) {
            next_states_L(i) = x_m(i, -1);
        }
        next_states = vertcat(next_states_1TL.T(), next_states_L.T()).T(); // 13*N+1
        // now we have x0, u0, next_states
        cout << "iter = " << mpciter++ << endl;
        cout<<"p1: "<<x0(7)<<endl;cout<<"p2: "<<x0(8)<<endl;cout<<"p3: "<<x0(9)<<endl;
    }
    finish = clock();
    double duration = (double)(finish - start) / CLOCKS_PER_SEC;
    cout<<"running time: "<<duration<<" seconds"<<endl;
    return 0;
}
