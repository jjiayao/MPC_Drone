addpath('/Users/barry_mac/Desktop/casadi-osx-matlabR2015a-v3.5.5')
import casadi.*
clear all
clc

%% Parameters
dt = 0.05;
N = 25;
rob_r = 0.5; % size of robot
m = 1.535;
g1 = 9.8066;
I1 = 0.029125; I2 = 0.029125; I3 = 0.055225;
alpha = 1;

%% system states, 13
q0 = SX.sym('q0'); q1 = SX.sym('q1'); q2 = SX.sym('q2'); q3 = SX.sym('q3');
o1 = SX.sym('o1'); o2 = SX.sym('o2'); o3 = SX.sym('o3');
p1 = SX.sym('p1'); p2 = SX.sym('p2'); p3 = SX.sym('p3'); 
v1 = SX.sym('v1'); v2 = SX.sym('v2'); v3 = SX.sym('v3');
states = [q0;q1;q2;q3;o1;o2;o3;p1;p2;p3;v1;v2;v3];
n_states = length(states);

%% control inputs, 4
t1 = SX.sym('t1'); t2 = SX.sym('t2'); t3 = SX.sym('t3'); f = SX.sym('f');
controls = [t1;t2;t3;f];
n_controls = length(controls);

%% right hand side of the system dynamics
rhs = [(-q1*o1 - q2*o2 - q3*o3) / 2 - alpha * (q0^2 + q1^2 + q2^2 + q3^2 - 1) * q0;
          (q0*o1 - q3*o2 + q2*o3) / 2 - alpha * (q0^2 + q1^2 + q2^2 + q3^2 - 1) * q1;
          (q3*o1 + q0*o2 - q1*o3) / 2 - alpha * (q0^2 + q1^2 + q2^2 + q3^2 - 1) * q2;
          (-q2*o1 + q1*o2 + q0*o3) / 2 - alpha * (q0^2 + q1^2 + q2^2 + q3^2 - 1) * q3;
          ((I2-I3)*o2*o3 + t1) / I1;
          ((I3-I1)*o1*o3 + t2) / I2;
          ((I1-I2)*o1*o2 + t3) / I3;
           v1;
           v2;
           v3;
           2 * (q0*q2+q1*q3) * f / m;
           2 * (q2*q3-q0*q1) * f / m;
           (q0^2 - q1^2 - q2^2 + q3^2) * f / m - g1];

%% nonlinear mapping function f(x,u)
f = Function('f', {states,controls}, {rhs});

% 4*N, Predicted controls
U = SX.sym('U', n_controls, N);
% initial states + reference states
P = SX.sym('P', n_states + n_states);
% controls reference
con_ref = [0;0;0;m*g1];
% 13*(N+1), Predicted states
X = SX.sym('X', n_states, (N+1));

obj = 0; % objective function
g = []; % constraints vector

Q = diag([1,1,1,1,1,1,1,1000,1000,1000,1,1,1]); % weighting on states
R = diag([1,1,1,10]); % weighting on controls

st = X(:,1); % initial states
g = [g;st-P(1:13)]; % initial condition constraints, P(1:13) initial condition on states

%% objective function
% single step
% Euler method
for k = 1:N
    st = X(:,k); % states
    con = U(:,k); % controls
    obj = obj + (st - P(14:26))'*Q*(st - P(14:26)) + (con-con_ref)'*R*(con-con_ref); % objective function
    st_next = X(:,k+1); % next states
    f_value = f(st,con); % rhs value of state equations
    st_next_euler = st + (dt*f_value); % predicted next states from euler method
    g = [g;st_next-st_next_euler]; % add new dynamic constraints 
end

% Runge-Kutta methods
% for k = 1:N
%     st = X(:,k); 
%     con = U(:,k);
%     obj = obj + (st - P(14:26))'*Q*(st - P(14:26)) + (con-con_ref)'*R*(con-con_ref);
%     st_next = X(:,k+1);
%     k1 = f(st, con); 
%     k2 = f(st + dt/2*k1, con); 
%     k3 = f(st + dt/2*k2, con); 
%     k4 = f(st + dt*k3, con); 
%     st_next_RK4=st +dt/6*(k1 +2*k2 +2*k3 +k4);
%     g = [g;st_next-st_next_RK4]; % compute constraints % new
% end

%% add constraints for collision avoidance
% obstacles position
obs_x1 = -8;
obs_y1 = 1;
obs_z1 = 0;
obs_x2 = 5;
obs_y2 = -2;
obs_z2 = 0;
obs_x3 = -3;
obs_y3 = -11;
obs_z3 = -1;
% obstacles size
obs_r_1 = 5;
obs_r_2 = 5;
obs_r_3 = 5;
% add new elements on g, 1 to N is system dynamic constraints, N+1 to 2N+1
% new constraint
% ball1
for k = 1:N+1
    g = [g; -sqrt((X(8,k)-obs_x1)^2 + (X(9,k)-obs_y1)^2 + (X(10,k)-obs_z1)^2) + rob_r + obs_r_1;];
end
% ball2
for k = 1:N+1
    g = [g; -sqrt((X(8,k)-obs_x2)^2 + (X(9,k)-obs_y2)^2 + (X(10,k)-obs_z2)^2) + rob_r + obs_r_2;];
end
%ball3
for k = 1:N+1
    g = [g; -sqrt((X(8,k)-obs_x3)^2 + (X(9,k)-obs_y3)^2 + (X(10,k)-obs_z3)^2) + rob_r + obs_r_3;];
end

%% Optimization Problem Formulation
% optimization variable contains states and controls
OPT_variables = [reshape(X, 13*(N+1),1); reshape(U,4*N,1)];
nlp_prob = struct('f', obj, 'x', OPT_variables, 'g', g, 'p', P);
% solver setting
opts = struct;
opts.ipopt.max_iter = 50;
opts.ipopt.print_level = 0;
opts.print_time = 0;
opts.ipopt.acceptable_tol = 1e-8;
opts.ipopt.acceptable_obj_change_tol = 1e-4;
solver = nlpsol('solver', 'ipopt', nlp_prob,opts);

args = struct;
% system dynamics
args.lbg(1:13*(N+1)) = 0;
args.ubg(1:13*(N+1)) = 0;
% obstacle constraints
% ball1
args.lbg(13*(N+1)+1:13*(N+1)+(N+1)) = -inf;
args.ubg(13*(N+1)+1:13*(N+1)+(N+1)) = 0;
% ball2
args.lbg(13*(N+1)+(N+1)+1: 13*(N+1)+(N+1)+(N+1)) = -inf;
args.ubg(13*(N+1)+(N+1)+1: 13*(N+1)+(N+1)+(N+1)) = 0;
% ball3
args.lbg(13*(N+1)+(N+1)+(N+1)+1: 13*(N+1)+(N+1)+(N+1)+(N+1)) = -inf;
args.ubg(13*(N+1)+(N+1)+(N+1)+1: 13*(N+1)+(N+1)+(N+1)+(N+1)) = 0;

% bound on q0, q1, q2, q3
args.lbx(1:13:13*(N+1),1) = 1; args.lbx(2:13:13*(N+1),1) = -inf; 
args.lbx(3:13:13*(N+1),1) = -inf; args.lbx(4:13:13*(N+1),1) = -inf;
args.ubx(1:13:13*(N+1),1) = 1; args.ubx(2:13:13*(N+1),1) = inf; 
args.ubx(3:13:13*(N+1),1) = inf; args.ubx(4:13:13*(N+1),1) = inf;
% bound on o1, o2, o3
args.lbx(5:13:13*(N+1),1) = -inf; args.lbx(6:13:13*(N+1),1) = -inf; args.lbx(7:13:13*(N+1),1) = -inf;
args.ubx(5:13:13*(N+1),1) = inf; args.ubx(6:13:13*(N+1),1) = inf; args.ubx(7:13:13*(N+1),1) = inf;
% bound on p1, p2, p3
args.lbx(8:13:13*(N+1),1) = -20; args.lbx(9:13:13*(N+1),1) = -20; args.lbx(10:13:13*(N+1),1) = -20;
args.ubx(8:13:13*(N+1),1) = 20; args.ubx(9:13:13*(N+1),1) = 20; args.ubx(10:13:13*(N+1),1) = 20;
% bound on v1, v2, v3
args.lbx(11:13:13*(N+1),1) = -inf; args.lbx(12:13:13*(N+1),1) = -inf; args.lbx(13:13:13*(N+1),1) = -inf;
args.ubx(11:13:13*(N+1),1) = inf; args.ubx(12:13:13*(N+1),1) = inf; args.ubx(13:13:13*(N+1),1) = inf;
% bound on t1, t2, t3, f
args.lbx(13*(N+1)+1:4:13*(N+1)+4*N,1) = -20; args.ubx(13*(N+1)+1:4:13*(N+1)+4*N,1) = 20;
args.lbx(13*(N+1)+2:4:13*(N+1)+4*N,1) = -20; args.ubx(13*(N+1)+2:4:13*(N+1)+4*N,1) = 20;
args.lbx(13*(N+1)+3:4:13*(N+1)+4*N,1) = -20; args.ubx(13*(N+1)+3:4:13*(N+1)+4*N,1) = 20;
args.lbx(13*(N+1)+4:4:13*(N+1)+4*N,1) = 0; args.ubx(13*(N+1)+4:4:13*(N+1)+4*N,1) = 20;

%% Start optimization
% initialization
t0 = 0;
x0 = [1;0;0;0;0;0;0;-15;0;0;0;0;0]; % initial states
xs = [1;0;0;0;0;0;0;11;-2;0;0;0;0]; % reference states
xx(:,1) = x0; % xx(:,1) initial states for first iterations
t(1) = t0;
u0 = zeros(N,4); % initial controls for N steps
X0 = repmat(x0,1,N+1)'; % initial states for N steps
mpciter = 0;
xx1 = [];
u_cl=[];

% calculation of each steps
tic
while(norm((x0(8:10,:)-xs(8:10,:))) > 0.01) % Distance between present point and goal point
    args.p = [x0;xs]; 
    args.x0 = [reshape(X0',13*(N+1),1); reshape(u0',4*N,1)]; % optimization variables
    % solve NLP
    sol = solver('x0', args.x0, 'lbx', args.lbx, 'ubx', args.ubx, 'lbg', args.lbg, 'ubg', args.ubg, 'p', args.p);
    % outputs from sol
    u = reshape(full(sol.x(13*(N+1)+1:end))',4,N)'; % predicted N steps controls
    xx1(:,1:13,mpciter+1) = reshape(full(sol.x(1:13*(N+1)))',13,N+1)'; % predicted N steps states
    u_cl= [u_cl ; u(1,:)]; % final controls
    t(mpciter+1) = t0;
    [t0, x0, u0] = shift(dt, t0, x0, u, f); % output new initial states and controls 
    xx(:,mpciter+2) = x0; % final states
    X0 = reshape(full(sol.x(1:13*(N+1)))',13,N+1)'; 
    X0 = [X0(2:end,:);X0(end,:)]; % new initial states
    display(mpciter)
    mpciter = mpciter + 1;
end
toc

%% Path plot
figure(1);
scatter3(xx(8,:), xx(9,:), xx(10,:));
hold on
grid on
[x,y,z] = ellipsoid(-8,1,0,5,5,5);
ball1 = mesh(x,y,z);
ball1.FaceColor = 'interp';
ball1.EdgeColor = 'none';
[l,m,n] = ellipsoid(5,-2,0,5,5,5);
ball2 = mesh(l,m,n);
ball2.FaceColor = 'interp';
ball2.EdgeColor = 'none';
axis equal
[o,p,q] = ellipsoid(-3,-11,-1,5,5,5);
ball3 = mesh(o,p,q);
ball3.FaceColor = 'interp';
ball3.EdgeColor = 'none';
axis equal

%% Control, Position, Velocity
figure(2)
% position
subplot(6,1,1)
t = 0:dt:dt*mpciter;
plot(t,xx(8,:),'.-'); 
xlabel('time')
ylabel('x')
grid on

subplot(6,1,2)
plot(t,xx(9,:),'.-') 
xlabel('time')
ylabel('y')
grid on

subplot(6,1,3)
plot(t,xx(10,:),'.-') 
xlabel('time')
ylabel('z')
grid on

% velocity
subplot(6,1,4)
plot(t,xx(11,:),'.-') 
xlabel('time')
ylabel('Vx')
grid on

subplot(6,1,5)
plot(t,xx(12,:),'.-') 
xlabel('time')
ylabel('Vy')
grid on

subplot(6,1,6)
plot(t,xx(13,:),'.-') 
xlabel('time')
ylabel('Vz')
grid on

% thrust
figure(3)
t1 = 0:dt:dt*(mpciter-1);
plot(t1,u_cl(:,4),'.-')
xlabel('time')
ylabel('thrust')
grid on