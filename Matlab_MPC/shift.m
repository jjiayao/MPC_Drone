function [t0, x0, u0] = shift(T, t0, x0, u,f)
st = x0;
con = u(1,:)'; % choose first step predicted controls as control input 
f_value = f(st,con); % output of rhs
st = st+ (T*f_value); % predicted next step states
x0 = full(st); % predicted next step states

t0 = t0 + T; 
u0 = [u(2:size(u,1),:);u(size(u,1),:)]; % create new initial controls
end