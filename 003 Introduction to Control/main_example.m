clear; close all; clc
%%
init = [4,0]';
[t,X] = ode45(@(t,X) system_fun(t,X), [0,10],init,'.');

figure(1)
plot(t,X(:,1)); hold on; plot(t,X(:,2));
legend('x','dx');
xlabel('time (s)')
