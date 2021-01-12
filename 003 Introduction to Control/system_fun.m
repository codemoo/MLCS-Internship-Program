function dX = system_fun(t,X)
    dX = zeros(2,1);
    x = X(1);
    dx = X(2);
    
    u = -4*dx;
    
    dX(1) = dx;
    dX(2) = -x + 2*dx + u;
end