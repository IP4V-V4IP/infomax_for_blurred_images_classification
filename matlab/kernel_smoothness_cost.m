
function Cost_K = kernel_smoothness_cost(beta)

    aux1 = conv(beta,[-1 2 -1],'valid'); % Laplacian filter, it detects the deviation from a linear response
    aux2 = 6*beta; %+ quadratic term
    Cost_K = sum(aux1.^2) + sum(abs(aux2).^2);

    assignin('base', 'Cost_K', Cost_K);

end