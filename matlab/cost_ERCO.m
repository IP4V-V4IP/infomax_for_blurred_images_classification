function Cost_T = cost_ERCO(beta, y, H, SNRe, lambda, M, gamma)

    % Entropy cost
    Cost_S = entropy_cost(SNRe, beta);

    % Out-of-range cost
    Cost_R = out_of_range_cost(y, M, H, beta);

    % And the total cost is a weighted sum of the previous
    Cost_T = Cost_S + lambda*Cost_R;

    assignin('base', 'Cost_S', Cost_S);
    assignin('base', 'Cost_R', Cost_R);
    assignin('base', 'Cost_T', Cost_T);

end