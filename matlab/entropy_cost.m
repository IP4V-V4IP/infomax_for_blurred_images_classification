
function Cost_S = entropy_cost(SNRe, beta)

    Cost_S = -max(log(SNRe),0)*(beta.*(beta>0)); % Works well
    Cost_S = Cost_S - 100*sum(beta.*(beta<0)); % Negative betas inhibition

    assignin('base', 'Cost_S', Cost_S);

end