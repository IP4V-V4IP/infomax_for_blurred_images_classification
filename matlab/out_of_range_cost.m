
function Cost_R = out_of_range_cost(y, M, H, beta)
    Hf = reshape(H*beta,size(y,1),size(y,2));

    Y = fft2(y);
    Z = Y.*exp(Hf);
    z = real(ifft2(Z));

    % Out-of-range cost
    Cost_R = sum((2*z(:)-1).^M); % x values outside [0,1] make this to grow quickly

    assignin('base', 'Cost_R', Cost_R);

end