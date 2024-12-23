
function SNRe = computeSNRe(y, Nor, H)

    J = size(H,2); % number of subbands
    Yf = fft2(y);
    SigmaNoi = 1/(sqrt(12)*Nor);
    for j = 1:J
        var_noi(j) = mean(H(:,j))*SigmaNoi^2;
        var_sig(j) = mean(H(:,j).*abs(Yf(:)).^2);
        SNRe(j) = var_sig(j)/var_noi(j);
    end    % j subband

    SNRe = SNRe/min(SNRe);

    assignin('base', 'SNRe', SNRe);

end