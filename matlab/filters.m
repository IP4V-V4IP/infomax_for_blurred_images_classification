function H = filters()
    Ny = 224;
    Nx = 224;
    J = 15;
    [u,v] = meshgrid(-Ny/2:Ny/2-1,-Ny/2:Ny/2-1);
    u = u/Ny;
    v = v/Nx;
    freq = sqrt(u.^2 + v.^2);
    ff = Ny/3*freq; % 256x256 -> 3x3 degrees, cycles per degree, not used here

    % Number of spectral rings in the maximal circle, including high- and low-pass residuals
    Mp = J-2;
    P = 1; % Overlapping factor (P = 1 => sine and cosine, squared)
    Q = 4; % logarithmic sampling
    ang = 0.56*pi*log2(2*(2^Q-1)*freq+1)/Q*(Mp-1)/P;
    H = zeros(Ny*Nx,Mp+2); % The filters, in the frequency domain
    for m = 0:Mp
            aux = fftshift(cos(ang - pi*m/(2*P)).^(2*P).*double(ang>=pi*(m-P)/(2*P)).*double(ang<=pi*(m+P)/(2*P)));
            H(:,m+1) = aux(:);
    end
    % High-pass residual (HPR)
    m = Mp+1;
    H(:,m+1) = 1 - squeeze(sum(H,2));
end