
function y_hat = deblur(y, residual, Nor, beta, H)

    [Ny,Nx] = size(y);
    Hf = exp(reshape(H*beta,Ny,Nx));
    Hf = conv2([1 4 6 4 1]/16,[1 4 6 4 1]/16,fftshift(Hf),'same');
    Hf = conv2([1 4 6 4 1]/16,[1 4 6 4 1]/16,Hf,'same');
    Hf = fftshift(Hf);
    Hf = Hf/Hf(1,1);

    Y = fft2(y);
    Z = Y.*Hf;
    z = real(ifft2(Z));
    z = z + residual/Nor;
    y_hat = min(max(z,0),1);

    assignin('base', 'y_hat', y_hat);


end