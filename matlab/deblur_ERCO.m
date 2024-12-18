function y_hat = deblur_ERCO(image_filename)

    J = 15; % Number of subbands
    lambda = 1e-4; % funciona bien
    M = 70;
    gamma = 5.0e-3; % bien, con ([-1 2 -1], [0 6 0])
    beta0 = zeros(J-1,1);
    lb = [];  ub = [];
    % No other linear constraints included here (but in the cost function yes)
    A = []; b = []; Aeq = []; beq = [];

    % Define a set of narrow band-pass isotropic filters
    H = filters();

    %% Load the image
    im = imread(fullfile(image_filename));
    im = double(squeeze(im(:,:,1)));

    %% Preprocess the image
    [x, y, residual, Nor] = preprocess(im);
    SNRe = computeSNRe(y, Nor, H);
    fun = @(beta) cost_ERCO([0;beta], y, H, SNRe, lambda, M, gamma);

    [beta, Cost] = fmincon(fun,beta0,A,b,Aeq,beq,lb,ub);
    beta_ext = [0;beta];
    y_hat = deblur(y, residual, Nor, beta_ext, H);

    assignin('base', 'Cost', Cost);
    assignin('base', 'beta', beta_ext)
    assignin('base', 'y_hat', y_hat)


end
