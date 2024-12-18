
function [x, y, res, Nor] = preprocess(im)

    %% Pre-process the image
    maxsigma = 15;
    im_ant = im;
    PSF = fspecial('gaussian',[],maxsigma);
    aux_I = im-mean(im(:));
    im = edgetaper(im-mean(im(:)),PSF) + mean(im(:));
    res = im_ant - im;
    Nor = 2^ceil(log2(max(im(:))))-1; % Number of levels
    x = im_ant/Nor; % max_value = 1
    y = im/Nor;

    assignin('base', 'x', x);
    assignin('base', 'y', y);
    assignin('base', 'res', res);
    assignin('base', 'Nor', Nor);
end