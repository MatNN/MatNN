function res = gaussian(dimensionVector, param)
%GAUSSIAN
%  Parameter:
%  mean
%  std
    default_param.mean = 0;
    default_param.std = 1;
    p = nn.utils.vararginHelper(default_param, param);
    res = randn(dimensionVector, 'single')*p.std+p.mean;
end