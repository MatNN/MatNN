function res = uniform(dimensionVector, param)
    default_param.min = -0.01;
    default_param.max = 0.01;
    p = vllab.utils.vararginHelper(default_param, param);
    res = rand(dimensionVector, 'single')*(p.max-p.min)+p.min;
end