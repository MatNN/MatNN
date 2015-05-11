function res = constant(dimensionVector, param)
    default_param.value = 0;
    p = vllab.utils.vararginHelper(default_param, param);
    res = ones(dimensionVector, 'single')*p.value;
end