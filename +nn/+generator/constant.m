function res = constant(dimensionVector, param)
    default_param.value = 0;
    p = nn.utils.vararginHelper(default_param, param);
    res = ones(dimensionVector, 'single')*single(p.value);
end