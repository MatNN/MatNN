function res = identity(dimensionVector, param)
%IDENTITY:
% Parameter:
% scale
  default_param.scale = 1;
  p = nn.utils.vararginHelper(default_param, param);
  tmp = eye([dimensionVector(1), dimensionVector(2)], 'single');
  res = repmat(tmp, 1, 1, dimensionVector(3), dimensionVector(4))*p.scale;
end