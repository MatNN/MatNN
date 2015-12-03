function moveTo(obj, varargin)

if numel(varargin)==0
    if numel(obj.gpus)>0
        dest = 'gpu';
    else
        dest = 'cpu';
    end
else
    dest = varargin{1};
end
for i=1:numel(obj.data.val)
    obj.data.val{i} = moveTo_private(dest,obj.data.val{i});
    obj.data.diff{i} = moveTo_private(dest,obj.data.diff{i});
end
for i=1:numel(obj.net.weights)
    obj.net.weights{i} = moveTo_private(dest,obj.net.weights{i});
end
for i=1:numel(obj.net.momentum)
    obj.net.momentum{i} = moveTo_private(dest,obj.net.momentum{i});
end
obj.net.learningRate = moveTo_private(dest,obj.net.learningRate);
obj.net.weightDecay = moveTo_private(dest,obj.net.weightDecay);

end



function va = moveTo_private(dest, va)

if strcmpi(dest, 'gpu')
    if ~isa(va, 'gpuArray')
        va = gpuArray(single(va));
    end
elseif strcmpi(dest, 'cpu')
    if isa(va, 'gpuArray')
        va = gather(va);
    end
else
    error('Unknown destination.');
end

end