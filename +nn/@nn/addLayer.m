function addLayer(obj, inputObj)
%ADDLAYER this is a protected method

if isstruct(inputObj)
    l = inputObj;
    try
        tmpHandle = str2func(['nn.layers.', l.type]);
        lobj = tmpHandle();
    catch
        tmpHandle = str2func(l.type);
        lobj = tmpHandle();
    end
    lobj.name  = l.name;
    lobj.origParams = l;
    lobj.net   = obj;

    for fe = {'top','bottom'}
        f = fe{1};
        if isfield(l,f)
            lobj.(f) = l.(f);
        end
    end

elseif isa(inputObj, 'nn.layers.template.BaseLayer')
    lobj = inputObj;
    lobj.net   = obj;
end

if any(strcmp(lobj.name, obj.layerNames))
    error('Layers must have different names.');
end
% Check layer name
if ismember(lobj.name, obj.layerNames)
    error('Layer name ''%s'' is already exists.', lobj.name);
else
    pickInds = [find(cellfun('isempty',obj.layerNames)), numel(obj.layerNames)+1];
    obj.layerNames{pickInds(1)} = lobj.name;
    obj.layerNamesInd.(lobj.name) = pickInds(1);
end

obj.layers{end+1} = lobj;
end