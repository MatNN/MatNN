function data = accumulateData(faceOpt, data, l, varargin)
dzdxEmpty = ~cellfun('isempty', varargin);

for b = find(dzdxEmpty)
    if any(data.connectId.(faceOpt.name){l.bottom(b)} == l.no) && ~any(data.replaceId.(faceOpt.name){l.bottom(b)} == l.no) && data.diffCount(l.bottom(b))
        data.diff{l.bottom(b)} = data.diff{l.bottom(b)} + varargin{b};
        data.diffCount(l.bottom(b)) = data.diffCount(l.bottom(b))+1;
    else
        data.diff(l.bottom(b)) = varargin(b);
        data.diffCount(l.bottom(b)) = 1;
    end
end

end