function printBar(~, varargin)% 1=alignment, 2=msg, 3 = varargin
    bar = '==========================================================================\n';
    if nargin == 3
        newStr = varargin{2};
        if strcmpi(varargin{1}, 'l')
            startInd = 1;
        elseif strcmpi(varargin{1}, 'c')
            startInd = max(floor((numel(bar)-1 - numel(newStr))/2),0);
        end
        if numel(newStr) >= (numel(bar)-2)
            fprintf([newStr, '\n']);
        else
            nnStr = bar;
            nnStr(startInd:numel(newStr)+startInd-1) = newStr;
            fprintf(nnStr);
        end
    elseif nargin >=4
        newStr = sprintf(varargin{2}, varargin{3:end});
        if strcmpi(varargin{1}, 'l')
            startInd = 1;
        elseif strcmpi(varargin{1}, 'c')
            startInd = max(floor((numel(bar)-1 - numel(newStr))/2),0);
        end
        if numel(newStr) >= (numel(bar)-2)
            fprintf([newStr, '\n']);
        else
            nnStr = bar;
            nnStr(startInd:numel(newStr)+startInd-1) = newStr;
            fprintf(nnStr);
        end
    else
        fprintf(bar);
    end
end