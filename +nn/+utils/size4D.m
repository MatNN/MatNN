function out = size4D(in)
    out = size(in);
    ns = numel(out);
    if ns == 3
        out = [out,1];
    elseif ns == 2
        out = [out,1,1];
    elseif ns == 1
        out = [out,1,1,1];
    end
end

% function out = size4D(in)
%     out = [1,1,1,1];
%     s = size(in);
%     out(1:numel(s)) = s;
% end
