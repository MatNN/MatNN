function out = size4D(in)
    out = [1,1,1,1];
    s = size(in);
    out(1:numel(s)) = s;
end