function forward(obj, data, net, face, opts)

for i = net.phase.(face)
    l = net.layers{i};
    l.no = i;
    l.obj.forward(obj, l, opts, data, net);
end

end