function net = movenet(net, destination)

switch destination
  case 'gpu', moveto = @(a) gpuArray(a) ;
  case 'cpu', moveto = @(a) gather(a) ;
  otherwise, error('Destination must be gpu or cpu.') ;
end
for w=1:numel(net.weights)
    net.weights{w}  = moveto(net.weights{w})  ;
    net.momentum{w} = moveto(net.momentum{w}) ;
end
end
