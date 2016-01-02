function run(obj)

% Calculate current flow, flow iteration
onceRunIterNum = 0;
for f = obj.flowOrder
    onceRunIterNum = onceRunIterNum + obj.flow.(f{1}).opts.iter;
end
if ~isempty(obj.globalIter)
    currentTotalIter = obj.globalIter+1; % continued from saved (iter+1)
else
    currentTotalIter = 1;
end

% calculate next iter
iter = floor((currentTotalIter-1) / onceRunIterNum)+1;
modIter = mod(currentTotalIter-1, onceRunIterNum)+1;

% Find current flow and current flow iter
for f = 1:numel(obj.flowOrder)
    if modIter > obj.flow.(obj.flowOrder{f}).opts.iter
        modIter = modIter-obj.flow.(obj.flowOrder{f}).opts.iter;
    else
        currentFace = f;
        currentIter = modIter;
        break;
    end
end
clearvars modIter;

% Start main loop
startTime = tic;
firstLoad = true;
globalIterNum = currentTotalIter;
for i = iter:obj.repeat
    if firstLoad
        f = currentFace;
        firstLoad = false;
    else
        f = 1;
    end
    while(f<=numel(obj.flowOrder))
        obj.printBar('l', 'Flow %s ', obj.flowOrder{f});
        obj.runFlow(obj.flow.(obj.flowOrder{f}).opts, obj.flow.(obj.flowOrder{f}).layers, i, globalIterNum, currentIter);
        globalIterNum = globalIterNum + (obj.flow.(obj.flowOrder{f}).opts.iter-currentIter)+1;
        f = f+1;
        currentIter = 1;
    end
end

% Report training/evaluation time
startTime = toc(startTime);
fprintf('Total running time: %.2fs\n', startTime);

end
