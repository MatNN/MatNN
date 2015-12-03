function run(obj)
    if obj.needReBuild
        obj.build();
    end

    % Calculate current phase, phase iteration
    onceRunIterNum = 0;
    for f = obj.phaseOrder
        onceRunIterNum = onceRunIterNum + obj.pha_opt.(f{1}).numToNext;
    end
    if ~isempty(obj.globalIter)
        currentTotalIter = obj.globalIter+1; % continued from saved (iter+1)
    else
        currentTotalIter = 1;
    end

    % calculate next iter
    iter = floor((currentTotalIter-1) / onceRunIterNum)+1;
    modIter = mod(currentTotalIter-1, onceRunIterNum)+1;

    % Find current phase and current phase iter
    for f = 1:numel(obj.phaseOrder)
        if modIter > obj.pha_opt.(obj.phaseOrder{f}).numToNext
            modIter = modIter-obj.pha_opt.(obj.phaseOrder{f}).numToNext;
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
    for i = iter:obj.repeatTimes
        if firstLoad
            f = currentFace;
            firstLoad = false;
        else
            f = 1;
        end
        
        while(f<=numel(obj.phaseOrder))
            disp('==========================================================================');
            obj.runPhase(obj.phaseOrder{f}, i, globalIterNum, currentIter);
            
            globalIterNum = globalIterNum + (obj.pha_opt.(obj.phaseOrder{f}).numToNext-currentIter)+1;
            f = f+1;
            currentIter = 1;
        end
        
    end

    % Report training/evaluation time
    startTime = toc(startTime);
    fprintf('Total running time: %.2fs\n', startTime);
end