function [connectId, replaceId, outId, srcId] = setConnectAndReplaceData(obj, phase)
    connectId = {};
    replaceId     = {};
    outId  = {};
    srcId  = {};
    for f=fieldnames(phase)'
        face = f{1};
        connectId.(face) = cell(1, numel(obj.data.names));
        replaceId.(face)     = cell(1, numel(obj.data.names)); % if any layer's btm and top use the same name
        outId.(face)  = [];
    end
    for f=fieldnames(phase)'
        face = f{1};
        currentLayerID = phase.(face);
        allBottoms = [];
        allTops    = [];
        srcs = [];
        outs = [];
        for i = currentLayerID
            if ~isempty(obj.net.layers{i}.bottom)
                allBottoms = [allBottoms, obj.net.layers{i}.bottom]; %#ok
                for b = 1:numel(obj.net.layers{i}.bottom)
                    btm = obj.net.layers{i}.bottom(b);
                    if any(obj.net.layers{i}.top == btm)
                        replaceId.(face){btm} = [replaceId.(face){btm}, i];
                    end
                    connectId.(face){btm} = [connectId.(face){btm}, i];


                    % only add first encountered data
                    if all(srcs ~= btm) && all(allTops ~= btm)
                        srcs = [srcs, btm]; %#ok
                    end

                    %delete used tops
                    outs = outs(~ismember(outs, btm));
                end

                
            end
            if ~isempty(obj.net.layers{i}.top)
                allTops = [allTops, obj.net.layers{i}.top]; %#ok
                outs = [outs, obj.net.layers{i}.top]; %#ok
            end
        end

        %totalBlobIDsInCurrentPhase = find(~cellfun('isempty', blobSizes.(face)));
        %allBottoms = unique(allBottoms);
        %outId.(face) = totalBlobIDsInCurrentPhase( ~ismember(totalBlobIDsInCurrentPhase, allBottoms) );
        %srcId.(face) = totalBlobIDsInCurrentPhase( ~ismember(totalBlobIDsInCurrentPhase, allTops) );
        outId.(face) = outs;
        srcId.(face) = srcs;
    end
end