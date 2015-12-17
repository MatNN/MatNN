function [holdIDs, topBottomIDs, outputId, inputId] = findDataID(~, data, layers)
%FINDDATAID layers ordering must be in execution order.
%
outputId = [];
inputId = [];
holdIDs = false(size(data.val));
topBottomIDs = [];

for i=1:numel(layers)
    btm = layers{i}.bottom;
    outputId = [outputId(~ismember(outputId, btm)), layers{i}.top];

    if all(~ismember(inputId, btm)) && all(~ismember(topBottomIDs, btm)) % all(srcs ~= btm) && all(allTops ~= btm)
        inputId = [inputId, btm];
    end
    holdIDs(unique(layers{i}.holdVars())) = true;
    topBottomIDs = [topBottomIDs, layers{i}.bottom, layers{i}.top];
end
topBottomIDs = unique(topBottomIDs, 'stable');

end
