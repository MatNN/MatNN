classdef net < handle
    properties
        name              = [];
        layers            = {}; %layer
        weights           = {}; %weights stores here
        weightsDiff       = {};
        weightsDiffCount  = [];
        phase             = {}; %A structure, each field is a phase name, eg. net.phase.train. And each field contains the IDs of layers.
        noSubPhase        = {};
        
        momentum          = {}; % number and size exactly the same as weights
        learningRate      = []; % learningRate of each weight
        weightDecay       = []; % weight Decay of each weight

        weightsNames      = {}; %weights names here                                           eg. {'conv1_w1', 'relu1', ...}
        weightsNamesIndex = {}; %Inverted index from weight name to weight index. A struct    eg. net.weightsNamesIndex.conv1_w1 = 1, ...
        weightsIsMisc     = []; %Stores res.(***), ***= field name. because there are layers use .weights to store miscs, not weights.

        layerNames        = {}; %Each layer's name
        layerNamesIndex   = {}; %Inverted index from layer name to layer index. A struct
    end
end