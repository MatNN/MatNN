classdef carrier < handle
    %CARRIER Just a handful class helps you carrying data
    %   Put your data in the variable 'data'
    
    properties
        data;
    end

    methods
    
        function obj = carrier(d)
            obj.data = d;
        end

        function d = pop(obj)
            d = obj.data;
            obj.data = [];
        end
    end

end

