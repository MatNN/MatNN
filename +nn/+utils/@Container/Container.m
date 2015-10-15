classdef Container < handle
    
    properties (Access = private)
        thing;
    end
    
    methods
        function obj = Container(v)
            obj.thing = v;
        end

        function setThing(obj, r)
            obj.thing = r;
        end
    
        function res = getThing(obj)
            res = obj.thing;
        end
    
        function clear(obj)
            obj.thing = [];
        end
    end
    
end
