function o = meanClassAccuracy()
%MEANCLASSLOSS Compute mean class accuracy for you
    o.setup    = @setup;
    o.forward  = @forward;
    o.backward = @backward;

    % save weights, 


    function resource = setup(layerParam)

    end


    function blob = forward(layerParam)

    end


    function backward(layerParam)

    end

end