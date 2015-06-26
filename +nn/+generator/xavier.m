function res = xavier(dimensionVector, ~)
    c = prod(dimensionVector)/dimensionVector(4);
    res = (rand(dimensionVector, 'single')-0.5)*2*sqrt(3/c);
end