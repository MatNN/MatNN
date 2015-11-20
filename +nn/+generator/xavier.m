function res = xavier(dimensionVector, ~)
    c = single(prod(dimensionVector)/dimensionVector(4));
    res = (rand(dimensionVector, 'single')-single(0.5))*single(2)*sqrt(single(3)/c);
end