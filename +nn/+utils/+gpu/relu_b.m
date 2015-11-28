function o = relu_b(in, out_diff, zero)
    o = arrayfun(@relu_kernel_b, in, out_diff, zero);
end

function o = relu_kernel_b(b, tf, z)
    if b > z
        o = tf;
    else
        o = z;
    end
end