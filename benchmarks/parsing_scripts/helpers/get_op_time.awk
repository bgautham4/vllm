#!/bin/awk -f

BEGIN {
    FPAT = "([^,]+)|(\"[^\"]+\")"
    printf("batch_size attention linear others\n")
}

BEGINFILE{
    match(FILENAME, /[0-9]+/)
    bsize = substr(FILENAME, RSTART, RLENGTH)
    delete op_times
}

# Identify if the operation is either 1)attention, 2)linear, 3)Other (activations/normalizations)
{
    if (NR==1)
        next
    kernel = strip_quotes($1)
    if (match(kernel, "attention") != 0) {
        op = "attention"
    }
    else if (match(kernel, "(cutlass)|(gemv)|(gemm)") != 0) {
        op = "linear"
    }
    else {
        op = "other"
    }
    dur = $5
    op_times[op] += dur
}

ENDFILE{
    for (op in op_times) {
        op_times[op] *= 1e-3
    }
    printf("%d %.4f %.4f %.4f\n", bsize, op_times["attention"], op_times["linear"], op_times["other"]);
}

function strip_quotes(str) {
    if (substr(str, 1, 1) != "\"")
        return str
    len = length(str)
    return substr(str, 2, len - 2)
}
