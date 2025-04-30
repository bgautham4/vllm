#!/bin/awk -f

BEGIN {
    FPAT = "([^,]+)|(\"[^\"]+\")"
    printf("batch_size kqv_proj rope paged_attention o_proj gated_up_proj down_proj others\n")
    ops[1] = "kqv_proj"
    ops[2] = "rope"
    ops[3] = "paged_attention"
    ops[4] = "o_proj"
    ops[5] = "gated_up_proj"
    ops[6] = "down_proj"

    #Re match op
    ops_re[1] = "(cutlass)|(gemv)|(gemm)"
    ops_re[2] = "rotary"
    ops_re[3] = "attention"
    ops_re[4] = "(cutlass)|(gemv)|(gemm)"
    ops_re[5] = "(cutlass)|(gemv)|(gemm)"
    ops_re[6] = "(cutlass)|(gemv)|(gemm)"

    N_OPS = length(ops)
}

BEGINFILE{
    match(FILENAME, /[0-9]+/)
    bsize = substr(FILENAME, RSTART, RLENGTH)
    delete op_durs
    delete trace_events
}

{
    if (NR==1)
        next
    ts = $NF
    kernel_name = strip_quotes($1)
    dur = $5
    #print dur
    if (kernel_name != "" && dur != "")
        trace_events[ts][kernel_name] = dur
}

ENDFILE {
    op_idx = 1
    asorti(trace_events, ts_sorted)
    for (i in ts_sorted) {
        time = ts_sorted[i]
        for (kernel in trace_events[time]) {
            dur = trace_events[time][kernel]
        }
        if (op_idx > N_OPS) {
            op_idx = 1
        }
        op = ops[op_idx]
        if (match(kernel, ops_re[op_idx]) != 0) {
            op_durs[op] += dur
            ++op_idx
            continue
        }
        op_durs["others"] += dur
    }
    for (op in op_durs) {
        op_durs[op] *= 1e-3
    }
    printf("%d %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n", bsize, op_durs["kqv_proj"], op_durs["rope"], op_durs["paged_attention"], op_durs["o_proj"], op_durs["gated_up_proj"], op_durs["down_proj"], op_durs["others"]);
}

function strip_quotes(str) {
    if (substr(str, 1, 1) != "\"")
        return str
    len = length(str)
    return substr(str, 2, len - 2)
}
