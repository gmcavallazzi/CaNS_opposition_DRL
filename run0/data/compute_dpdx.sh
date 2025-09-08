#!/bin/bash

# Simple convergence detection for forcing.out with time range
# Usage: ./compute_dpdx.sh [start_time] [end_time]

start_time=${1:-0}
end_time=${2:-999999999}

awk -v tstart="$start_time" -v tend="$end_time" '
BEGIN { converged = 0; count = 0 }
/^#/ || /^$/ { next }
$1 >= tstart && $1 <= tend {
    count++
    time[count] = $1
    value[count] = $2
    
    if (count >= 50 && !converged) {
        # Check last 30 points for stability
        start = count - 29
        sum = 0
        for (i = start; i <= count; i++) sum += value[i]
        mean = sum / 30
        
        # Calculate std dev
        sum_sq = 0
        for (i = start; i <= count; i++) {
            diff = value[i] - mean
            sum_sq += diff * diff
        }
        std_dev = sqrt(sum_sq / 29)
        
        # Check convergence (relative std dev < 1%)
        if (std_dev / (mean < 0 ? -mean : mean) < 0.01) {
            conv_start = start
            converged = 1
        }
    }
}
END {
    if (converged) {
        sum = 0
        for (i = conv_start; i <= count; i++) {
            sum += value[i]
        }
        printf "%.6e\n", sum/(count-conv_start+1)
    } else {
        print "NOT_CONVERGED"
    }
}
' forcing.out
