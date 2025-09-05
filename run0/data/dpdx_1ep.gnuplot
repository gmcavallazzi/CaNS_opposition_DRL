set autoscale
set grid
set title 'dPdx convergence'
set xlabel 'Time'
set ylabel 'dP/dx'
set yrange [-0.004:-0.0015]
set xrange [:149]

while(1) {
    plot 'forcing.out' using 1:2 with lines notitle lw 3
    #plot 'tau.out' using 0:1 with lines notitle lw 3
    pause mouse
    if (MOUSE_KEY == -3) {
        break
    }
    reread
}
