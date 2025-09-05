#set terminal dumb 200,30
set autoscale
set grid
set title 'Grid'
set xlabel 'n'
set ylabel 'z'
#set yrange [-0.0045:-0.0012]
#set xrange [600:]
plot 'grid.out' using 0:2 with lines notitle lw 3
pause -1
