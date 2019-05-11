reset
set terminal pdfcairo monochrome enhanced dashed font "Times-Roman,20" linewidth 2
set output "fprhp_unpatched.pdf"
set key samplen 2 top center horizontal spacing .75 width -1 font ",18"
set xlabel "Num of Attack" offset 0,0.5
set ylabel "fpr(%)" offset 1,0
unset grid
set xtics 2
set ytics 20
set yrange [-10:100]
set xrange [0:16]


set style data lines


plot "fpr1_100" u 1:2 title "BiDi" w lines, "fpr1_100" u 1:3 title "Ngram" w lines, "fpr1_100" u 1:4 title "ens-SVM" w lines
#, "acc" u 1:4 title "ens-SVM" w lines
set terminal xterm
set output
unset terminal
replot
replot
