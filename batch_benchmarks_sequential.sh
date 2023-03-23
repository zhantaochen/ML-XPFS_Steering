conda activate sqt

for nl in {2.0,1.0,0.5} ; do
    for pw in {0.1,0.2,0.4}; do
        echo "Running benchmarks for noise_level=$nl, pulse_width=$pw"
        python benchmarks_sequential.py -nl "$nl" -pw "$pw"
    done
done
