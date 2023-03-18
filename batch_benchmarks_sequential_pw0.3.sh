conda activate sqt

for nl in {10.0,5.0,1.0} ; do
    echo "Running benchmarks for noise_level=$nl, pulse_width=0.3"
    python benchmarks_sequential.py -nl "$nl" -pw "0.3"
done
