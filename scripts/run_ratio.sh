# input to wrapper
source venv/bin/activate


for seed in 0 1 2 3 4
do
    for input in credit 
    do
        for miss in MNAR1var MCAR 
        do
            for miss_ratio in 0.20 0.4 0.6 0.80
            do
                python src/main_preprocess.py --n 5000 --miss-ratio "${miss_ratio}" --data-file "data/${input}" --miss-type "${miss}" --seed "${seed}" 
            done
        done
    done
done
