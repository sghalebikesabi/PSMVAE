# input to wrapper
source venv/bin/activate


for seed in 0 1 2 3 4
do
    for input in letter breast adult credit letter spam wine
    do
        for miss in MNAR1var MCAR MNAR1varMCAR MAR MNAR2var #label MAR MNAR1var MNAR1varMCAR MNAR2var
        do
            for miss_ratio in 0.20 0.80
            do
                python src/main_preprocess.py --miss-ratio "${miss_ratio}" --data-file "data/${input}" --miss-type "${miss}" --seed "${seed}" 
            done
        done
    done
done
