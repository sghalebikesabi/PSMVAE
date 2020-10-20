# input to wrapper
source venv/bin/activate


for seed in 0 1 2 3 4
do
    for m in 3 6 9 12 15 
    do
        for miss in MNAR1var MCAR
        do
            for miss_ratio in 0.20 
            do
                input=credit
                python src/main_preprocess.py --m "${m}" --n 5000 --miss-ratio "${miss_ratio}" --data-file "data/${input}" --miss-type "${miss}" --seed "${seed}" 
            done
        done
    done
done
