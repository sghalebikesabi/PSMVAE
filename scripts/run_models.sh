
set -e



source venv/bin/activate
for ((seed=0; seed<=4; seed++))
do
    for input in breast letter wine letter adult credit spam 
    do
        for miss_type in MCAR MNAR1var #MAR MNAR1var MNAR1varMCAR MNAR2var
        do
            for miss_ratio in 20
            do
                for model_class in PSMVAE_a PSMVAE_b PSMVAEwoM DLGMVAE GMVAE VAE missForest mice gain mean
                do
                    data_file="data/${input}/data_${seed}"
                    miss_file="data/${input}/miss_data/${miss_type}_uniform_frac_${miss_ratio}_seed_${seed}"
                    wandb_run_name="${model_class}_${input}_${miss_type}_uniform_frac_${miss_ratio}_seed_${seed}"
                    python src/main.py --wandb-tag 'run_new' --max-epochs 10 --num-samples 100 --compl-data-file "${data_file}" --miss-data-file "${miss_file}" --seed "${seed}" --model-class "${model_class}" --wandb-run-name ${wandb_run_name}
                done
            done
        done
    done
done

