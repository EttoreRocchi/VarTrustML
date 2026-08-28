#!/usr/bin/env bash
#
# run_cross_dataset.sh
# --------------------
# Standalone Phase 2: per-variant cross-dataset, across the 3 samples, using each
# variant's best model (highest mean MCC in the per-sample compare-models runs).
#
# Use this to run the cross-dataset analysis WITHOUT waiting for the combined_*
# compare-models jobs to finish: cross-dataset only depends on the per-sample
# results (HG002/NA12878/REACH_<SV>/model_metrics_comparison.csv), so it is
# independent of the (slow) combined_DEL job.
#
# Prerequisite: the 12 per-sample compare-models runs are complete.
set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

CONT="SVLEN_CALLER,MAXQV,CG_CONTENT,COVERAGE_MOSDEPTH,coverage_inside,mean_insert_inside,sd_insert_inside,mean_mapq_inside,n_clipped_inside,n_split_inside,n_discordant_inside,coverage_left,coverage_right,mean_flank_insert,mean_flank_mapq,DELTA_insert,DELTA_mapq,clipped_ratio,split_ratio,discordant_ratio"
SEED=42
NJOBS="${NJOBS:-16}"
export CONT SEED NJOBS

mkdir -p results_logs

run_cd() {
    local sv="$1"
    local best
    best=$(python scripts/select_best_model.py --results-dir results --variant "$sv" \
        --samples HG002 NA12878 REACH --metric "Matthews Corr. Coef.")
    echo ">>> cross-dataset ${sv} (best=${best})"
    vartrustml cross-dataset "HG002_${sv}.csv" "NA12878_${sv}.csv" "REACH_${sv}.csv" \
        --data-dir data --output-dir "results/cross_dataset_${sv}" --target-column state \
        --continuous "$CONT" --models "$best" \
        --hpo-method grid --seed "$SEED" --n-outer-splits 10 --n-inner-splits 5 \
        --calibrate-model --calibration isotonic --calibration-cv 3 \
        --optimize-threshold --threshold-method auto \
        --bootstrap-iters 1000 --ci-level 0.95 --ci-method bca --nan-strategy median \
        --callers MANTA,SMOOVE,DELLY --default-combinations \
        --cv-scheme "${CV_SCHEME:-both}" \
        --n-jobs "$NJOBS" --verbose 1 \
        > "results_logs/cd_${sv}.log" 2>&1
    echo "    done: cross-dataset ${sv}"
}
export -f run_cd

printf '%s\n' DEL DUP INS INV | xargs -I{} -P 4 bash -c 'run_cd "$@"' _ {}
echo "Cross-dataset done. Results under results/cross_dataset_*/"
