#!/usr/bin/env bash
#
# run_compare_models.sh
# ---------------------
# Standalone Phase 1: compare-models for all 16 subdatasets (12 per-sample +
# 4 combined pools), in parallel, with caller comparison.
#
# Re-running this REUSES existing per-fold checkpoints (it logs
# "Skipping fold N - using checkpoint" and does NOT retrain), so it just
# regenerates the reports + plots with the current code -- fast.
#
# Independent of the cross-dataset phase (different output dirs), so it can run
# concurrently with scripts/run_cross_dataset.sh.
#
# Tune with PAR (concurrent jobs) and NJOBS (cores per job); PAR*NJOBS cores.
set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

CONT="SVLEN_CALLER,MAXQV,CG_CONTENT,COVERAGE_MOSDEPTH,coverage_inside,mean_insert_inside,sd_insert_inside,mean_mapq_inside,n_clipped_inside,n_split_inside,n_discordant_inside,coverage_left,coverage_right,mean_flank_insert,mean_flank_mapq,DELTA_insert,DELTA_mapq,clipped_ratio,split_ratio,discordant_ratio"
MODELS="XGBoost,Random Forest,MLP,CatBoost,Logistic Regression,KNN"
CALLERS="MANTA,SMOOVE,DELLY"
SEED=42
NJOBS="${NJOBS:-16}"
PAR="${PAR:-6}"
export CONT MODELS CALLERS SEED NJOBS

mkdir -p results_logs

run_cm() {
    local stem="$1"
    echo ">>> compare-models ${stem}"
    vartrustml compare-models "${stem}.csv" \
        --data-dir data --output-dir results --target-column state \
        --continuous "$CONT" --models "$MODELS" \
        --hpo-method grid --seed "$SEED" --n-outer-splits 10 --n-inner-splits 5 \
        --calibrate-model --calibration isotonic --calibration-cv 3 \
        --optimize-threshold --threshold-method auto \
        --compare-callers --callers "$CALLERS" --default-combinations \
        --bootstrap-iters 1000 --ci-level 0.95 --ci-method bca --nan-strategy median \
        --comparison-metric "Matthews Corr. Coef." --n-jobs "$NJOBS" --verbose 1 \
        > "results_logs/cm_${stem}.log" 2>&1
    echo "    done: ${stem}"
}
export -f run_cm

printf '%s\n' \
    HG002_DEL HG002_DUP HG002_INS HG002_INV \
    NA12878_DEL NA12878_DUP NA12878_INS NA12878_INV \
    REACH_DEL REACH_DUP REACH_INS REACH_INV \
    combined_DEL combined_DUP combined_INS combined_INV \
    | xargs -I{} -P "$PAR" bash -c 'run_cm "$@"' _ {}

echo "compare-models done. Results under results/<dataset>/"
