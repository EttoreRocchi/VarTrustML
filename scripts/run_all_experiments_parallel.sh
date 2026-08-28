#!/usr/bin/env bash
#
# run_all_experiments_parallel.sh
# -------------------------------
# Throughput-oriented variant of run_all_experiments.sh: runs the independent
# per-subdataset jobs CONCURRENTLY to fill all CPU cores, instead of one at a
# time. Same experiments and settings as the sequential script.
#
# WHY THIS LAYOUT
#   `--n-jobs` drives GridSearchCV (grid x inner folds). The estimators are now
#   single-threaded (XGBoost n_jobs=1, CatBoost thread_count=1 in model_registry),
#   so GridSearchCV --n-jobs is the ONLY parallelism layer: total CPU usage is
#   exactly PAR*NJOBS cores -- predictable, no oversubscription. The env vars
#   below additionally cap BLAS threads (numpy / MLP / LogReg) to 1.
#
# KEEP THE MACHINE USABLE (e.g. browsing): set PAR*NJOBS BELOW your core count to
#   leave headroom. The defaults use 6*16 = 96 cores on a 128-core box (32 free).
#
# TUNE:
#   PAR=6 NJOBS=16  ./scripts/run_all_experiments_parallel.sh   # 96 cores (default, 32 free)
#   PAR=7 NJOBS=16  ./scripts/run_all_experiments_parallel.sh   # 112 cores (faster, 16 free)
#   PAR=4 NJOBS=16  ./scripts/run_all_experiments_parallel.sh   # 64 cores (very responsive)
#
# MONITOR: per-job logs land in results_logs/.  Watch with:
#   tail -f results_logs/*.log    and    htop   (load avg should sit ~= PAR*NJOBS)
#
# Prerequisites: subdatasets + combined pools built in data/ (see the sequential
# script header). Phase 2 (cross-dataset) depends on Phase 1 having finished.
set -euo pipefail
cd "$(dirname "$0")/.."

# --- 1 thread per process: let --n-jobs be the only parallelism layer ---
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# --- hide the GPU: XGBoost 2.x inits a CUDA context even with device="cpu";
#     concurrent jobs then exhaust GPU memory (cudaErrorMemoryAllocation). Force CPU. ---
export CUDA_VISIBLE_DEVICES=""

CONT="SVLEN_CALLER,MAXQV,CG_CONTENT,COVERAGE_MOSDEPTH,coverage_inside,mean_insert_inside,sd_insert_inside,mean_mapq_inside,n_clipped_inside,n_split_inside,n_discordant_inside,coverage_left,coverage_right,mean_flank_insert,mean_flank_mapq,DELTA_insert,DELTA_mapq,clipped_ratio,split_ratio,discordant_ratio"
MODELS="XGBoost,Random Forest,MLP,CatBoost,Logistic Regression,KNN"
CALLERS="MANTA,SMOOVE,DELLY"
SEED=42
NJOBS="${NJOBS:-16}"     # cores per job
PAR="${PAR:-6}"          # concurrent jobs  (PAR*NJOBS = 96 cores; leaves 32 free)
export CONT MODELS CALLERS SEED NJOBS

mkdir -p results_logs

# compare-models on a single subdataset -> results/<stem>/  (with caller comparison)
run_cm() {
    local stem="$1"
    echo ">>> [compare-models] ${stem} (njobs=${NJOBS})"
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

# cross-dataset for one SV type, across the 3 samples, using that variant's best model
run_cd() {
    local sv="$1"
    local best
    best=$(python scripts/select_best_model.py --results-dir results --variant "$sv" \
        --samples HG002 NA12878 REACH --metric "Matthews Corr. Coef.")
    echo ">>> [cross-dataset] ${sv} (best=${best})"
    vartrustml cross-dataset "HG002_${sv}.csv" "NA12878_${sv}.csv" "REACH_${sv}.csv" \
        --data-dir data --output-dir "results/cross_dataset_${sv}" --target-column state \
        --continuous "$CONT" --models "$best" \
        --hpo-method grid --seed "$SEED" --n-outer-splits 10 --n-inner-splits 5 \
        --calibrate-model --calibration isotonic --calibration-cv 3 \
        --optimize-threshold --threshold-method auto \
        --bootstrap-iters 1000 --ci-level 0.95 --ci-method bca --nan-strategy median \
        --callers MANTA,SMOOVE,DELLY --default-combinations --cv-scheme both \
        --n-jobs "$NJOBS" --verbose 1 \
        > "results_logs/cd_${sv}.log" 2>&1
    echo "    done: cross-dataset ${sv}"
}
export -f run_cd

echo "############################################################"
echo "# Phase 1: compare-models (16 jobs, ${PAR} at a time)"
echo "############################################################"
# Biggest subdatasets first for better load balancing.
printf '%s\n' \
    combined_DEL REACH_DEL NA12878_DEL HG002_DEL \
    combined_INS combined_DUP combined_INV \
    NA12878_DUP REACH_DUP HG002_INS NA12878_INS REACH_INS \
    HG002_DUP NA12878_INV REACH_INV HG002_INV \
    | xargs -I{} -P "$PAR" bash -c 'run_cm "$@"' _ {}

echo "############################################################"
echo "# Phase 2: cross-dataset (4 jobs, per variant)"
echo "############################################################"
printf '%s\n' DEL DUP INS INV | xargs -I{} -P 4 bash -c 'run_cd "$@"' _ {}

echo "############################################################"
echo "# DONE. Results under results/  (per-job logs in results_logs/)"
echo "############################################################"
