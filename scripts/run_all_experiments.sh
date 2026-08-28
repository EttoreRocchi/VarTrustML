#!/usr/bin/env bash
#
# run_all_experiments.sh
# ----------------------
# Full experimental run for VarTrustML on the 3 datasets (HG002, NA12878, REACH).
#
#   1) compare-models : one run per sample (its 4 SV-type subdatasets), with
#                       caller comparison (MANTA / SMOOVE / DELLY + default combos)
#   2) compare-models : one run per SV type on the COMBINED pool of the 3 samples,
#                       with caller comparison (robustness of a single shared model)
#   3) cross-dataset  : one run per SV type across the 3 samples, using that
#                       variant's best model (highest mean MCC in step 1)
#
# Prerequisites: subdatasets and combined pools already built into data/ with
#   python scripts/extract_sv_subdatasets.py --save-subdatasets -o data \
#       --target-column state data/HG002.csv data/NA12878.csv data/REACH.csv
#   python scripts/make_combined_subdatasets.py data/HG002.csv data/NA12878.csv \
#       data/REACH.csv -o data --prefix combined --target-column state
#
# Settings:
#   - HPO            : grid search (deterministic, fully reproducible)
#   - Models         : all 6 (XGBoost, CatBoost, Random Forest, LogReg, MLP, KNN)
#   - Calibration    : isotonic (cv=3)
#   - Threshold      : Youden's J, method=auto
#   - Bootstrap CIs  : 1000 iterations, 95% level, BCa method
#   - NaN handling   : median imputation in pipeline (--nan-strategy median)
#   - HTML reports   : enabled (default)
#   - CV             : 10 outer / 5 inner folds
#   - Seed           : 42
#   - Parallel jobs  : 64
set -euo pipefail

cd "$(dirname "$0")/.."

# Hide the GPU: XGBoost 2.x inits a CUDA context even with device="cpu", and
# concurrent fits exhaust GPU memory (cudaErrorMemoryAllocation). Force CPU.
export CUDA_VISIBLE_DEVICES=""

CONT="SVLEN_CALLER,MAXQV,CG_CONTENT,COVERAGE_MOSDEPTH,coverage_inside,mean_insert_inside,sd_insert_inside,mean_mapq_inside,n_clipped_inside,n_split_inside,n_discordant_inside,coverage_left,coverage_right,mean_flank_insert,mean_flank_mapq,DELTA_insert,DELTA_mapq,clipped_ratio,split_ratio,discordant_ratio"
MODELS="XGBoost,Random Forest,MLP,CatBoost,Logistic Regression,KNN"
CALLERS="MANTA,SMOOVE,DELLY"
NJOBS=64
SEED=42

echo "############################################################"
echo "# 1) COMPARE-MODELS  (per sample, with caller comparison)"
echo "############################################################"

for DS in HG002 NA12878 REACH; do
    echo ">>> compare-models: ${DS}"
    vartrustml compare-models \
        "${DS}_DEL.csv" "${DS}_DUP.csv" "${DS}_INS.csv" "${DS}_INV.csv" \
        --data-dir data \
        --output-dir results \
        --target-column state \
        --continuous "$CONT" \
        --models "$MODELS" \
        --hpo-method grid \
        --seed "$SEED" \
        --n-outer-splits 10 \
        --n-inner-splits 5 \
        --calibrate-model \
        --calibration isotonic \
        --calibration-cv 3 \
        --optimize-threshold \
        --threshold-method auto \
        --compare-callers \
        --callers "$CALLERS" \
        --default-combinations \
        --bootstrap-iters 1000 \
        --ci-level 0.95 \
        --ci-method bca \
        --nan-strategy median \
        --comparison-metric "Matthews Corr. Coef." \
        --n-jobs "$NJOBS" \
        --verbose 1
done

echo "############################################################"
echo "# 2) COMPARE-MODELS  (combined pool of the 3 samples, per SV type)"
echo "############################################################"

for SV in DEL DUP INS INV; do
    echo ">>> compare-models (combined): ${SV}"
    vartrustml compare-models \
        "combined_${SV}.csv" \
        --data-dir data \
        --output-dir results \
        --target-column state \
        --continuous "$CONT" \
        --models "$MODELS" \
        --hpo-method grid \
        --seed "$SEED" \
        --n-outer-splits 10 \
        --n-inner-splits 5 \
        --calibrate-model \
        --calibration isotonic \
        --calibration-cv 3 \
        --optimize-threshold \
        --threshold-method auto \
        --compare-callers \
        --callers "$CALLERS" \
        --default-combinations \
        --bootstrap-iters 1000 \
        --ci-level 0.95 \
        --ci-method bca \
        --nan-strategy median \
        --comparison-metric "Matthews Corr. Coef." \
        --n-jobs "$NJOBS" \
        --verbose 1
done

echo "############################################################"
echo "# 3) CROSS-DATASET  (per SV type; best model per variant by mean MCC)"
echo "############################################################"

for SV in DEL DUP INS INV; do
    # Pick this variant's best model by mean MCC across the 3 per-sample runs
    # (step 1 must have completed and written model_metrics_comparison.csv).
    BEST=$(python3 scripts/select_best_model.py \
        --results-dir results --variant "$SV" \
        --samples HG002 NA12878 REACH \
        --metric "Matthews Corr. Coef.")
    echo ">>> cross-dataset: ${SV}  (best model by mean MCC: ${BEST})"
    vartrustml cross-dataset \
        "HG002_${SV}.csv" "NA12878_${SV}.csv" "REACH_${SV}.csv" \
        --data-dir data \
        --output-dir "results/cross_dataset_${SV}" \
        --target-column state \
        --continuous "$CONT" \
        --models "$BEST" \
        --hpo-method grid \
        --seed "$SEED" \
        --n-outer-splits 10 \
        --n-inner-splits 5 \
        --calibrate-model \
        --calibration isotonic \
        --calibration-cv 3 \
        --optimize-threshold \
        --threshold-method auto \
        --bootstrap-iters 1000 \
        --ci-level 0.95 \
        --ci-method bca \
        --nan-strategy median \
        --n-jobs "$NJOBS" \
        --verbose 1
done

echo "############################################################"
echo "# DONE. Results under results/"
echo "############################################################"
