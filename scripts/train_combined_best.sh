#!/usr/bin/env bash
#
# train_combined_best.sh
# ----------------------
# Fit and save the deployable (calibrated) model for each SV type on the
# COMBINED three-sample pool, using the best model per variant as identified by
# the combined compare-models run (results/combined_<SV>/, RESULTS.md 3.1):
#
#   DEL -> CatBoost   DUP -> CatBoost   INS -> XGBoost   INV -> CatBoost
#
# Flags mirror, one-to-one, the settings of the combined compare-models run that
# produced results/combined_<SV>/ (see scripts/run_all_experiments.sh step 2 and
# results/combined_DEL/experiment_config.json):
#   - 20 continuous features (CONT), target=state
#   - grid HPO, inner CV = 5 folds        -> --cv-folds 5
#   - seed 42                             -> --seed 42
#   - isotonic calibration, cv = 3        -> --calibrate-model --calibration-method isotonic --calibration-cv 3
#   - threshold optimization, auto        -> --optimize-threshold --threshold-method auto
#   - median NaN imputation               -> train default
#
# Note: compare-models nested 10x5 CV, bootstrap CIs, caller comparison and the
# MCC comparison-metric are evaluation/selection-only and have no equivalent in
# `train`, whose job is to emit the single fitted *_calibrated_model.joblib.
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=""

CONT="SVLEN_CALLER,MAXQV,CG_CONTENT,COVERAGE_MOSDEPTH,coverage_inside,mean_insert_inside,sd_insert_inside,mean_mapq_inside,n_clipped_inside,n_split_inside,n_discordant_inside,coverage_left,coverage_right,mean_flank_insert,mean_flank_mapq,DELTA_insert,DELTA_mapq,clipped_ratio,split_ratio,discordant_ratio"
NJOBS=64
SEED=42

declare -A BEST=( [DEL]=CatBoost [DUP]=CatBoost [INS]=XGBoost [INV]=CatBoost )

for SV in DEL DUP INS INV; do
    MODEL="${BEST[$SV]}"
    echo ">>> train (combined best): ${SV}  model=${MODEL}"
    vartrustml train "data/combined_${SV}.csv" \
        --model "$MODEL" \
        --target state \
        --continuous "$CONT" \
        --cv-folds 5 \
        --seed "$SEED" \
        --calibrate-model \
        --calibration-method isotonic \
        --calibration-cv 3 \
        --optimize-threshold \
        --threshold-method auto \
        --n-jobs "$NJOBS" \
        -o "results/fitted_model_combined_${SV}"
done

echo "############################################################"
echo "# DONE. Fitted models under results/fitted_model_combined_<SV>/"
echo "############################################################"
