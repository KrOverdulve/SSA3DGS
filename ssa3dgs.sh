#!/bin/bash
# Usage:
#   ssa3dgs.sh <data_root> <output_root> <subset> <dens> <artifact> <exp> <remove> <lr> <from_iter> <l1_reg> <tv_reg> [init]
#
# data_root : the dataset itself. For subset=360v2: the prep root holding one
#             subfolder per artifact (watermark/ ui/ ... clean/). For subset=my:
#             the folder that directly contains the real scenes
#             (<scene>/undistorted/{images,sparse}).
# output_root : where result trees are written.
# subset    : 360v2 | my   -- which dataset layout to train on
# dens      : adc | mcmc          -- densification. adc = train.py default (ADC),
#             mcmc = train.py mcmc (per-scene --strategy.cap-max).
# artifact  : prep-root subdir (watermark|ui|dashboard|rain|mud|clean).
#             "clean" is the no-artifact prep folder -> Clean (Ref) upper bound.
#             Ignored when subset=my.
# exp       : output dir tag (e.g. l1_0.05). Ignored when remove=0.
# remove    : 1 to enable --remove_watermark (Ours), 0 for the clean/baseline run.
# lr/from_iter/l1_reg/tv_reg : overlay hyperparams (ignored when remove=0).
# init      : overlay init, statistics (default) | constant. Ignored when remove=0.
#
# Environment:
#   GPU : CUDA device index. Default: 0
#
# All 360v2 runs read the PRE-SCALED 360_v2_prep folders (written by
# prepare_watermark_dataset.py), so data_factor stays 1.
# Output tree (densification is explicit in the path):
#   $OUTPUT_ROOT/clean_<dens>/360_v2_prep/<scene>/
#   $OUTPUT_ROOT/no_remove_<dens>/<artifact>/360_v2_prep/<scene>/
#   $OUTPUT_ROOT/ours_<dens>/<exp>/<artifact>-<lr>-<from>-<l1>-<tv>/360_v2_prep/<scene>/
#   $OUTPUT_ROOT/my_no_remove_<dens>/<scene>/
#   $OUTPUT_ROOT/my_ours_<dens>/<exp>/<lr>-<from>-<l1>-<tv>/<scene>/
set -euo pipefail

DATA_ROOT=$1
OUTPUT_ROOT=$2
SUBSET=$3
DENS=$4
ARTIFACT=$5
EXP=$6
REMOVE=$7
HYPERPAR_LR=$8
FROM_ITER=$9
L1_REG=${10}
TV_REG=${11}
INIT=${12:-statistics}

GPU=${GPU:-0}

case "$DENS" in
    adc)  STRATEGY="default" ;;
    mcmc) STRATEGY="mcmc" ;;
    *)    echo "Unknown dens: $DENS (expected adc | mcmc)" >&2; exit 1 ;;
esac

if [ "$REMOVE" = "1" ]; then
    REMOVE_FLAGS="--remove_watermark --watermark_lr $HYPERPAR_LR --watermark_from_iter $FROM_ITER --watermark_l1_reg $L1_REG --watermark_tv_reg $TV_REG --watermark_init $INIT"
else
    REMOVE_FLAGS=""
fi

print_stats() {
    local label=$1
    local result_dir=$2
    for SCENE_DIR_OUT in "$result_dir"/*/; do
        [ -d "$SCENE_DIR_OUT" ] || continue
        local SCENE
        SCENE=$(basename "$SCENE_DIR_OUT")
        echo "=== Eval Stats ($label $SCENE) ==="
        for STATS in "$SCENE_DIR_OUT"stats/val*.json; do
            [ -e "$STATS" ] || continue
            echo "$STATS"
            cat "$STATS"
            echo
        done
        echo "=== Train Stats ($label $SCENE) ==="
        for STATS in "$SCENE_DIR_OUT"stats/train*_rank0.json; do
            [ -e "$STATS" ] || continue
            echo "$STATS"
            cat "$STATS"
            echo
        done
    done
}

run_scene() {
    local label=$1
    local data_dir=$2
    local result_dir=$3
    local cap_max=$4    # only used by mcmc

    echo "Running $label [$STRATEGY]"
    if [ "$STRATEGY" = "mcmc" ]; then
        CUDA_VISIBLE_DEVICES=$GPU python train.py mcmc --disable_viewer \
            --strategy.cap-max "$cap_max" \
            $REMOVE_FLAGS \
            --data_dir "$data_dir" \
            --result_dir "$result_dir"
    else
        CUDA_VISIBLE_DEVICES=$GPU python train.py default --disable_viewer \
            $REMOVE_FLAGS \
            --data_dir "$data_dir" \
            --result_dir "$result_dir"
    fi
}

run_360v2() {
    local scene_dir="$DATA_ROOT"
    if [ ! -d "$scene_dir/$ARTIFACT" ]; then
        echo "ERROR: $scene_dir/$ARTIFACT does not exist. data_root must be the prep root" >&2
        echo "holding one subfolder per artifact (see prepare_watermark_dataset.py)." >&2
        exit 1
    fi
    local scenes="bicycle bonsai counter garden kitchen room stump"
    local result_dir
    if [ "$REMOVE" = "1" ]; then
        result_dir="$OUTPUT_ROOT/ours_$DENS/$EXP/$ARTIFACT-$HYPERPAR_LR-$FROM_ITER-$L1_REG-$TV_REG/360_v2_prep"
    elif [ "$ARTIFACT" = "clean" ]; then
        result_dir="$OUTPUT_ROOT/clean_$DENS/360_v2_prep"
    else
        result_dir="$OUTPUT_ROOT/no_remove_$DENS/$ARTIFACT/360_v2_prep"
    fi

    for SCENE in $scenes; do
        case "$SCENE" in
            bicycle) CAP_MAX=5900000 ;;
            bonsai)  CAP_MAX=1300000 ;;
            counter) CAP_MAX=1200000 ;;
            garden)  CAP_MAX=5200000 ;;
            kitchen) CAP_MAX=1800000 ;;
            room)    CAP_MAX=1500000 ;;
            stump)   CAP_MAX=4750000 ;;
        esac

        # Per-scene crash isolation: a diverged/failed scene (e.g. MCMC on the
        # most view-inconsistent artifacts) must NOT abort the remaining scenes.
        run_scene "360v2 $ARTIFACT/$SCENE" \
            "$scene_dir/$ARTIFACT/$SCENE/" \
            "$result_dir/$SCENE/" \
            "$CAP_MAX" \
            || echo "WARNING: scene 360v2 $ARTIFACT/$SCENE failed (exit $?); continuing with next scene" >&2
    done

    print_stats "360v2 $ARTIFACT" "$result_dir"
}

run_my() {
    local scene_dir="$DATA_ROOT"
    local result_dir
    if [ "$REMOVE" = "1" ]; then
        result_dir="$OUTPUT_ROOT/my_ours_$DENS/$EXP/$HYPERPAR_LR-$FROM_ITER-$L1_REG-$TV_REG"
    else
        result_dir="$OUTPUT_ROOT/my_no_remove_$DENS"
    fi

    # All COLMAP scenes: every subfolder that carries an undistorted reconstruction.
    local N_SCENES=0
    for SCENE_PATH in "$scene_dir"/*/; do
        [ -d "${SCENE_PATH}undistorted/sparse" ] || continue
        N_SCENES=$((N_SCENES + 1))
        local SCENE CAP_MAX
        SCENE=$(basename "$SCENE_PATH")
        CAP_MAX=3000000
        run_scene "my $SCENE" \
            "${SCENE_PATH}undistorted/" \
            "$result_dir/$SCENE/" \
            "$CAP_MAX" \
            || echo "WARNING: scene my $SCENE failed (exit $?); continuing with next scene" >&2
    done

    if [ "$N_SCENES" = "0" ]; then
        echo "ERROR: no scenes found under $scene_dir." >&2
        echo "data_root must directly contain the scene folders, each with an" >&2
        echo "undistorted COLMAP reconstruction: <scene>/undistorted/{images,sparse}." >&2
        exit 1
    fi

    print_stats "my" "$result_dir"
}

case "$SUBSET" in
    360v2) run_360v2 ;;
    my)    run_my ;;
    *)     echo "Unknown subset: $SUBSET (expected 360v2 | my)" >&2; exit 1 ;;
esac
