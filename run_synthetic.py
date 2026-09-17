"""Run the full synthetic Mip-NeRF 360 benchmark of the paper, locally and
sequentially (one GPU). This replaces the cluster job generator; every run is a
plain `ssa3dgs.sh` invocation.

Runs, per densification in DENS (Tab. 1):
  no_remove  : 3DGS on the corrupted views, no overlay  -> Baseline 3DGS
  main       : SSA-3DGS, statistics init, lambda_sparse = 0.05 (mud: 0.01)
and, per densification in ABLATION_DENS (Tab. 3, paper: MCMC):
  no_reg     : SSA-3DGS without the sparsity regularizer (lambda_sparse = 0)
  const_init : SSA-3DGS with the constant overlay initialization

    python run_synthetic.py <data_root> <output_root>

data_root is the benchmark root holding one subfolder
per artifact (watermark/ ui/ ...); results are written under
output_root (default: outputs). GPU selection: GPU=<idx> (default 0).

Edit the constants below to narrow the grid (e.g. a single artifact). A full
grid is 7 scenes x 30k iterations per run; expect several GPU-days in total.
"""
import argparse
import subprocess
import sys

DENS = ["adc", "mcmc"]          # main benchmark (Tab. 1)
ABLATION_DENS = ["mcmc"]        # regularizer/init ablations (Tab. 3)
ARTIFACTS = ["watermark", "ui", "dashboard", "rain", "mud"]
LR = 5e-3                       # overlay learning rate (paper Sec. 4.1)
FROM_ITER = 0


def l1_for(artifact):
    # lambda_sparse = 0.05, except mud, which occludes large contiguous
    # regions and prefers the weaker prior 0.01 (paper Sec. 4.1).
    return 0.01 if artifact == "mud" else 0.05


def sh(dens, artifact, exp, remove, l1, tv=0.0, init="statistics"):
    # ssa3dgs.sh <data> <out> <subset> <dens> <artifact> <exp> <remove> <lr> <from> <l1> <tv> [init]
    return ["360v2", dens, artifact, exp,
            str(remove), str(LR), str(FROM_ITER), str(l1), str(tv), init]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_root",
                    help="Synthetic benchmark root, one subfolder per artifact")
    ap.add_argument("output_root",
                    help="Where result trees are written")
    args = ap.parse_args()
    prefix = ["bash", "ssa3dgs.sh", args.data_root, args.output_root]

    commands = []
    for dens in DENS:
        for artifact in ARTIFACTS:
            commands.append(sh(dens, artifact, "_", 0, 0))
        for artifact in ARTIFACTS:
            commands.append(sh(dens, artifact, "main", 1, l1_for(artifact)))
    for dens in ABLATION_DENS:
        for artifact in ARTIFACTS:
            commands.append(sh(dens, artifact, "no_reg", 1, 0.0))
        for artifact in ARTIFACTS:
            commands.append(sh(dens, artifact, "const_init", 1, l1_for(artifact),
                               init="constant"))

    failures = []
    for n, tail in enumerate(commands, 1):
        cmd = prefix + tail
        print(f"[{n}/{len(commands)}] {' '.join(cmd)}", flush=True)
        if subprocess.run(cmd).returncode != 0:
            failures.append(" ".join(cmd))

    print(f"\n{len(commands) - len(failures)}/{len(commands)} runs completed.")
    if failures:
        print("Failed runs:")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
