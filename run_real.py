"""Train SSA-3DGS on every real scene with the paper settings (Tab. 2 "Ours"):
ADC densification, statistics init, lambda_sparse = 0.01, overlay lr 5e-3.
Scenes are auto-discovered under <data_root>: every subfolder with an
undistorted COLMAP reconstruction (<scene>/undistorted/{images,sparse}).

    python run_real.py <data_root> <output_root>

GPU selection: GPU=<idx> (default 0).
"""
import argparse
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data_root", help="Folder directly containing the real scenes")
    ap.add_argument("output_root", help="Where result trees are written")
    args = ap.parse_args()

    # ssa3dgs.sh <data> <out> <subset> <dens> <artifact> <exp> <remove> <lr> <from> <l1> <tv> [init]
    cmd = ["bash", "ssa3dgs.sh", args.data_root, args.output_root,
           "my", "adc", "_", "l1_0.01", "1", "0.005", "0", "0.01", "0.0", "statistics"]
    print(" ".join(cmd), flush=True)
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
