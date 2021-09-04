import sys
import subprocess
import os
import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=1,
        help="No. of total jobs created"
    )
    parser.add_argument(
        "--num-procs",
        type=int,
        default=1,
        help="No. of workers used per job"
    )
    parser.add_argument(
        "--out-path",
        default=".",
        help="Directory to save predictions"
    )
    return parser
if __name__== "__main__":
    parser = get_parser()
    args = parser.parse_args()

    for idx in range(args.num_jobs):
        command = (
            f"name='job-{idx}' cpu_request={args.num_procs} cpu_limit={args.num_procs} bash_command='python main.py --num-jobs={args.num_jobs} --job-idx={idx} --num-procs={args.num_procs}' genkubejob | kubectl apply -f -"
        )
        with open("run.sh", "w") as f:
            f.write(command)
        f.close()
        p = subprocess.Popen(["bash", "run.sh"])
        p.wait()
