"""Run bounded B3 map ablations and safeguarded 60-step checks locally."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out",type=Path,required=True)
parser.add_argument("--workers",type=int,default=3)
args=parser.parse_args()
if args.out.exists():
    raise FileExistsError(args.out)
args.out.mkdir(parents=True)
os.environ.update(OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
tasks=[]
for case in ("rect","swept30","elliptic"):
    for mode in ("old","attachment","material","both"):
        tasks.append((case,mode,8,True))
for case in ("rect","swept30","elliptic"):
    tasks.append((case,"both",60,False))
running=[];completed=[]
while tasks or running:
    while tasks and len(running)<args.workers:
        case,mode,iterations,unguarded=tasks.pop(0)
        label=case+"_"+mode+("_raw8" if unguarded else "_guarded60")
        folder=args.out/label
        command=[sys.executable,"-m","tools.wake_material_ablation","--case",case,"--mode",mode,
                 "--iterations",str(iterations),"--potential-backend","torch","--device","cuda",
                 "--block","512","--out",str(folder)]
        if unguarded:
            command.append("--unguarded")
        with (args.out/(label+".log")).open("w",encoding="utf-8") as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        running.append((label,process,command))
    for label,process,command in list(running):
        code=process.poll()
        if code is not None:
            completed.append(dict(label=label,exit_code=code,command=command))
            running.remove((label,process,command))
            print(label,"exit",code,flush=True)
    (args.out/"queue.json").write_text(json.dumps(dict(pending=len(tasks),
        running=[dict(label=label,pid=process.pid) for label,process,_ in running],
        completed=completed),indent=2),encoding="utf-8")
    if tasks or running:
        time.sleep(2)
