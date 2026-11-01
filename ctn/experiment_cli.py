import argparse, json
from .training import run_experiment

def main():
    p=argparse.ArgumentParser(description="Train CTN structured-feature baselines")
    p.add_argument("--data",default="data/generated"); p.add_argument("--output",default="results")
    p.add_argument("--epochs",type=int,default=15); p.add_argument("--seeds",type=int,nargs="+",default=[11,23,37])
    a=p.parse_args(); print(json.dumps(run_experiment(a.data,a.output,tuple(a.seeds),a.epochs),ensure_ascii=False,indent=2))

if __name__=="__main__": main()
