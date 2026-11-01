import argparse,json
from .dataset import generate_v2

def main():
 p=argparse.ArgumentParser(description="Generate versioned CTN V2 causal data")
 p.add_argument("--output",default="data/v2/v2.2"); p.add_argument("--train",type=int,default=50000); p.add_argument("--validation",type=int,default=10000); p.add_argument("--test",type=int,default=10000); p.add_argument("--seed",type=int,default=20261102)
 a=p.parse_args(); stats,hashes,_=generate_v2(a.output,a.train,a.validation,a.test,a.seed); print(json.dumps({"statistics":stats["splits"],"hashes":hashes},ensure_ascii=False,indent=2))

if __name__=="__main__": main()
