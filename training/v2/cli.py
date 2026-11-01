import argparse,json
from .runner import run_suite

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',default='data/v2/v2.2');p.add_argument('--output',default='reports/phase4/v1');p.add_argument('--smoke',action='store_true');p.add_argument('--epochs',type=int,default=20);a=p.parse_args()
 r=run_suite(a.data,a.output,a.smoke,epochs=a.epochs);print(json.dumps({"device":r['device'],"runs":len(r['records']),"params":{x['model']:x['parameters'] for x in r['records'][:4]}},indent=2))

if __name__=='__main__':main()
