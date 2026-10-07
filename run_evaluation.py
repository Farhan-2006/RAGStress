"""One-command executed, resumable paired evaluation on fixed SciFact gold."""
import argparse,json
from src.cli import build
from src.generator import Generator
from src.stance import Stance
from evaluation.study import run_study,assert_frozen,OUT
from evaluation.analysis import analyze

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--limit',type=int,default=50)
    parser.add_argument('--no-resume',action='store_true')
    parser.add_argument('--report-only',action='store_true',help='Rebuild analysis from a completed real run without rerunning answering.')
    args=parser.parse_args()
    if args.report_only:
        manifest=assert_frozen()
        records=json.loads((OUT/'raw_results.json').read_text(encoding='utf8'))
        protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf8'))
        if protocol['freeze']['files']!=manifest['files']:
            raise RuntimeError('Saved results describe the earlier answering version. Use the evaluated snapshot for its report, or --no-resume to benchmark the paragraph revision.')
        if len(records)!=protocol['examples']:
            raise RuntimeError('The saved experiment is incomplete')
        if any(r['signature']!=protocol['signature'] for r in records):
            raise RuntimeError('Saved records belong to different experimental protocols')
        summaries=analyze(records,protocol)
        print(json.dumps(summaries,indent=2))
        return
    retriever=build();generator=Generator('local');classifier=Stance(retriever)
    records,protocol=run_study(retriever,generator,classifier,args.limit,not args.no_resume)
    summaries=analyze(records,protocol)
    print(json.dumps(summaries,indent=2))

if __name__=='__main__':main()
