"""Run the VIPE camera evaluator."""
from execution_metrics import execution_metrics,read_views
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'vipe_eval_20261030'))
from score import main
if __name__=='__main__':main('gate')
