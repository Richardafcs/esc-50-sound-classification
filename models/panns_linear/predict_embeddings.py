"""Apply the saved linear head to frozen Cnn14_16k 2048-dimensional embeddings."""
from pathlib import Path
import argparse, json
import numpy as np


def predict(embeddings, head):
    z=np.asarray(embeddings,dtype=np.float32).copy()
    z-=head['mean']
    z/=head['scale']
    logits=z@head['coef'].T+head['intercept']
    return head['classes'][np.argmax(logits,axis=1)]


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features',type=Path,required=True,help='NPZ with embeddings and filenames')
    parser.add_argument('--head',type=Path,default=Path(__file__).with_name('linear_head.npz'))
    args=parser.parse_args()
    with np.load(args.features) as features, np.load(args.head) as head:
        predictions=predict(features['embeddings'],head)
        names=json.loads(Path(__file__).with_name('class_names.json').read_text())
        print(json.dumps([{'filename':str(f),'prediction':int(p),'category':names[int(p)]}
                          for f,p in zip(features['filenames'],predictions)],indent=2))
