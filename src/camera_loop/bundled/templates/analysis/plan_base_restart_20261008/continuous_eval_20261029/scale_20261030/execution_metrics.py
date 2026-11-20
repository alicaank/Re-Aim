"""Camera motion diagnostics for the 80-frame pilot."""
from PIL import Image
import numpy as np

def execution_metrics(measured,commanded):
    x=commanded[:,:3,3]-commanded[0,:3,3];y=measured[:,:3,3]-measured[0,:3,3]
    denom=float(np.linalg.norm(x[-1]));energy=float((x*x).sum())
    if denom<1e-4 or energy<1e-6:raise ValueError('Insufficient commanded motion for execution ratio')
    ratio=float(np.linalg.norm(y[-1])/denom)
    gain=float((x*y).sum()/energy)
    return dict(endpoint_execution_ratio=ratio,projected_gain=gain,
        path_length_ratio=float(np.linalg.norm(np.diff(y,axis=0),axis=1).sum()/np.linalg.norm(np.diff(x,axis=0),axis=1).sum()),
        endpoint_position_error=float(np.linalg.norm(y[-1]-x[-1])),
        mean_position_error=float(np.linalg.norm(y[1:]-x[1:],axis=1).mean()),
        passed=bool(np.isfinite(measured).all() and 0<ratio<=1.3 and gain>0),
        gate_definition='norm(t80-t0 measured) / norm(t80-t0 commanded) <= 1.3; finite poses and positive projected motion required',
        expected_ratio_hypothesis=1.05,upper_limit=1.3)

def read_views(source,scene,ids):
    if (source/f'{scene}_frames').exists():
        return [np.asarray(Image.open(source/f'{scene}_frames'/f'frame_{i:05d}.png').convert('RGB')) for i in ids]
    if (source/'frames.npz').exists():
        with np.load(source/'frames.npz') as z:return list(z['frames'][ids])
    import cv2
    path=source if source.is_file() else source/'review_full.mp4'
    cap=cv2.VideoCapture(str(path));assert cap.isOpened(),path;out=[]
    for i in range(max(ids)+1):
        ok,f=cap.read();assert ok,(path,i)
        if i in ids:out.append(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))
    cap.release();return out
