"""Frozen-input affine investigations. Images pull back; probability mass pushes forward."""
import numpy as np
from scipy.ndimage import affine_transform

def specification(raw=None):
    raw=raw or {};v={k:float(raw.get(k,d)) for k,d in [('dx',0),('dy',0),('angle',0),('scale',1)]}
    if not all(np.isfinite(list(v.values()))) or not .25<=v['scale']<=4 or max(abs(v['dx']),abs(v['dy']))>56 or abs(v['angle'])>360:raise ValueError('Invalid affine transform')
    return v

def matrix(shape,spec):
    h,w=shape;a=np.deg2rad(spec['angle']);M=spec['scale']*np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    centre=np.array([(w-1)/2,(h-1)/2]);return M,centre-M@centre+np.array([spec['dx'],spec['dy']])

def image(x,spec):
    x=np.asarray(x);M,b=matrix(x.shape[-2:],spec);inv=np.linalg.inv(M);swap=np.array([[0,1],[1,0]])
    return np.stack([affine_transform(c,swap@inv@swap,offset=-swap@inv@b,order=1,mode='grid-constant',cval=0,prefilter=False) for c in x]).astype(x.dtype)

def transport(field,spec):
    if field is None:return dict(field=None,retained_mass=None,lost_mass=None,status='undefined_source')
    a=np.asarray(field,float);h,w=a.shape;M,b=matrix(a.shape,spec);yy,xx=np.indices(a.shape);xy=M@np.stack([xx.ravel(),yy.ravel()])+b[:,None];out=np.zeros_like(a)
    x0=np.floor(xy[0]).astype(int);y0=np.floor(xy[1]).astype(int);fx=xy[0]-x0;fy=xy[1]-y0
    for ox,oy,weight in [(0,0,(1-fx)*(1-fy)),(1,0,fx*(1-fy)),(0,1,(1-fx)*fy),(1,1,fx*fy)]:
        x=x0+ox;y=y0+oy;ok=(x>=0)&(x<w)&(y>=0)&(y<h);np.add.at(out,(y[ok],x[ok]),a.ravel()[ok]*weight[ok])
    retained=float(out.sum());return dict(field=(out/retained).tolist() if retained>1e-15 else None,retained_mass=retained,lost_mass=max(0.,float(a.sum())-retained),status='available' if retained>1e-15 else 'zero_retained_mass')

def hellinger(a,b):
    if a is None or b is None:return None
    return float(np.linalg.norm(np.sqrt(a)-np.sqrt(b))/np.sqrt(2))
