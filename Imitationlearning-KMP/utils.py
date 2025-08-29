# utils.py
import os, csv, time
import numpy as np
from scipy.io import loadmat
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from contextlib import contextmanager

def load_demo_pos_2xT(mat_path: str, demo_idx: int = 0):
    m = loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    demos = m['demos']
    try:
        pos_raw = demos[demo_idx].pos
    except Exception:
        pos_raw = demos['pos'][demo_idx]
    a = np.asarray(pos_raw)
    while a.dtype == object and a.size == 1:
        a = np.asarray(a.item())
    a = np.squeeze(a)
    if a.ndim == 2:
        if a.shape[0] == 2: pos = a
        elif a.shape[1] == 2: pos = a.T
        else: raise ValueError(f"Unexpected pos shape {a.shape}")
    else:
        ax = [i for i,s in enumerate(a.shape) if s==2]
        if not ax: raise ValueError(f"No axis=2 in pos shape {a.shape}")
        pos = np.moveaxis(a, ax[0], 0).reshape(2,-1)
    return np.asarray(pos, dtype=np.float64)

def _resample_traj(t, pos, T=200):
    t = np.asarray(t).reshape(-1)
    t0, t1 = float(t[0]), float(t[-1])
    t_rs = np.linspace(t0, t1, T)
    pos_rs = np.zeros((pos.shape[0], T))
    for d in range(pos.shape[0]):
        f = interp1d(t, pos[d], kind='linear', fill_value='extrapolate', assume_sorted=True)
        pos_rs[d] = f(t_rs)
    dt = (t1 - t0) / (T-1) if T>1 else 1.0
    return t_rs, pos_rs, dt

def _smooth(pos, window=11, poly=3):
    if (window%2==0) or (window>=pos.shape[1]): return pos.copy()
    out = np.zeros_like(pos)
    for d in range(pos.shape[0]):
        out[d] = savgol_filter(pos[d], window_length=window, polyorder=poly, mode='interp')
    return out

def _normalize(pos, mode='zscore'):
    if mode=='zscore':
        mean=pos.mean(axis=1, keepdims=True); std=pos.std(axis=1, keepdims=True)+1e-8
        pos_n=(pos-mean)/std; stats={'mode':'zscore','mean':mean,'std':std}
    elif mode=='minmax':
        mn=pos.min(axis=1,keepdims=True); mx=pos.max(axis=1,keepdims=True); rng=(mx-mn)+1e-8
        pos_n=(pos-mn)/rng*2-1; stats={'mode':'minmax','min':mn,'max':mx}
    else:
        pos_n=pos.copy(); stats={'mode':'none'}
    return pos_n, stats

def denormalize(pos_n, stats):
    m=stats.get('mode','none')
    if m=='zscore': return pos_n*stats['std']+stats['mean']
    if m=='minmax': return (pos_n+1.0)*(stats['max']-stats['min'])/2.0 + stats['min']
    return pos_n

def _derivs(pos, dt):
    vel = np.gradient(pos, axis=1)/dt
    acc = np.gradient(vel, axis=1)/dt
    return vel, acc

def preprocess_demo(pos, dt0=0.005, T=200, smooth=True, win=11, poly=3, norm='zscore'):
    L=pos.shape[1]; t=np.linspace(0.0, dt0*(L-1), L)
    t_rs, pos_rs, dt_rs = _resample_traj(t, pos, T=T)
    pos_sm = _smooth(pos_rs, window=win, poly=poly) if smooth else pos_rs
    pos_n, stats = _normalize(pos_sm, mode=norm)
    vel_n, acc_n = _derivs(pos_n, dt_rs)
    return {'t':t_rs,'dt':dt_rs,'pos':pos_n,'vel':vel_n,'acc':acc_n,'pos_raw':pos_rs,'stats':stats}

def load_letter_all(letter='G', root='../2Dletters', T=200, dt=0.005,
                    smooth=True, win=11, poly=3, norm='zscore'):
    m = loadmat(f'{root}/{letter}.mat', struct_as_record=False, squeeze_me=True)
    demos = m['demos']
    N = (len(demos) if isinstance(demos, (list,tuple)) else demos.shape[-1])
    outs=[]
    for i in range(N):
        try: pos_raw = demos[i].pos
        except: pos_raw = demos['pos'][i]
        pos = np.asarray(pos_raw)
        while pos.dtype==object and pos.size==1: pos=np.asarray(pos.item())
        pos = np.squeeze(pos)
        if pos.ndim==2:
            if pos.shape[0]==2: pass
            elif pos.shape[1]==2: pos=pos.T
            else: raise ValueError(f"pos shape {pos.shape}")
        else:
            ax=[k for k,s in enumerate(pos.shape) if s==2]
            if not ax: raise ValueError(f"pos shape {pos.shape}")
            pos=np.moveaxis(pos, ax[0],0).reshape(2,-1)
        outs.append(preprocess_demo(pos, dt0=dt, T=T, smooth=smooth, win=win, poly=poly, norm=norm))
    return outs  # list of dicts

def split_indices(N, mode='holdout', ratio=0.8, loo_test=None):
    idx=np.arange(N)
    if mode=='holdout':
        ntr=int(round(N*ratio)); return idx[:ntr].tolist(), idx[ntr:].tolist()
    elif mode=='loo':
        assert loo_test is not None
        return [i for i in idx if i!=loo_test], [loo_test]
    else:
        raise ValueError("mode must be 'holdout' or 'loo'")

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred)**2)))

def endpoint_err(y_true, y_pred):
    return float(np.linalg.norm(y_true[:,-1] - y_pred[:,-1]))

def jerk_score(pos, dt):
    vel = np.gradient(pos, axis=1)/dt
    acc = np.gradient(vel, axis=1)/dt
    jerk = np.gradient(acc, axis=1)/dt
    return float(np.mean(np.linalg.norm(jerk, axis=0)))

@contextmanager
def timer():
    t0=time.perf_counter(); yield lambda: (time.perf_counter()-t0)*1000.0  # ms

def append_csv(path, header, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = (not os.path.exists(path))
    with open(path, 'a', newline='', encoding='utf-8') as f:
        w=csv.writer(f)
        if write_header: w.writerow(header)
        w.writerow(row)
