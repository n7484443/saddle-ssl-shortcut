"""Bounded paired controls, separate from the historical ImageNet protocol."""
import argparse
import ctypes
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import scaling_cache as sc

ROOT = sc.ROOT
STATUS = ROOT / 'train_theory_status.json'
STATE = dict(pid=os.getpid(), status='starting')
ALPHAS = np.linspace(.2, 2, 10)
EXTENTS = np.array([1,4,9,16])/16
RANK_RTOL = 1e-10
CONDITIONS = ['original', 'shuffle0', 'shuffle1', 'shuffle2', 'common', 'common_centered']
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False


def status(**updates):
    STATE.update(updates, updated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    sc.write_json(STATUS, STATE)


def load_class(rows):
    transform = transforms.Compose([transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                                    transforms.CenterCrop(224), transforms.ToTensor()])
    raw = []
    for row in rows:
        with Image.open(row['path']) as im: raw.append(transform(im.convert('RGB')))
    raw = torch.stack(raw)
    work = raw.double()
    mean64 = work.mean((0,2,3))
    mean = mean64.float().view(3,1,1)
    std = (work.square().mean((0,2,3))-mean64.square()).clamp_min(1e-12).sqrt().float().view(3,1,1)
    # Reuse the original alignment implementation after fixing raw RGB statistics.
    build,args = sc.notebook_cell('BaseClassDataset')
    ns = dict(CROP=224, Dataset=Dataset, F=F, Image=Image, TRANSFORM=transform,
              torch=torch, ui_align=type('Value',(),{'value':True})())
    base, = build(**{a:ns[a] for a in args})
    dataset = base(rows,mean,std)
    loaded = [dataset[i] for i in range(len(rows))]
    x = torch.stack([a for a,b in loaded]); geom = torch.stack([b for a,b in loaded])
    mask = torch.zeros((len(rows),1,224,224), dtype=torch.bool)
    patches112, patches28 = [], []
    for i,(bx,by,w,h) in enumerate(geom.tolist()):
        mask[i,:,by:by+h,bx:bx+w] = True
        patch = x[i:i+1,:,by:by+h,bx:bx+w]
        patches112.append(F.interpolate(patch,(112,112),mode='bilinear',align_corners=False,antialias=True)[0])
        patches28.append(F.interpolate(patch,(28,28),mode='bilinear',align_corners=False,antialias=True)[0])
    return dict(x=x, o=x*mask, b=x*(~mask), patches112=torch.stack(patches112),
                patches28=torch.stack(patches28), mean=mean, std=std)


def centered_canvas(patches):
    n,c,h,w = patches.shape
    canvas = torch.zeros((n,c,224,224),dtype=patches.dtype,device=patches.device)
    top,left = (224-h)//2,(224-w)//2
    canvas[:,:,top:top+h,left:left+w] = patches
    return canvas


def spectral(o, b, device='cpu'):
    o = o.reshape(len(o),-1).to(device=device,dtype=torch.float64)
    b = b.reshape(len(b),-1).to(device=device,dtype=torch.float64)
    g = (o@o.T)/len(o); g = (g+g.T)/2
    h = (o@b.T)/len(o)
    lam,u = torch.linalg.eigh(g)
    lam,u = lam.flip(0),u.flip(1)
    valid = lam > lam[0]*RANK_RTOL
    rank = int(valid.sum())
    if rank == 0: raise ValueError('No numerically positive object mode')
    k = min(rank,100); lam,u = lam[:k],u[:,:k]
    projection = h.T@u
    f = torch.stack([((projection+float(a)*lam*u).square().sum(0)/lam) for a in ALPHAS])
    coefficients = torch.stack([lam,2*(projection*u).sum(0),projection.square().sum(0)/lam])
    return dict(f_alpha=f.cpu().numpy(), coefficients=coefficients.cpu().numpy(),
                eigenvalues=lam.cpu().numpy(), gram=g.cpu().numpy(), rank=rank,
                relative_overlap=float(h.norm()/g.norm()),
                total_energy=float(o.square().sum()/len(o)))


def class_controls(rows, device='cpu', preview=None):
    data = load_class(rows)
    o,b = data['o'],data['b']
    common_o = centered_canvas(data['patches112'])
    common_b = b.clone(); common_b[:,:,56:168,56:168] = 0
    variants = {'original':(o,b), 'common':(common_o,common_b),
                'common_centered':(common_o-common_o.mean(0),common_b-common_b.mean(0))}
    pairings = {}
    for seed in (0,1,2):
        idx = np.random.default_rng(seed).permutation(len(rows))
        name = f'shuffle{seed}'
        variants[name] = (o,b[idx])
        pairings[name] = dict(indices=idx.tolist(), image_ids=[rows[i]['stem'] for i in idx],
                              self_pairings=int((idx==np.arange(len(rows))).sum()))
    arrays, metadata = {}, dict(n=len(rows), pairings=pairings, conditions={})
    for name in CONDITIONS:
        result = spectral(*variants[name],device)
        arrays[name+'_alpha'] = result['f_alpha']
        arrays[name+'_coefficients'] = result['coefficients']
        metadata['conditions'][name] = {k:result[k] for k in ('rank','relative_overlap','total_energy')}
        if name.startswith('common'):
            if result['relative_overlap'] > 1e-12: raise AssertionError('Common support leaked')
            np.testing.assert_allclose(result['f_alpha']/result['f_alpha'][4], ALPHAS[:,None]**2*np.ones((1,result['f_alpha'].shape[1])), rtol=1e-10, atol=1e-10)
    for kind in ('repeat','bilinear'):
        results, patches = [], []
        for r in (1,2,3,4):
            if kind == 'repeat': patch = data['patches28'].repeat_interleave(r,2).repeat_interleave(r,3)
            else: patch = F.interpolate(data['patches28'],(28*r,28*r),mode='bilinear',align_corners=False,antialias=True)
            results.append(spectral(centered_canvas(patch),common_b,device))
            patches.append(patch)
        minrank = min(len(result['eigenvalues']) for result in results)
        arrays[kind+'_extent'] = np.stack([r['f_alpha'][4,:minrank] for r in results])
        arrays[kind+'_eigenvalues'] = np.stack([r['eigenvalues'][:minrank] for r in results])
        arrays[kind+'_energy'] = np.asarray([r['total_energy'] for r in results])
        metadata['conditions'][kind] = dict(ranks=[r['rank'] for r in results],
            retained_modes=minrank, support_pixels=[(28*r)**2 for r in (1,2,3,4)],
            relative_overlaps=[r['relative_overlap'] for r in results])
        if kind == 'repeat':
            errors = [np.max(abs(r['gram']-(i+1)**2*results[0]['gram'])) / np.max(abs(r['gram'])) for i,r in enumerate(results)]
            metadata['repeat_gram_scaling_max_error'] = float(max(errors))
            if max(errors)>1e-10: raise AssertionError('Coordinate repetition violated Gram scaling')
            np.testing.assert_allclose(arrays[kind+'_extent']/arrays[kind+'_extent'][-1], EXTENTS[:,None]*np.ones((1,minrank)), rtol=1e-8, atol=1e-10)
    if preview is not None:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2,4,figsize=(12,6), constrained_layout=True)
        images = [data['x'][0],o[0],b[0],o[0]+b[pairings['shuffle0']['indices'][0]],
                  common_o[0],common_b[0],centered_canvas(data['patches28'])[0],
                  centered_canvas(data['patches28'].repeat_interleave(4,2).repeat_interleave(4,3))[0]]
        labels = ['Aligned original','Object bbox component','Original background','Shuffled composite',
                  'Common 112x112 object','Common reserved background','28x28 base patch','4x coordinate repetition']
        for ax,img,label in zip(axes.flat,images,labels):
            ax.imshow((img*data['std']+data['mean']).clamp(0,1).permute(1,2,0).numpy())
            ax.set_title(label,fontsize=10); ax.axis('off')
        fig.suptitle('Display clips RGB only; computations use unclipped standardized components.')
        fig.savefig(preview); plt.close(fig)
    return arrays,metadata


def summarize(directory, protocol):
    per_class, checks = {}, {}
    for cls in protocol['selected_images']:
        with np.load(directory / f'{cls}.npz',allow_pickle=False) as saved:
            per_class[cls] = {k:saved[k] for k in saved.files}
        checks[cls] = json.loads((directory / f'{cls}.json').read_text(encoding='utf-8'))
    summaries = {}
    for name in CONDITIONS+['repeat','bilinear']:
        is_amp = name in CONDITIONS; x = ALPHAS if is_amp else EXTENTS
        key = name+('_alpha' if is_amp else '_extent'); ref=4 if is_amp else -1
        curves, slopes, r2s, modes = [], [], [], []
        for cls,arrays in per_class.items():
            values = arrays[key]; curve = np.median(values/values[ref],axis=1)
            slope,r2 = sc.fit_power(x,curve)
            curves.append(curve); slopes.append(float(slope[0])); r2s.append(float(r2[0])); modes.append(values.shape[1])
        idx = np.random.default_rng(20260924).integers(0,len(slopes),(2000,len(slopes)))
        ci = np.quantile(np.asarray(slopes)[idx].mean(1),[.025,.975]).tolist()
        summaries[name] = dict(quantity='amplitude' if is_amp else 'extent',
            mean_class_curve_exponent=float(np.mean(slopes)),median_r2=float(np.median(r2s)),
            class_exponents=dict(zip(per_class,slopes)), conditional_class_bootstrap_95=ci,
            actual_modes=dict(zip(per_class,modes)),x=x.tolist(),class_median_curves=np.asarray(curves).T.tolist())
    original = np.array(list(summaries['original']['class_exponents'].values()))
    for name in ('shuffle0','shuffle1','shuffle2'):
        changed = np.array(list(summaries[name]['class_exponents'].values()))
        idx=np.random.default_rng(20260924).integers(0,len(original),(2000,len(original)))
        delta=abs(original[idx].mean(1)-2)-abs(changed[idx].mean(1)-2)
        summaries[name]['paired_distance_reduction_95']=np.quantile(delta,[.025,.975]).tolist()
    result=dict(key=directory.name,protocol=protocol,summary=summaries,checks=checks)
    sc.write_json(directory/'summary.json',result)
    sc.write_json(ROOT/'cache/theory_control_results.json',result)
    lines=['# ImageNet theory-aligned controls','',
           'Exploratory paired subset: first 20 eligible primary classes, at most first 100 primary image IDs per class. No outcome-based exclusions. Float64 Gram products, positive eigenvalue threshold 1e-10 times the largest; K≤100 and effective rank. Different numerical rank handling from historical code is explicit.', '',
           '| Condition | Quantity | Mean class-curve exponent | Median R² | Conditional class-bootstrap 95% |','|---|---|---:|---:|---|']
    for name,s in summaries.items():
        lo,hi=s['conditional_class_bootstrap_95']
        lines.append(f"| {name} | {s['quantity']} | {s['mean_class_curve_exponent']:.6f} | {s['median_r2']:.6f} | [{lo:.6f}, {hi:.6f}] |")
    lines += ['', 'Common support amplitude=2 and coordinate-repetition extent=1 are algebraically enforced controls, not independent discoveries. Background permutation may reduce association but does not enforce orthogonal subspaces. Permuted composites can overlap original object regions and are additive standardized-component interventions, not natural image replacements.', '',
              'All conditions share fixed raw RGB statistics within each subset class. Common_centered additionally subtracts each component pixel mean. Extent is a separate four-point grid e=[1,4,9,16]/16 with fixed background outside the largest reserved region. Bilinear is applied to the same 28x28 baseline patch as repetition; it tests interpolation effects, not a historical ImageNet extent replication.', '',
              'Pixel areas are reserved support sizes, not segmentation masks. Previews inverse-standardize and clip only for display. Confidence intervals condition on this fixed class subset; they are not population-wide or selection-corrected.', '',
              f"Protocol, image IDs, permutations, spectra and diagnostics: `{directory}`", '']
    (ROOT.parent/'output/imagenet_theory_control_results.md').write_text('\n'.join(lines),encoding='utf-8')
    make_plot(result, ROOT.parent/'output/imagenet_theory_controls')
    return result


def make_plot(result,path=None):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),constrained_layout=True)
    for name,s in result['summary'].items():
        ax=axes[0] if s['quantity']=='amplitude' else axes[1]
        curve=np.exp(np.log(np.asarray(s['class_median_curves'])).mean(axis=1))
        ax.loglog(s['x'],curve,'o-',label=f"{name}: {s['mean_class_curve_exponent']:.3f}",markersize=3)
    for ax,x,theory,label in zip(axes,(ALPHAS,EXTENTS),(2,1),('Amplitude','Extent')):
        ax.loglog(x,x**theory,'k:',label=f'Theory {theory}')
        ax.set(xlabel=label,ylabel='Normalized class-median strength')
        ax.grid(True,which='both',alpha=.2); ax.legend(fontsize=7)
    fig.suptitle('Paired theory controls — common-support results are algebraic checks')
    if path:
        for ext in ('png','pdf'): fig.savefig(path.with_suffix('.'+ext))
        plt.close(fig)
    return fig


def main():
    status(status='waiting_for_sensitivity')
    while True:
        source=ROOT/'train_sensitivity_status.json'
        state=json.loads(source.read_text(encoding='utf-8')) if source.exists() else {}
        if state.get('status')=='ready_for_structural_controls': break
        status(sensitivity_status=state.get('status')); time.sleep(30)
    frozen=json.loads((ROOT/'cache/train_candidate_settings.json').read_text(encoding='utf-8'))
    baseline=frozen['baseline']
    original=json.loads((ROOT/'cache/train_runs'/baseline['key']/'protocol.json').read_text(encoding='utf-8'))
    selected={c:ids[:100] for c,ids in list(original['selected_images'].items())[:20]}
    records=json.loads((ROOT/'cache/train_records.json').read_text(encoding='utf-8'))
    lookup={r['stem']:r for r in records['records']}
    protocol=dict(primary_run=baseline['key'],selected_images=selected,normalization='fixed subset class RGB; common_centered also component pixel-centering',
        amplitudes=ALPHAS.tolist(),extents=EXTENTS.tolist(),pairing_seeds=[0,1,2],
        rank_rtol=RANK_RTOL,max_k=100,gram_precision='float64',conditions=CONDITIONS,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_kernel_sha256=hashlib.sha256(sc.SOURCE.read_bytes()).hexdigest(),dataset=records['identity'],
        bootstrap_seed=20260924,bootstrap_repeats=2000)
    directory=ROOT/'cache/theory_controls'/sc.digest(protocol)
    sc.write_json(directory/'protocol.json',protocol)
    status(status='computing_theory_controls',key=directory.name,total_classes=len(selected),completed_classes=0)
    import matplotlib
    matplotlib.use('Agg')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    for i,(cls,ids) in enumerate(selected.items(),1):
        target=directory/f'{cls}.npz'; meta=directory/f'{cls}.json'
        if not (target.exists() and meta.exists()):
            arrays,metadata=class_controls([lookup[s] for s in ids],device,
                directory/'example_components.png' if i==1 else None)
            temp=target.with_suffix('.npz.tmp')
            with temp.open('wb') as stream: np.savez_compressed(stream,**arrays)
            temp.replace(target); sc.write_json(meta,metadata)
        status(completed_classes=i)
        print(f'Theory controls: {i}/{len(selected)} classes',flush=True)
    summarize(directory,protocol)
    status(status='ready_for_review',completed_classes=len(selected),directory=str(directory))


def validate():
    rng=np.random.default_rng(17)
    o=torch.tensor(rng.normal(size=(8,32)),dtype=torch.float64)
    b=torch.tensor(rng.normal(size=(8,32)),dtype=torch.float64)
    oo=torch.cat([o,torch.zeros_like(o)],1); bb=torch.cat([torch.zeros_like(b),b],1)
    r=spectral(oo,bb)
    np.testing.assert_allclose(r['f_alpha']/r['f_alpha'][4], ALPHAS[:,None]**2*np.ones((1,8)),atol=1e-12)
    # Compare general Gram formula to direct feature-space Rayleigh quotients.
    mixed=spectral(o,b)
    lam,u=torch.linalg.eigh(o@o.T/8); lam,u=lam.flip(0),u.flip(1)
    v=o.T@u/torch.sqrt(8*lam)
    direct=np.stack([((a*o+b)@v).square().mean(0).numpy() for a in ALPHAS])
    np.testing.assert_allclose(mixed['f_alpha'],direct,rtol=1e-11,atol=1e-11)
    records=json.loads((ROOT/'cache/train_records.json').read_text(encoding='utf-8'))
    config=dict(sc.DEFAULTS,grouping='classwise',min_class_images=20)
    groups,_=sc.select_groups(records['records'],config)
    rows=next(iter(groups.values()))[:8]
    directory=ROOT.parent/'tmp/rebuttal_audit/theory_control_pilot'; directory.mkdir(exist_ok=True)
    arrays,meta=class_controls(rows,'cpu',directory/'example_components.png')
    sc.write_json(directory/'validation.json',dict(status='passed',sample_ids=[r['stem'] for r in rows],checks=meta))
    print('PASS: exact amplitude control, direct Rayleigh equivalence, common-support separation, repeat Gram/eigenvalue scaling, real train-image pilot.',flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(); parser.add_argument('--validate',action='store_true'); args=parser.parse_args()
    if args.validate: validate()
    else:
        with (ROOT/'train_theory.lock').open('a+b') as lock:
            lock.seek(0)
            if not lock.read(1): lock.write(b'0'); lock.flush()
            lock.seek(0); msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
            try: main()
            except Exception as exc:
                status(status='failed',error=repr(exc)); raise
            finally: ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
