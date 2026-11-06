from pathlib import Path
import argparse,hashlib,json,os,pickle,time,gc,sys,atexit
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
from PIL import Image,ImageDraw,__version__ as PIL_VERSION
import xml.etree.ElementTree as ET
from tqdm import tqdm
STEPS=['0_200','2_1400','5_3400','10_6800','20_13400','30_20200','40_26800','50_33400','60_40200','70_46800','80_53600','99_66799']
class LinearClassifier(nn.Module):
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)

def parse_bbox_annotation(xml_path):
    if not os.path.exists(xml_path):
        return []

    tree = ET.parse(xml_path)
    root = tree.getroot()

    bboxes = []
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        bboxes.append([xmin, ymin, xmax, ymax])

    return bboxes

class MaskedImageDataset(Dataset):
    def __init__(self, base_dataset, data_list, transform):
        self.base_dataset = base_dataset
        self.data_list = data_list
        self.transform = transform
        self.preprocess()

    def preprocess(self):
        print("Preprocessing bboxes...")
        for data in tqdm(self.data_list):
            img = Image.open(data['image_path'])
            orig_w, orig_h = img.size

            bboxes = parse_bbox_annotation(data['xml_path'])

            data['orig_size'] = (orig_w, orig_h)
            data['bboxes'] = bboxes

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]

        img = Image.open(data['image_path']).convert('RGB')
        orig_w, orig_h = data['orig_size']
        bboxes = data['bboxes']

        # Apply transform to image
        img_tensor = self.transform(img)
        H, W = img_tensor.shape[1], img_tensor.shape[2]  # 224, 224

        # Create bbox mask on PIL Image, then transform it
        bbox_mask_pil = Image.new('L', (orig_w, orig_h), 0)
        draw = ImageDraw.Draw(bbox_mask_pil)

        for bbox in bboxes:
            xmin, ymin, xmax, ymax = bbox
            draw.rectangle([xmin, ymin, xmax, ymax], fill=255)

        # Apply same transform to mask (without normalization)
        mask_transform = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])

        bbox_mask_tensor = mask_transform(bbox_mask_pil)  # (1, 224, 224)
        bbox_mask_tensor = (bbox_mask_tensor > 0.5).float()  # Binarize

        # Create 3-channel mask
        mask = 1 - bbox_mask_tensor.repeat(3, 1, 1)  # 0 where bbox, 1 elsewhere

        object_tensor = img_tensor * (1 - mask)
        background_tensor = img_tensor * mask

        return img_tensor, object_tensor, background_tensor, data['class'], mask, idx

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def atomic_json(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp')
    with open(tmp,'w',encoding='utf8') as f:json.dump(data,f,ensure_ascii=False,indent=2)
    os.replace(tmp,path)

def load_samples(path):
    if path.suffix.lower()=='.jsonl':
        with open(path,encoding='utf8') as f:rows=[json.loads(line) for line in f if line.strip()]
    else:
        data=json.loads(path.read_text(encoding='utf8'));rows=data if isinstance(data,list) else data['samples']
    for i,row in enumerate(rows):
        assert int(row['cache_idx'])==i,(i,row)
        assert 0<=int(row['class'])<1000
        assert Path(row['image_path']).is_file() and Path(row['xml_path']).is_file()
    return rows

def load_pair(root,step):
    backbone_path=root/(step+'.pth');head_path=root/step/'final_checkpoint.pth'
    model=torchvision.models.resnet50(weights=None);model.fc=nn.Identity()
    state=torch.load(backbone_path,map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True)
    linear=LinearClassifier(2048,1000)
    head=torch.load(head_path,map_location='cpu',weights_only=True)
    linear.load_state_dict({k.replace('module.',''):v for k,v in head.items()},strict=True)
    del state,head
    return model,linear,{'backbone_path':str(backbone_path),'backbone_sha256':sha(backbone_path),
      'head_path':str(head_path),'head_sha256':sha(head_path)}

def infer_step(model,linear,loader,expected_n):
    device=torch.device('cuda')
    model.to(device).eval();linear.to(device).eval()
    model_grad=torch.nn.Sequential(model,linear)
    lists={k:[] for k in ['class','conf','top1','top5','idx','pred_indices','top5_indices']}
    started=time.time();torch.cuda.reset_peak_memory_stats();batch_timings=[];tie_count=0
    with torch.no_grad(),torch.amp.autocast(device_type='cuda',dtype=torch.bfloat16):
        for orig_batch,object_batch,background_batch,classes,_,indices in tqdm(loader):
            batch_started=time.perf_counter();B=orig_batch.size(0)
            combined=torch.cat([orig_batch,object_batch,background_batch],dim=0).to(device,non_blocking=True)
            classes=classes.to(device,non_blocking=True)
            logits_all=model_grad(combined)
            classes_3x=classes.repeat(3);idx_3x=torch.arange(3*B,device=device)
            log_probs=logits_all-logits_all.logsumexp(dim=1,keepdim=True)
            conf_all=log_probs[idx_3x,classes_3x].exp().view(3,B)
            preds=logits_all.argmax(dim=1).view(3,B)
            corr_top1=(preds==classes)
            top5=logits_all.topk(5,dim=1,sorted=False)[1].view(3,B,5)
            corr_top5=(top5==classes.view(1,B,1)).any(dim=2)
            assert torch.isfinite(logits_all).all() and torch.isfinite(conf_all).all()
            # BF16 ties can make argmax's index absent from topk's selected
            ties=corr_top1&~corr_top5
            if bool(ties.any()):
                true_scores=logits_all[idx_3x,classes_3x].view(3,B)
                selected_scores=torch.gather(logits_all,1,top5.reshape(3*B,5)).view(3,B,5)
                assert torch.all(true_scores[ties]==selected_scores.min(dim=2).values[ties])
                tie_count+=int(ties.sum().item())
            lists['class'].append(classes.cpu())
            lists['conf'].append(conf_all.cpu().to(torch.float32))
            lists['top1'].append(corr_top1.cpu());lists['top5'].append(corr_top5.cpu())
            lists['idx'].append(indices.cpu())
            lists['pred_indices'].append(preds.cpu());lists['top5_indices'].append(top5.cpu())
            batch_timings.append({'samples':int(B),'seconds':time.perf_counter()-batch_started})
    classes=np.concatenate([x.numpy() for x in lists['class']])
    indices=np.concatenate([x.numpy() for x in lists['idx']])
    conf=np.concatenate([x.numpy() for x in lists['conf']],axis=1)
    top1=np.concatenate([x.numpy() for x in lists['top1']],axis=1)
    top5=np.concatenate([x.numpy() for x in lists['top5']],axis=1)
    assert len(classes)==expected_n and np.array_equal(indices,np.arange(expected_n))
    arrays={'class':classes,'cache_idx':indices,
       'pred_indices':np.concatenate([x.numpy() for x in lists['pred_indices']],axis=1),
       'top5_indices':np.concatenate([x.numpy() for x in lists['top5_indices']],axis=1)}
    for j,view in enumerate(['orig','obj','bg']):
        arrays[view+'_top1']=top1[j];arrays[view+'_top5']=top5[j];arrays['conf_'+view]=conf[j]
    report={'seconds':time.time()-started,'samples':expected_n,
      'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved(),
      'accuracy':{k:float(arrays[k].mean()) for k in arrays if k.endswith(('top1','top5'))},
      'batch_compute_timings':batch_timings,'warm_batch_seconds':batch_timings[0]['seconds'],
      'bf16_argmax_top5_tie_disagreements':tie_count,
      'steady_batch_mean_seconds':float(np.mean([b['seconds'] for b in batch_timings[1:]])) if len(batch_timings)>1 else None,
      'timing_scope':'Per-batch H2D+forward+metrics+D2H; total includes DataLoader waits; first batch is warmup.'}
    model.cpu();linear.cpu();del model_grad;gc.collect();torch.cuda.empty_cache()
    return arrays,report

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--samples',type=Path,required=True)
    p.add_argument('--models',type=Path,default=Path('models/bt-imagenet'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',nargs='+',choices=STEPS,default=STEPS)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--no-pin-memory',action='store_true',help='Disable pinned host memory for Windows/CUDA transfer compatibility.')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--smoke-samples',type=int,default=256)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--run',action='store_true')
    args=p.parse_args()
    if not args.run:
        print('Prepared only. Main must pass --run to execute inference.');return
    assert args.batch_size>0 and args.workers>=0
    workspace=Path(__file__).resolve().parents[4]
    output=args.out.resolve();output.relative_to(workspace/'output')
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise RuntimeError('Output is nonempty; use a new directory or explicit --resume.')
    output.mkdir(parents=True,exist_ok=True)
    lock=output/'dispatch.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w',encoding='utf8') as f:json.dump({'pid':os.getpid(),'started':time.time()},f)
    atexit.register(lambda:lock.unlink(missing_ok=True))
    rows=load_samples(args.samples)
    if args.smoke:rows=rows[:args.smoke_samples];args.steps=args.steps[:1]
    sample_hash=sha(args.samples)
    policy={'sample_manifest_sha256':sample_hash,'sample_count':len(rows),'smoke':args.smoke,'steps':list(args.steps),
      'batch_size':args.batch_size,'original_source_batch_size':512,'workers':args.workers,'pin_memory':not args.no_pin_memory,
      'sample_selection':'first fixed prefix of sorted-WNID/filename manifest' if args.smoke else 'all samples in manifest order',
      'autocast':'bfloat16','confidence_arithmetic':'original BF16; float32 cast only after CPU for NumPy',
      'mask_policy':'original PIL rectangle union; bicubic256/CenterCrop224; >0.5; multiply normalized RGB',
      'adapter_sha256':sha(__file__),'source_snapshot_sha256':sha(Path(__file__).parent/'source_statics_train.py.txt')}
    runtime={'torch':torch.__version__,'torchvision':torchvision.__version__,'numpy':np.__version__,'pillow':PIL_VERSION,'python':sys.version,
      'cuda_version':torch.version.cuda,'device':torch.cuda.get_device_name(0),
      'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
      'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'float32_matmul_precision':torch.get_float32_matmul_precision(),
      'cudnn_version':torch.backends.cudnn.version(),'cudnn_deterministic':torch.backends.cudnn.deterministic,
      'cudnn_benchmark':torch.backends.cudnn.benchmark,'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
      'CUBLAS_WORKSPACE_CONFIG':os.environ.get('CUBLAS_WORKSPACE_CONFIG')}
    if (output/'run_policy.json').exists():
        previous=json.loads((output/'run_policy.json').read_text(encoding='utf8'))
        assert previous['policy']==policy,'Resume policy differs from existing output.'
        assert previous['runtime']==runtime,'Resume runtime differs from existing output.'
    atomic_json(output/'run_policy.json',{'policy':policy,'runtime':runtime,'steps':args.steps})
    sidecar=output/'sample_indices.json'
    if not sidecar.exists():atomic_json(sidecar,[{k:r[k] for k in ['cache_idx','original_idx','class','image_path','xml_path']} for r in rows])
    transform=transforms.Compose([transforms.Resize(256,interpolation=3),transforms.CenterCrop(224),transforms.ToTensor(),
       transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
    dataset=MaskedImageDataset(None,[dict(r) for r in rows],transform)
    loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=not args.no_pin_memory,persistent_workers=args.workers>0)
    completed=[]
    for step in args.steps:
        target=output/(step+'.npz');meta_path=output/(step+'.json')
        if args.resume and target.exists() and meta_path.exists():
            prior=json.loads(meta_path.read_text(encoding='utf8'))
            assert prior['policy']==policy and prior['runtime']==runtime and prior['npz_sha256']==sha(target)
            assert prior['inputs']['backbone_sha256']==sha(args.models/(step+'.pth'))
            assert prior['inputs']['head_sha256']==sha(args.models/step/'final_checkpoint.pth')
            completed.append(prior);print('RESUME_VALIDATED_SKIP',step,flush=True);continue
        if target.exists() or meta_path.exists():raise RuntimeError('Incomplete/foreign step outputs exist: '+step)
        model,linear,inputs=load_pair(args.models,step)
        arrays,performance=infer_step(model,linear,loader,len(rows))
        tmp=target.with_suffix('.npz.tmp')
        with open(tmp,'wb') as f:np.savez_compressed(f,**arrays)
        os.replace(tmp,target)
        record={'step':step,'status':'complete','policy':policy,'runtime':runtime,'inputs':inputs,'npz_sha256':sha(target),'performance':performance}
        atomic_json(meta_path,record);completed.append(record)
        atomic_json(output/'progress.json',{'completed_steps':[x['step'] for x in completed],'requested_steps':args.steps})
        print(json.dumps({'step':step,'status':'complete',**performance}),flush=True)
        del model,linear,arrays;gc.collect()
    cache={}
    compatibility_fields=['class','orig_top1','orig_top5','obj_top1','obj_top5','bg_top1','bg_top5','conf_orig','conf_obj','conf_bg']
    for step in args.steps:
        with np.load(output/(step+'.npz'),allow_pickle=False) as z:
            payload={k:z[k] for k in compatibility_fields}
        cache[step]={i:{k:payload[k][i] for k in compatibility_fields} for i in range(len(rows))}
    compatibility=output/('inference_smoke_cache.pkl' if args.smoke else 'inference_cache.pkl')
    tmp=compatibility.with_suffix('.pkl.tmp')
    with open(tmp,'wb') as f:pickle.dump(cache,f,protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp,compatibility)
    atomic_json(output/'completion.json',{'status':'SMOKE_COMPLETE' if args.smoke else 'INFERENCE_COMPLETE',
       'steps':args.steps,'sample_count':len(rows),'policy':policy,'compatibility_cache':str(compatibility),
       'compatibility_cache_sha256':sha(compatibility),'completed':completed,'training':False})

if __name__=='__main__':main()
