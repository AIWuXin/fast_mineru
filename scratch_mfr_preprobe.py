"""MFR 预处理管道细分：crop_margin / resize+pad / normalize / format+batch 逐步骤计时。"""
import os,time,sys;os.environ.setdefault("PYTHONUTF8","1")
import numpy as np
import torch

# 挂探针：在 batch_predict 内插细粒度计时
from mineru.model.mfr.pp_formulanet_plus_m import predict_formula as PF
_orig_bp = PF.FormulaRecognizer.batch_predict
_rec = {"crop_margin":0.0,"resize_pad":[0.0,0],"norm":0.0,"fmt_batch":0.0,"h2d":0.0,"net":0.0,"calls":0,"n":0}

def probed(self, images_mfd_res, images, batch_size=64, interline_enable=True):
    from mineru.model.mfr.pp_formulanet_plus_m.predict_formula import build_mfr_batch_groups
    if not images_mfd_res: return []
    mfl=[];bl=[];ii=[]
    t=time.perf_counter()
    # 公式裁剪(crop_margin 是其中大头，在 UniMERNetImgDecode.img_decode 内)
    for mfd_res,img in zip(images_mfd_res,images):
        fl,ct=self._build_formula_items(mfd_res,img,interline_enable=interline_enable)
        for fi,(x0,y0,x1,y1) in ct:
            bbox_img=img[y0:y1,x0:x1]; area=(x1-x0)*(y1-y0)
            ii.append((area,len(mfl),bbox_img));mfl.append(bbox_img);bl.append(fi)
    if not ii: return [[] for _ in images_mfd_res]
    ii.sort(key=lambda x:x[0])
    sorted_images=[x[2] for x in ii]
    sorted_indices=[x[1] for x in ii]
    frb=max(1,batch_size//2)
    # 步骤1: UniMERNetImgDecode(crop_margin+resize+pad，CPU)
    t1=time.perf_counter()
    bi=self.pre_tfs["UniMERNetImgDecode"](imgs=sorted_images)
    _rec["crop_margin"]+=time.perf_counter()-t1
    _rec["n"]+=len(sorted_images)
    # 步骤2: UniMERNetTestTransform(normalize，CPU)
    t2=time.perf_counter()
    bi=self.pre_tfs["UniMERNetTestTransform"](imgs=bi)
    _rec["norm"]+=time.perf_counter()-t2
    # 步骤3: LatexImageFormat(pad to 16x multiple，CPU)
    t3=time.perf_counter()
    bi=self.pre_tfs["LatexImageFormat"](imgs=bi)
    _rec["fmt_batch"]+=time.perf_counter()-t3
    # 步骤4: ToBatch+stack -> H2D
    inp0=self.pre_tfs["ToBatch"](imgs=bi)
    torch.cuda.synchronize();t4=time.perf_counter()
    inp=torch.from_numpy(inp0[0]).to(self.device);torch.cuda.synchronize()
    _rec["h2d"]+=time.perf_counter()-t4
    # net
    _amp=(not str(self.device).startswith("cpu") and os.environ.get("MFR_INFERENCE_PRECISION","fp16").lower()!="fp32")
    rec_f=[]
    t5=time.perf_counter()
    sorted_areas=[x[0] for x in ii]
    batch_groups=build_mfr_batch_groups(sorted_areas,frb)
    with torch.no_grad():
        with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=_amp):
            for bg in batch_groups:
                bp=[self.net(inp[bg])];bp=[p.reshape([-1]) for p in bp[0]];bp=[x.cpu().numpy() for x in bp]
                rec_f+=self.post_op(bp)
    torch.cuda.synchronize();_rec["net"]+=time.perf_counter()-t5
    _rec["calls"]+=1
    index_mapping={n:o for n,o in enumerate(sorted_indices)}
    unsorted=[""]*len(rec_f)
    for n,l in enumerate(rec_f):unsorted[index_mapping[n]]=l
    for res,l in zip(bl,unsorted):res["latex"]=l
    return [[] for _ in images_mfd_res]  # 保持格式
PF.FormulaRecognizer.batch_predict=probed

from fast_mineru.config import PipelineConfig;from fast_mineru.pipeline import FastMineruPipeline;from pathlib import Path
pdf=Path(r"D:/project/MinerU/input/三维显示图像的串扰评价方法与理论建模（特邀）_激光与光电子学进展.pdf")
cfg=PipelineConfig(output_dir=Path("scratch_mfrpp"),no_render=True,warmup_pages=1)
pipe=FastMineruPipeline(cfg)
t0=time.perf_counter();pipe.process(pdf);pipe.close()
wall=time.perf_counter()-t0
r=_rec
print(f"\n=== process wall {wall:.2f}s MFR calls={r['calls']} formulas={r['n']} ===")
print(f"  crop_margin+resize+pad(CPU):     {r['crop_margin']*1000:8.1f} ms  ({r['crop_margin']/wall*100:.1f}%)")
print(f"  normalize(CPU):                  {r['norm']*1000:8.1f} ms")
print(f"  format+stack to [N,1,H,W](CPU):  {r['fmt_batch']*1000:8.1f} ms")
print(f"  H2D:                             {r['h2d']*1000:8.1f} ms")
print(f"  net(encoder+decoder)+post+D2H:   {r['net']*1000:8.1f} ms")
print(f"  预处理合计(CPU):                 {(r['crop_margin']+r['norm']+r['fmt_batch'])*1000:8.1f} ms")
print(f"  预处理 每公式平均:               {(r['crop_margin']+r['norm']+r['fmt_batch'])/max(1,r['n'])*1e6:5.0f} μs")
