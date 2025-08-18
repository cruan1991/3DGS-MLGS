#!/usr/bin/env python3
"""
Hybrid COLMAP ↔ 3D Gaussian correspondence analysis (GPU-ready, plotting & layer-aware)
=====================================================================================
新增功能
--------
* **自定义层级**：`--layers 10,25,50,75,90`（百分位列表）
* **分层统计**：每层高斯数、α 均值、Pearson / Spearman（启用 `--corr_layers`）
* **频谱指标**：
  * *SH 能谱*（低频 l≤1 vs 高频 l>1 能量占比）
  * *Scale–Freq 2D 直方图*（log(scale) × SH 阶能量）
* **可视化** (`--plots`)：调用 *plot_utils.py* 生成
  * 空间 XY 热力图
  * 尺度直方图
  * 热点 3D scatter（梯度↑ & 密度↑）
  * Scale–Freq 2D hist

安装依赖
```
conda install -c conda-forge faiss-gpu cuml pykeops matplotlib seaborn plotly -y
```
用例
```
python correspondence_analysis_hybrid.py \
  --colmap points3D.ply --gaussians teacher.ply \
  --device cuda --sample 1.0 --layers 10,25,50,75,90 --corr_layers --plots
```
"""
from __future__ import annotations
import argparse, sys, warnings, time, json, os
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
import importlib

warnings.filterwarnings("ignore")

try:
    import torch, pykeops.torch as keops
except Exception:
    torch = None; keops = None
try:
    import faiss
except Exception:
    faiss = None
try:
    import cudf, cuml
except Exception:
    cudf = cuml = None

sys.path.append('/shared/user59/workspace/cihan/3dgs_Vincent/my_3dgs')

# 尝试导入3DGS相关模块，如果失败则使用模拟版本
try:
from scene.gaussian_model import GaussianModel  # type: ignore
from scene import dataset_readers               # type: ignore
    FULL_3DGS_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入完整的3DGS模块 ({e})，将使用简化模式")
    FULL_3DGS_AVAILABLE = False

# PLY文件处理始终可用
from plyfile import PlyData                     # type: ignore

# ───────────────────────── PLOT UTILS DYNAMIC IMPORT ─────────────────────────
plot_utils = None
try:
    plot_utils = importlib.import_module('plot_utils')
except ModuleNotFoundError:
    pass  # 若用户没要求 --plots，可忽略

# ─────────────────────────────────── HELPERS ──────────────────────────────────

# 简化的BasicPointCloud类用于模拟
class SimpleBasicPointCloud:
    def __init__(self, points, colors, normals):
        self.points = points
        self.colors = colors  
        self.normals = normals

def simple_fetchPly(path):
    """简化的PLY加载器，当dataset_readers不可用时使用"""
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    if len(vertices) == 0:
        positions = np.array([]).reshape(0, 3)
        colors = np.array([]).reshape(0, 3)
        normals = np.array([]).reshape(0, 3)
    else:
        positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return SimpleBasicPointCloud(points=positions, colors=colors, normals=normals)

def safe_ply_sh_degree(ply_path: str) -> int:
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']
    f_rest = [p for p in vertex.properties if p.name.startswith('f_rest_')]
    if not f_rest:
        return 0
    max_idx = max(int(p.name.split('_')[-1]) for p in f_rest)
    return int(np.sqrt((max_idx + 4) / 3) - 1)


def load_colmap(ply: str) -> np.ndarray:
    if not os.path.exists(ply):
        raise FileNotFoundError(f"COLMAP PLY文件不存在: {ply}")
    
    try:
        if FULL_3DGS_AVAILABLE:
            point_cloud = dataset_readers.fetchPly(ply)
        else:
            point_cloud = simple_fetchPly(ply)
            
        if point_cloud is None or point_cloud.points is None:
            raise ValueError(f"无法从PLY文件读取点云数据: {ply}")
        
        points = point_cloud.points.astype(np.float32)
        print(f"成功加载COLMAP点云: {len(points)} 个点 from {ply}")
        return points
    except Exception as e:
        raise RuntimeError(f"加载COLMAP PLY文件失败 {ply}: {str(e)}")


def simple_load_gaussians(ply: str, clean=True):
    """简化的高斯数据加载器，直接从PLY文件读取"""
    if not os.path.exists(ply):
        raise FileNotFoundError(f"高斯PLY文件不存在: {ply}")
    
    try:
        plydata = PlyData.read(ply)
        vertex = plydata['vertex']
        property_names = [p.name for p in vertex.properties]
        
        # 读取基本位置信息
        xyz = np.stack((np.asarray(vertex["x"]),
                       np.asarray(vertex["y"]),
                       np.asarray(vertex["z"])), axis=1).astype(np.float32)
        
        # 读取opacity
        if 'opacity' in property_names:
            alpha = np.asarray(vertex["opacity"]).astype(np.float32).reshape(-1)
        else:
            alpha = np.ones(len(xyz), dtype=np.float32)
        
        # 读取scale信息
        scale_names = [p for p in property_names if p.startswith("scale_")]
        if scale_names:
            scale = np.zeros((len(xyz), len(scale_names)), dtype=np.float32)
            for idx, attr_name in enumerate(sorted(scale_names)):
                scale[:, idx] = np.asarray(vertex[attr_name])
        else:
            # 如果没有scale信息，使用默认值
            scale = np.ones((len(xyz), 3), dtype=np.float32) * 0.01
        
        # 读取SH特征
        f_dc_names = [p for p in property_names if p.startswith("f_dc_")]
        f_rest_names = [p for p in property_names if p.startswith("f_rest_")]
        
        if f_dc_names and f_rest_names:
            # 计算SH度数
            max_idx = max([int(p.split('_')[-1]) for p in f_rest_names])
            sh = int(np.sqrt((max_idx + 4) / 3) - 1)
            
            # 读取DC分量 (l=0)
            features_dc = np.zeros((len(xyz), 3, 1), dtype=np.float32)
            for i, attr_name in enumerate(sorted(f_dc_names)):
                if i < 3:
                    features_dc[:, i, 0] = np.asarray(vertex[attr_name])
            
            # 读取其他分量 (l>0)
            if sh > 0 and f_rest_names:
                features_extra = np.zeros((len(xyz), len(f_rest_names)), dtype=np.float32)
                for idx, attr_name in enumerate(sorted(f_rest_names, key=lambda x: int(x.split('_')[-1]))):
                    features_extra[:, idx] = np.asarray(vertex[attr_name])
                
                # 重塑为 (N, 3, SH_coeffs_except_DC)
                sh_coeffs_except_dc = (sh + 1) ** 2 - 1
                features_extra = features_extra.reshape((len(xyz), 3, sh_coeffs_except_dc))
                
                # 拼接DC和其他分量: (N, 3, total_SH_coeffs)
                feats = np.concatenate([features_dc, features_extra], axis=2)
            else:
                feats = features_dc
        elif f_dc_names:
            # 只有DC分量，sh=0
            sh = 0
            features_dc = np.zeros((len(xyz), 3, 1), dtype=np.float32)
            for i, attr_name in enumerate(sorted(f_dc_names)):
                if i < 3:
                    features_dc[:, i, 0] = np.asarray(vertex[attr_name])
            feats = features_dc
        else:
            # 没有SH特征
            sh = 0
            feats = None
        
        print(f"成功加载高斯数据: {len(xyz)} 个高斯球 from {ply} (简化模式)")
        
        if clean:
            mask = ~(np.isnan(xyz).any(1) | np.isnan(scale).any(1) | np.isnan(alpha))
            xyz, scale, alpha = xyz[mask], scale[mask], alpha[mask]
            feats = feats[mask] if feats is not None else None
            print(f"清理后剩余: {len(xyz)} 个有效高斯球")
        
        return xyz, scale, alpha, feats, sh
    except Exception as e:
        raise RuntimeError(f"加载高斯PLY文件失败 {ply}: {str(e)}")

def load_gaussians(ply: str, clean=True):
    if not os.path.exists(ply):
        raise FileNotFoundError(f"高斯PLY文件不存在: {ply}")
    
    try:
        if FULL_3DGS_AVAILABLE:
            # 完整模式：使用3DGS的GaussianModel
    sh = safe_ply_sh_degree(ply)

    try:
        g = GaussianModel(sh, device='cpu')
    except TypeError:
        g = GaussianModel(sh)
        if hasattr(g, 'to'):
            g.to('cpu')

    g.load_ply(ply, use_train_test_exp=False)

    # ---- 关键改动：加 .detach() ----
    xyz   = g.get_xyz     .detach().cpu().numpy().astype(np.float32)
    scale = g.get_scaling .detach().cpu().numpy().astype(np.float32)
    alpha = g.get_opacity .detach().cpu().numpy().astype(np.float32).reshape(-1)
    feats = (g.get_features.detach().cpu().numpy().astype(np.float32)
            if sh > 0 else None)

    # --------------------------------
            print(f"成功加载高斯数据: {len(xyz)} 个高斯球 from {ply}")

    if clean:
        mask  = ~(np.isnan(xyz).any(1) | np.isnan(scale).any(1) | np.isnan(alpha))
        xyz, scale, alpha = xyz[mask], scale[mask], alpha[mask]
        feats = feats[mask] if feats is not None else None
                print(f"清理后剩余: {len(xyz)} 个有效高斯球")

    return xyz, scale, alpha, feats, sh
        else:
            # 简化模式：直接从PLY读取
            return simple_load_gaussians(ply, clean)
    except Exception as e:
        raise RuntimeError(f"加载高斯PLY文件失败 {ply}: {str(e)}")




def gentle_filter(scale: np.ndarray, pct: float):
    mags = np.linalg.norm(scale, axis=1)
    lo, hi = np.percentile(mags,[pct,100-pct])
    return (mags>=lo)&(mags<=hi)&(scale.min(1)>1e-6)


def stratified_sample(idxs: np.ndarray, bins: List[np.ndarray], ratio: float):
    if ratio>=0.999:
        return idxs
    sample=[]
    for b in bins:
        if len(b):
            n=max(1,int(len(b)*ratio))
            sample.append(np.random.choice(b,n,False))
    return np.concatenate(sample)

# ─────────────────────────── GPU helpers (Faiss / KeOps) ─────────────────────

def faiss_count(points: np.ndarray, ref: np.ndarray, r: float, gpu=True):
    r2 = r*r
    index = faiss.IndexFlatL2(3)
    if gpu:
        res = faiss.StandardGpuResources(); index=faiss.index_cpu_to_gpu(res,0,index)
    index.add(ref)
    lim,_,_=index.range_search(points,r2)
    return np.diff(lim).astype(np.int32)


def keops_scale_density(colmap: np.ndarray, g_xyz: np.ndarray, g_r: np.ndarray):
    P = keops.LazyTensor(torch.from_numpy(colmap).cuda()[:,None,:])
    G = keops.LazyTensor(torch.from_numpy(g_xyz).cuda()[None,:,:])
    dist2=((P-G)**2).sum(-1)
    inside=(dist2<torch.from_numpy(g_r**2).cuda()[None,:])
    return inside.sum(1).cpu().numpy()

# ──────────────────────────────── ANALYZER ───────────────────────────────────
class Analyzer:
    def __init__(self, colmap_ply:str, gauss_ply:str, args):
        self.args=args; self.device=args.device
        self.colmap=load_colmap(colmap_ply)
        
        # 添加COLMAP数据检查
        if len(self.colmap) == 0:
            raise ValueError(f"COLMAP点云数据为空，请检查文件路径: {colmap_ply}")
        
        g_xyz, g_scale, g_alpha, g_feat, sh = load_gaussians(gauss_ply)
        
        # 添加高斯数据检查
        if len(g_xyz) == 0:
            raise ValueError(f"高斯数据为空，请检查文件路径: {gauss_ply}")
            
        if args.gentle>0:
            mask=gentle_filter(g_scale,args.gentle);g_xyz,g_scale,g_alpha=(g_xyz[mask],g_scale[mask],g_alpha[mask])
            g_feat=g_feat[mask] if g_feat is not None else None
            
        # 再次检查过滤后的数据
        if len(g_xyz) == 0:
            raise ValueError(f"经过gentle filter后高斯数据为空，请调整--gentle参数 (当前: {args.gentle})")
            
        self.g_xyz,self.g_scale,self.g_alpha,self.g_feat,self.sh=g_xyz,g_scale,g_alpha,g_feat,sh

        mags = (g_scale if g_scale.ndim == 1 
            else np.linalg.norm(g_scale, axis=1))
        self.layers=np.array(args.layers)
        bins=[np.where(mags<np.percentile(mags,self.layers[0]))[0]]
        for lo,hi in zip(self.layers[:-1],self.layers[1:]):
            bins.append(np.where((mags>=np.percentile(mags,lo))&(mags<np.percentile(mags,hi)))[0])
        bins.append(np.where(mags>=np.percentile(mags,self.layers[-1]))[0])
        self.bins=bins
        g_idx=stratified_sample(np.arange(len(mags)),bins,args.sample)
        
        # 修复采样问题：确保有足够的数据进行采样
        colmap_sample_size = max(1, int(len(self.colmap) * args.sample))
        if colmap_sample_size > len(self.colmap):
            colmap_sample_size = len(self.colmap)
            print(f"警告: 采样比例过大，使用全部COLMAP数据 ({len(self.colmap)} 个点)")
            
        c_idx = np.random.choice(len(self.colmap), colmap_sample_size, False) if colmap_sample_size < len(self.colmap) else np.arange(len(self.colmap))
        
        self.colmap_s,self.g_xyz_s,self.g_scale_s=self.colmap[c_idx],g_xyz[g_idx],g_scale[g_idx]

    def basic(self):
        return{'count':{'colmap':int(len(self.colmap)),'gauss':int(len(self.g_xyz))},
               'density_ratio':float(len(self.g_xyz)/len(self.colmap)),
               'bounds':{'colmap':np.ptp(self.colmap,0).tolist(),'gauss':np.ptp(self.g_xyz,0).tolist()}}

    def scale_stats(self):
        mags=np.linalg.norm(self.g_scale, axis=1)
        stats={'mean':float(mags.mean()),'std':float(mags.std()),
               'percentiles':{p:float(np.percentile(mags,p)) for p in [25,50,75,90]}}
        # SH spectrum
        if self.sh>0 and self.g_feat is not None:
            L=int(np.sqrt(self.g_feat.shape[2])-1)
            energy=np.square(self.g_feat).sum(1) if self.g_feat.ndim==3 else np.square(self.g_feat)
            per_l=[]
            idx=0
            for l in range(L+1):
                n=(2*l+1)
                per_l.append(energy[:,idx:idx+n].sum(1)); idx+=n
            per_l=np.stack(per_l,1)  # (N,L+1)
            low=np.sum(per_l[:,:2],1);high=np.sum(per_l[:,2:],1)+1e-9
            stats['sh_low_high_ratio']=float((low/high).mean())
            self.scale_freq=(np.log10(mags+1e-6),np.log10(per_l[:,2:].sum(1)+1e-9))
        else:
            # 如果没有SH特征，设置为None
            self.scale_freq = None
        return stats

    def densities(self):
        r=self.args.radius
        if self.device=='cuda' and faiss:
            trad=faiss_count(self.colmap_s,self.g_xyz_s,r,True)
        else:
            from sklearn.neighbors import KDTree;trad=KDTree(self.g_xyz_s).query_radius(self.colmap_s,r,count_only=True)
        if self.device=='cuda' and keops:
            scale_aware=keops_scale_density(self.colmap_s,self.g_xyz_s,self.g_scale_s.max(1))
        else:
            eff=self.g_scale_s.max(1);scale_aware=[((np.linalg.norm(self.g_xyz_s-p,axis=1)<=eff).sum()) for p,eff in zip(self.colmap_s,eff)]
        self.density_vec=np.array(trad)
        return{'traditional_mean':float(np.mean(trad)),'scale_aware_mean':float(np.mean(scale_aware))}

    def correlations(self):
        from sklearn.neighbors import KDTree
        tree=KDTree(self.colmap_s)
        _,nn=tree.query(self.colmap_s,k=self.args.kgrad+1)
        strengths=np.zeros(len(self.colmap_s))
        from sklearn.decomposition import PCA
        pca=PCA(3)
        for i,ind in enumerate(nn):
            vecs=self.colmap_s[ind[1:]]-self.colmap_s[i]
            if vecs.var():
                pca.fit(vecs);strengths[i]=pca.explained_variance_[0]
        rho_p=pearsonr(strengths,self.density_vec)[0]
        out={'global_pearson':float(rho_p)}
        if self.args.corr_layers:
            # 重新实现分层相关性分析：基于COLMAP点的密度进行分层
            layer_metrics = {}
            
            # 使用density_vec（基于COLMAP采样点）进行分层
            density_percentiles = [0] + list(self.layers) + [100]
            
            for i, (lo, hi) in enumerate(zip(density_percentiles[:-1], density_percentiles[1:])):
                # 基于密度的百分位进行分层
                density_lo = np.percentile(self.density_vec, lo)
                density_hi = np.percentile(self.density_vec, hi)
                
                # 在采样的COLMAP点中找到该密度范围的点
                if hi == 100:  # 最后一层包含等于上限的点
                    mask_density = (self.density_vec >= density_lo) & (self.density_vec <= density_hi)
                else:
                    mask_density = (self.density_vec >= density_lo) & (self.density_vec < density_hi)
                
                if mask_density.any() and np.sum(mask_density) > 1:  # 至少需要2个点才能计算相关性
                    try:
                        # 计算该层的相关性
                        layer_strengths = strengths[mask_density]
                        layer_densities = self.density_vec[mask_density]
                        
                        # 确保有足够的变化性来计算相关性
                        if len(set(layer_strengths)) > 1 and len(set(layer_densities)) > 1:
                            pearson_corr = pearsonr(layer_strengths, layer_densities)[0]
                            spearman_corr = spearmanr(layer_strengths, layer_densities)[0]
                            
                            layer_metrics[f'density_{lo}-{hi}'] = {
                                'pearson': float(pearson_corr) if not np.isnan(pearson_corr) else 0.0,
                                'spearman': float(spearman_corr) if not np.isnan(spearman_corr) else 0.0,
                                'count': int(np.sum(mask_density)),
                                'density_range': [float(density_lo), float(density_hi)]
                            }
                    except Exception as e:
                        print(f"警告: 密度层 {lo}-{hi} 相关性计算失败: {e}")
                        
            out['layers'] = layer_metrics
        self.grad=strengths
        return out

    def layer_counts(self):
        mags=np.linalg.norm(self.g_scale,axis=1)
        counts=[];alphas=[]
        pivots=[0]+list(self.layers)+[100]
        for lo,hi in zip(pivots[:-1],pivots[1:]):
            mask=(mags>=np.percentile(mags,lo))&(mags<np.percentile(mags,hi))
            counts.append(int(mask.sum()))
            alphas.append(float(self.g_alpha[mask].mean()))
        return{'layers':list(zip([f'{pivots[i]}-{pivots[i+1]}' for i in range(len(counts))],counts,alphas))}

    # 2D hist
    def scale_freq_hist(self,out_dir):
        if self.sh>0 and plot_utils and hasattr(self, 'scale_freq') and self.scale_freq is not None:
            xs, ys = self.scale_freq
            # 确保xs和ys是数组且长度相同
            if isinstance(xs, np.ndarray) and isinstance(ys, np.ndarray) and len(xs) > 0 and len(ys) > 0 and len(xs) == len(ys):
            plot_utils.plot_2d_hist(xs,ys,out_dir/'scale_freq.png','log(scale)','log(high_freq_energy)')
            else:
                print(f"警告: scale_freq数据格式不正确，跳过scale_freq直方图生成")
        else:
            print(f"警告: 无SH特征数据或plot_utils不可用，跳过scale_freq直方图生成")

    # plot visuals
    def plots(self,out_dir):
        if not plot_utils: return
        plot_utils.heatmap_xy(self.g_xyz,out_dir/'xy_heatmap.png')
        plot_utils.scale_hist(np.linalg.norm(self.g_scale,axis=1),out_dir/'scale_hist.png')
        # hotspot scatter (top 2% grad & density)
        qg,qc=np.quantile(self.grad,0.98),np.quantile(self.density_vec,0.98)
        mask=(self.grad>qg)&(self.density_vec>qc)
        plot_utils.scatter3d(self.colmap_s[mask],out_dir/'hotspots.png')
        self.scale_freq_hist(out_dir)

    def run(self):
        rep={'sample_ratio':self.args.sample,
             'basic':self.basic(),
             'scale':self.scale_stats(),
             'density':self.densities(),
             'corr':self.correlations(),
             'layer_counts':self.layer_counts()}
        out_dir=Path(self.args.out)
        if self.args.plots:
            self.plots(out_dir)
        return rep

# ───────────────────────────── CLI ───────────────────────────────────────────

def parse_layers(s:str)->List[int]:
    return[float(i) for i in s.split(',')]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--colmap',required=True);ap.add_argument('--gaussians',required=True)
    ap.add_argument('--sample',type=float,default=0.02)
    ap.add_argument('--kgrad',type=int,default=12)
    ap.add_argument('--radius',type=float,default=0.1)
    ap.add_argument('--gentle',type=float,default=0.0)
    ap.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    ap.add_argument('--layers',type=parse_layers,default=[10,25,50,75,90])
    ap.add_argument('--corr_layers',action='store_true')
    ap.add_argument('--plots',action='store_true')
    ap.add_argument('--out',default='hybrid_report')
    args=ap.parse_args()

    Path(args.out).mkdir(exist_ok=True)
    A=Analyzer(args.colmap,args.gaussians,args)
    rep=A.run()
    with open(Path(args.out)/'summary.json','w') as f:
        json.dump(rep,f,indent=2)
    print('✅ summary saved:',Path(args.out)/'summary.json')

if __name__=='__main__':
    main()
