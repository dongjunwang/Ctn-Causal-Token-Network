from __future__ import annotations
import torch
from torch import nn

class Heads(nn.Module):
    def __init__(self,h):
        super().__init__(); self.y_num=nn.Linear(h,6); self.y_phase=nn.Linear(h,2); self.s_num=nn.Linear(h,4); self.s_phase=nn.Linear(h,2)
    def forward(self,z): return {"y_num":self.y_num(z),"y_phase":self.y_phase(z),"s_num":self.s_num(z),"s_phase":self.s_phase(z)}

class XY(nn.Module):
    def __init__(self,dx,dc,ds,domain_dim=3,h=64):
        super().__init__(); self.net=nn.Sequential(nn.Linear(dx+domain_dim,h),nn.GELU(),nn.Linear(h,h),nn.GELU()); self.heads=Heads(h)
    def forward(self,x,c,s,d): return self.heads(self.net(torch.cat([x,d],-1)))

class MLPFull(nn.Module):
    def __init__(self,dx,dc,ds,domain_dim=3,h=64):
        super().__init__(); self.net=nn.Sequential(nn.Linear(dx+dc+ds+domain_dim,h),nn.GELU(),nn.Linear(h,h),nn.GELU()); self.heads=Heads(h)
    def forward(self,x,c,s,d): return self.heads(self.net(torch.cat([x,c,s,d],-1)))

class TransformerFull(nn.Module):
    def __init__(self,dx,dc,ds,domain_dim=3,h=32):
        super().__init__(); self.proj=nn.ModuleList([nn.Linear(dx,h),nn.Linear(dc,h),nn.Linear(ds,h),nn.Linear(domain_dim,h)]); self.pos=nn.Parameter(torch.zeros(1,4,h)); layer=nn.TransformerEncoderLayer(h,4,h*2,batch_first=True,dropout=0.,activation="gelu"); self.encoder=nn.TransformerEncoder(layer,1); self.heads=Heads(h)
    def forward(self,x,c,s,d):
        z=torch.stack([p(v) for p,v in zip(self.proj,(x,c,s,d))],1)+self.pos
        return self.heads(self.encoder(z).mean(1))

class CTN(nn.Module):
    """Separate X/C/S encoders, gated pair relations, and dedicated state transition."""
    def __init__(self,dx,dc,ds,domain_dim=3,h=32,ablation=None):
        super().__init__(); self.ablation=ablation or set()
        self.xe=nn.Sequential(nn.Linear(dx,h),nn.GELU()); self.ce=None if "no_c" in self.ablation else nn.Sequential(nn.Linear(dc,h),nn.GELU()); self.se=None if "no_s" in self.ablation else nn.Sequential(nn.Linear(ds,h),nn.GELU()); self.de=nn.Linear(domain_dim,h)
        self.g_xc=nn.Parameter(torch.ones(())); self.g_xs=nn.Parameter(torch.ones(())); self.g_cs=nn.Parameter(torch.ones(())); self.g_all=nn.Parameter(torch.ones(()))
        self.relation=nn.Sequential(nn.Linear(h*2,h),nn.GELU(),nn.Linear(h,h),nn.GELU())
        self.transition=None if "no_transition" in self.ablation else nn.Sequential(nn.Linear(h*3,h),nn.GELU(),nn.Linear(h,h),nn.GELU())
        self.fallback=nn.Sequential(nn.Linear(h*3,h),nn.GELU())
        self.heads=Heads(h); self.state_num=nn.Linear(h,4); self.state_phase=nn.Linear(h,2); self.activations={}
    def forward(self,x,c,s,d):
        xe=self.xe(x); ce=self.ce(c) if self.ce else torch.zeros_like(xe); se=self.se(s) if self.se else torch.zeros_like(xe); de=self.de(d)
        xc=xe*ce if self.ce and "no_xc" not in self.ablation else torch.zeros_like(xe)
        xs=xe*se if self.se and "no_xs" not in self.ablation else torch.zeros_like(xe)
        cs=ce*se if self.ce and self.se and "no_cs" not in self.ablation else torch.zeros_like(xe)
        all3=xe*ce*se if self.ce and self.se else torch.zeros_like(xe)
        rel=xe+self.g_xc*xc+self.g_xs*xs+self.g_cs*cs+self.g_all*all3
        rz=self.relation(torch.cat([rel,de],-1))
        if self.transition is not None: sz=self.transition(torch.cat([xe,ce,se],-1)); z=rz+sz
        else: sz=self.fallback(torch.cat([xe,ce,se],-1)); z=rz+sz
        self.activations={"gate_xc":float(self.g_xc.detach()),"gate_xs":float(self.g_xs.detach()),"gate_cs":float(self.g_cs.detach()),"gate_all":float(self.g_all.detach()),"mean_abs_xc":float(xc.detach().abs().mean()),"mean_abs_xs":float(xs.detach().abs().mean()),"mean_abs_cs":float(cs.detach().abs().mean())}
        out=self.heads(rz); out["s_num"]=self.state_num(sz); out["s_phase"]=self.state_phase(sz)
        return out
