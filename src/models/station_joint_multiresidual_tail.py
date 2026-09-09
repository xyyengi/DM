"""Joint wind/solar multiresolution residual tail for Station-24.

The module is an additive epsilon correction on a frozen Raw body.  It uses an
orthogonal Haar projection so slow and fast corrections have exact, measurable
roles.  Forecast decomposition is condition-only; no future actual/residual is
accepted by this interface.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels: int, requested: int) -> int:
    for value in range(min(channels, requested),0,-1):
        if channels%value==0:return value
    return 1


def lag_difference(value: torch.Tensor, lag: int) -> torch.Tensor:
    result=torch.zeros_like(value);result[...,lag:]=value[...,lag:]-value[...,:-lag]
    return result


def moving_average_same(value: torch.Tensor, width: int) -> torch.Tensor:
    """Same-length average of a fully available forecast trajectory."""
    if width < 1:
        raise ValueError("moving-average width must be positive")
    left = (width - 1) // 2
    right = width - 1 - left
    padded = F.pad(value, (left, right), mode="replicate")
    return F.avg_pool1d(
        padded.reshape(-1, 1, padded.shape[-1]), width, stride=1
    ).reshape_as(value)


class HaarMultiresolution(nn.Module):
    """Orthogonal projection onto blockwise Haar scaling space.

    At level=3, low contains the >~8 h block structure and high contains the
    exact complementary details. Sequence length must be divisible by 2**level.
    """
    def __init__(self,levels: int=3):
        super().__init__();self.levels=int(levels)
        if self.levels<1:raise ValueError('Haar levels must be positive')

    def low(self,value: torch.Tensor) -> torch.Tensor:
        width=2**self.levels
        if value.shape[-1]%width:raise ValueError('time length must be divisible by Haar width')
        shape=value.shape;blocks=value.reshape(*shape[:-1],shape[-1]//width,width)
        return blocks.mean(-1,keepdim=True).expand_as(blocks).reshape_as(value)

    def split(self,value: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        low=self.low(value);return low,value-low


@dataclass
class JointMultiresidualOutput:
    correction: torch.Tensor
    slow_correction: torch.Tensor
    fast_correction: torch.Tensor
    forecast_slow: torch.Tensor
    forecast_fast: torch.Tensor
    local_raw: torch.Tensor
    system_raw: torch.Tensor


class JointMultiresolutionResidualTail(nn.Module):
    """One joint tail, not three separately routed experts.

    Local type channels and a shared system channel are combined before one
    orthogonal slow/fast reconstruction. The diffusion noise is the stochastic
    source; this module contains no binary issue/onset classifier.
    """
    def __init__(self,hidden_channels: int,station_features: torch.Tensor,
                 adjacency: torch.Tensor,station_capacities: torch.Tensor,
                 channels: int=24,groups: int=8,haar_levels: int=3):
        super().__init__();self.station_count=int(station_features.shape[0])
        self.haar=HaarMultiresolution(haar_levels)
        graph=adjacency.float();degree=graph.sum(-1).clamp(min=1e-8)
        graph=degree.rsqrt()[:,None]*graph*degree.rsqrt()[None,:]
        self.register_buffer('graph',graph);self.register_buffer('station_features',station_features.float())
        cap=station_capacities.float().clamp(min=1e-8);wind=station_features[:,0].float();solar=station_features[:,1].float()
        self.register_buffer('wind_mask',wind);self.register_buffer('solar_mask',solar)
        self.register_buffer('wind_weight',cap*wind/(cap*wind).sum().clamp(min=1e-8))
        self.register_buffer('solar_weight',cap*solar/(cap*solar).sum().clamp(min=1e-8))
        norm=lambda:nn.GroupNorm(_groups(channels,groups),channels)
        self.hidden=nn.Sequential(nn.Conv1d(hidden_channels,channels,1),norm(),nn.SiLU())
        self.fast_condition=nn.Sequential(nn.Conv1d(6,channels,5,padding=2),norm(),nn.SiLU())
        self.slow_condition=nn.Sequential(nn.Conv1d(6,channels,5,padding=2),norm(),nn.SiLU())
        self.fast_fuse=nn.Sequential(nn.Conv1d(2*channels,channels,5,padding=2),norm(),nn.SiLU(),nn.Conv1d(channels,channels,3,padding=2,dilation=2),nn.SiLU())
        self.slow_fuse=nn.Sequential(nn.Conv1d(2*channels,channels,5,padding=2),norm(),nn.SiLU(),nn.Conv1d(channels,channels,3,padding=4,dilation=4),nn.SiLU())
        # Two physical types are channels of one joint correction, not routes.
        self.local_fast=nn.Conv1d(channels,2,1);self.local_slow=nn.Conv1d(channels,2,1)
        self.system=nn.Sequential(nn.Conv1d(8,channels,5,padding=2),norm(),nn.SiLU(),nn.Conv1d(channels,channels,3,padding=1),nn.SiLU())
        self.system_fast=nn.Conv1d(channels,2,1);self.system_slow=nn.Conv1d(channels,2,1)
        self.system_loading=nn.Linear(station_features.shape[1],2,bias=True)
        for head in (self.local_fast,self.local_slow,self.system_fast,self.system_slow):
            nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
        nn.init.normal_(self.system_loading.weight,mean=0,std=.02);nn.init.zeros_(self.system_loading.bias)

    def forecast_components(self,forecast: torch.Tensor) -> tuple[torch.Tensor,torch.Tensor]:
        return self.haar.split(forecast)

    def _conditions(self,forecast,recent_error,recent_error_mask):
        b,s,l=forecast.shape;low,high=self.forecast_components(forecast)
        d1=lag_difference(forecast,1);d3=lag_difference(forecast,3);d6=lag_difference(forecast,6)
        fast=torch.stack([high,d1,d3,d6,lag_difference(d1,1),torch.einsum('ij,bjt->bit',self.graph,d3)],2)
        r6=torch.zeros(b,s,device=forecast.device,dtype=forecast.dtype);r24=torch.zeros_like(r6)
        if recent_error is not None:
            avail=torch.ones_like(r6) if recent_error_mask is None else recent_error_mask[...,0].to(forecast.dtype)
            r6=recent_error[...,-min(6,recent_error.shape[-1]):].mean(-1)*avail;r24=recent_error.mean(-1)*avail
        neighbor=torch.einsum('ij,bj->bi',self.graph,r24)
        # The 24 h forecast trend is a genuinely coarser known-ahead condition;
        # it is not an actual/residual-derived target and therefore adds no
        # future-observation leakage at inference time.
        day_trend=moving_average_same(forecast,24)
        slow=torch.stack([low,day_trend,lag_difference(low,8),r6[...,None].expand(-1,-1,l),r24[...,None].expand(-1,-1,l),neighbor[...,None].expand(-1,-1,l)],2)
        def agg(x,w):return torch.einsum('s,bst->bt',w,x)
        system=torch.stack([agg(low,self.wind_weight),agg(low,self.solar_weight),agg(high,self.wind_weight),agg(high,self.solar_weight),agg(d3,self.wind_weight),agg(d3,self.solar_weight),torch.einsum('s,bs->b',self.wind_weight,r24)[:,None].expand(-1,l),torch.einsum('s,bs->b',self.solar_weight,r24)[:,None].expand(-1,l)],1)
        return fast,slow,system,low,high

    def forward(self,hidden,forecast,recent_error=None,recent_error_mask=None,route=0.0):
        b,s,c,l=hidden.shape
        if forecast.shape!=(b,s,l) or s!=self.station_count:raise ValueError('joint tail shape mismatch')
        fast,slow,system,forecast_low,forecast_high=self._conditions(forecast,recent_error,recent_error_mask)
        h=self.hidden(hidden.reshape(b*s,c,l));ff=self.fast_fuse(torch.cat([h,self.fast_condition(fast.reshape(b*s,6,l))],1));sf=self.slow_fuse(torch.cat([h,self.slow_condition(slow.reshape(b*s,6,l))],1))
        type_mask=torch.stack([self.wind_mask,self.solar_mask],-1)[None,:,:,None]
        local_fast=(self.local_fast(ff).reshape(b,s,2,l)*type_mask).sum(2);local_slow=(self.local_slow(sf).reshape(b,s,2,l)*type_mask).sum(2)
        sy=self.system(system);modes_fast=self.system_fast(sy);modes_slow=self.system_slow(sy)
        loading=torch.tanh(self.system_loading(self.station_features))
        common_fast=torch.einsum('sk,bkt->bst',loading,modes_fast);common_slow=torch.einsum('sk,bkt->bst',loading,modes_slow)
        slow_corr=self.haar.low(local_slow+common_slow);_,fast_corr=self.haar.split(local_fast+common_fast)
        if isinstance(route,(int,float)):route_tensor=torch.full((b,1,1),float(route),device=hidden.device,dtype=hidden.dtype)
        else:
            route_tensor=route.to(hidden.dtype)
            if route_tensor.ndim==1:route_tensor=route_tensor[:,None,None]
        if route_tensor.shape!=(b,1,1):raise ValueError('route must be scalar, [B], or [B,1,1]')
        if torch.any((route_tensor < 0) | (route_tensor > 1)):
            raise ValueError('route values must lie in [0, 1]')
        return JointMultiresidualOutput(route_tensor*(slow_corr+fast_corr),route_tensor*slow_corr,route_tensor*fast_corr,forecast_low,forecast_high,local_slow+local_fast,common_slow+common_fast)
