"""Versioned joint full-trajectory model inheriting Station24 condition priors.

No body/tail routing. Graphs and thresholds are frozen train-only data assets,
not pretrained network weights. Structural losses supervise x0 reconstructions,
not a proper score of the final sampled ensemble.
"""
import torch
from torch import nn
import torch.nn.functional as F
from .station_diffusion_ts import StationDiffusionTS


class StationDiffusionTSInherited(StationDiffusionTS):
    VERSION = "diffusion_ts_joint_inherited_v2"
    CONDITION_KEYS = ("forecast", "recent_error", "recent_error_mask", "calendar", "lead", "node_state")

    def __init__(self, config, assets):
        super().__init__(config)
        width = config["d_model"]
        for name in ("geographic", "historical", "station_features", "capacities"):
            self.register_buffer(name, torch.as_tensor(assets[name], dtype=torch.float32))
        for name in ("geographic", "historical"):
            adjacency = getattr(self, name)
            if adjacency.shape != (24,24) or not torch.isfinite(adjacency).all() or (adjacency < 0).any() or (adjacency.sum(1) <= 0).any():
                raise ValueError(f"invalid {name} graph")
        self.local_stem = nn.Linear(1+4+5, 8)
        self.node_projection = nn.Linear(24*8, width)
        self.graph_projection = nn.Linear(24*8, width)
        self.graph_logits = nn.Parameter(torch.tensor([2., 0.]))
        self.graph_gate = nn.Parameter(torch.tensor(-1.))
        self.calendar_embedding = nn.Linear(10, width)
        self.encoder_film = nn.ModuleList([nn.Linear(width, 2*width) for _ in self.backbone.encoder.blocks])
        self.decoder_film = nn.ModuleList([nn.Linear(width, 2*width) for _ in self.backbone.decoder.blocks])
        self.encoder_norm = nn.ModuleList([nn.LayerNorm(width, elementwise_affine=False) for _ in self.encoder_film])
        self.decoder_norm = nn.ModuleList([nn.LayerNorm(width, elementwise_affine=False) for _ in self.decoder_film])
        # Small, nonzero modulation avoids suppressing upstream gradients at start.
        for layer in list(self.encoder_film)+list(self.decoder_film):
            nn.init.normal_(layer.weight, std=.01); nn.init.zeros_(layer.bias)

    @staticmethod
    def modulate(x, c, affine, norm):
        gamma, beta = affine(c).chunk(2, -1)
        return x + norm(x)*torch.tanh(gamma) + beta

    def spatial_condition(self, batch, forecast):
        b,t,s = forecast.shape
        state = batch["node_state"].float().permute(0,3,1,2)
        static = self.station_features[None,None].expand(b,t,-1,-1)
        nodes = F.silu(self.local_stem(torch.cat([forecast[...,None]*2-1, state, static], -1)))
        graphs = torch.stack([self.geographic, self.historical])
        graphs = graphs/graphs.sum(-1,keepdim=True)
        graph = torch.einsum("k,kij->ij",self.graph_logits.softmax(0),graphs)
        neighbors = torch.einsum("ij,btjc->btic", graph, nodes)
        return self.node_projection(nodes.flatten(2)) + self.graph_gate.sigmoid()*self.graph_projection(neighbors.flatten(2))

    def components(self, noisy, timestep, batch):
        forecast,recent,mask = self.conditions(batch)
        c = self.forecast_embedding(forecast*2-1) + self.spatial_condition(batch,forecast)
        c = c + self.calendar_embedding(torch.cat([batch["calendar"],batch["lead"]],1).float().transpose(1,2))
        emb = self.backbone.emb(noisy)
        x = self.backbone.pos_enc(emb+c)
        for block,affine,norm in zip(self.backbone.encoder.blocks,self.encoder_film,self.encoder_norm):
            x = self.modulate(x,c,affine,norm)
            x,_ = block(x,timestep)
        history = self.history_embedding(torch.cat([recent,mask],-1)) + self.history_position
        memory = torch.cat([x,history],1)
        x = self.backbone.pos_dec(emb+c)
        means=[]; trends=[]; seasons=[]
        for block,affine,norm in zip(self.backbone.decoder.blocks,self.decoder_film,self.decoder_norm):
            x = self.modulate(x,c,affine,norm)
            x,m,tr,se = block(x,memory,timestep)
            means.append(m); trends.append(tr); seasons.append(se)
        residual=self.backbone.inverse(x); mean=residual.mean(1,keepdim=True)
        trend=self.backbone.combine_m(torch.cat(means,1))+mean+sum(trends)
        seasonal=self.backbone.combine_s(sum(seasons).transpose(1,2)).transpose(1,2)
        return trend.float(), seasonal.float(), (residual-mean).float()

    def structure_losses(self, prediction, clean, valid, forecast):
        def average(x,m):
            return (x*m).sum((1,2))/m.sum((1,2)).clamp(min=1)
        ramps=[]
        for lag in (1,3,6):
            m=valid[:,lag:]*valid[:,:-lag]
            real=clean[:,lag:]-clean[:,:-lag]
            pred=prediction[:,lag:]-prediction[:,:-lag]
            ramps.append(average((pred-real).abs(),m))
        slow=[]
        for window in (12,24):
            p=prediction.unfold(1,window,1).mean(-1)
            y=clean.unfold(1,window,1).mean(-1)
            m=valid.unfold(1,window,1).amin(-1)
            slow.append(average((p-y).abs(),m))
        # Signed cross-station residual products, evaluated at the same hour.
        # This explicitly constrains co-anomalies, not just a weekly correlation.
        p=(prediction-forecast)*.5; y=(clean-forecast)*.5
        pair_mask=valid[:,:,:,None]*valid[:,:,None,:]
        pair_mask=pair_mask*(1-torch.eye(24,device=clean.device))[None,None]
        pairs=(p[:,:,:,None]*p[:,:,None,:]-y[:,:,:,None]*y[:,:,None,:]).abs()
        joint=(pairs*pair_mask).sum((1,2,3))/pair_mask.sum((1,2,3)).clamp(min=1)
        groups=[]
        for index in (0,1):
            w=self.capacities*self.station_features[:,index]
            groups.append(w/w.sum())
        groups.append(self.capacities/self.capacities.sum())
        weights=torch.stack(groups)
        agg_error=torch.einsum("bts,gs->btg",prediction-clean,weights).abs()
        agg_valid=(valid[:,:,None,:]+(weights==0)[None,None]).amin(-1).clamp(max=1)
        aggregate=average(agg_error,agg_valid)
        return {"ramp":sum(ramps)/3, "slow":sum(slow)/2, "synchrony":joint, "aggregate":aggregate}

    def forward(self,batch,timestep=None,noise=None):
        clean=batch["actual"].float().transpose(1,2)*2-1
        valid=batch["valid_mask"].float().transpose(1,2)
        if timestep is None: timestep=torch.randint(self.num_steps,(len(clean),),device=clean.device)
        if noise is None: noise=torch.randn_like(clean)
        a=self.alpha_bar[timestep,None,None]
        pred=self.predict(a.sqrt()*clean+(1-a).sqrt()*noise,timestep,batch)
        with torch.amp.autocast(clean.device.type,enabled=False):
            pred=pred.float()
            rec=((pred-clean).abs()*valid).sum((1,2))/valid.sum((1,2)).clamp(min=1)
            delta=torch.fft.fft(pred,dim=1,norm="forward")-torch.fft.fft(clean,dim=1,norm="forward")
            complete=valid.amin(1)[:,None,:]
            fft=((delta.real.abs()+delta.imag.abs())*complete).sum((1,2))/(complete.sum((1,2))*168).clamp(min=1)
            parts={"reconstruction":rec,"fourier":fft,**self.structure_losses(pred,clean,valid,batch["forecast"].float().transpose(1,2)*2-1)}
            weights={"reconstruction":1.,"fourier":self.config["fourier_weight"],**self.config["structure_weights"]}
            weighted={k:(v*self.loss_weight[timestep]).mean() for k,v in parts.items()}
            loss=sum(weights[k]*v for k,v in weighted.items())
        return loss,{k:v.detach() for k,v in weighted.items()}
