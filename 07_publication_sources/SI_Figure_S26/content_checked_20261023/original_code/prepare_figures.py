from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,'C:/Users/Cheng Wang/.codex/skills/figure-style')
from kernel import apply_figure_style
apply_figure_style(sizes=(10,9,8))
H=Path(__file__).parent;A=H.parent/'defect_322_analysis_20260914_v1';rng=np.random.default_rng(25)
d=pd.read_csv(A/'diagnostic_counts.csv');rad=pd.read_csv(A/'radial_counts.csv');coverage=pd.read_csv(A/'condition_coverage.csv')
fig=plt.figure(figsize=(6.35,5.65));gs=fig.add_gridspec(2,2,left=.12,right=.98,bottom=.09,top=.91,wspace=.40,hspace=.57)
axs=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,:])]
colors=['#287a9c','#bd7138']
for i,mode in enumerate(['fixed','finite']):
    g=d[(d.tol==2)&(d.alpha==1.05)&(d.delta==0)&(d.rho==1)&(d['mode']==mode)&(d.cohort=='original_endpoint')]
    assert len(g)==32
    axs[0].scatter(i+rng.uniform(-.14,.14,32),g.vacancy_pct,s=10,alpha=.6,c=colors[i]);axs[0].plot([i-.24,i+.24],[g.vacancy_pct.median()]*2,c=colors[i],lw=2)
    r=rad[(rad.tol==2)&(rad.delta==0)&(rad.rho==1)&(rad['mode']==mode)&((rad.hi-rad.lo)==.5)].copy();r['p']=np.where(r.eligible>=10,100*r.missing/r.eligible,np.nan);p=r.pivot(index='id',columns='lo',values='p').dropna()
    for _,row in p.iterrows():axs[1].plot([3*i,3*i+1],row.values,c=colors[i],lw=.55,alpha=.25)
    axs[1].plot([3*i,3*i+1],p.median().values,'o-',c=colors[i],lw=2,ms=3,label=mode.capitalize())
axs[0].set(xticks=[0,1],xticklabels=['Fixed','Finite'],ylabel='Missing interior BDC (%)',title='Same 50,000-event endpoint',ylim=(-.3,6.7),xlim=(-.5,1.5))
axs[1].set(xticks=[0,1,3,4],xticklabels=['Inner','Outer','Inner','Outer'],ylabel='Missing interior BDC (%)',title='Radial distribution',ylim=(-.3,11.4));axs[1].legend(loc='upper right',fontsize=8)
for i,alpha in enumerate([.85,1.05,1.25]):
    g=d[(d.tol==2)&(d.alpha==alpha)&(d.delta==2)&(d.rho==1)&(d['mode']=='fixed')&(d.cohort=='size_followon')]
    axs[2].scatter(i+rng.uniform(-.1,.1,len(g)),g.vacancy_pct,s=10,c='#70559a',alpha=.6);axs[2].plot([i-.18,i+.18],[g.vacancy_pct.median()]*2,c='#70559a',lw=2)
axs[2].set(xticks=[0,1,2],xticklabels=['0.85','1.05','1.25'],ylabel='Missing interior BDC (%)',xlabel='Envelope parameter α',title='N = 1000: geometric-analysis sensitivity',ylim=(-.4,10),xlim=(-.45,2.45))
for ax,l in zip(axs,'ABC'):
    b=ax.get_position();fig.text(b.x0-.08,b.y1+.025,l,weight='bold',fontsize=12)
for ext in ['png','pdf','svg']:fig.savefig(H/f'Figure_S58.{ext}',bbox_inches=None)
plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(6.35,3.8),layout='constrained')
for ax,mode in zip(axs,['fixed','finite']):
    mat=np.zeros((6,3),int)
    for r in coverage[coverage['mode']==mode].itertuples():mat[int(r.delta/2),[.1,1,3].index(r.rho)]=int(r.original_endpoint+r.size_followon)
    im=ax.imshow(mat,vmin=0,vmax=32,cmap='Blues',aspect='auto')
    for i in range(6):
        for j in range(3):ax.text(j,i,str(mat[i,j]),ha='center',va='center',color='white' if mat[i,j]>18 else 'black',fontsize=9)
    ax.set(xticks=range(3),xticklabels=['0.1','1','3'],yticks=range(6),yticklabels=range(0,12,2),xlabel=r'$\rho_0$',ylabel=r'$\Delta=U_D/(RT)$',title=mode.capitalize()+' pool')
fig.colorbar(im,ax=axs,label='Available structures / 32',shrink=.8)
for ext in ['png','pdf','svg']:fig.savefig(H/f'Figure_S59.{ext}')
plt.close(fig)
