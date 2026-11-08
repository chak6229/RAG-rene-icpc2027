"""Standalone scientific figure of the predeclared paired configuration changes."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/rene_cqc_training_fidelity_20261020'


def main():
    result=json.loads((OUT/'RESULTS.json').read_text(encoding='utf-8-sig'))
    assert json.loads((OUT/'EFFECT_AUDIT.json').read_text(encoding='utf-8-sig'))['passed']
    records={(r['domain'],r['metric'],r['arm']):r for r in result['primary_changes']}
    arms=['O','S','N','W','P','PN','PS']
    labels=['Optimizer: Adafactor','Schedule: linear 100,000','Length: 128','Weight decay: 0',
        'Public hyperparameter bundle','Progress-scaled schedule','Public bundle, seed 42']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(13,8.2),sharey=True)
    for i,metric in enumerate(['S3','top1_gain']):
        for j,domain in enumerate(['source','cosqa']):
            ax=axes[i,j]
            for y,arm in enumerate(arms):
                row=records[domain,metric,arm];point=row['mean'];bounds=row['family28_ci95_group_bootstrap']
                tbounds=row['family28_ci95_cluster_t'];color='#246ca6' if metric=='S3' else '#b45d26'
                ax.plot(tbounds,[y,y],color='#aaa',linewidth=1.3,zorder=1)
                ax.plot(bounds,[y,y],color=color,linewidth=3,zorder=2)
                ax.scatter([point],[y],s=27,color=color,zorder=3)
            ax.axvline(0,color='#444',linewidth=.9,linestyle='--')
            ax.set_yticks(np.arange(len(arms)),labels)
            ax.set_ylim(6.6,-.6);ax.grid(axis='x',alpha=.16)
            ax.set_title(('Source' if domain=='source' else 'CoSQA development')+' | '+
                ('ENTR best3 - exact RAND best3' if metric=='S3' else 'ENTR Top1 - NoRewrite'),fontsize=11)
            ax.set_xlabel('Paired change relative to budget-matched local configuration L')
            ax.spines[['top','right']].set_visible(False)
    fig.suptitle('CQC training configuration sensitivity at the same data and update budget',fontsize=14,y=.98)
    fig.text(.035,.02,'Each arm: 32,768 shared examples; 1,024 updates; original t5-base initialization. 512 queries / groups per domain.\n'
        'Thick: group bootstrap; thin gray: cluster-t. Both are 95% intervals adjusted for 28 predefined comparisons.\n'
        'Intervals condition on the trained weights. best3 uses relevance labels; it is an opportunity metric.',fontsize=9,va='bottom')
    fig.tight_layout(rect=[0,.12,1,.95])
    for suffix in ('png','pdf'):
        path=OUT/f'cqc_configuration_effects.{suffix}'
        assert not path.exists(),'Do not overwrite a delivered figure'
        fig.savefig(path,dpi=200,bbox_inches='tight')
    plt.close(fig)
    print('Scientific figure exported: PNG and PDF')


if __name__=='__main__':main()
