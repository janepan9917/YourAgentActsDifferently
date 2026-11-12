"""Table 7: results with three different user models (Ultimatum and Bike Sale).

Agents: Claude Haiku 4.5, DeepSeek V4 Flash and GPT-5.6 Terra; user models: GPT-5.6 Terra
(the main experiments), GPT-5.6 Sol and Claude Sonnet 5; 100 With User runs per cell
(50 per side) against the same 100 No User runs. Reads data/processed/agent_runs.csv.gz.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paper_stats as st  # noqa: E402
import paper_style as ps  # noqa: E402
from _shared import random_effects_meta, welch_mean_difference_ci  # noqa: E402
from table3_payout import heatmap_color, heatmap_limit  # noqa: E402
from table4_misaligned_behavior import rate_heatmap_color, rate_heatmap_limit  # noqa: E402

from analysis.inputs import load_agent_runs  # noqa: E402

USERS = ['gpt-5.6-terra','gpt-5.6-sol','claude-sonnet-5']
MODELS = ['claude-haiku-4-5','deepseek-ai/DeepSeek-V4-Flash-0731','gpt-5.6-terra']
TOKENS = dict(zip(MODELS,['claudehaiku45','deepseek_v4_flash','gpt56terra']))
MODEL_NAMES = dict(zip(MODELS,['Claude Haiku 4.5','DeepSeek V4 Flash','GPT-5.6 Terra']))
GAMES = ['ultimatum','pb_sanity_negot_01']
NAMES = {'ultimatum':'Ultimatum','pb_sanity_negot_01':'Bike Sale'}
USER_IDS=['MB_OFFER_STATE_MISREPORTING','MB_HALLUCINATED_ACTION','MB_UNSUPPORTED_URGENCY']
CP_IDS=['MB_OTHER_AGENT_OFFER_STATE_MISUNDERSTANDING','MB_OTHER_AGENT_UNSUPPORTED_URGENCY','MB_EXTRA_PROTOCOL_CONTRACTING']
BEHAVIORS=USER_IDS+CP_IDS
SUFFIX='_misbehavior_single_gpt56sol_medium'


def _payout_matrix(frame):
    """One row per run: ungated payouts of agent_1 and agent_2."""
    wide = frame.pivot_table(index='game_id', columns='role', values='payout_ungated', aggfunc='first')
    return wide[['agent_1', 'agent_2']].to_numpy(dtype=float)


def _label_rows(frame, condition):
    rows = []
    for r in frame.itertuples():
        for b in BEHAVIORS:
            rows.append(dict(game=r.game, model=r.agent_model, condition=condition, game_id=r.game_id,
                             role=r.role, behavior_id=b, flagged=bool(getattr(r, b))))
    return rows


def main():
    runs = load_agent_runs()
    base_runs = runs[(runs.condition == 'no_user') & runs.game.isin(GAMES) & runs.agent_model.isin(MODELS)]
    robust = runs[(runs.condition == 'user_model_robustness') & runs.user_model.isin(USERS)]
    bases = {}; episode_rows = []; inventory = []; role_stats = []
    for game in GAMES:
        for model in MODELS:
            d = base_runs[(base_runs.game == game) & (base_runs.agent_model == model)]
            assert d.game_id.nunique() == 100, (game, model)
            bases[(game, model)] = _payout_matrix(d)
            episode_rows += _label_rows(d, 'no_user')
            inventory.append(dict(condition='no_user', game=game, model=model, n=100))
    for (game, model, user), d in robust.groupby(['game', 'agent_model', 'user_model']):
        base = bases[(game, model)]
        for i in range(2):
            side = d[d.direction == f'user_on_agent_{i+1}']
            assert side.game_id.nunique() == 50, (game, model, user, i)
            arr = _payout_matrix(side)
            scale = max(*abs(base.mean(axis=0)), *abs(arr.mean(axis=0)))
            assert scale > 0
            diff, se, lo, hi = welch_mean_difference_ci(base[:, i], arr[:, i])
            role_stats.append(dict(game=game, model=model, user=user, role=i+1, mean=diff/scale*100, se=se/scale*100,
                                   scale=scale, n_with=50, n_no=100, raw_diff=diff))
            episode_rows += _label_rows(side[side.selected], user)
        inventory.append(dict(condition=user, game=game, model=model, n=100))
    assert len({(e['game'], e['model'], e['condition']) for e in inventory}) == 24
    role_df=pd.DataFrame(role_stats)
    cells=[]
    for (g,m,u),d in role_df.groupby(['game','model','user']):
        assert len(d)==2
        cells.append(dict(game=g,model=m,user=u,**st.z_summary(d['mean'].mean(),np.sqrt((d.se**2).sum())/2)))
    cell=pd.DataFrame(cells)
    payout=[]
    for game in GAMES:
        for model in MODELS:
            payout.append(dict(game=game,model=model,values={u:cell.query('game==@game and model==@model and user==@u').iloc[0].to_dict() for u in USERS}))
        pooled={}
        for u in USERS:
            d=cell.query('game==@game and user==@u'); r=random_effects_meta(d['mean'].to_numpy(),d.se.to_numpy())
            pooled[u]={**r,'mean':r['estimate']}
        payout.append(dict(game=game,model='Pooled',values=pooled))
    payout.append(dict(game='Both games',model='Mean',values={u:st.combine(cell.query('user==@u')['mean'],cell.query('user==@u').se) for u in USERS}))
    frame=pd.DataFrame(episode_rows)
    episodes=pd.read_csv(ps.DATA/'behavior_episodes.csv')
    episodes=episodes[episodes.game.isin(GAMES)&episodes.model.isin(TOKENS.values())&episodes.condition.isin(['no_user','single_user'])&episodes.behavior_id.isin(BEHAVIORS)]
    current=frame[frame.condition.isin(['no_user','gpt-5.6-terra'])].copy()
    current['model']=current.model.map(TOKENS)
    current['condition']=current.condition.replace({'gpt-5.6-terra':'single_user'})
    keys=['game','model','condition','role','game_id','behavior_id']
    a=current.set_index(keys).flagged.sort_index(); b=episodes.set_index(keys).present.sort_index()
    assert a.index.equals(b.index), (len(a),len(b),a.index.difference(b.index)[:4].tolist(),b.index.difference(a.index)[:4].tolist())
    assert np.array_equal(a.to_numpy(),b.to_numpy()), (a[a!=b].head(8).to_dict(),b[a!=b].head(8).to_dict())
    mb_reference_matched=len(a)
    groups=[('User-facing',[(b,n) for b,n in zip(USER_IDS,['Misreporting','Hallucinated action','Unsupported urgency'])]+[('any_user','Any')]),
            ('Counterparty-facing',[(b,n) for b,n in zip(CP_IDS,['Misunderstanding','Unsupported urgency','Out-of-scope terms'])]+[('any_cp','Any')]),
            ('All behaviors',[('any_all','Any')])]
    mb=[]; mb_cells={}
    for family,bs in groups:
        for b,name in bs:
            ids=USER_IDS if b=='any_user' else CP_IDS if b=='any_cp' else BEHAVIORS if b=='any_all' else [b]
            units=frame[frame.behavior_id.isin(ids)].groupby(['game','model','condition','role','game_id']).flagged.any().reset_index()
            c=st.binomial_cells(units,['game','model','condition']); mb_cells[b]=c
            values={}
            for cond in ['no_user']+USERS:
                r=st.two_stage(c[c.condition==cond])[1]; r.pop('pvalue'); r['test_p']=None
                if cond!='no_user' and family!='User-facing':
                    diff=st.paired_cells(c,'condition',cond,'no_user',['game','model'])
                    test=st.two_stage(diff)[1];r.update(test_p=test['pvalue'],difference=test['mean'])
                values[cond]=r
            mb.append(dict(family=family,behavior=b,name=name,values=values))
    bygame={}
    for b in ['any_cp','any_all']:
        c=mb_cells[b]; bygame[b]={}
        for g in GAMES:
            bygame[b][g]={}
            for u in USERS:
                d=st.paired_cells(c[c.game==g],'condition',u,'no_user',['game','model'])
                bygame[b][g][u]=st.combine(d.est,d.se)
    # Consistency check: on this subset the Terra cells equal the main-table values.
    ref=pd.read_csv(ps.DATA/'payout_change_by_model.csv')
    checked=0
    for r in role_stats:
        if r['user']!='gpt-5.6-terra':continue
        game=r['game']
        q=ref[(ref.game==game)&(ref.model==TOKENS[r['model']])&(ref.user_role==f"user_{r['role']}")]
        assert len(q)==1
        assert np.allclose([r['mean'],r['se']],[q.iloc[0]['mean'],q.iloc[0]['se']],atol=1e-7),(r,q)
        checked+=1
    out = dict(inventory=inventory, terra_reference_cells_matched=checked, role_payouts=role_stats,
               payout=payout, mb=mb, mb_by_game=bygame, mb_reference_agent_behavior_rows_matched=mb_reference_matched)
    (ps.DATA / 'table7_user_model_robustness.json').write_text(json.dumps(out, indent=2, default=float), encoding='utf-8')
    render(payout, mb)
    print('Saved table 7')


def fmt(r,test=True):
    v=round(float(r['mean']),1);v=0.0 if v==0 else v
    hw=(r['ci_high']-r['ci_low'])/2
    p=r.get('test_p') if 'test_p' in r else r['pvalue']
    star=r'\textsuperscript{*}' if test and p is not None and p<.05 else ''
    return rf'\Value{{{v:.1f}}}{{{hw:.1f}}}{{{star}}}'


def render(payout,mb):
    header=r'''% Code: analysis/paper/table7_user_model_robustness.py
\begingroup
\microtypesetup{protrusion=false}
\definecolor{PayoutIncrease}{HTML}{79B48C}
\definecolor{PayoutDecrease}{HTML}{D98783}
\definecolor{MBRateHigh}{HTML}{D98783}
\newcommand{\Value}[3]{\makebox[2.6em][r]{#1}\,$\pm$\,#2\rlap{#3}}
'''
    limit=heatmap_limit([r['mean'] for x in payout for r in x['values'].values()])
    lines=[];colors=[]
    for i,row in enumerate(payout,start=3):
        label=NAMES.get(row['game'],row['game'])
        if i>3 and row['game']==payout[i-4]['game']:label=''
        cells=[label,MODEL_NAMES.get(row['model'],row['model'])]
        for j,u in enumerate(USERS,start=3):
            r=row['values'][u];cells.append(fmt(r));colors.append(rf'cell{{{i}}}{{{j}}}={{bg={heatmap_color(r["mean"],limit)}}},')
        if i in (7,11):lines.append(r'\midrule')
        lines.append(' & '.join(cells)+r' \\')
    a=header+r'''\begin{tblr}{width=\linewidth,colspec={Q[l] Q[l] *{3}{X[c]}},colsep=4pt,rowsep=2pt,
row{1,2}={font=\bfseries},column{Z}={rightsep=6pt},
'''+ '\n'.join(colors)+r'''}
\toprule
\SetCell[r=2]{l,m} Game & \SetCell[r=2]{l,m} Agent model & \SetCell[c=3]{c} User model & & \\
\cmidrule[lr]{3-5}
 & & GPT-5.6 Terra & GPT-5.6 Sol & Claude Sonnet 5 \\
\midrule
'''+ '\n'.join(lines)+r'''
\bottomrule
\end{tblr}
\endgroup
'''
    ps.save_table(a,'user_model_payout_body')
    ratevalues=[r['mean'] for row in mb for r in row['values'].values()]
    limit=rate_heatmap_limit(ratevalues);peak=max(ratevalues);lines=[];colors=[];idx=2;last=None
    for row in mb:
        if row['family']!=last:
            if last is not None:lines.append(r'\midrule')
            lines.append(r'\SetCell[c=5]{l} \textbf{'+row['family']+r'} & & & & \\');idx+=1;last=row['family']
        cells=[row['name']]
        for j,c in enumerate(['no_user']+USERS,start=2):
            r=row['values'][c]
            cells.append('---' if c=='no_user' and row['family']=='User-facing' else fmt(r,c!='no_user'))
            if cells[-1]!='---':colors.append(rf'cell{{{idx}}}{{{j}}}={{bg={rate_heatmap_color(r["mean"],limit,peak_rate=peak)}}},')
        lines.append(' & '.join(cells)+r' \\');idx+=1
    b=header+r'''\begin{tblr}{width=\linewidth,colspec={Q[l] *{4}{X[c]}},colsep=4pt,rowsep=2pt,row{1}={font=\bfseries},column{Z}={rightsep=6pt},
'''+ '\n'.join(colors)+r'''}
\toprule
Behavior & \nouser{} & GPT-5.6 Terra & GPT-5.6 Sol & Claude Sonnet 5 \\
\midrule
'''+ '\n'.join(lines)+r'''
\bottomrule
\end{tblr}
\endgroup
'''
    ps.save_table(b,'user_model_mb_body')




if __name__ == '__main__':
    main()
