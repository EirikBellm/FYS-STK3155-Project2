"""Build the Part e result tables from saved runs.

No model fitting or model selection occurs.

LLM-assisted: Codex implemented the saved-file loading, table formatting and
presentation.
"""
import json
import numpy as np
import pandas as pd
from part_e_classification import OUT,NAMES

summary=pd.read_csv(OUT/'search_summary.csv')
configs=json.loads((OUT/'sampled_configurations.json').read_text())
config_table=pd.DataFrame([{**c,'hidden':str(c['hidden']),'depth':len(c['hidden']),
                            'width':c['hidden'][0]} for c in configs])
search=summary[summary.family=='manual'].merge(config_table,on='id')
search.to_csv(OUT/'search_results.csv',index=False)
picks=json.loads((OUT/'frozen_selection.json').read_text())
match=json.loads((OUT/'library_match.json').read_text())
test=pd.read_csv(OUT/'test_scores.csv').set_index('model')
rows=[]
for model,pick in picks.items():
    s=summary[summary.id==pick['config']['id']].iloc[0]
    rows.append({'model':model,'config_id':pick['config']['id'],
                 'hidden':str(pick['config']['hidden']),
                 'activation':pick['config']['activation'] if pick['config']['hidden'] else 'none (softmax only)',
                 'gamma':pick['config']['gamma'],
                 'penalty':pick['config']['penalty'],'lmbd':pick['config']['lmbd'],
                 'cv_macro_f1':s.macro_f1,'cv_f1_fold_sd':s.macro_f1_fold_sd,
                 'cv_accuracy':s.accuracy,'cv_cross_entropy':s.cross_entropy,
                 'train_cv_macro_f1':s.train_macro_f1,'train_cv_ce':s.train_cross_entropy,
                 'cv_fit_seconds':s.total_seconds,'refit_epochs':pick['epochs'],
                 **{f'test_{k}':float(test.loc[model,k]) for k in ['macro_f1','accuracy','cross_entropy']}})
comparison=pd.DataFrame(rows)
comparison.to_csv(OUT/'model_comparison.csv',index=False)
per=pd.read_csv(OUT/'final/manual_per_class.csv',index_col=0).loc[NAMES].copy()
per.index.name='letter'
counts=pd.read_csv(OUT/'development_manifest.csv').label.value_counts()
per['development_support']=counts.loc[NAMES].to_numpy()
per['test_support']=per['support'].astype(int)
per.to_csv(OUT/'selected_per_class.csv')
cm=np.loadtxt(OUT/'final/manual_confusion.csv',delimiter=',')
normalized=cm/cm.sum(1,keepdims=True)
confusions=[]
for i in range(24):
    for j in range(24):
        if i!=j:confusions.append({'true':NAMES[i],'predicted':NAMES[j],
                                 'count':int(cm[i,j]),'fraction_of_true':float(normalized[i,j])})
pd.DataFrame(confusions).sort_values('count',ascending=False).to_csv(OUT/'largest_confusions.csv',index=False)
matched=summary[summary.id==match['manual_id']]
matched.to_csv(OUT/'library_matched_manual_scores.csv',index=False)
print(comparison.to_string(index=False))

print('Part e tables refreshed. Use present_report_readable.py for the report figures.')
