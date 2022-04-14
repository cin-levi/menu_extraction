import json
import numpy as np

kfold_report = 'D:\\menu_extraction\\full_report.json'

with open(kfold_report, 'r', encoding='utf-8') as f:
    data = json.load(f)

best_epochs = []

for report in data:
    best_report_idx = np.argmax([report[epoch]['w']['f'] for epoch in report])
    best_epochs.append(report[str(best_report_idx)])

avg_micro_p = np.average([x['micro_ner_p'] for x in best_epochs])
avg_micro_r = np.average([x['micro_ner_r'] for x in best_epochs])
avg_micro_f1 = np.average([x['micro_ner_f1'] for x in best_epochs])

avg_micro_p_price = np.average([x['p']['p'] for x in best_epochs])
avg_micro_r_price = np.average([x['p']['r'] for x in best_epochs])
avg_micro_f1_price = np.average([x['p']['f'] for x in best_epochs])

avg_micro_p_vintage = np.average([x['v']['p'] for x in best_epochs])
avg_micro_r_vintage = np.average([x['v']['r'] for x in best_epochs])
avg_micro_f1_vintage = np.average([x['v']['f'] for x in best_epochs])

avg_micro_p_wine = np.average([x['w']['p'] for x in best_epochs])
avg_micro_r_wine = np.average([x['w']['r'] for x in best_epochs])
avg_micro_f1_wine = np.average([x['w']['f'] for x in best_epochs])


final_report = {'avg_micro_p': avg_micro_p,
                'avg_micro_r': avg_micro_r,
                'avg_micro_f1': avg_micro_f1,
                'avg_micro_p_price': avg_micro_p_price,
                'avg_micro_r_price': avg_micro_r_price,
                'avg_micro_f1_price': avg_micro_f1_price,

                'avg_micro_p_vintage': avg_micro_p_vintage,
                'avg_micro_r_vintage': avg_micro_r_vintage,
                'avg_micro_f1_vintage': avg_micro_f1_vintage,

                'avg_micro_p_wine': avg_micro_p_wine,
                'avg_micro_r_wine': avg_micro_r_wine,
                'avg_micro_f1_wine': avg_micro_f1_wine,
                }
for key in final_report:
    print(f"{key} : {final_report[key]}")
