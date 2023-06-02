import json
import pickle
import os
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import numpy as np
from supervenn import supervenn
from collections import OrderedDict

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_name = {'luke': "studio-ousia/luke-base",
              'bert_both': "bert-base-cased",
              'roberta': "roberta-base"}

dataset = 'TREx'
print(dataset)
with open(f"./{dataset}_data.json") as f:
    data = json.load(f)

with open(f"./cardinality_information_{dataset}.json") as f:
    card = json.load(f)

categories = list(set(card.values()))
if '1:1' in categories:
    card_order = {'1:1': [], 'N:1': [], 'N:M': [] } #, 'Total':[]}
else:
    card_order = {c : [] for c in categories}

for c in card:
    card_order[card[c]].append(c)
    #card_order['Total'].append(c)


template_names = ['simple',
                  'compound',
                  'simple_compound',
                  'complex',
                  'simple_complex',
                  'compound_complex',
                  'simple_compound_complex']

"""{'bert_both': "bert-base-cased",
           'roberta': "roberta-base",
           'luke': "studio-ousia/luke-base",
           }"""

models = {'bert_both': "bert-base-cased",
           'roberta': "roberta-base",
          'luke': "studio-ousia/luke-base",
          }

name = 'bert_both'
m = models[name]

# total and per cardinality

# appendix per relation
# performance accuracy
# ranking correlation
# performance graph with additional information no information --> one information averaged --> both information
# wasserstein distance between opposing for
# average isotropy for different subjects averaged over seperated addtional information

opposing = {'compound': 'simple_compound',
            'complex': 'simple_complex',
            'compound_complex': 'simple_compound_complex'}


def calc_accuracy(accuracy, acc_type):

    import matplotlib.pyplot as plt
    template_typologies = ['simple', 'compound', 'complex', 'compound_complex']
    template_control = ["simple_" + t if not 'simple' in t else t for t in template_typologies]

    complete_acc = {}
    values_typologies_all = {m: [] for m in accuracy.keys()}
    values_control_all = {m: [] for m in accuracy.keys()}
    support = []
    total_all = {m : {t: 0 for t in template_names} for m in accuracy.keys()}.copy()
    correct_all = {m : {t: 0 for t in template_names} for m in accuracy.keys()}.copy()

    for m in accuracy.keys():
        cardis = list(accuracy[m].keys())
        #if not 'total' in cardis:
        #    cardis.append('total')
        for k in cardis:  # + ['total']:
            correct = {t: 0 for t in template_names}.copy()
            total = {t: 0 for t in template_names}.copy()
            for a in accuracy[m][k]:

                for t in template_names:
                    values = a[1][acc_type].copy()
                    correct[t] += values[t]['correct']
                    total[t] += values[t]['total']
                    total_all[m][t] += values[t]['total']
                    correct_all[m][t] += values[t]['correct']

            xlabels = ['relation', 'relation+1', 'relation+2']
            values_typologies = [correct[t] / total[t] for t in template_typologies]
            values_typologies_all[m].append(values_typologies)
            support.append(total['simple'])
            values_control = [correct[t] / total[t] for t in template_control]
            values_control_all[m].append(values_control)

    for m in accuracy.keys():
        values_typologies = [correct_all[m][t] / total_all[m][t] for t in template_typologies]
        values_typologies_all[m].append(values_typologies)
        support.append(total_all[m]['simple'])
        values_control = [correct_all[m][t] / total_all[m][t] for t in template_control]
        values_control_all[m].append(values_control)

    return values_typologies_all, values_control_all, support

def experiment_one(accuracy, acc_type):
    typologies, control, support = calc_accuracy(accuracy, acc_type)
    print(support)
    if '1:1' in categories:
        cardis = ['1:1', 'N:1', 'N:M', 'Total']
    else:
        cardis = categories
        if not 'Total' in cardis:
            cardis.append('Total')
    models = ['bert_both', 'roberta', 'luke']
    lines = []
    print(cardis)
    for j, c in enumerate(cardis):
        line = f"{c} & "
        for i in [0, 1, 2]:
            if i == 0:
                line += " & ".join([str(typologies[m][j][i])[1:6] for m in models]) + " & "
            else:
                line += " & ".join([f"{str(typologies[m][j][i])[1:6]} & {str(control[m][j][i])[1:6]}" for m in models]) + " & "
        lines.append(line + "\\\\")

    for l in lines:
        print(l)

def calc_acc(ctrl, typ):
    for m in typ:
        for c in typ[m]:
            for r in typ[m][c]:
                typ[m][c][r] = sum(typ[m][c][r])/len(typ[m][c][r])
    for m in ctrl:
        for c in ctrl[m]:
            for r in ctrl[m][c]:
                if r == 'relation':
                    continue
                ctrl[m][c][r] = sum(ctrl[m][c][r])/len(ctrl[m][c][r])
    return typ, ctrl

def combination_accuracy(subjects, acc_type='highest_acc', prob_type='highest_prob'):
    if '1:1' in categories:
        cardis = ['1:1', 'N:1', 'N:M', 'Total']
    else:
        cardis = categories
        if not 'Total' in cardis:
            cardis.append('Total')
    import matplotlib.pyplot as plt
    template_typologies = ['simple', 'compound', 'complex', 'compound_complex']
    template_control = ["simple_" + t if not 'simple' in t else t for t in template_typologies]
    information_gain = ['relation', 'relation+1', 'relation+2']

    acc_combined = {m: {c: {t: [] for t in information_gain}.copy() for c in cardis} for m in subjects.keys()}.copy()
    acc_combined_control = {m: {c: {t: [] for t in information_gain}.copy() for c in cardis} for m in subjects.keys()}.copy()

    for m in subjects:
        for c in subjects[m]:
            for s in subjects[m][c]:
                acc_combined[m][c]['relation'].append(s[1][acc_type]['simple'])
                acc_combined[m]['Total']['relation'].append(s[1][acc_type]['simple'])

                if torch.max(s[prob_type]['compound']) > torch.max(s[prob_type]['complex']):
                    acc_combined[m][c]['relation+1'].append(s[1][acc_type]['compound'])
                    acc_combined[m]['Total']['relation+1'].append(s[1][acc_type]['compound'])
                else:
                    acc_combined[m][c]['relation+1'].append(s[1][acc_type]['complex'])
                    acc_combined[m]['Total']['relation+1'].append(s[1][acc_type]['complex'])

                if torch.max(s[prob_type]['simple_compound']) > torch.max(s[prob_type]['simple_complex']):
                    acc_combined_control[m][c]['relation+1'].append(s[1][acc_type]['simple_compound'])
                    acc_combined_control[m]['Total']['relation+1'].append(s[1][acc_type]['simple_compound'])
                else:
                    acc_combined_control[m][c]['relation+1'].append(s[1][acc_type]['simple_complex'])
                    acc_combined_control[m]['Total']['relation+1'].append(s[1][acc_type]['simple_complex'])

                acc_combined[m][c]['relation+2'].append(s[1][acc_type]['compound_complex'])
                acc_combined[m]['Total']['relation+2'].append(s[1][acc_type]['compound_complex'])

                acc_combined_control[m][c]['relation+2'].append(s[1][acc_type]['simple_compound_complex'])
                acc_combined_control[m]['Total']['relation+2'].append(s[1][acc_type]['simple_compound_complex'])

    return acc_combined, acc_combined_control

def experiment_two(subjects, acc_type='highest_acc', prob_type='highest_prob'):
    typologies, control = combination_accuracy(subjects, acc_type=acc_type, prob_type=prob_type)
    if '1:1' in categories:
        cardis = ['1:1', 'N:1', 'N:M', 'Total']
    else:
        cardis = categories
    models = ['bert_both', 'roberta', 'luke']
    typologies, control = calc_acc(control, typologies)
    lines = []
    print(typologies)
    for j, c in enumerate(cardis):
        line = f"{c} & "
        for i in ['relation', 'relation+1', 'relation+2']:
            if i == 'relation':
                line += " & ".join([str(typologies[m][c][i])[1:6] for m in models]) + " & "
            else:
                line += " & ".join([f"{str(typologies[m][c][i])[1:6]} & {str(control[m][c][i])[1:6]}" for m in models]) + " & "
        lines.append(line + "\\\\")

    for l in lines:
        print(l)

def prompt_recall(subjects, acc_type):
    #acc_type = 'inverse_acc'
    correct_information = {m: {t: set() for t in template_names}.copy() for m in subjects}

    i = 0
    for m in subjects:
        for k in subjects[m]:
            for s in subjects[m][k]:
                for t in correct_information[m]:
                    if s[1][acc_type][t]:
                        correct_information[m][t].add(i)
                i += 1
    return correct_information

def experiment_three(subjects, acc_type):
    lines = []
    real_names = {'simple': 'simple',
                  'simple_compound': 'domain control',
                  'compound': 'compound',
                  'simple_complex': 'range control',
                  'complex': 'complex',
                  'simple_compound_complex': 'combined control',
                  'compound_complex': 'compound complex'}
    correct_information = prompt_recall(subjects, acc_type)
    for m in correct_information:
        range_or_domain_specific = set().union(*[correct_information[m]['compound'], correct_information[m]['complex']])
        range_or_domain_control = set().union(*[correct_information[m]['simple_compound'], correct_information[m]['simple_complex']])
        colors_specific = ['#D79B00', '#6C8EBF', '#B85450', '#9673A6']
        colors_control = ['#D79B00', '#D79B00','#D79B00','#D79B00']
        display_information_specific = OrderedDict({'simple': correct_information[m]['simple'],
                                                    'compound': correct_information[m]['compound'],
                                                    'complex': correct_information[m]['complex'],
                                                    'compound complex': correct_information[m]['compound_complex'],})
        display_information_control = OrderedDict({
                               'simple': correct_information[m]['simple'],
                               'domain': correct_information[m]['simple_compound'],
                               'range': correct_information[m]['simple_complex'],
                               'combined': correct_information[m]['simple_compound_complex'],})
        #plt.figure(figsize=(20, 10))
        """chunks_ordering = 'occurrence'
        supervenn(list(display_information_specific.values()), list(display_information_specific.keys()), widths_minmax_ratio=0.1,
                  sets_ordering=None, chunks_ordering=chunks_ordering, rotate_col_annotations=True, col_annotations_area_height=2, fontsize=25, color_cycle=colors_specific)
        plt.ylabel('', fontsize=1)
        plt.xlabel('Correctly Classified Triples', fontsize=30)
        plt.savefig(f'{dataset}_{m}_{acc_type}_{chunks_ordering}_specific_supervenn_two.png')
        plt.clf()
        plt.figure(figsize=(20, 10))
        chunks_ordering = 'occurrence'
        supervenn(list(display_information_control.values()), list(display_information_control.keys()), widths_minmax_ratio=0.1,
                  sets_ordering=None, chunks_ordering=chunks_ordering, rotate_col_annotations=True, col_annotations_area_height=2, fontsize=25, color_cycle=colors_control)
        plt.ylabel('', fontsize=1)
        plt.xlabel('Correctly Classified Triples', fontsize=30)
        plt.savefig(f'{dataset}_{m}_{acc_type}_{chunks_ordering}_control_supervenn_two.png')
        plt.clf()"""

        plt.figure(figsize=(20, 10))
        chunks_ordering = 'size'
        supervenn(list(display_information_specific.values()), list(display_information_specific.keys()), widths_minmax_ratio=0.1,
                  sets_ordering=None, chunks_ordering=chunks_ordering, rotate_col_annotations=True, col_annotations_area_height=2, fontsize=25, color_cycle=colors_specific)
        plt.ylabel('', fontsize=1)
        plt.xlabel('Correctly Predicted Triples', fontsize=30)
        plt.savefig(f'{dataset}_{m}_{acc_type}_{chunks_ordering}_specific_supervenn_two.png')
        plt.clf()
        plt.figure(figsize=(20, 10))
        chunks_ordering = 'size'
        supervenn(list(display_information_control.values()), list(display_information_control.keys()), widths_minmax_ratio=0.1,
                  sets_ordering=None, chunks_ordering=chunks_ordering, rotate_col_annotations=True, col_annotations_area_height=2, fontsize=25, color_cycle=colors_control)
        plt.ylabel('', fontsize=1)
        plt.xlabel('Correctly Predicted Triples', fontsize=30)
        plt.savefig(f'{dataset}_{m}_{acc_type}_{chunks_ordering}_control_supervenn_two.png')
        plt.clf()
    for name in correct_information:
        ground_truth = correct_information[name]['simple']
        line = name + " & "
        for t in template_names[1:]:
            recall = str(len(correct_information[name][t].intersection(ground_truth)) / len(ground_truth))[1:6]
            line += f" {recall} &"
        line += f" {len(ground_truth)} \\\\"
        lines.append(line)

    for l in lines:
        print(l)


def plot_entropy(overview_spec, overview_control):
    from matplotlib import rcParams
    rcParams.update({'figure.autolayout': True})
    xlabels = ['relation', 'range or domain', 'combined']
    fig, axs = plt.subplots(2,2, sharex='col', sharey='all')
    right_name = {'bert_both' : 'BERT',
                  'roberta': 'RoBERTa',
                  'luke': 'Luke'}

    for m in overview_spec:
        axs[0, 0].plot(range(len(xlabels)), [np.mean(overview_spec[m]['entropy_by_prob']['Total'][r]) for r in overview_spec[m]['entropy_by_prob']['Total']],
                       label = f"{right_name[m]} - N = {len(overview_spec[m]['entropy']['Total']['relation'])}")
        axs[0, 0].set_ylabel("High Confidence \n Completion \n Average Entropy")
        axs[0, 0].set_title("Specific Syntax")
        axs[0, 0].grid(True)
        axs[0, 0].legend()
    for m in overview_spec:
        axs[0, 1].plot(range(len(xlabels)), [np.mean(overview_control[m]['entropy_by_prob']['Total'][r]) for r in overview_control[m]['entropy_by_prob']['Total']])
        axs[0, 1].set_title("Control Syntax")
        axs[0, 1].grid(True)
    for m in overview_spec:
        axs[1, 0].plot(range(len(xlabels)), [np.mean(overview_spec[m]['entropy']['Total'][r]) for r in overview_spec[m]['entropy']['Total']])
        axs[1, 0].set_ylabel("High Quality \n Completion \n Average Entropy")
        axs[1, 0].set_xticks(range(len(xlabels)), xlabels)
        axs[1, 0].set_xticks(range(len(xlabels)), xlabels, rotation=45)
        axs[1, 0].grid(True)
    for m in overview_spec:
        axs[1, 1].plot(range(len(xlabels)), [np.mean(overview_control[m]['entropy']['Total'][r]) for r in overview_control[m]['entropy']['Total']])
        axs[1, 1].set_xticks(range(len(xlabels)), xlabels, rotation=45)
        axs[1, 1].grid(True)
    plt.savefig(f"{dataset}_response_confidence_given_information_quality.png")
    plt.show()

def experiment_four(subjects, acc_types, entropy_types):
    template_typologies = ['simple', 'compound', 'complex', 'compound_complex']
    template_control = ["simple_" + t if not 'simple' in t else t for t in template_typologies]
    relation_amount = {'simple': 'relation',
                       'compound': 'relation+1',
                       'complex': 'relation+1',
                       'compound_complex': 'relation+2',
                       'simple_compound': 'relation+1',
                       'simple_complex': 'relation+1',
                       'simple_compound_complex': 'relation+2'}
    relation_amounts = ['relation', 'relation+1', 'relation+2']
    if '1:1' in categories:
        specific_templates_right = {m : {entropy_type :
                                        {'1:1': {t: [] for t in relation_amounts}.copy(),
                                         'N:1': {t: [] for t in relation_amounts}.copy(),
                                         'N:M': {t: [] for t in relation_amounts}.copy(),
                                         'Total': {t: [] for t in relation_amounts}.copy()}.copy()
                                    for entropy_type in entropy_types}
                               for m in subjects}
        unspecific_templates_right = {m : {entropy_type :
                                        {'1:1': {t: [] for t in relation_amounts}.copy(),
                                         'N:1': {t: [] for t in relation_amounts}.copy(),
                                         'N:M': {t: [] for t in relation_amounts}.copy(),
                                         'Total': {t: [] for t in relation_amounts}.copy()}.copy()
                                    for entropy_type in entropy_types}
                               for m in subjects}
    else:
        specific_templates_right = {m : {entropy_type :
                                        {c: {t: [] for t in relation_amounts}.copy()
                                         for c in categories}.copy()
                                    for entropy_type in entropy_types}
                               for m in subjects}
        unspecific_templates_right = {m : {entropy_type :
                                        {c: {t: [] for t in relation_amounts}.copy()
                                         for c in categories}.copy()
                                    for entropy_type in entropy_types}
                               for m in subjects}

    for m in subjects:
        for c in subjects[m]:
            for s in subjects[m][c]:
                if all([s[10][acc_type]['simple'] for acc_type in acc_types]):
                    for entropy_type in entropy_types:
                        for t in template_names:
                            if t == 'simple':
                                specific_templates_right[m][entropy_type][c][relation_amount[t]].append(s[entropy_type][t].item())
                                specific_templates_right[m][entropy_type]['Total'][relation_amount[t]].append(s[entropy_type][t].item())
                                unspecific_templates_right[m][entropy_type][c][relation_amount[t]].append(s[entropy_type][t].item())
                                unspecific_templates_right[m][entropy_type]['Total'][relation_amount[t]].append(s[entropy_type][t].item())
                            elif t in template_typologies:
                                specific_templates_right[m][entropy_type][c][relation_amount[t]].append(s[entropy_type][t].item())
                                specific_templates_right[m][entropy_type]['Total'][relation_amount[t]].append(s[entropy_type][t].item())
                            else:
                                unspecific_templates_right[m][entropy_type][c][relation_amount[t]].append(s[entropy_type][t].item())
                                unspecific_templates_right[m][entropy_type]['Total'][relation_amount[t]].append(s[entropy_type][t].item())

    plot_entropy(specific_templates_right, unspecific_templates_right)

def make_results_comparable(r, comparison_results):
    new_subs = []
    new_accuracy = {}
    for i, s in enumerate(r['subjects']):
        if all([c_r[i] for c_r in comparison_results]):
            for k in [1, 10, 50, 100, 200, 500, 1000]:
                if not new_accuracy.get(k):
                    new_accuracy[k] = {}
                for acc_type in s[k]:
                    if not new_accuracy[k].get(acc_type):
                        new_accuracy[k][acc_type] = {}
                    for t in s[k][acc_type]:
                        if not new_accuracy[k][acc_type].get(t):
                            new_accuracy[k][acc_type][t] = []
                        new_accuracy[k][acc_type][t].append(s[k][acc_type][t])
            new_subs.append(s)
        else:
            new_subs.append(False)
    r['accuracy'] = {k: {acc_type:
                             {t: {'correct': sum(subs), 'total': len(subs)}
                                for t, subs in subs_per.items()}
                                for acc_type, subs_per in k_at_acc.items()}
                     for k, k_at_acc in new_accuracy.items()}
    r['subjects'] = new_subs
    return r

def load_data():
    res_dict = {}
    acc_dict = {}
    rel_dict = {}
    for name, m in models.items():
        if '1:1' in categories:
            res_dict[name] = {'1:1': [], 'N:1': [], 'N:M': []}.copy()
            acc_dict[name] = {'1:1': [], 'N:1': [], 'N:M': []}.copy()
        else:
            res_dict[name] = {c: [] for c in categories}.copy()
            acc_dict[name] = {c: [] for c in categories}.copy()

        rel_dict[name] = {}

        for k, relations in card_order.items():
            for p in tqdm(relations):
                try:
                    results = pickle.load(
                        open(f"/data_ssds/disk11/slinzbach/LAMA/results/{dataset}/{name}/{p}.p", "rb"))
                    comparison = []

                    for comp, comp_path in models.items():
                        if name == comp:
                            continue
                        cm = pickle.load(open(f"/data_ssds/disk11/slinzbach/LAMA/results/{dataset}/{comp}/{p}.p", "rb"))
                        comparison.append(cm[comp_path][p]['subjects'])
                    results[m][p] = make_results_comparable(results[m][p], comparison)
                except Exception as e:
                    print(e)
                    continue
                acc_dict[name][k].append(results[m][p]['accuracy'])
                rel_dict[name][p] = results[m][p]['subjects']
                res_dict[name][k] += results[m][p]['subjects']
                del results
                """results = {s: {k : num for k, num in v.items() if k in [1, 10, 50, 100, 200, 500, 1000]}
                           for s, v in results.items()}"""
            # subjects.update(results)
    for name in res_dict:
        res_dict[name] = {k: [x for x in v if x] for k, v in res_dict[name].items()}

        for k, subs in res_dict[name].items():
            for s in subs:
                if not s:
                    continue
                for t in ['compound_complex', 'simple_compound_complex']:
                    entropies = s['entropy'][t]
                    s['entropy'][t] = entropies[0]
                    s['entropy_inverse'][t] = entropies[1]
                    s['entropy_by_prob'][t] = entropies[2]
                    s['entropy_inverse_by_prob'][t] = entropies[3]
    return res_dict, acc_dict, rel_dict

res_dict, acc_dict, rel_dict = load_data()

print("EXPERIMENT 1")

print("Completion 1")
experiment_one(acc_dict, 'acc')
print("Completion 2")
experiment_one(acc_dict, 'highest_acc')

print("EXPERIMENT 2")

print("Completion 1")
experiment_two(res_dict, acc_type='acc', prob_type='probability')

print("Completion 1 Combination 2")
experiment_two(res_dict, acc_type='acc')

print("Completion 2")
experiment_two(res_dict)

print("EXPERIMENT 3")

print("Completion 1")
experiment_three(res_dict, acc_type='acc')
print("Completion 2")
experiment_three(res_dict, acc_type='highest_acc')

print("EXPERIMENT 4")

print("Completion 1")
experiment_four(res_dict, ['acc', 'highest_acc'], ['entropy', 'entropy_by_prob'])
