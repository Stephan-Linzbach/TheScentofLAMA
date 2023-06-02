import itertools
import json
import os
from tqdm import tqdm
from timeit import default_timer as timer
import pickle

from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
import torch
import numpy as np
from datasets import Dataset
from datasets import disable_caching
from scipy.stats import wasserstein_distance
from scipy.special import kl_div
from IsoScore import IsoScore
from sklearn.metrics.pairwise import cosine_similarity

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

dataset = 'GoogleRE' # 'TREx', 'CNET'
model_name =  {'bert_both': "bert-base-cased",
               'roberta': "roberta-base",
               'luke': "studio-ousia/luke-base",
               } # {'bert_prob': "bert-base-cased",} #{'bert': "bert-base-cased",} # {'luke': "studio-ousia/luke-base",} #
              #
              # # "microsoft/deberta-base", "studio-ousia/luke-base", "google/electra-base-generator", "bert-base-cased", "roberta-base"


def get_model_environment(model_name):
    global allowed_letters
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).cuda().eval()

    mask = tokenizer.mask_token
    mask_id = tokenizer.mask_token_id
    vocab = {}
    all_letters = set().union(*[set(k) for k in tokenizer.get_vocab().keys()])
    inter = all_letters.intersection(allowed_letters)
    print(all_letters)
    print(inter)
    vocab_sorted = {k: v for k, v in sorted(tokenizer.get_vocab().items(), key=lambda item: item[1])}
    vocab_words = np.array(["".join([letter for letter in word if letter in inter]) for word in vocab_sorted.keys()])
    print(vocab_words)
    vocab = [False for _ in range(len(vocab_sorted.keys()))]
    for i, s in enumerate(vocab_words):
        if s in allowed_vocab:
            vocab[i] = True
    return tokenizer, model, mask, mask_id, vocab, vocab_words


with open(f"./{dataset}_data.json") as f:
    data = json.load(f)

with open("./common_vocab_cased.txt") as f:
    allowed_vocab = set([x[:-1] for x in f.readlines()])

with open(f"./exclusion_information_{dataset}.json") as f:
    exclusion = json.load(f)

allowed_letters = set()

for p in data:
    for s, o in tqdm(list(zip(data[p]['subjects'], data[p]['objects']))):
        allowed_letters = allowed_letters.union(*[set(s), set(o)])

exclusion = {k: {kk : set(vv) for kk, vv in v.items()} for k, v in exclusion.items()}

tokenizer_specifics = []

template_names = ['simple',
                  'compound',
                  'simple_compound',
                  'complex',
                  'simple_complex']


def tokenize(s):
    return tokenizer(s, padding=True, truncation=True, return_tensors='pt')

@torch.no_grad()
def get_results(encodings, model, vocab, vocab_words, exclusion, masked_positions):
    vocab_exclude = vocab
    if exclusion:
        vocab_exclude = torch.tensor(np.where(np.isin(vocab_words, exclude), False, vocab))
    start = timer()
    answers = model(**encodings, output_hidden_states=True)
    end = timer()
    #print("inference", end - start)
    hidden_states = answers['hidden_states'][-1][masked_positions[0], masked_positions[1], :].cpu()
    logits = answers['logits'][masked_positions[0], masked_positions[1], :].cpu()
    logits = logits[:, vocab_exclude]
    logits = torch.nn.Softmax(1)(logits)
    del answers
    return logits, hidden_states, vocab_exclude


def create_sample(template, name, sub, mask, range, poss_range, domain, poss_domain):
    result_templates = [template]

    if name in ['complex', 'simple_complex']:
        result_templates += [template.replace(range, p_r) for p_r in poss_range]
    if name in ['compound', 'simple_compound']:
        result_templates += [template.replace(domain, p_d) for p_d in poss_domain]

    results_t = []
    for t in result_templates:
        t = t.replace("[X]", sub)
        t = t[0].upper() + t[1:]
        t = t.replace("[Y]", mask)
        results_t.append(t)
    return results_t

def KL(P,Q):
    epsilon = 0.00000000001

    # You may want to instead make copies to avoid changing the np arrays.
    P = P+epsilon
    Q = Q+epsilon

    divergence = torch.sum(P*torch.log(P/Q))
    return divergence

def wasserstein_gain(distros, t):
    opposing = {'simple': ['compound',
                           'simple_compound',
                          'complex',
                          'simple_complex',
                          'compound_complex',
                          'simple_compound_complex'],
                'compound':['compound_complex', 'simple_compound_complex', 'complex', 'simple_compound', 'simple_complex'],
                'complex': ['compound_complex', 'simple_compound_complex','simple_compound', 'simple_complex'],
                'simple_compound':['compound_complex', 'simple_compound_complex'],
                'simple_complex': ['compound_complex', 'simple_compound_complex'],
                'compound_complex': ['simple_compound_complex'],
                'simple_compound_complex': ['compound_complex']}

    results = {t : {}}

    for c in opposing[t]:
        distro_t = distros[t].view(-1,)
        distro_s = distros[c].view(-1,)
        results[t][c] = wasserstein_distance(distro_s, distro_t)

    return results

def kl_gain(distros, t):
    opposing = {'simple': ['compound',
                           'simple_compound',
                          'complex',
                          'simple_complex',
                          'compound_complex',
                          'simple_compound_complex'],
                'compound':['compound_complex', 'simple_compound_complex', 'complex', 'simple_compound', 'simple_complex'],
                'complex': ['compound_complex', 'simple_compound_complex','simple_compound', 'simple_complex'],
                'simple_compound':['compound_complex', 'simple_compound_complex'],
                'simple_complex': ['compound_complex', 'simple_compound_complex'],
                'compound_complex': ['simple_compound_complex'],
                'simple_compound_complex': ['compound_complex']}

    results = {t : {}}

    for c in opposing[t]:
        distro_t = distros[t].view(-1,)
        distro_s = distros[c].view(-1,)
        results[t][c] = KL(distro_s, distro_t)

    return results

def wasserstein(distros, t):
    opposing = {'simple': 'simple',
                'compound': 'simple_compound',
                'complex': 'simple_complex',
                'compound_complex': 'simple_compound_complex',
                'simple_compound': 'compound',
                'simple_complex': 'complex',
                'simple_compound_complex': 'compound_complex'}

    comparison = opposing[t]

    distro_t = distros[t].view(-1,)
    distro_s = distros[comparison].view(-1,)

    return wasserstein_distance(distro_t, distro_s)

def get_performance(logits, obj, reference_vocab, vocab_words, counts, hidden_states, performance, _range=None, domain=None,  finding=False):
    start = timer()
    sorted_args = torch.flip(torch.argsort(logits, dim=1), [1])

    end = timer()
    #print("flip", end - start)
    rel_vocab = vocab_words[reference_vocab]
    #print(rel_vocab.shape)
    ranking = torch.stack((sorted_args == obj).nonzero(as_tuple=True), dim=1)

    prob = logits[ranking[:, 0], sorted_args[ranking[:, 0], ranking[:, 1]]]
    entropy = torch.sum(logits*(-torch.log2(logits)), dim=1)
    highest = torch.max(logits, dim=1)[0]
    ranking = ranking[:, 1]
    for value, _tensor in {'ranking': ranking,
                           'entropy': entropy,
                           'probability': prob,
                           'distros': logits,
                           'hidden_states': hidden_states,
                           'highest_prob': highest}.items():
        if performance.get(value):
            for t, v in counts.items():
                performance[value][t] = _tensor[v[0]:v[1]].cpu()
        else:
            performance[value] = {t: _tensor[v[0]:v[1]].cpu() for t, v in counts.items()}.copy()

    #print(domain)
    #print([performance['probability'][t] for t in ['compound', 'simple_compound']])
    #print([torch.argmax(performance['probability'][t]).item() for t in ['compound', 'simple_compound']])
    #print(_range)
    opposing = {'simple': 'simple',
                'compound': 'simple_compound',
                'complex': 'simple_complex',
                'compound_complex': 'simple_compound_complex',
                'simple_compound': 'compound',
                'simple_complex': 'complex',
                'simple_compound_complex': 'compound_complex'}
    if finding:
        performance['best_range_by_prob'] = {t: _range[torch.argmax(performance['highest_prob'][t]).item()] for t in ['complex', 'simple_complex']}
        performance['best_domain_by_prob'] = {t: domain[torch.argmax(performance['highest_prob'][t]).item()] for t in ['compound', 'simple_compound']}
        performance['entropy_by_prob'] = {t: performance['entropy'][t][torch.argmax(performance['highest_prob'][t]).item()].cpu() for t in counts.keys()}
        performance['entropy_inverse_by_prob'] = {t: performance['entropy'][t][torch.argmax(performance['highest_prob'][opposing[t]]).item()].cpu() for t in counts.keys()}
        performance['distros_by_prob'] = {t: performance['distros'][t][torch.argmax(performance['highest_prob'][t]).item()].cpu() for t in counts.keys()}
        performance['hidden_states_by_prob'] = {t: performance['hidden_states'][t][torch.argmax(performance['highest_prob'][t]).item():torch.argmax(performance['highest_prob'][t]).item()+1].cpu() for t in counts.keys()}

        performance['best_range'] = {t: _range[torch.argmax(performance['probability'][t]).item()] for t in ['complex', 'simple_complex']}
        performance['best_domain'] = {t: domain[torch.argmax(performance['probability'][t]).item()] for t in ['compound', 'simple_compound']}
        performance['entropy_inverse'] = {t: performance['entropy'][t][torch.argmax(performance['probability'][opposing[t]]).item()].cpu() for t in counts.keys()}
        performance['entropy'] = {t: performance['entropy'][t][torch.argmax(performance['probability'][t]).item()].cpu() for t in counts.keys()}
        performance['distros'] = {t: performance['distros'][t][torch.argmax(performance['probability'][t]).item()].cpu() for t in counts.keys()}
        performance['hidden_states'] = {t: performance['hidden_states'][t][torch.argmax(performance['probability'][t]).item():torch.argmax(performance['probability'][t]).item()+1].cpu() for t in counts.keys()}

    #del performance['distros']#, performance['hidden_states']
    """if performance.get('wasserstein'):
        for t, v in counts.items():
            if 'simple' in t:
                continue
            performance['wasserstein'][t] = wasserstein(performance['distros'], t)
    else:
        performance['wasserstein'] = {t: wasserstein(performance['distros'], t) for t in counts.keys() if 'simple' not in t}.copy()

    if 'compound_complex' in performance['ranking']:
        performance['wasserstein_gain'] = {t: wasserstein_gain(performance['distros'], t) for t in ['simple',
                                                                                                      'compound',
                                                                                                      'simple_compound',
                                                                                                      'complex',
                                                                                                      'simple_complex',
                                                                                                      'compound_complex',
                                                                                                      'simple_compound_complex']}.copy()
        performance['kl_gain'] = {t: kl_gain(performance['distros'], t) for t in ['simple',
                                                                                                      'compound',
                                                                                                      'simple_compound',
                                                                                                      'complex',
                                                                                                      'simple_complex',
                                                                                                      'compound_complex',
                                                                                                      'simple_compound_complex']}.copy()"""


    for k in [1, 10, 50, 100, 200, 500, 1000]:
        if k not in performance:
            performance[k] = {'acc': {}, 'inverse_acc': {}, 'highest_acc': {}, 'inverse_highest_acc': {}}.copy()
        start = timer()
        if k == 0:
            k += 1
            #performance[f"{k}_pred"].append([rel_vocab[i] for i in s_a[:k]])
        pred = [obj in s_a for s_a in sorted_args[:, :k]]
        for t, v in counts.items():
            if t in ['compound_complex', 'simple_compound_complex']:
                performance[k]['acc'][t] = pred[v[0]:v[1]][0]
                performance[k]['inverse_acc'][t] = pred[v[0]:v[1]][1]
                performance[k]['highest_acc'][t] = pred[v[0]:v[1]][2]
                performance[k]['inverse_highest_acc'][t] = pred[v[0]:v[1]][3]

            else:
                performance[k]['acc'][t] = pred[v[0]:v[1]][torch.argmax(performance['probability'][t]).item()]
                performance[k]['inverse_acc'][t] = pred[v[0]:v[1]][torch.argmax(performance['probability'][opposing[t]]).item()]
                performance[k]['highest_acc'][t] = pred[v[0]:v[1]][torch.argmax(performance['highest_prob'][t]).item()]
                performance[k]['inverse_highest_acc'][t] = pred[v[0]:v[1]][torch.argmax(performance['highest_prob'][opposing[t]]).item()]

        end = timer()
        #print(f"K={k}", end - start)
    performance = {k: v if not torch.is_tensor(v) else v.cpu() for k, v in performance.items()}
    return performance, ranking


def clean(tok, specs):
    for s in specs:
        tok = tok.replace(s, "")
    return tok


def create_combined(temp, t, sub, mask, _range, replace_range, _domain, replace_domain):
    temps = []
    for r_d, r_r in zip(replace_domain, replace_range):
        t = temp.replace(_domain, r_d)
        t = t.replace(_range, r_r)
        t = t.replace("[X]", sub)
        t = t[0].upper() + t[1:]
        t = t.replace("[Y]", mask)
        temps.append(t)
    return temps

tokenizer_specifics = {'bert-base-cased': []}

def random_cosine(points, num_samples=1000):
    import numpy as np
    points = points.detach().numpy()
    cos_sim = []
    for _ in range(num_samples):
        p1 = np.reshape(points[np.random.randint(len(points))],(1,-1))
        p2 = np.reshape(points[np.random.randint(len(points))],(1,-1))
        cos_sim.append(cosine_similarity(p1,p2))
    return sum(cos_sim)[0][0]/num_samples

def calc_isotropy(hidden_states, t_n):
    isoscores = {}
    for t in t_n:
        isoscores[t] = []
        for i in range(hidden_states[0][t].size()[0]):
            h_s_i = torch.stack(tuple(h_s[t][i,:] for h_s in hidden_states))
            isoscores[t].append(random_cosine(h_s_i))

    return isoscores

for name, m in model_name.items():
    try:
        if len(results.get(m).get('subjects')) == 41:
            continue
    except:
        pass
    if m not in results:
        results[m] = {}
    tokenizer, model, mask, mask_id, vocab, vocab_words = get_model_environment(m)
    for rel, p in enumerate(data): #enumerate(data):
        results[m] = {}
        try:
            if results[m][p]['subjects']:
                continue
        except:
            pass
        print(p)

        results[m][p] = {}
        results[m][p]['subjects'] = []
        for s, o in tqdm(list(zip(data[p]['subjects'], data[p]['objects']))):
            #print(s, o)
            template_names = ['simple',
                              'compound',
                              'simple_compound',
                              'complex',
                              'simple_complex']
            sentences = [create_sample(data[p][t],
                                       t,
                                       s,
                                       mask,
                                       data[p]['range'],
                                       data[p]['possible_range'] if 'possible_range' in data[p] else [],
                                       data[p]['domain'],
                                       data[p]['possible_domain'] if 'possible_domain' in data[p] else []) for t in template_names]

            counts = {t: (sum(len(sentences[j]) for j in range(0, i)), sum(len(sentences[j]) for j in range(0, i+1))) for i, t in enumerate(template_names)}
            sentences = list(itertools.chain(*sentences))
            _range = [data[p]['range']]
            if 'possible_range' in data[p]:
                _range += data[p]['possible_range']
            domain = [data[p]['domain']]
            if 'possible_domain' in data[p]:
                domain += data[p]['possible_domain']
            results[m][p]['range'] = _range
            results[m][p]['domain'] = domain
            encodings = {k: v.cuda() for k, v in tokenize(sentences).items()}
            masked_positions = (encodings['input_ids'] == mask_id).nonzero(as_tuple=True)
            exclude = exclusion[p][s].difference({o})
            logits, hidden_states, reference_vocab = get_results(encodings, model, vocab, vocab_words, exclude, masked_positions)
            start = timer()
            try:
                o_id = np.where(vocab_words[reference_vocab] == o)[0][0]
            except IndexError:
                print(o)
                results[m][p]['subjects'].append(False)
                continue
            #print(o_id)
            performance, ranks = get_performance(logits,
                                                 o_id,
                                                 reference_vocab,
                                                 vocab_words,
                                                 counts,
                                                 hidden_states, {}.copy(), _range, domain, True)
            performance['object'] = o
            performance['subject'] = s
            end = timer()
            #print("performance", end - start)
            results[m][p]['subjects'].append(performance)

            template_names = ['compound_complex', 'simple_compound_complex']
            _domain_info = ['compound', 'simple_compound']
            _range_info = ['complex', 'simple_complex']
            opposing = {'simple': 'simple',
                        'compound': 'simple_compound',
                        'complex': 'simple_complex',
                        'compound_complex': 'simple_compound_complex',
                        'simple_compound': 'compound',
                        'simple_complex': 'complex',
                        'simple_compound_complex': 'compound_complex'}
            sentences = [create_combined(data[p][t],
                                       t,
                                       s,
                                       mask,
                                       data[p]['range'],
                                       [performance['best_range'][r],
                                        performance['best_range'][opposing[r]],
                                        performance['best_range_by_prob'][r],
                                        performance['best_range_by_prob'][opposing[r]]],
                                       data[p]['domain'],
                                       [performance['best_domain'][d],
                                        performance['best_domain'][opposing[d]],
                                        performance['best_domain_by_prob'][d],
                                        performance['best_domain_by_prob'][opposing[d]]],) for t, r, d in
                                    zip(template_names, _range_info, _domain_info)]
            o = performance['object']
            counts = {t: (sum(len(sentences[j]) for j in range(0, i)), sum(len(sentences[j]) for j in range(0, i+1))) for i, t in enumerate(template_names)}
            sentences = list(itertools.chain(*sentences))
            encodings = {k: v.cuda() for k, v in tokenize(sentences).items()}
            masked_positions = (encodings['input_ids'] == mask_id).nonzero(as_tuple=True)
            exclude = exclusion[p][s].difference({o})
            logits, hidden_states, reference_vocab = get_results(encodings, model, vocab, vocab_words, exclude,
                                                                 masked_positions)
            o_id = np.where(vocab_words[reference_vocab] == o)[0][0]
            new_performance, ranks = get_performance(logits,
                                                     o_id,
                                                     reference_vocab,
                                                     vocab_words,
                                                     counts,
                                                     hidden_states,
                                                     performance)
            del new_performance['distros'], new_performance['distros_by_prob']
            results[m][p]['subjects'][-1] = new_performance

        acc = {}
        template_names = ['simple',
                          'compound',
                          'simple_compound',
                          'complex',
                          'simple_complex',
                          'compound_complex',
                          'simple_compound_complex']

        results[m][p]['isotropy'] = calc_isotropy(
            [s['hidden_states'] for s in results[m][p]['subjects'] if s],
            template_names)
        for s in results[m][p]['subjects']:
            if not s:
                continue
            del s['hidden_states']
        for k in [1, 10, 50, 100, 200, 500, 1000]:
            acc[k] = {}
            for accs in ['acc','inverse_acc','highest_acc', 'inverse_highest_acc']:
                acc[k][accs] = {}
                for t in template_names:
                    acc[k][accs][t] = {}
                    acc[k][accs][t]['correct'] = sum([s[k][accs][t] for s in results[m][p]['subjects'] if s])
                    acc[k][accs][t]['total'] = len([x for x in results[m][p]['subjects'] if x])
        results[m][p]['accuracy'] = acc.copy()
        results[m][p]['total'] = len([x for x in results[m][p]['subjects'] if x])
        pickle.dump(results, open(f"/results/{dataset}/{name}/{p}.p", "wb"))


#results = pickle.load(open(f"/data_ssds/disk11/slinzbach/LAMA/results/bert_both/P1001.p", "rb"))
