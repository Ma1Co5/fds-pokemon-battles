"""Submission 1 - final model: hidden P2 knock-outs, P2 team reconstruction,
out-of-fold encoders and a soft-voting ensemble (logistic regression + two HistGradientBoosting).

Run from the repository root:
    python src/model_final.py [output_csv]
The output file defaults to submission_final.csv.
"""
import sys
OUTPUT_CSV = sys.argv[1] if len(sys.argv) > 1 else "submission_final.csv"

#blocco 1
#data import
import os, json, time
import numpy as np, pandas as pd

base = next((r for r, _, fs in os.walk("/kaggle/input") if "train.jsonl" in fs), ".")
load = lambda p: [json.loads(l) for l in open(os.path.join(base, p))]
train, test = load("train.jsonl"), load("test.jsonl")
print(base, len(train), len(test))




#blocco 2
#import library
import numpy as np, pandas as pd
from collections import Counter, defaultdict
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
 
 
#type table
def load_type_chart(path):
    chart = defaultdict(dict)
    for r in pd.read_csv(path).itertuples():
        chart[r.attacking_type][r.defending_type] = float(r.multiplier)
    return chart
 
 
#Multiplier of an attack of type `atk` against a Pokemon with types `defs`.
def eff(atk, defs):
    m = 1.0
    for d in defs:
        if d and d != "notype":   #skip empty types
            m *= CHART.get(atk, {}).get(d, 1.0)   #multiply by the value in the table
    return m
 
#Best damage multiplier the attacker can get against the defender
def best_stab(att, defs):
    v = [eff(t, defs) for t in att if t and t != "notype"]   #multiplier of each type of the attacker
    return max(v) if v else 1.0
 
 
#Who has the type advantage, Pokemon a or Pokemon b?
#   positive -> a is better, negative -> b is better, 0 -> even
def ladv(a, b):
    return np.log2(max(best_stab(a, b), .25)) - np.log2(max(best_stab(b, a), .25))
 
 
#Finding what each move does by looking at what happens after it is used.
#The roles are: heal, boost, boom (user faints), sleep, status, phaze (opponent forced out).
#A move gets a role if that rate is above a threshold. Only moves used at least `min_uses` times are considered.
def derive_move_roles(battles, min_uses=30):
    use = Counter(); cat = {}; fx = defaultdict(Counter); E = ("heal", "boost", "boom", "sleep", "status", "phaze")
    for b in battles:
        tl = b["battle_timeline"]
        for i in range(1, len(tl)):
            for s, o in (("p1", "p2"), ("p2", "p1")):
                m = tl[i][f"{s}_move_details"]
                if not m: continue
                # a0/a1 = user before/after the move, b0/b1 = opponent before/after
                a0, a1 = tl[i - 1][f"{s}_pokemon_state"], tl[i][f"{s}_pokemon_state"]
                b0, b1 = tl[i - 1][f"{o}_pokemon_state"], tl[i][f"{o}_pokemon_state"]
                k = m["name"]; use[k] += 1; cat[k] = m["category"]
                if a0["name"] == a1["name"]:
                    fx[k]["heal"] += a1["hp_pct"] - a0["hp_pct"] > 0.2
                    fx[k]["boost"] += sum(a1["boosts"].values()) > sum(a0["boosts"].values())
                fx[k]["boom"] += a1["status"] == "fnt"
                if b0["name"] == b1["name"]:
                    fx[k]["sleep"] += b1["status"] == "slp" and b0["status"] != "slp"
                    fx[k]["status"] += b1["status"] not in ("nostatus", "fnt", "slp") and b0["status"] == "nostatus"
                    #opponent changed Pokemon although it was not KO and it still made a move -> it was forced out
                fx[k]["phaze"] += b0["name"] != b1["name"] and b0["status"] != "fnt" and tl[i][f"{o}_move_details"] is not None
    rate = {k: {e: fx[k][e] / use[k] for e in E} for k in use if use[k] >= min_uses}
    base = {e: np.median([r[e] for r in rate.values()]) for e in E}
    thr = {"heal": .3, "boost": .3, "boom": .3, "sleep": .2, "status": .2, "phaze": 2 * base["phaze"]}
    roles = {e: {k for k, r in rate.items() if r[e] >= thr[e] and (e != "phaze" or cat[k] == "status")} for e in E}
    roles["phaze"] -= roles["boost"]   #a move that boosts the user is not a phazing move
    roles["top"] = [k for k, _ in use.most_common(50)]        #the 50 most used moves
    #the 10 most frequent volatile effects
    roles["effects"] = [e for e, _ in Counter(e for b in battles for t in b["battle_timeline"] for s in ("p1", "p2")
                                               for e in t[f"{s}_pokemon_state"]["effects"] if e != "noeffect").most_common(10)]
    return roles
 
 
#Making a small "Pokedex": saving the types, speed and total stats of each Pokemon.
def build_species_db(battles):
    db = {}
    for b in battles:
        for p in list(b["p1_team_details"]) + [b["p2_lead_details"]]:
            db.setdefault(p["name"], {"types": p["types"], "spe": p["base_spe"],
                                      "bst": sum(p[k] for k in ("base_hp", "base_atk", "base_def", "base_spa", "base_spd", "base_spe"))})
    return db
 
 
#Guessing which P2 Pokemon were KO even if the data does not say "fnt".
#We say a P2 Pokemon is KO if:
#  - it left the field and never came back,
#  - P1 attacked in that turn,
#  - the new P2 Pokemon did not lose HP (so the attack hit the old one),
#  - the old one had less than 60% HP.
def hidden_p2_kos(tl):
    S = [t["p2_pokemon_state"] for t in tl]; M1 = [t["p1_move_details"] for t in tl]; n = len(tl)
    lasthp = {S[0]["name"]: S[0]["hp_pct"]}; out = {}
    for i in range(1, n):
        A, B = S[i - 1], S[i]   # A = previous P2 Pokemon, B = current one
        if A["name"] != B["name"] and A["status"] != "fnt":
            later = any(S[j]["name"] == A["name"] for j in range(i, n))
            if not later and M1[i] is not None and M1[i]["category"] != "status" \
                    and B["hp_pct"] >= lasthp.get(B["name"], 1.0) - 0.02 and A["hp_pct"] < 0.6:
                out[A["name"]] = i
        lasthp[B["name"]] = B["hp_pct"]
    return out
 
 
#Finding every time side s changes Pokemon (with the old not being KO).
def _switch_events(tl, s):
    o = "p1" if s == "p2" else "p2"
    S = [t[f"{s}_pokemon_state"] for t in tl]; n = len(tl)
    last = {S[0]["name"]: S[0]["hp_pct"]}; out = []   #last HP seen for each Pokemon
    for i in range(1, n):
        A, B = S[i - 1], S[i]   # A = old Pokemon, B = new Pokemon
        if A["name"] != B["name"] and A["status"] != "fnt":   #Pokemon changed when the old one is not KO
            mo = tl[i][f"{o}_move_details"]; ms = tl[i][f"{s}_move_details"]
            key = (mo is not None and mo["category"] != "status", B["hp_pct"] >= last.get(B["name"], 1.0) - 0.02,
                   ms is not None, min(3, int(A["hp_pct"] * 4)), min(2, (n - i) // 10))
            out.append((i, A["name"], key, any(S[j]["name"] == A["name"] for j in range(i, n))))   #last value: does A come back?
        last[B["name"]] = B["hp_pct"]   #remember the HP of the new Pokemon
    return out
 
 
#Learning how often a P2 "switch" is really a hidden KO.
def fit_ko_model(battles, prior=20):
    q1 = defaultdict(lambda: [0, 0]); r2 = defaultdict(lambda: [0, 0])
    for b in battles:
        for _, _, k, back in _switch_events(b["battle_timeline"], "p1"): q1[k][0] += back; q1[k][1] += 1   #P1 counts
        for _, _, k, back in _switch_events(b["battle_timeline"], "p2"): r2[k][0] += back; r2[k][1] += 1   #P2 counts
    g1 = sum(v[0] for v in q1.values()) / sum(v[1] for v in q1.values())   #overall P1 come-back rate
    model = {}
    for k, (bk, nk) in r2.items():
        q = (q1[k][0] + prior * g1) / (q1[k][1] + prior) if k in q1 else g1
        model[k] = (float(np.clip(1 - ((bk + 1) / (nk + 2)) / max(q, 1e-3), 0, .98)), q)
    return model
 
 
 
#Checking for one battle the chance that each P2 Pokemon is KO (the pokemon left and never come back), and in which turn.
#If it is marked "fnt", the chance is 1.
#If it left the field and never came back, we use BAYES' THEOREM
def p2_ko_probs(tl, model):
    probs = {}
    for i, nm, k, back in _switch_events(tl, "p2"):
        if back: continue   #it came back, so it was not KO
        p, q = model.get(k, (0.0, 0.6))
        pk = p / (p + (1 - p) * (1 - q)) if p > 0 else 0.0   #the chance of KO given that it never came back
        if pk > probs.get(nm, (0, 0))[0]: probs[nm] = (pk, i)   #keeping the highest chance
    for i, t in enumerate(tl):
        x = t["p2_pokemon_state"]
        if x["status"] == "fnt" and x["name"] not in probs: probs[x["name"]] = (1.0, i)   #KO shown in the data
        elif x["status"] == "fnt": probs[x["name"]] = (1.0, min(i, probs[x["name"]][1]))   #keeping the earliest turn
    return probs
 
#Slope of a line through the values: if > 0 means it goes up over time
def _slope(v):
    return float(np.polyfit(np.arange(len(v)), v, 1)[0])
 
 
#feature of one battle
def battle_features(b, db, R, kom, cands):
    tl = b["battle_timeline"]; n = len(tl); f = {}
    S = {s: [t[f"{s}_pokemon_state"] for t in tl] for s in ("p1", "p2")}   #Pokemon state for each turn
    M = {s: [t[f"{s}_move_details"] for t in tl] for s in ("p1", "p2")}   #move for each turn
    hp = {s: np.array([x["hp_pct"] for x in S[s]], dtype=float) for s in S}   #HP for each turn
    nm = {s: [x["name"] for x in S[s]] for s in S}   #Pokemon name for each turn
    hidden = hidden_p2_kos(tl); kp = p2_ko_probs(tl, kom)   #hidden P2 KOs
 
    #HP difference between the two Pokemon on the field (if > 0 means that P1 ahead)
    d = hp["p1"] - hp["p2"]
    f.update(hp_diff_mean=d.mean(), hp_diff_last=d[-1], hp_diff_last5=d[-5:].mean(), hp_diff_last10=d[-10:].mean(),
             hp_diff_first10=d[:10].mean(), hp_diff_slope=_slope(d), hp_frac_ahead=float((d > 0).mean()))
 
    #Same features for P1 and for P2
    for s in ("p1", "p2"):
        o = "p2" if s == "p1" else "p1"   # the opponent
 
        # --- what is left of the team (Pokemon never seen is a pokemon alive, full HP and healthy)---
        last = {}
        for x in S[s]: last[x["name"]] = x   #last state of each Pokemon
        alive = [v for v in last.values() if v["status"] != "fnt"]; unseen = 6 - len(last)
        f[f"res_{s}_alive"] = len(alive) + unseen   #Pokemon still alive
        f[f"res_{s}_team_hp"] = sum(v["hp_pct"] for v in alive) + unseen   #total HP of the team
        f[f"res_{s}_healthy"] = sum(v["status"] == "nostatus" for v in alive) + unseen   #alive with no status
        f[f"res_{s}_asleep"] = sum(v["status"] == "slp" for v in alive)
        f[f"res_{s}_par"] = sum(v["status"] == "par" for v in alive)
        f[f"res_{s}_poisoned"] = sum(v["status"] in ("tox", "psn") for v in alive)
        f[f"res_{s}_frz"] = sum(v["status"] == "frz" for v in alive)
        f[f"res_{s}_low_hp"] = sum(v["hp_pct"] < 0.34 for v in alive)   #alive but almost KO
        f[f"res_{s}_unseen"] = unseen
        f[f"res_{s}_hp_last"] = hp[s][-1]   #HP of the Pokemon on the field at the end
        f[f"res_{s}_active_status_last"] = int(S[s][-1]["status"] not in ("nostatus", "fnt"))
 
        # --- KOs ---
        ft = [i for i, x in enumerate(S[s]) if x["status"] == "fnt"]   #turns with a KO
        f[f"ko_{s}_fainted"] = len({nm[s][i] for i in ft})   #how many Pokemon are KO
        f[f"ko_{s}_fainted_last10"] = len({nm[s][i] for i in ft if i >= n - 10})   #number of KOs in the last 10 turns
        f[f"ko_{s}_first_faint_turn"] = ft[0] + 1 if ft else n + 1   #turn of the first KO
        f[f"ko_{s}_self_ko"] = sum(1 for i in ft if M[s][i] and M[s][i]["name"] in R["boom"])   #number of KO by Explosion
  
        #---status---
        for st in ("slp", "par", "tox", "psn", "brn", "frz"):
            f[f"st_{s}_{st}_turns"] = sum(x["status"] == st for x in S[s])
        f[f"st_{s}_status_turns"] = sum(x["status"] not in ("nostatus", "fnt") for x in S[s])
        on, prev = 0, {}
        for x in S[s]:
            on += x["status"] not in ("nostatus", "fnt") and prev.get(x["name"], "nostatus") == "nostatus"
            prev[x["name"]] = x["status"]
        f[f"st_{s}_status_received"] = on
        f[f"st_{s}_mons_statused"] = len({x["name"] for x in S[s] if x["status"] not in ("nostatus", "fnt")})
         #turns lost because asleep or frozen 
        f[f"st_{s}_wasted_turns"] = sum(1 for x, m in zip(S[s], M[s]) if x["status"] in ("slp", "frz") and not (m and m["name"] == "sleeptalk"))
 
        #---switches---
        sw = fo = 0   # sw = all switches, fo = forced (after a KO)
        for i in range(1, n):
            if nm[s][i] != nm[s][i - 1]:
                sw += 1; fo += S[s][i - 1]["status"] == "fnt"
        f[f"sw_{s}_switches"] = sw; f[f"sw_{s}_forced"] = fo; f[f"sw_{s}_voluntary"] = sw - fo
        f[f"sw_{s}_n_seen"] = len(set(nm[s])); f[f"sw_{s}_null_moves"] = sum(m is None for m in M[s])
 
        #---moves---
        mv = [m for m in M[s] if m]; mvn = [m["name"] for m in mv]; k = max(1, len(mv))   #moves used and their names
        f[f"mv_{s}_status_frac"] = sum(m["category"] == "status" for m in mv) / k   #share of status moves
        f[f"mv_{s}_attack_frac"] = sum(m["category"] != "status" for m in mv) / k   #share of attacks
        f[f"mv_{s}_prio"] = sum(m["priority"] > 0 for m in mv)   #priority moves (such as Quick Attack)
        for r in ("heal", "sleep", "boost", "phaze", "boom", "status"):
            f[f"mv_{s}_{r}_moves"] = sum(x in R[r] for x in mvn)
        f[f"mv_{s}_n_distinct"] = len(set(mvn))
        for x in R["top"]:
            f[f"mv_{s}_{x}"] = mvn.count(x)   #one column for each of the 50 most used moves
 
        
        #---damage---
        drops = [max(0.0, hp[o][i - 1] - hp[o][i]) for i in range(1, n) if nm[o][i] == nm[o][i - 1]]   #HP lost by the opponent each turn
        f[f"dmg_{s}_dealt"] = float(sum(drops)); f[f"dmg_{s}_big_hits"] = sum(x > 0.3 for x in drops)   #total damage
        f[f"dmg_{s}_healed"] = float(sum(max(0.0, hp[s][i] - hp[s][i - 1]) for i in range(1, n) if nm[s][i] == nm[s][i - 1]))   #HP healed
        f[f"rt_{s}_dmg_last10"] = float(sum(max(0.0, hp[o][i - 1] - hp[o][i]) for i in range(n - 10, n) if nm[o][i] == nm[o][i - 1]))   #damage in the last 10 turns
 
        #---boosts and effects--
        pos = [sum(v for v in x["boosts"].values() if v > 0) for x in S[s]]   #positive boosts in each turn
        f[f"bst_{s}_max"] = max(pos); f[f"bst_{s}_last"] = sum(S[s][-1]["boosts"].values())
        f[f"bst_{s}_atk_max"] = max(x["boosts"]["atk"] for x in S[s])
        for e in R["effects"]:
            f[f"eff_{s}_{e}"] = sum(e in x["effects"] for x in S[s])
 
        # --- Pokemon on the field at the end ---
        x = S[s][-1]; alive_now = x["status"] != "fnt"
        f[f"ctl_{s}_boost_x_hp"] = max(x["boosts"]["atk"], x["boosts"]["spa"]) * x["hp_pct"] if alive_now else 0.0   #a pokemon boosted and healthy is a dangerous pokemon
        f[f"ctl_{s}_asleep_last"] = int(x["status"] == "slp")
        f[f"ctl_{s}_hp_x_healthy"] = x["hp_pct"] * (x["status"] == "nostatus")
 
    #---Spikes---
    on = {"p1": 0, "p2": 0}; tu = {"p1": 0, "p2": 0}; hu = {"p1": 0, "p2": 0}
    for i in range(n):
        for s, o in (("p1", "p2"), ("p2", "p1")):
            m = M[s][i]
            if m and m["name"] == "spikes": on[o] = 1   #if s puts Spikes on the opponent's side
            if m and m["name"] == "rapidspin": on[s] = 0   #if s removes Spikes from its own side
        for s in ("p1", "p2"):
            tu[s] += on[s]; hu[s] += i > 0 and on[s] and nm[s][i] != nm[s][i - 1]
    for s in ("p1", "p2"):
        f[f"hz_spikes_on_{s}_last"] = on[s]; f[f"hz_spikes_on_{s}_turns"] = tu[s]; f[f"hz_switch_into_spikes_{s}"] = hu[s]
    f["hz_diff_spikes_last"] = on["p2"] - on["p1"]; f["hz_diff_turns"] = tu["p2"] - tu["p1"]; f["hz_diff_hurt"] = hu["p2"] - hu["p1"]
 
    # --- matchup in each turn: who has the type advantage and who is faster---
    adv, spd = [], []
    for i in range(n):
        a, c = db.get(nm["p1"][i]), db.get(nm["p2"][i])   #Pokedex info of the two Pokemon
        if a and c:
            adv.append(ladv(a["types"], c["types"]))   #type advantage
            # +1 if P1 is faster, -1 if P2 is faster (paralysis = speed / 4)
            spd.append(np.sign(a["spe"] * (.25 if S["p1"][i]["status"] == "par" else 1) - c["spe"] * (.25 if S["p2"][i]["status"] == "par" else 1)))
    f["mu_type_adv_mean"] = float(np.mean(adv)) if adv else 0.0
    f["mu_type_adv_last"] = adv[-1] if adv else 0.0
    f["mu_speed_adv_mean"] = float(np.mean(spd)) if spd else 0.0
    f["mu_speed_adv_last"] = spd[-1] if spd else 0.0
    #Lead Pokemon of each side
    team = b["p1_team_details"]; l2 = b["p2_lead_details"]
    f["lead_type_adv"] = ladv(team[0]["types"], l2["types"])   #type advantage of the leads
    f["lead_speed_diff"] = team[0]["base_spe"] - l2["base_spe"]   #speed difference of the leads
    f["lead_team_answers"] = sum(best_stab(l2["types"], p["types"]) <= .5 and best_stab(p["types"], l2["types"]) >= 1 for p in team)   #P1 Pokemon good against the P2 lead
 
 
    #Counting alive Pokemon and total HP of side s, looking only at the first `upto` turns.
    #For P2, Pokemon that are probably KO count less/zero
    def resources(s, upto, prob=False):
        last = {}
        for x in S[s][:upto]: last[x["name"]] = x   #last state of each Pokemon up to turn `upto`
        a = h = 0.0   # a = alive, h = HP
        for k, x in last.items():
            if x["status"] == "fnt": continue   #skip KO Pokemon
            w = 1.0   #weight of this Pokemon
            if s == "p2":
                if prob and k in kp and kp[k][1] < upto: w = 1 - kp[k][0]
                elif not prob and k in hidden and hidden[k] < upto: w = 0.0
            a += w; h += w * x["hp_pct"]
        u = 6 - len(last)
        return a + u, h + u, last
 
    #Situation at turns 10, 15, 20, 25, 30:
    tr = []
    for c in (10, 15, 20, 25, 30):
        a1, h1, _ = resources("p1", c); a2, h2, _ = resources("p2", c, prob=True)
        f[f"tr_hp_diff_t{c}"] = h1 - h2; f[f"tr_alive_diff_t{c}"] = a1 - a2; tr.append(h1 - h2)
    f["tr_hp_change_last10"] = tr[-1] - tr[-3]; f["tr_hp_change_last5"] = tr[-1] - tr[-2]   #who gained in the last turns
    #Final situation with hidden KOs fixed
    a2o, h2o, _ = resources("p2", n)
    a1, h1, last1 = resources("p1", n); a2, h2, last2 = resources("p2", n, prob=True)
    f["kp_p2_exp_ko"] = sum(p for p, _ in kp.values())   #expected number of P2 KOs
    f["kp_diff_ko"] = f["ko_p1_fainted"] - f["kp_p2_exp_ko"]
    f["kp_diff_alive"] = a1 - a2; f["kp_diff_team_hp"] = h1 - h2
    f["kx_diff_alive_rule"] = a1 - a2o; f["kx_diff_team_hp_rule"] = h1 - h2o
 
    #P2 team is only partly known, so we try all the possible full teams and take the average, giving more weight to common teams.
    p1a = [(p["types"], p["base_spe"], 1.0 if p["name"] not in last1 else last1[p["name"]]["hp_pct"])
           for p in team if p["name"] not in last1 or last1[p["name"]]["status"] != "fnt"]   # alive P1 Pokemon
    acc = dict(rc_threat_on_p2=0.0, rc_threat_on_p1=0.0, rc_speed_edge=0.0, rc_p2_hidden_bst=0.0, rc_p2_alive_full=0.0)
    for key, w in cands:   #key = possible P2 team, w = its weight
        p2a = []   #alive P2 Pokemon of this team
        for k in key.split("|"):
            if k not in db: continue
            x = last2.get(k)
            if x is not None and x["status"] == "fnt": continue   #skip KO Pokemon
            wk = 1 - kp[k][0] if k in kp else 1.0   #chance it is still alive
            p2a.append((db[k]["types"], db[k]["spe"], (1.0 if x is None else x["hp_pct"]) * wk, x is None, k))
        if not p2a or not p1a: continue
        acc["rc_threat_on_p2"] += w * np.mean([np.mean([best_stab(a, t) >= 2 for a, _, _ in p1a]) * h for t, _, h, _, _ in p2a])   # P1 can hit P2 super-effectively
        acc["rc_threat_on_p1"] += w * np.mean([np.mean([best_stab(t, a) >= 2 for t, _, _, _, _ in p2a]) * h for a, _, h in p1a])   # P2 can hit P1 super-effectively
        acc["rc_speed_edge"] += w * (max(s for _, s, _ in p1a) - max(s for _, s, _, _, _ in p2a))   # fastest P1 - fastest P2
        acc["rc_p2_hidden_bst"] += w * sum(db[k]["bst"] for _, _, _, hid, k in p2a if hid) / 600   # strength of P2 Pokemon not seen yet
        acc["rc_p2_alive_full"] += w * sum(h for _, _, h, _, _ in p2a)   # HP left for P2
    f.update(acc); f["rc_known"] = int(bool(cands))   # 0 if we found no possible team
 
    # P1 minus P2 for the most important features (if > 0 , it is good for P1)
    for g, k in [("res", "alive"), ("res", "team_hp"), ("res", "healthy"), ("res", "asleep"), ("res", "hp_last"),
                 ("ko", "fainted"), ("ko", "fainted_last10"), ("st", "status_turns"), ("st", "status_received"),
                 ("st", "wasted_turns"), ("st", "slp_turns"), ("st", "par_turns"), ("sw", "switches"), ("sw", "n_seen"),
                 ("sw", "forced"), ("dmg", "dealt"), ("dmg", "big_hits"), ("dmg", "healed"), ("rt", "dmg_last10"),
                 ("bst", "max"), ("bst", "last"), ("mv", "status_frac"), ("mv", "heal_moves"), ("mv", "sleep_moves"),
                 ("ctl", "boost_x_hp"), ("ctl", "hp_x_healthy"), ("ctl", "asleep_last")]:
        f[f"{g}_diff_{k}"] = f[f"{g}_p1_{k}"] - f[f"{g}_p2_{k}"]   #add the differences P1 - P2
    return f
 
 
#teams and columns for encoders
def team_key(names):
    return "|".join(sorted(names))
 
#All P1 teams in the data and how many times each one appears.
def build_team_pool(battles):
    pool = Counter(team_key(p["name"] for p in b["p1_team_details"]) for b in battles)
    by_mon = defaultdict(set)
    for k in pool:
        for m in k.split("|"): by_mon[m].add(k)
    return pool, by_mon
 
#Possible full teams of P2: teams that contain all the P2 Pokemon we saw, keeping the 30 most common;
def p2_candidates(b, pool, by_mon, max_c=30):
    seen = {b["p2_lead_details"]["name"]} | {t["p2_pokemon_state"]["name"] for t in b["battle_timeline"]}
    cand = None
    for m in seen:
        cand = set(by_mon.get(m, set())) if cand is None else cand & by_mon.get(m, set())   # keep teams that have all of them
    cand = sorted(cand or [], key=lambda k: (-pool[k], k))[:max_c]   # most common first
    tot = sum(pool[k] for k in cand)
    return [(k, pool[k] / tot) for k in cand]
 
 
#Text columns used only by the encoders
def encoder_columns(b, cands):
    p2 = {b["p2_lead_details"]["name"]} | {t["p2_pokemon_state"]["name"] for t in b["battle_timeline"]}   # P2 Pokemon seen
    return {"tq_p1_key": team_key(p["name"] for p in b["p1_team_details"]), "tq_p2_cands": cands,   # P1 team, possible P2 teams
            "pm_p1": "|".join(p["name"] for p in b["p1_team_details"]), "pm_p2": "|".join(sorted(p2)),   # P1 Pokemon, P2 Pokemon
            "pm_lead": b["p1_team_details"][0]["name"] + "|" + b["p2_lead_details"]["name"]}   # the two leads
 
ENC_COLS = ["tq_p1_key", "tq_p2_cands", "pm_p1", "pm_p2", "pm_lead"]
 
 
class _OOF(TransformerMixin, BaseEstimator):
    def fit_transform(self, X, y):
        y = np.asarray(y); self.fit(X, y); E = np.zeros((len(X), len(self.names_)))
        for tr, va in StratifiedKFold(self.inner_folds, shuffle=True, random_state=self.random_state).split(X, y):
            E[va] = self._enc(X.iloc[va], self._stats(X.iloc[tr], y[tr]))
        return self._out(X, E)
 
    def fit(self, X, y):
        self.st_ = self._stats(X, np.asarray(y)); return self
 
    def transform(self, X):
        return self._out(X, self._enc(X, self.st_))
 
    def _out(self, X, E):
        o = X.copy()
        for j, nm in enumerate(self.names_): o[nm] = E[:, j]
        return o
 
#Strength score for every team and every Pokemon 
#if the output bestscore is > 0 means that P1 is stronger.
class BTRating(_OOF):
    names_ = ["bt_score"]
 
    def __init__(self, C=1.0, inner_folds=5, random_state=0):
        self.C = C; self.inner_folds = inner_folds; self.random_state = random_state
 
    def _design(self, X, ti, si):
        #table: one row per battle, one column per team and per Pokemon
        r, c, v = [], [], []   #row, column, value
        for i, (k1, cands, s1, s2) in enumerate(zip(X["tq_p1_key"], X["tq_p2_cands"], X["pm_p1"], X["pm_p2"])):
            if k1 in ti: r.append(i); c.append(ti[k1]); v.append(1.0)   #P1 team = +1
            for k, w in cands:
                if k in ti: r.append(i); c.append(ti[k]); v.append(-w)
            a = [x for x in s1.split("|") if x in si]; b = [x for x in s2.split("|") if x in si]
            for x in a: r.append(i); c.append(si[x]); v.append(1.0 / max(1, len(a)))
            for x in b: r.append(i); c.append(si[x]); v.append(-1.0 / max(1, len(b)))
        return sparse.csr_matrix((v, (r, c)), shape=(len(X), len(ti) + len(si)))
 
    def _stats(self, X, y):
        keys = sorted(set(X["tq_p1_key"]) | {k for c in X["tq_p2_cands"] for k, _ in c})   # all teams
        sps = sorted({x for v in X["pm_p1"] for x in v.split("|")} | {x for v in X["pm_p2"] for x in v.split("|")})   # all Pokemon
        ti = {k: i for i, k in enumerate(keys)}; si = {x: i + len(ti) for i, x in enumerate(sps)}
        return ti, si, LogisticRegression(C=self.C, max_iter=3000).fit(self._design(X, ti, si), y)
 
    def _enc(self, X, st):
        ti, si, m = st
        return m.decision_function(self._design(X, ti, si))[:, None]   #strength score
 
#How often each team wins
class TeamQualityEncoder(_OOF):
    names_ = ["tq_p1_quality", "tq_p2_quality", "tq_diff_quality", "tq_p1_n", "tq_p2_n", "tq_p2_known"]
 
    def __init__(self, smoothing=2, inner_folds=5, random_state=0):
        self.smoothing = smoothing; self.inner_folds = inner_folds; self.random_state = random_state
 
    def _stats(self, X, y):
        w, c = defaultdict(float), defaultdict(int)   #wins and games for each team
        for k, yy in zip(X["tq_p1_key"], y): w[k] += yy; c[k] += 1
        return w, c
 
    def _enc(self, X, st):
        w, c = st; m = self.smoothing
        q = lambda k: (w.get(k, 0) + .5 * m) / (c.get(k, 0) + m) - .5
        p1 = np.array([q(k) for k in X["tq_p1_key"]])
        p2 = np.array([sum(wt * q(k) for k, wt in cs) if cs else 0.0 for cs in X["tq_p2_cands"]])
        return np.c_[p1, p2, p1 - p2, [c.get(k, 0) for k in X["tq_p1_key"]],
                     [sum(wt * c.get(k, 0) for k, wt in cs) if cs else 0.0 for cs in X["tq_p2_cands"]],
                     [float(len(cs) > 0) for cs in X["tq_p2_cands"]]]
 
#Checking how often P1 wins when Pokemon X (P1) meets Pokemon Y (P2).
class PairMatchupEncoder(_OOF):
    names_ = ["pm_mean", "pm_max", "pm_min", "pm_lead_vs_lead"]
 
    def __init__(self, smoothing=30, inner_folds=5, random_state=0):
        self.smoothing = smoothing; self.inner_folds = inner_folds; self.random_state = random_state
 
    def _stats(self, X, y):
        w, c, wl, cl = defaultdict(float), defaultdict(int), defaultdict(float), defaultdict(int)   #wins/games for pairs and leads
        for a, b2, ld, yy in zip(X["pm_p1"], X["pm_p2"], X["pm_lead"], y):
            for s in a.split("|"):
                for t in b2.split("|"): w[(s, t)] += yy; c[(s, t)] += 1   #every P1 Pokemon vs every P2 Pokemon
            wl[ld] += yy; cl[ld] += 1   #lead vs lead
        return w, c, wl, cl
 
    def _enc(self, X, st):
        w, c, wl, cl = st; m = self.smoothing; out = []
        for a, b2, ld in zip(X["pm_p1"], X["pm_p2"], X["pm_lead"]):
            v = [(w.get((s, t), 0) + .5 * m) / (c.get((s, t), 0) + m) - .5 for s in a.split("|") for t in b2.split("|")]
            out.append((np.mean(v), np.max(v), np.min(v), (wl.get(ld, 0) + .5 * m) / (cl.get(ld, 0) + m) - .5))
        return np.array(out)
 
 
class DropCols(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None): return self
    def transform(self, X): return X.drop(columns=[c for c in ENC_COLS if c in X.columns])
#---Building the dataset---
t = time.time()
CHART = load_type_chart(os.path.join(base, "type_chart_gen2.csv"))   #type chart
allb = train   #battles used to learn the helper tables
db = build_species_db(allb)   #Pokedex: types, speed, total stats of each Pokemon
R = derive_move_roles(allb)   #what each move does
kom = fit_ko_model(allb)   #model for hidden P2 KOs
pool, by_mon = build_team_pool(allb)   #list of known teams, and teams for each Pokemon

#All the features of one battle
def feats(b):
    c = p2_candidates(b, pool, by_mon)   #possible full teams of P2
    return {**battle_features(b, db, R, kom, c), **encoder_columns(b, c)}

X = pd.DataFrame([feats(b) for b in train])   #training table: one row per battle
X_test = pd.DataFrame([feats(b) for b in test]).reindex(columns=X.columns)   #test table, same columns in the same order as train
y = np.array([int(b["player_won"]) for b in train])   #what we want to predict: 1 = P1 won, 0 = P1 lost
print(X.shape, X_test.shape, f"{time.time() - t:.0f}s")   #table sizes and time taken
print({k: sorted(v) for k, v in R.items() if k not in ("top", "effects")})   #check: moves found for each role



#blocco 3
#---model and cross validation---
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

#Gradient boosting
def hgb(**k):
    p = dict(learning_rate=0.03,      #how much each new tree counts
             max_iter=500,            #number of trees
             max_leaf_nodes=15,       #size of each tree
             min_samples_leaf=40,     #at least 40 battles in each leaf
             l2_regularization=1.0,   #penalty for extreme values
             max_features=0.5,        #each tree uses only 50% of the features
             random_state=0)          #same result every time we run it
    p.update(k)   # eplace the defaults with the values we pass
    return HistGradientBoostingClassifier(**p)


def enc():
    return [BTRating(1.0), TeamQualityEncoder(2), PairMatchupEncoder(30), DropCols()]

#The final model
def make_model():
    return VotingClassifier([
        # Logistic regression
        ("lr", make_pipeline(*enc(), StandardScaler(), LogisticRegression(C=0.01, max_iter=5000))),
        #Gradient boosting
        ("hgb", make_pipeline(*enc(), hgb())),
        #Slower and more careful gradient boosting
        ("hgb2", make_pipeline(*enc(), hgb(learning_rate=0.01, max_iter=1500, max_leaf_nodes=7,
                                           min_samples_leaf=100, max_features=0.3, random_state=1))),
    ], voting="soft")

# 5-fold cross-validation: split the battles in 5 parts, train on 4, test on the 5th, and repeat 5 times.
s = cross_val_score(make_model(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42), scoring="accuracy", n_jobs=-1)
print(f"CV 5-fold: {s.mean():.4f} ± {s.std():.4f}")


#blocco 4
#---final model and submission---
model = make_model().fit(X, y)
proba = model.predict_proba(X_test)[:, 1]   #chance that P1 wins each test battle
sub = pd.DataFrame({"battle_id": [b["battle_id"] for b in test],   #id of each test battle
                    "player_won": (proba >= 0.5).astype(int)})   # 1 if chance >= 50%, otherwise 0
sub.to_csv(OUTPUT_CSV, index=False)
assert len(sub) == len(test) and sub.battle_id.is_unique and set(sub.player_won) <= {0, 1}
print(sub.shape, sub.player_won.mean().round(3))
print(sub.head())
