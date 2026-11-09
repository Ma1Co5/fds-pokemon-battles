"""Submission 2 - base, extended and advanced features; logistic regression (L1) or Random Forest, whichever scores higher in 5-fold CV.

Run from the repository root:
    python src/model_advanced_features.py [output_csv]
The output file defaults to submission_advanced_features.csv.
"""
import sys
OUTPUT_CSV = sys.argv[1] if len(sys.argv) > 1 else "submission_advanced_features.csv"
#blocco 1
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#blocco 2
DATA_DIR=next((r for r, _, fs in os.walk("/kaggle/input") if "train.jsonl" in fs), ".")  # folder with the competition data

def data_file(nome, estensione):
    return os.path.join(DATA_DIR, f"{nome}{estensione}")

#blocco 3
import json
from collections import defaultdict

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


SEED=42

STATISTICHE=["base_hp", "base_atk", "base_def", "base_spa", "base_spd", "base_spe"]


def carica_match(path):
    """Legge un file .jsonl e carica ogni singola riga come una lista di dizionari (ogni dizionario rappresenta un singolo match)."""
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def dizionario_moltiplicatori(path):
    """ritorna {(attacking_type, defending_type): multiplier}, coppie che mancano corrispondono a un x1."""
    tabella=pd.read_csv(path)
    return {(a,d): m for a, d, m in zip(tabella.attacking_type, tabella.defending_type, tabella.multiplier)}


def fit_species_types(battaglie_osservate):
    """name -> types, crea un dizioanrio con i nomi dei pokemon nei dati osservati associati ai tipi"""
    tipi={}
    for match in battaglie_osservate:
        for p in match["p1_team_details"] + [match["p2_lead_details"]]:  #prende lista di ogni pokemon del primo giocatore e la unisce con il pokemon del giocatore 2
            tipi[p["name"].lower().strip()]=p.get("types", ["notype"])   #se non ha tipo metto notype: dizionario con chiave il nome del pokemon in minuscolo e senza spazi e come valori una lista di tipi
    return tipi


def fit_species_stats(battaglie_osservate):
    """name -> base stats (dizionario). Per ogni pokemon vengono associate le statistiche di base, per ciascuna in un dizionario"""
    stats={}
    for b in battaglie_osservate:
        for p in b["p1_team_details"] + [b["p2_lead_details"]]:
            stats[p["name"].lower().strip()]={s: p[s] for s in STATISTICHE}
    return stats


def efficacia(type_chart, attack_type, defend_types):
    """Moltiplicatore del danno di una mossa a seconda dei tipi dei pokemon."""
    m=1.0  #no moltiplicatore
    for d in defend_types:   #per ogni tipo del pokemon di di difesa verifico i tipi del pokemon che attacca, lo cerco neldizionario creato e modifico il moltiplicatore
        if d != "notype":
            m*=type_chart.get((attack_type, d), 1.0)
    return m


def miglior_moltiplicatore(type_chart, att_types, def_types):    #vantaggio di tipo: aumenta probabilità che un pokemon possa fare danni superefficaci
    """Miglior moltiplicatore che un pokemon che attacca può ottenere con le mosse del suo stesso tipo sul pokemno che difende"""
    valori=[efficacia(type_chart, t, def_types) for t in att_types if t != "notype"]
    return max(valori) if valori else 1.0


#blocco 4
def features_di_base(battles, type_chart, species_types):
    """Una riga per battaglia: differenze tra statistiche, vantagio di tipo nei turni e analisi al turno 30."""
    righe=[]
    for b in battles:
        team=b.get("p1_team_details", [])
        lead1=team[0] if team else {}
        lead2=b.get("p2_lead_details", {})
        features={}

        # differenze tra le statistiche di base dei team leaders
        features["lead_diff_spa"]=lead2.get("base_spa", 0)-lead1.get("base_spa", 0)
        features["lead_diff_spd"]=lead2.get("base_spd", 0)-lead1.get("base_spd", 0)
        features["lead_diff_atk"]=lead2.get("base_atk", 0)-lead1.get("base_atk", 0)
        features["lead_diff_def"]=lead2.get("base_def", 0)-lead1.get("base_def", 0)

        # tenere traccia degli hp: name -> ultima frazione di hp conosciuta nei 30 turni ---
        p1_hp={p["name"].lower().strip(): 1.0 for p in team}
        lead2_name=lead2.get("name", "unknown").lower().strip()
        p2_hp={lead2_name: 1.0}
        tipi_conosciuti=species_types.copy()

        for p in team:
           nome=p["name"].lower().strip()
           tipi=p.get("types", ["notype"])
           tipi_conosciuti[nome]=tipi
        
        prec1=lead1.get("name", "").lower().strip()  #Pokémon che era in campo nel turno prevedente
        prec2=lead2_name

        p1_effs,p2_effs=[],[]
        for t in b.get("battle_timeline", []):
            s1,s2=t["p1_pokemon_state"], t["p2_pokemon_state"]
            n1,n2=s1["name"].lower().strip(), s2["name"].lower().strip()   #nuovo Pokemon in campo in questo momento
            m1,m2=t.get("p1_move_details"),t.get("p2_move_details")   #mossa usata dal giocatore. Se vale None,il giocatore non ha attaccato ma ha effettuato uno switch
            p1_hp[n1]=s1["hp_pct"]
            p2_hp[n2]=s2["hp_pct"]

            #Se un giocatore decide di cambiare pokemon, questa azione avviene all'inizio del turno, prima che l'avversario sferri il suo colpo. Di conseguenza, l'attacco avversario colpirà il pokemon appena entrato.
            target1=n1 if m1 is None else prec1  #il bersaglio
            target2=n2 if m2 is None else prec2
            p1_effs.append(efficacia(type_chart, m1["type"] if m1 else "notype", tipi_conosciuti.get(target2, ["notype"])))
            p2_effs.append(efficacia(type_chart, m2["type"] if m2 else "notype", tipi_conosciuti.get(target1, ["notype"])))
            prec1,prec2=n1,n2

        # metriche di confronto utili
        features["p1_mean_move_effectiveness"]=np.mean(p1_effs) if p1_effs else 1.0   #fa la media dei moltiplicatori di danno
        features["p2_mean_move_effectiveness"] = np.mean(p2_effs) if p2_effs else 1.0 #stessa cosa per il giocatore 2
        features["p1_advantage_turn_ratio"]=np.mean([a > c for a, c in zip(p1_effs, p2_effs)]) if p1_effs else 0.5  #percentuale di chi è in vantaggio in ogni turno (es. 0.75 il giocatore 1 è in vantaggio al 75% mentre se non ho moltiplicatori da confrontare allora mettiamo il valore neutro di parità ovvero 0.5)
        features["mean_net_type_advantage"]=np.mean([a - c for a, c in zip(p1_effs, p2_effs)]) if p1_effs else 0.0  #calcola in ogni turno la diffrenza tra i moltiplicatori e fa la media (se numero positivo domina il giocatore 1)

        # info raccolte nei turni
        # Ppokemon che non si vedono nei primi 30 turni si assume siano integri
        unseen1=max(0, 6-len(p1_hp))
        unseen2=max(0, 6-len(p2_hp))
        p1_vivi=sum(hp>0 for hp in p1_hp.values())+unseen1
        p2_vivi=sum(hp>0 for hp in p2_hp.values())+unseen2
        p1_total_hp=(sum(p1_hp.values())+unseen1)*100
        p2_total_hp=(sum(p2_hp.values())+unseen2)*100
        features["p1_final_alive"], features["p2_final_alive"]=p1_vivi,p2_vivi
        features["final_alive_diff"]=p1_vivi-p2_vivi
        features["p1_final_total_hp"], features["p2_final_total_hp"]=round(p1_total_hp, 2), round(p2_total_hp, 2)
        features["final_total_hp_diff"]=round(p1_total_hp-p2_total_hp, 2)

        features["battle_id"]=b.get("battle_id")   #salva l'id della battaglia
        if "player_won" in b:   #controllo che io stia analizzando i dati di training
            features["player_won"]=int(b["player_won"])
        righe.append(features)   #aggiungo tutte queste features al dizionario creato
    return pd.DataFrame(righe).fillna(0)   #converto il tutto in una tabella con pandas (in caso di errori mette 0)


#blocco 5
STATUS=["slp", "par", "frz", "tox", "psn", "brn"]
EFFECTS=["substitute", "confusion", "leechseed", "trapped", "protect", "perish"]
HEAL_MOVES={"recover", "softboiled", "milkdrink", "rest", "moonlight", "morningsun", "synthesis"}
SETUP_MOVES={"swordsdance", "curse", "growth", "agility", "amnesia", "bellydrum", "meditate",
               "sharpen", "defensecurl", "withdraw", "harden", "barrier", "acidarmor", "doubleteam"}
STATUS_MOVES={"toxic", "thunderwave", "sleeppowder", "spore", "hypnosis", "lovelykiss", "sing",
                "stunspore", "glare", "poisonpowder", "confuseray", "swagger"}

def forza(n):
    """
    Calcola la somma  'n'.
    Legge i dati dal dizionario globale 'species_stats'.
    """
    nome=n.lower().strip()
    return sum(species_stats[nome].values()) if nome in species_stats else 425  ###CAPIRE COME FARE SE NON HO LISTA CON TUTTE LE STATISTICHE

def summary(last, team_names):
    """Stato dei due team all'ultimo turno"""
    out={}
    for s in ("p1", "p2"):
        visti=last[s]
        fnt=[n for n, s in visti.items() if s["status"]=="fnt" or s["hp"]==0]
        vivi=[n for n in visti if n not in fnt]
        n_unseen=max(0, 6-len(visti))
        hp_seen=sum(visti[n]["hp"] for n in vivi)
        out[f"{s}_n_visti"]=len(visti)
        out[f"{s}_n_fnt"]=len(fnt)
        out[f"{s}_hp_est"]=hp_seen+n_unseen  #stesse cose viste prima
        for stato in STATUS:
            out[f"{s}_end_{stato}"]=sum(visti[n]["status"]==stato for n in vivi)  #conta pokemon con quello stato per i due giocatori nei 30 turni
        out[f"{s}_nstato_alterato"]=sum(visti[n]["status"] in STATUS for n in vivi)   #conteggio generale degli stati alterati

        # calcolo la forza di una squadra (tenendo conto gli hp e la forza dei pokemon sopravvissuti)
        strength=sum(visti[n]["hp"]*forza(n) for n in vivi)
        if s=="p1":
            strength+=sum(forza(n) for n in team_names if n not in visti)
        else:
            strength+=n_unseen*425  #Valore medio se non ho statistiche del pokemon
        out[f"{s}_strength"]=strength
        out[f"{s}_mean_hp_alive"]=(hp_seen+n_unseen)/max(1, 6-len(fnt))
    out["strength_diff"]=out["p1_strength"]-out["p2_strength"]
    out["stato_diff"]=out["p1_nstato_alterato"]-out["p2_nstato_alterato"]
    out["slp_diff"]=out["p1_end_slp"]-out["p2_end_slp"]  #due stati molto significativi
    out["par_diff"]=out["p1_end_par"]-out["p2_end_par"]
    return out


def features_estese(battles, type_chart, species_types, species_stats):
    """Extra features ricavate da quelle di base"""
    mean_stats={s: np.mean([v[s] for v in species_stats.values()]) for s in STATISTICHE}
    righe=[]
    for b in battles:
        team=b["p1_team_details"]
        lead1,lead2=team[0],b["p2_lead_details"]
        team_names=[p["name"].lower().strip() for p in team]
        tipi_conosciuti=species_types.copy()
        for p in team:
           nome=p["name"].lower().strip()
           tipi=p.get("types", ["notype"])
           tipi_conosciuti[nome]=tipi
        known_stats=species_stats.copy()
        for p in team:
            nome=p["name"].lower().strip()
            statistiche_pokemon={}
            for s in STATISTICHE:
                statistiche_pokemon[s]=p[s]
            known_stats[nome]=statistiche_pokemon
        types_of=lambda n: [t for t in tipi_conosciuti.get(n, ["notype"]) if t != "notype"]
        stats_of=lambda n: known_stats.get(n, mean_stats)
        f={"battle_id": b["battle_id"]}

        #feature su statistiche
        team_df=pd.DataFrame(team)
        for s in STATISTICHE:    #altre statistiche utili come massimo e media di ogni valore
            f[f"p1_team_mean_{s}"]=team_df[s].mean()
            f[f"p1_team_max_{s}"]=team_df[s].max()
            f[f"p2_lead_{s}"]=lead2[s]
        f["p1_team_mean_potenza"]=team_df[STATISTICHE].sum(axis=1).mean()
        f["lead_diff_potenza"]=sum(lead1[s] for s in STATISTICHE)-sum(lead2[s] for s in STATISTICHE)
        f["lead_diff_velocita"]=lead1["base_spe"]-lead2["base_spe"]
        f["team_faster_than_p2lead"]=int((team_df["base_spe"]>lead2["base_spe"]).sum())
        lt, p2t=types_of(lead1["name"].lower().strip()), [t for t in lead2["types"] if t != "notype"]  #tipi dei due leader
        f["lead_off_mult"]=miglior_moltiplicatore(type_chart, lt, p2t)
        f["lead_def_mult"]=miglior_moltiplicatore(type_chart, p2t, lt)
        f["team_n_se_vs_p2lead"]=sum(miglior_moltiplicatore(type_chart, types_of(n), p2t)>=2 for n in team_names)  #contatore superefficaci (numero di mosse superefficaci contro il pokemon iniziale del giocatore 2)
        f["team_n_weak_to_p2lead"] = sum(miglior_moltiplicatore(type_chart, p2t, types_of(n))>= 2 for n in team_names) #stessa cosa ma mostra quanto è debole il player 1
        f["team_n_types"]=len({t for n in team_names for t in types_of(n)})  #numero di tipi diversi per capire la varietà e la facilità di contrasto
        f["p1_n_leftovers"] = sum(p.get("item")=="leftovers" for p in team)  #informazione tattica: più ci sono e più partite saranno lunghe e sarà più difficile perdere hp per il player1

        #info su sequenza dei turni: trasforma partita in una serie di numeri significativi
        last={"p1":{}, "p2":{}}  #memoria durante la partita: ricorda ogni singolo pokemon sceso in campo e gli hp
        prev_active, prev_hp = {"p1": None, "p2": None}, {"p1": None, "p2": None}  #chi era in campo il turno prima e hp
        cont=defaultdict(float) #dizionario con dei contatori utili
        first_faint={"p1":31, "p2":31} #istante in cui avviene la prima kill (posta di default a 31)
        faint_turn={"p1": {}, "p2": {}} #turno in cui viene ucciso un pokemon
        hp_curve,snapshots=[],{}  #hp curve registra la differenza di hp per ogni turno e snapshots salva lo stato della partita in precisi istanti

        for t in b.get("battle_timeline", []):
            turno=t["turn"]
            # s=giocatore attuale ("p1" o "p2"), opp=avversario
            for s, opp in (("p1", "p2"), ("p2", "p1")): 
                stato_pok=t[f"{s}_pokemon_state"]
                nome=stato_pok["name"].lower().strip()
                hp=stato_pok["hp_pct"]
                #Controllo degli switch
                if prev_active[s] is not None and nome!=prev_active[s]:    #pokemon diverso
                    # Se il Pokemon precedente non era andato K.O. è uno switch voluto
                    if last[s].get(prev_active[s], {}).get("status") != "fnt":   #non K.O.
                        cont[f"{s}_switches"]+=1   #aggiorno contatore                   
                elif prev_hp[s] is not None:                            
                    delta=hp-prev_hp[s]   #differenza di hp
                    if delta<0:
                        #Ha perso vita: contiamo i danni subiti e quelli inflitti dall'avversario
                        cont[f"{s}_danno_subito"]-=delta
                        cont[f"{opp}_danno_fatto"]-=delta
                    elif delta>0:
                        # Ha guadagnato vita
                        cont[f"{s}_cura"]+=delta    
                # Aggiorniamo la memoria 'last' con lo stato attuale del Pokemon
                last[s][nome]={"hp": hp, "status": stato_pok["status"], "boosts": stato_pok.get("boosts", {})}
                #Controllo K.O. (Faint)
                if stato_pok["status"]=="fnt":
                    # Registra il primo K.O. subito dal player se non l'avevo già fatto
                    first_faint[s]=min(first_faint[s], turno)
                    # Registra in che turno
                    faint_turn[s].setdefault(nome, turno)  
                #Controllo problemi di stato e condizioni
                if stato_pok["status"] in STATUS:
                    cont[f"{s}_stati_alterati"]+=1
                for effetto in stato_pok.get("effects", []):
                    if effetto in EFFECTS:
                        cont[f"{s}_eff_{effetto}"]+=1
                # Salviamo il Pokemon corrente e i suoi HP per il turno successivo
                prev_active[s], prev_hp[s]=nome,hp
                #Anali=si mosse usate
                mossa=t.get(f"{s}_move_details")
                if mossa is None:
                    # Se non esegue mosse ha switchato o non ha attaccato
                    cont[f"{s}_no_move"]+=1
                    continue
                #Tipo di mossa
                categoria=mossa["category"]
                cont[f"{s}_m_{categoria}"]+=1
                cont[f"{s}_m_cura"]+=mossa["name"] in HEAL_MOVES
                cont[f"{s}_setup_moves"]+=mossa["name"] in SETUP_MOVES
                cont[f"{s}_status_moves"]+=mossa["name"] in STATUS_MOVES
                cont[f"{s}_priority_moves"]+=(mossa.get("priority", 0)>0)
                # Se mossa offensiva (Fisica o Speciale)
                if categoria!="status" and (mossa.get("base_power", 0)>0):
                    bersaglio=t[f"{opp}_pokemon_state"]["name"].lower().strip()
                    tipi_bersaglio=types_of(bersaglio)
                    moltiplicatore=efficacia(type_chart, mossa["type"], tipi_bersaglio)
                    cont[f"{s}_bp_sum"]+=mossa["base_power"]
                    cont[f"{s}_se_hits"]+=moltiplicatore>=2 # mosse superefficaci
                    cont[f"{s}_nse_hits"]+=0<moltiplicatore<1 #poco efficace
                    cont[f"{s}_immune_hits"]+=moltiplicatore==0  # Immune
                    cont[f"{s}_same_type_moves"]+=mossa["type"] in types_of(nome) # Attacco dello stesso tipo
            snap=summary(last, team_names)
            hp_curve.append(snap["p1_hp_est"]-snap["p2_hp_est"])
            if turno in (10,20):   #cattura situazione in particolari turni
                snapshots[turno]=snap

        final=summary(last, team_names)
        f.update(final)
        f["hp_diff_t30"]=final["p1_hp_est"]-final["p2_hp_est"]  #differenza hp al turno 30
        f["faint_diff_t30"]=final["p2_n_fnt"]-final["p1_n_fnt"]  #differenza di pokemon sconfitti turno 30
        for turn in (10, 20):
            snap=snapshots.get(turn, final)
            f[f"hp_diff_t{turn}"]=snap["p1_hp_est"]-snap["p2_hp_est"]  #differenza hp a turno 10 e 20
            f[f"faint_diff_t{turn}"]=snap["p2_n_fnt"]-snap["p1_n_fnt"]
        f["hp_diff_trend"]=f["hp_diff_t30"]-f["hp_diff_t10"]
        f["hp_diff_mean"]=float(np.mean(hp_curve))  #media differenza hp ad ogni turno
        if hp_curve:
            f["diff_hp_ultimi_5_turni"]=hp_curve[-1]-hp_curve[max(0, len(hp_curve) - 6)]   # 1. Trend degli HP negli ultimi 5 turni
        else:
            f["diff_hp_ultimi_5_turni"]=0
        # 2. Analisi del primo K.O.
        f["p1_primo_ko"]=first_faint["p1"]
        f["p2_primo_ko"]=first_faint["p2"]
        f["diff_primo_ko"]=first_faint["p1"]-first_faint["p2"]  #conta i turni di distanza tra i primi K.O.
        for s in ("p1", "p2"):  #per ogni player
            # Conta i K.O. avvenuti dopo il turno 20
            f[f"{s}_ko_ultimi_10"]=sum(turno_ko>20 for turno_ko in faint_turn[s].values())
            # Conta i potenziamenti dei Pokémon non K.O.
            f[f"{s}_boost"]=sum(sum(v.get("boosts", {}).values()) for v in last[s].values() if v.get("status") != "fnt")                                     
        f["diff_ko_ultimi_10"]=f["p2_ko_ultimi_10"]-f["p1_ko_ultimi_10"]
        f["diff_boost"]=f["p1_boost"]-f["p2_boost"]
        chiavi=["switches", "danno_subito", "danno_fatto", "cura", "stati_alterati", "no_move", "m_physical","m_special", "m_status", "m_cura", "setup_moves", "status_moves", "priority_moves", "se_hits", "nse_hits", "immune_hits", "same_type_moves"]+[f"eff_{e}" for e in EFFECTS]
        for s in ("p1", "p2"):
            for k in chiavi:
                f[f"{s}_{k}"]=cont.get(f"{s}_{k}", 0.0)
            # Calcolo della potenza media degli attacchi usati
            attacchi=cont.get(f"{s}_m_physical", 0)+cont.get(f"{s}_m_special", 0)  #definiti quando ho creato categoria
            f[f"{s}_mean_potenza"]=cont.get(f"{s}_bp_sum", 0)/attacchi if attacchi>0 else 0.0
        # 5. Calcolo delle differenze tra i due giocatori
        chiavi_per_differenze=["switches", "danno_fatto", "cura", "stati_alterati", "se_hits", "status_moves","m_cura", "setup_moves", "m_status"]
        for k in chiavi_per_differenze:
            f[f"diff_{k}"]=f.get(f"p1_{k}", 0)-f.get(f"p2_{k}", 0)
        #Pokémon attivi all'ultimo turno
        ultimo_turno=b["battle_timeline"][-1]
        attivo1=ultimo_turno["p1_pokemon_state"]
        attivo2=ultimo_turno["p2_pokemon_state"]
        nome1=attivo1["name"].lower().strip()
        nome2=attivo2["name"].lower().strip()
        f["act_hp_diff"]=attivo1.get("hp_pct", 0.0)-attivo2.get("hp_pct", 0.0)
        f["act_boost_diff"]=sum(attivo1.get("boosts", {}).values())-sum(attivo2.get("boosts", {}).values())
        f["act_spe_diff"]=stats_of(nome1).get("base_spe", 0)-stats_of(nome2).get("base_spe", 0)  #velocità
        adv_p1=miglior_moltiplicatore(type_chart, types_of(nome1), types_of(nome2))   #vantaggio tipo tra ultimi due pokemon
        adv_p2=miglior_moltiplicatore(type_chart, types_of(nome2), types_of(nome1))
        f["p1_mult"]=adv_p1
        f["p2_mult"]=adv_p2
        # Verifica problemi di stato dell'ultimo pokemon (paralisi, veleno, ecc.)
        f["act_p1_bad_status"]=int(attivo1.get("status") in STATUS)
        f["act_p2_bad_status"]=int(attivo2.get("status") in STATUS)
        #HP finali di ogni Pokémon
        #Per il Giocatore 1 (di cui conosciamo tutta la squadra in anticipo)
        for n in team_names:
            st=last["p1"].get(n)
            # Se 'st' è None, non è mai sceso in campo, quindi ha ancora il 100% di vita.
            # Se lo status è "fnt" (K.O.), la sua vita è 0, altrimenti prendo gli HP correnti.
            f[f"sp1_{n}"]=1 if st is None else (0 if st.get("status")=="fnt" else st.get("hp", 0))
        # Per il Giocatore 2 (conosco solo i Pokémon effettivamente scesi in campo)
        for n, st in last["p2"].items():
            f[f"sp2_{n}"]=0 if st.get("status")=="fnt" else st.get("hp", 0)
            f[f"sp2seen_{n}"]=1  # Feature che considera quel pokemon come visto in campo
        # Aggiungiamo tutte queste feature alle colonne che già avevamo
        righe.append(f)
    return pd.DataFrame(righe)


def costruisci_features(battles, type_chart, species_types, species_stats, colonne_specie=None, supporto_minimo=30):
    """
    Fonde le feature di base e quelle estese. 
    Per evitare l'overfitting cancello le colonne dei pokemon che compaiono in meno di 30 partite per evitare conclusioni errate.
    """
    #Generazione dei due set di feature
    df_base=features_di_base(battles, type_chart, species_types)
    df_estese=features_estese(battles, type_chart, species_types, species_stats)
    # Troviamo le colonne che iniziano per sp1_, sp2_ o sp2seen_
    colonne_sp=[c for c in df_estese.columns if c.startswith(("sp1_", "sp2_", "sp2seen_"))]
    # Se 'colonne_specie' è None, significa che stiamo addestrando il modello (fase di Train)
    if colonne_specie is None:
        # Teniamo SOLO i Pokémon che compaiono in almeno 'supporto_minimo' di partite (es. 30).
        # Evita overfitting
        colonne_specie=[c for c in colonne_sp if df_estese[c].notna().sum()>=supporto_minimo]
    # Rimuoviamo dal DataFrame le colonne dei Pokémon troppo rari (o che non sono nel Test set)
    df_estese=df_estese.drop(columns=[c for c in colonne_sp if c not in colonne_specie])
    # I Pokémon non presenti in una determinata partita avranno "NaN" (Not a Number). Li portiamo a 0
    df_estese=df_estese.reindex(columns=[c for c in df_estese.columns if c not in colonne_specie]+colonne_specie)
    df_estese[colonne_specie]=df_estese[colonne_specie].fillna(0.0)
    #Unione dei due DataFrame
    # Rimuoviamo da 'df_estese' eventuali feature che erano già state calcolate in 'df_base'
    df_estese=df_estese.drop(columns=[c for c in df_estese.columns if c in df_base.columns and c!="battle_id"])
    # Uniamo i due DataFrame usando "battle_id" come chiave di aggancio. 
    # Qualsiasi dato mancante residuo viene convertito a 0.
    df_finale=df_base.merge(df_estese, on="battle_id", how="left").fillna(0)
    # NUOVO: feature avanzate (HP pesati per lo stato, spikes, momentum, matchup dei sopravvissuti...)
    df_avanzate=features_avanzate(battles, type_chart, species_types, species_stats)
    df_avanzate=df_avanzate.drop(columns=[c for c in df_avanzate.columns if c in df_finale.columns and c!="battle_id"])
    df_finale=df_finale.merge(df_avanzate, on="battle_id", how="left").fillna(0)
    #Target (Etichetta finale)
    # Estraiamo la colonna "player_won" e la rimettiamo in ultimissima posizione, per avere il DataFrame ordinato: ID -> Features -> Target.
    target=df_finale.pop("player_won") if "player_won" in df_finale else None
    if target is not None:
        df_finale["player_won"]=target
    # Restituisce il dataset pronto e la lista dei Pokémon "frequenti" salvata
    return df_finale, colonne_specie


#blocco 6
#Quanto vale un pokemon in base al suo stato (1.0 = vale tutto, 0.1 = quasi niente)
#Infatti, per esempio: un pokemon al 100% ma paralizzato vale come uno sano al 75%
PESO_STATO={"nostatus": 1.0, "par": 0.75, "psn": 0.9, "brn": 0.8, "tox": 0.8, "slp": 0.35, "frz": 0.1}
PESO_SONNO_REST=0.7
BOOM_MOVES={"explosion", "selfdestruct"}
SLEEP_MOVES={"spore", "sleeppowder", "hypnosis", "lovelykiss", "sing"}
PHAZE_MOVES={"roar", "whirlwind"}


def features_avanzate(battles, type_chart, species_types, species_stats):
    """
    Feature nuove:
    - HP "effettivi" pesati per lo stato alterato
    - Mosse importanti del Gen 2, contate per ogni giocatore:
        * Spikes: mette delle punte sul campo avversario, ogni pokemon che entra perde HP
        * Rapid Spin: toglie le Spikes dal proprio campo
        * Explosion / Selfdestruct: il pokemon si sacrifica per fare un danno enorme
        * Rest: il pokemon si cura tutto ma si addormenta per 2 turni
        * Sleep Talk: permette di attaccare anche mentre si dorme
        * Roar / Whirlwind: costringono l'avversario a cambiare pokemon
    - Andamento della partita:
        * pendenza HP: la partita sta girando a favore di P1 o di P2?
        * % turni in vantaggio: per quanto tempo P1 ha avuto più HP
        * ultimi 10 turni: chi ha guadagnato o perso terreno di recente
        * ultimi K.O.: chi ha perso pokemon di recente
    - Matchup dei pokemon ANCORA VIVI (non solo dei lead) e velocità dei sopravvissuti
    - Boost offensivi ancora attivi (Curse / Belly Drum / Swords Dance)
    """

    #Valori medi dei pokemon: li uso quando non conosco le statistiche di un pokemon
    mean_bst=np.mean([sum(v.values()) for v in species_stats.values()])   #forza media 
    mean_spe=np.mean([v["base_spe"] for v in species_stats.values()])     #velocità media
    righe=[]
    for b in battles:
        team=b["p1_team_details"]
        team_names=[p["name"].lower().strip() for p in team]
        tipi=dict(species_types)
        stats=dict(species_stats)
        # Aggiungo tipi e statistiche dei pokemon di P1 che conosco sicuramente
        for p in team:
            n=p["name"].lower().strip()
            tipi[n]=p.get("types", ["notype"])
            stats[n]={s: p[s] for s in STATISTICHE}
        types_of=lambda n: [t for t in tipi.get(n, ["notype"]) if t != "notype"]
        bst=lambda n: sum(stats[n].values()) if n in stats else mean_bst
        spe=lambda n: stats[n]["base_spe"] if n in stats else mean_spe
        timeline=b.get("battle_timeline", [])
        lead2=b["p2_lead_details"]["name"].lower().strip()

        f={"battle_id": b["battle_id"]}
        f["n_turni"]=len(timeline)   #se < 30 la battaglia potrebbe essere già finita

        last={"p1": {}, "p2": {}}
        dorme_da_rest={"p1": set(), "p2": set()}
        spikes_sul_lato={"p1": 0, "p2": 0}   #spikes presenti sul lato di s
        cont=defaultdict(float)
        curva_hp, curva_vivi=[], []
        ko_turni={"p1": [], "p2": []}

        for t in timeline:   #scorro la partita turno per turno
            turno=t["turn"]
            #guardo prima P1 e poi P2 (s = giocatore che sto guardando, opp = il suo avversario)
            for s, opp in (("p1", "p2"), ("p2", "p1")):
                st=t[f"{s}_pokemon_state"]   # tato del pokemon in campo
                n=st["name"].lower().strip()   #nome del pokemon
                prima=last[s].get(n)   #come stava questo pokemon l'ultima volta che l'ho visto
                if st["status"]=="fnt" and (prima is None or prima["status"] != "fnt"):
                    ko_turni[s].append(turno)
                last[s][n]={"hp": 0.0 if st["status"]=="fnt" else st["hp_pct"], "status": st["status"], "boosts": st.get("boosts", {})}
                #Se non dorme più, lo tolgo dalla lista di chi dorme per Rest
                if st["status"] != "slp":
                    dorme_da_rest[s].discard(n)
                mossa=t.get(f"{s}_move_details")
                if mossa is None:   #se nessuna mossa (quindi ha cambiato pokemon): passo oltre
                    continue
                nm=mossa["name"]
                if nm=="rest":   #se usa Rest: si addormenta "volontariamente"
                    dorme_da_rest[s].add(n)
                    cont[f"{s}_rest"]+=1
                if nm=="spikes":   #se mette le Spikes sul campo dell'avversario
                    spikes_sul_lato[opp]=1
                    cont[f"{s}_spikes"]+=1
                if nm=="rapidspin":   #se toglie le Spikes dal proprio campo
                    spikes_sul_lato[s]=0
                    cont[f"{s}_spin"]+=1
                #conto le altre mosse importanti (True vale 1, False vale 0)
                cont[f"{s}_boom"]+=nm in BOOM_MOVES
                cont[f"{s}_sleep_moves"]+=nm in SLEEP_MOVES
                cont[f"{s}_phaze"]+=nm in PHAZE_MOVES
                cont[f"{s}_sleeptalk"]+=nm=="sleeptalk"
                cont[f"{s}_acc_bassa"]+=(mossa.get("accuracy", 1.0) or 1.0)<0.9

            #Fine turno: calcolo la situazione della partita
            #HP totali di ogni giocatore (i pokemon mai visti contano come 100% = 1)
            hp1=sum(v["hp"] for v in last["p1"].values())+(6-len(last["p1"]))
            hp2=sum(v["hp"] for v in last["p2"].values())+max(0, 6-len(last["p2"]))
            curva_hp.append(hp1-hp2)   #verifico la differenza di HP in questo turno (se positiva allora P1 avanti)
            curva_vivi.append(len(ko_turni["p2"])-len(ko_turni["p1"]))   #guardo la differenza di K.O. fatti (se positiva allora P1 avanti)

        # --- HP effettivi pesati per lo stato ---
        #restituisce solo i pokemon e ancora vivi (non K.O. e con HP > 0)
        def vivi(s):
            return {n: v for n, v in last[s].items() if v["status"] != "fnt" and v["hp"]>0}
        vivi1, vivi2=vivi("p1"), vivi("p2")
        unseen1=[n for n in team_names if n not in last["p1"]]   #pokemon di P1 mai entrati in campo
        unseen2=max(0, 6-len(last["p2"]))   #quanti pokemon di P2 non si sono mai visti
        for s, vv, n_unseen in (("p1", vivi1, len(unseen1)), ("p2", vivi2, unseen2)):
            hp_eff, utili, slp_avv, low, full, max_off, boost_pos=0.0, 0, 0, 0, n_unseen, 0, 0
            for n, v in vv.items():
                stato=v["status"]
                #Scelgo il peso: il sonno da Rest pesa meno di quello causato dall'avversario
                if stato=="slp" and n in dorme_da_rest[s]:
                    peso=PESO_SONNO_REST
                else:
                    peso=PESO_STATO.get(stato, 1.0)
                    slp_avv+=stato=="slp"   #conto chi dorme per colpa dell'avversario
                hp_eff+=v["hp"]*peso   #HP "veri" = HP × peso dello stato
                #Un pokemon non congelato e non addormentato dall'avversario e' un pokemon utile
                utili+=stato not in ("frz",) and not (stato=="slp" and n not in dorme_da_rest[s])
                low+=v["hp"]<0.3
                full+=v["hp"]>=0.99
                bo=v.get("boosts", {})
                max_off=max(max_off, bo.get("atk", 0), bo.get("spa", 0))
                boost_pos+=sum(x for x in bo.values() if x>0)
            f[f"{s}_hp_eff"]=hp_eff+n_unseen
            f[f"{s}_vivi_utili"]=utili+n_unseen
            f[f"{s}_slp_non_rest"]=slp_avv
            f[f"{s}_frz"]=sum(v["status"]=="frz" for v in vv.values())
            f[f"{s}_low_hp"]=low
            f[f"{s}_full_hp"]=full
            f[f"{s}_max_boost_off"]=max_off
            f[f"{s}_boost_pos"]=boost_pos
        for k in ("hp_eff", "vivi_utili", "slp_non_rest", "frz", "low_hp", "full_hp", "max_boost_off", "boost_pos"):
            f[f"diff_{k}"]=f[f"p1_{k}"]-f[f"p2_{k}"]
        #Quota di P1 sul totale (0.5 = parità, 1 = tutto di P1)
        tot=f["p1_hp_eff"]+f["p2_hp_eff"]
        f["quota_hp_eff_p1"]=f["p1_hp_eff"]/tot if tot>0 else 0.5
        tot_v=f["p1_vivi_utili"]+f["p2_vivi_utili"]
        f["quota_vivi_utili_p1"]=f["p1_vivi_utili"]/tot_v if tot_v>0 else 0.5

        # --- mosse chiave del Gen 2 ---
        #Spikes ancora presenti a fine partita (1 = sì, 0 = no)
        f["spikes_su_p1"]=spikes_sul_lato["p1"]   #Spikes sul campo di P1
        f["spikes_su_p2"]=spikes_sul_lato["p2"]   #Spikes sul campo di P2
        f["diff_spikes_attive"]=spikes_sul_lato["p2"]-spikes_sul_lato["p1"]     #se positivo allora vantaggio per P1
        #Quante volte ogni giocatore ha usato le mosse importanti
        for k in ("rest", "spikes", "spin", "boom", "sleep_moves", "phaze", "sleeptalk", "acc_bassa"):
            f[f"p1_{k}"]=cont.get(f"p1_{k}", 0.0)
            f[f"p2_{k}"]=cont.get(f"p2_{k}", 0.0)
            f[f"diff_{k}"]=f[f"p1_{k}"]-f[f"p2_{k}"]

        
        # --- andamento (momentum) ---
        if curva_hp:
            c=np.array(curva_hp, dtype=float)
            f["frac_turni_p1_avanti"]=float(np.mean(c>0))   # % di turni in cui P1 aveva più HP
            f["pendenza_hp"]=float(np.polyfit(np.arange(len(c)), c, 1)[0]) if len(c)>1 else 0.0
            f["hp_diff_ultimi_10"]=c[-1]-c[max(0, len(c)-11)]   #quanto è cambiata la differenza negli ultimi 10 turni
            f["hp_diff_min"], f["hp_diff_max"]=c.min(), c.max()   #momento peggiore e migliore per P1
            f["vivi_diff_ultimi_10"]=curva_vivi[-1]-curva_vivi[max(0, len(curva_vivi)-11)]
        else:
            f.update({"frac_turni_p1_avanti": 0.5, "pendenza_hp": 0.0, "hp_diff_ultimi_10": 0.0,
                      "hp_diff_min": 0.0, "hp_diff_max": 0.0, "vivi_diff_ultimi_10": 0})
        ultimo=timeline[-1]["turn"] if timeline else 30
        for s in ("p1", "p2"):
            f[f"{s}_ko_ultimi_5"]=sum(x>ultimo-5 for x in ko_turni[s])   #K.O. subiti negli ultimi 5 turni
            #Da quanti turni non perde un pokemon
            f[f"{s}_turni_da_ultimo_ko"]=ultimo-max(ko_turni[s]) if ko_turni[s] else 31
        f["diff_ko_ultimi_5"]=f["p2_ko_ultimi_5"]-f["p1_ko_ultimi_5"]
        # +1 = P2 ha perso tutti e 6 i pokemon, -1 = li ha persi P1, 0 = partita ancora aperta
        f["battaglia_finita"]=int(len(ko_turni["p2"])>=6)-int(len(ko_turni["p1"])>=6)

        # --- matchup dei sopravvissuti ---
        vivi1_nomi=list(vivi1)+unseen1   #pokemon vivi di P1
        vivi2_nomi=list(vivi2) if last["p2"] else [lead2]   #pokemon vivi visti di P2
        # P1: valori calcolati sui suoi pokemon vivi
        f["p1_spe_max_vivi"]=max((spe(n) for n in vivi1_nomi), default=0)   # il più veloce
        f["p1_bst_vivi_mean"]=float(np.mean([bst(n) for n in vivi1_nomi])) if vivi1_nomi else 0.0   # forza media
        if vivi2_nomi:
            # P2: stessi calcoli sui suoi pokemon vivi che conosco
            f["p2_spe_max_vivi"]=max(spe(n) for n in vivi2_nomi)
            f["p2_bst_vivi_mean"]=float(np.mean([bst(n) for n in vivi2_nomi]))
        elif unseen2>0:
            #se tutti i pokemon visti di P2 sono K.O. ma ne ha ancora di nascosti: uso valori medi
            f["p2_spe_max_vivi"]=mean_spe
            f["p2_bst_vivi_mean"]=mean_bst
        else:
            #se P2 non ha più pokemon
            f["p2_spe_max_vivi"], f["p2_bst_vivi_mean"]=0, 0.0
        if vivi1_nomi and vivi2_nomi:
            #per ogni pokemon di P2 cerco il colpo migliore che P1 può tirargli, poi faccio la media
            #(2 = P1 ha un colpo superefficace per tutti, 0.5 = P1 fa poco danno a tutti)
            f["p1_copertura"]=np.mean([max(miglior_moltiplicatore(type_chart, types_of(a), types_of(d)) for a in vivi1_nomi) for d in vivi2_nomi])
            #Lo stesso al contrario
            f["p2_copertura"]=np.mean([max(miglior_moltiplicatore(type_chart, types_of(a), types_of(d)) for a in vivi2_nomi) for d in vivi1_nomi])
        else:
            f["p1_copertura"], f["p2_copertura"]=1.0, 1.0
        f["diff_copertura"]=f["p1_copertura"]-f["p2_copertura"]
        f["diff_spe_max_vivi"]=f["p1_spe_max_vivi"]-f["p2_spe_max_vivi"]
        f["diff_bst_vivi"]=f["p1_bst_vivi_mean"]-f["p2_bst_vivi_mean"]
        if timeline:
            att2=timeline[-1]["p2_pokemon_state"]["name"].lower().strip()
            f["p1_vivi_piu_veloci_di_att2"]=sum(spe(n)>spe(att2) for n in vivi1_nomi)
        else:
            f["p1_vivi_piu_veloci_di_att2"]=0
        f["p2_n_rivelati"]=len(last["p2"])
        righe.append(f)
    return pd.DataFrame(righe).fillna(0)


#blocco 7
train=carica_match(data_file("train", ".jsonl"))
test=carica_match(data_file("test", ".jsonl"))
type_chart=dizionario_moltiplicatori(data_file("type_chart_gen2", ".csv"))
print(f"{len(train)} battaglie di train, {len(test)} battaglie di test, {len(type_chart)} combinazioni di tipo caricate")

#Esplorazione della prima battaglia nel dataset
prima_battaglia=train[0]
print("Info di una battaglia:", sorted(prima_battaglia.keys()))
print("Squadra del Giocatore 1:", [p["name"] for p in prima_battaglia["p1_team_details"]])
print("Pokemon iniziale giocatore 2:", prima_battaglia["p2_lead_details"]["name"])
print("Primo turno:", prima_battaglia["battle_timeline"][0])
print("Il Giocatore 1 ha vinto?", bool(prima_battaglia.get("player_won", False)))


#blocco 8

#Regressione Logistica
def crea_regressione_logistica():
    """
    Crea laRegressione Logistica. 
    Scalare prima i dati (con StandardScaler)
    """
    # C=0.03 con penalità L1: la penalità L1 azzera i coefficienti delle feature poco utili
    return LogisticRegression(C=0.03, penalty="l1", solver="liblinear", max_iter=5000, random_state=SEED)

#Modello Random Forest
def crea_random_forest():
    """
    Crea la Random Forest. Gli alberi decisionali non hanno bisogno di alcuno scaling sui dati.
    """
    return RandomForestClassifier(
        n_estimators=500,    # Numero di alberi nella foresta
        max_depth=12,        # Profondità massima degli alberi (evita overfitting)
        random_state=SEED, 
        n_jobs=-1            # Usa tutti i processori del computer
    )

def valuta_modello_cv(X, y, modello, n_divisioni=5, scala_dati=False):
    """
    Esegue la Cross-Validation a 5 fold gestendo lo scaling (se richiesto). Restituisce le probabilità stimatee i punteggi finali delle metriche.
    """
    y=np.asarray(y)
    probabilita_stimate=np.zeros(len(y))
    divisione=StratifiedKFold(n_divisioni, shuffle=True, random_state=SEED)
    # Ciclo sui fold della Cross-Validation
    for train_idx, val_idx in divisione.split(X, y):
        #Separazione dei dati del fold corrente in Training e Validazione
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr = y[train_idx]
        #Se richiesto, scaliamo i dati
        if scala_dati:
            scaler=StandardScaler()
            X_tr=scaler.fit_transform(X_tr)
            X_va=scaler.transform(X_va)
        #Addestriamo il modello sul training set
        modello.fit(X_tr, y_tr)
        # 4. Salviamo la probabilità predetta per la parte di validazione (fuori campione)
        probabilita_stimate[val_idx]=modello.predict_proba(X_va)[:, 1]
    #Calcoliamo le metriche finali sulle predizioni
    punteggi={
        "accuratezza": accuracy_score(y, probabilita_stimate> 0.5),
        "logloss": log_loss(y, probabilita_stimate),
        "auc": roc_auc_score(y, probabilita_stimate)}
    return probabilita_stimate, punteggi
species_types=fit_species_types(train)
species_stats=fit_species_stats(train)
train_df,species_columns=costruisci_features(train, type_chart, species_types, species_stats)
test_df, _=costruisci_features(test, type_chart, species_types, species_stats, colonne_specie=species_columns)
all_features=[c for c in train_df.columns if c not in ("battle_id", "player_won")]
X,y=train_df[all_features], train_df["player_won"].to_numpy()
risultati = {}
modelli_da_testare={"Regressione Logistica": (crea_regressione_logistica(), True), "Random Forest": (crea_random_forest(), False)}
#Ciclo di addestramento e valutazione per ogni modello
for nome_modello, (modello, scala_dati) in modelli_da_testare.items():
    print(f"In corso: {nome_modello}->")
    prob_stimate, metriche=valuta_modello_cv(X, y, modello, n_divisioni=5, scala_dati=scala_dati)
    risultati[nome_modello]=metriche
    print(f"-> Risultati {nome_modello}: {risultati[nome_modello]}\n")
#Confronto
print("RIEPILOGO PRESTAZIONI")
for nome, metriche in risultati.items():
    print(f"{nome} -> Acc: {metriche['accuratezza']:.4f} | LogLoss: {metriche['logloss']:.4f} | AUC: {metriche['auc']:.4f}")


#blocco 9


#Scelta automatica del mdello migiore tra Random Forest e Regressione Logistica sulla base dell'accuratezza più alta.
# Controlliamo quale modello ha ottenuto l'accuratezza più alta
acc_rf=risultati.get("Random Forest", {}).get("accuratezza", 0)
acc_lr=risultati.get("Regressione Logistica", {}).get("accuratezza", 0)
if acc_rf>=acc_lr:
    MIGLIOR_MODELLO = "Random Forest"
else:
    MIGLIOR_MODELLO="Regressione Logistica"
#Preparazione delle matrici di dati
X_train_completo=train_df[all_features]
y_train_completo=train_df["player_won"].to_numpy()
X_test_completo=test_df[all_features]
#Addestramento sul 100% dei dati
if MIGLIOR_MODELLO=="Random Forest":
    modello_finale=crea_random_forest()
    print("Addestramento della Random Forest sull'intero dataset")
    modello_finale.fit(X_train_completo, y_train_completo)
    previsioni_test=modello_finale.predict_proba(X_test_completo)[:,1]

elif MIGLIOR_MODELLO=="Regressione Logistica":
    #Scaliamo i dati usando tutto il train
    scaler=StandardScaler()
    X_train_scalato=scaler.fit_transform(X_train_completo)
    X_test_scalato=scaler.transform(X_test_completo)
    modello_finale=crea_regressione_logistica()
    print("Addestramento della Regressione Logistica sull'intero dataset")
    modello_finale.fit(X_train_scalato, y_train_completo)
    previsioni_test=modello_finale.predict_proba(X_test_scalato)[:, 1]
#Converto probabilità in codice binario
test_pred=(previsioni_test>0.5).astype(int)

#Creazione e salvataggio file submission
bozza_submission=pd.read_csv(data_file("sample_submission", ".csv"))
df_predizioni=pd.DataFrame({"battle_id": test_df["battle_id"], "player_won": test_pred})
submission=bozza_submission[["battle_id"]].merge(df_predizioni, on="battle_id", how="left")
# Controlli
assert submission["player_won"].notna().all() and len(submission)==len(bozza_submission), \
    "Attenzione: errore nelle dimensioni o presenza di valori mancanti nel file di submission!"
submission["player_won"]=submission["player_won"].astype(int)
#Salvataggio nella cartella di lavoro di Kaggle
percorso_output=OUTPUT_CSV
submission.to_csv(percorso_output, index=False)
print(f"\nFile salvato con successo in: {percorso_output}")
print(f"Righe totali: {len(submission)}")
print(f"Frazione prevista di vittorie del Giocatore 1: {submission['player_won'].mean():.3f}")
print(submission.head())
