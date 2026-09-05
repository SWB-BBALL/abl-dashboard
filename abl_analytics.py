import os
import re
import json
import time
import random
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup

BASE_URL = "https://ascension-basketball.com/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}
ALLSTAR_TEAMS = {'east', 'west'}    # exhibition game team names — skipped everywhere
MIN_DELAY = 1.5
MAX_DELAY = 3.0
TOTAL_GAMES = 82
N_SIMS = 10000
REPLACEMENT_LEVEL = -2.0   # BPM of a replacement player
CURRENT_SEASON = "2004"     # Only pull stats from this season year — update each season


# ══════════════════════════════════════════════════════════════════════════════
# BPM 2.0 MATH
# ══════════════════════════════════════════════════════════════════════════════

POS_REG  = {'intercept':2.130,'pct_trb':8.668,'pct_stl':-2.486,'pct_pf':0.992,'pct_ast':-3.536,'pct_blk':1.667}
ROLE_REG = {'intercept':6.00,'pct_ast':-6.642,'pct_tpts':-8.544}

BPM_COEFFS  = {'pts_adj':(0.860,0.860),'threepm':(0.389,0.389),'ast':(0.580,1.034),'to':(-0.964,-0.964),'orb':(0.613,0.181),'drb':(0.116,0.181),'stl':(1.369,1.008),'blk':(1.327,0.703),'pf':(-0.367,-0.367)}
BPM_ROLE_COEFFS = {'fga':(-0.560,-0.780),'fta':(-0.246,-0.343)}

OBPM_COEFFS = {'pts_adj':(0.605,0.605),'threepm':(0.477,0.477),'ast':(0.476,0.476),'to':(-0.579,-0.882),'orb':(0.606,0.422),'drb':(-0.112,0.103),'stl':(0.177,0.294),'blk':(0.725,0.097),'pf':(-0.439,-0.439)}
OBPM_ROLE_COEFFS = {'fga':(-0.330,-0.472),'fta':(-0.145,-0.208)}

def _interp(ct, x):
    c1,c5=ct; t=(x-1.0)/4.0; return c1+t*(c5-c1)
def _pos_constant(pos):   return 0.0 if pos>=3.0 else -0.818*(3.0-pos)/2.0
def _obpm_pos_constant(p):return 0.0 if p>=3.0  else -1.698*(3.0-p)/2.0
def _role_constant(r):    return (r-3.0)*(2.774/2.0)
def _obpm_role_constant(r):return (r-3.0)*(0.860/2.0)

def _estimate_position(p, team):
    tm_min=team.get('min',1); p_min=p['min']
    if p_min<=0 or tm_min<=0: return 3.0
    def pct(s):
        tr=team.get(s,0)/tm_min; pr=p.get(s,0)/p_min
        return (pr/tr)*(p_min/tm_min) if tr>0 else 0.0
    raw=(POS_REG['intercept']+POS_REG['pct_trb']*pct('trb')+POS_REG['pct_stl']*pct('stl')
         +POS_REG['pct_pf']*pct('pf')+POS_REG['pct_ast']*pct('ast')+POS_REG['pct_blk']*pct('blk'))
    return float(np.clip((raw*p_min+3.0*50)/(p_min+50),1.0,5.0))

def _estimate_role(p, team):
    tm_min=team.get('min',1); p_min=p['min']
    if p_min<=0 or tm_min<=0: return 3.0
    def pct(s):
        tr=team.get(s,0)/tm_min; pr=p.get(s,0)/p_min
        return (pr/tr)*(p_min/tm_min) if tr>0 else 0.0
    tsa_p=p.get('fga',0)+0.44*p.get('fta',0); tsa_t=team.get('fga',0)+0.44*team.get('fta',0)
    if tsa_t>0 and tsa_p>0:
        thr=max(team.get('pts',0)/tsa_t-0.33,0)
        thresh_pts=max(p.get('pts',0)/max(tsa_p,1)-thr,0)*tsa_p
    else: thresh_pts=0.0
    tm_pts=team.get('pts',1)
    pct_tpts=(thresh_pts/p_min)/(tm_pts/tm_min)*(p_min/tm_min) if tm_pts>0 else 0.0
    raw=ROLE_REG['intercept']+ROLE_REG['pct_ast']*pct('ast')+ROLE_REG['pct_tpts']*pct_tpts
    return float(np.clip((raw*p_min+4.0*50)/(p_min+50),1.0,5.0))

def _pts_adj_100(s100, team100):
    LEAGUE=1.05
    tsa_t=team100.get('fga_100',0)+0.44*team100.get('fta_100',0)
    tsa_p=s100.get('fga_100',0)+0.44*s100.get('fta_100',0)
    adj=(LEAGUE-team100['pts_100']/tsa_t)*tsa_p if tsa_t>0 else 0.0
    return s100['pts_100']+adj

def _raw_score(s100, pos, role, pts_adj, coeffs, rc_, pc_fn, rc_fn):
    return (_interp(coeffs['pts_adj'],pos)*pts_adj+_interp(coeffs['threepm'],pos)*s100.get('threepm_100',0)
            +_interp(coeffs['ast'],pos)*s100['ast_100']+_interp(coeffs['to'],pos)*s100['to_100']
            +_interp(coeffs['orb'],pos)*s100['orb_100']+_interp(coeffs['drb'],pos)*s100['drb_100']
            +_interp(coeffs['stl'],pos)*s100['stl_100']+_interp(coeffs['blk'],pos)*s100['blk_100']
            +_interp(coeffs['pf'],pos)*s100['pf_100']
            +_interp(rc_['fga'],role)*s100.get('fga_100',0)+_interp(rc_['fta'],role)*s100.get('fta_100',0)
            +pc_fn(pos)+rc_fn(role))

def compute_bpm(player, team_counting, team_100):
    p_min=player['min']
    if p_min<=0: return None
    pace=team_counting.get('pace',95.0); pf=(48.0/pace)*100.0
    s100={
        'pts_100':player['pts']/p_min*pf,'ast_100':player['ast']/p_min*pf,
        'to_100':player['to']/p_min*pf,'orb_100':player['orb']/p_min*pf,
        'drb_100':player['drb']/p_min*pf,'stl_100':player['stl']/p_min*pf,
        'blk_100':player['blk']/p_min*pf,'pf_100':player['pf']/p_min*pf,
        'fga_100':player['fga']/p_min*pf,'fta_100':player['fta']/p_min*pf,
        'threepm_100':player.get('threepm',0)/p_min*pf,
    }
    pos=_estimate_position(player,team_counting); role=_estimate_role(player,team_counting)
    pts_adj=_pts_adj_100(s100,team_100)
    raw_bpm =_raw_score(s100,pos,role,pts_adj,BPM_COEFFS, BPM_ROLE_COEFFS, _pos_constant,      _role_constant)
    raw_obpm=_raw_score(s100,pos,role,pts_adj,OBPM_COEFFS,OBPM_ROLE_COEFFS,_obpm_pos_constant,_obpm_role_constant)
    return {'pos':round(pos,2),'role':round(role,2),'raw_bpm':round(raw_bpm,3),'raw_obpm':round(raw_obpm,3)}

def apply_team_adjustment(df, team_ratings):
    df=df.copy()
    df['w_raw']=df['raw_bpm']*df['min']; df['w_rawO']=df['raw_obpm']*df['min']
    t=df.groupby('team').agg(tot_w=('w_raw','sum'),tot_wO=('w_rawO','sum'),tot_m=('min','sum')).reset_index()
    t['team_avg_raw']=t['tot_w']/t['tot_m']; t['team_avg_rawO']=t['tot_wO']/t['tot_m']
    df=df.merge(t[['team','team_avg_raw','team_avg_rawO']],on='team',how='left')
    def get_rating(tn):
        tl=tn.lower()
        for k,v in team_ratings.items():
            if k.lower() in tl or tl in k.lower(): return v['rating']
        return 0.0
    df['team_rating']=df['team'].apply(get_rating)
    raw_d=df['raw_bpm']-df['raw_obpm']; df['w_rawD']=raw_d*df['min']
    t2=df.groupby('team').agg(tot_wD=('w_rawD','sum'),tot_m2=('min','sum')).reset_index()
    t2['team_avg_rawD']=t2['tot_wD']/t2['tot_m2']
    df=df.merge(t2[['team','team_avg_rawD']],on='team',how='left')
    df['o_share']=(df['team_avg_rawO']/df['team_avg_raw'].replace(0,np.nan).fillna(1)).clip(-3,3)
    adj=df['team_rating']-df['team_avg_raw']
    df['adj_bpm']=df['raw_bpm']+adj; df['adj_obpm']=df['raw_obpm']+adj*df['o_share'].fillna(0.5)
    df['adj_dbpm']=df['adj_bpm']-df['adj_obpm']
    shrink=df['min']/(df['min']+250)
    df['bpm']=(df['adj_bpm']*shrink).round(2)
    df['obpm']=(df['adj_obpm']*shrink).round(2)
    df['dbpm']=(df['adj_dbpm']*shrink).round(2)
    # VORP: (BPM - replacement_level) * (min / (pace * 48)) * (1/10)
    # Simplified: BPM above replacement * fraction of team possessions used
    # Standard formula: VORP = (BPM - (-2.0)) * (MP / (Pace * 48)) * team_games / 82 * (82 / team_games)
    # Simplified to: (BPM + 2) * min / 2500  (approximates per-season value)
    df['vorp']=((df['bpm']-REPLACEMENT_LEVEL)*df['min']/2500.0).round(2)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# PLAYOFF SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def simulate_season(teams_data, hca, unplayed_games, n_sims=N_SIMS):
    """
    Monte Carlo season simulation using the real remaining schedule.
    Each remaining game: P(home wins) = logistic((home_rating - away_rating + hca) / spread)
    spread ~= 11 pts (typical game-to-game std dev).
    Returns per-team: projected wins, playoff% (top 16 of 30).
    """
    print(f"\n[sim] Running {n_sims:,} simulations on real remaining schedule ({len(unplayed_games)} games)...")
    SPREAD        = 11.0
    PLAYOFF_SPOTS = 16

    team_names = list(teams_data.keys())
    ratings    = {t: teams_data[t]['rating'] for t in team_names}
    current_w  = {t: teams_data[t]['wins']   for t in team_names}
    known      = set(team_names)

    # Use the actual remaining schedule — filter to teams we have ratings for
    sched = [(g['home'], g['away']) for g in unplayed_games
             if g['home'] in known and g['away'] in known]

    if not sched:
        print("  Warning: no remaining schedule found — projections will equal current wins.")

    # Pre-compute win probability per game — same every sim since ratings are fixed
    probs = []
    for home, away in sched:
        diff   = ratings[home] - ratings[away] + hca
        p_home = 1.0 / (1.0 + np.exp(-diff / SPREAD))
        probs.append((home, away, p_home))

    playoff_counts = {t: 0   for t in team_names}
    proj_wins_sum  = {t: 0.0 for t in team_names}
    rng = np.random.default_rng(42)

    for _ in range(n_sims):
        sim_w = dict(current_w)
        rolls = rng.random(len(probs))
        for (home, away, p_home), roll in zip(probs, rolls):
            if roll < p_home:
                sim_w[home] += 1
            else:
                sim_w[away] += 1

        ranked      = sorted(team_names, key=lambda t: sim_w[t], reverse=True)
        playoff_set = set(ranked[:PLAYOFF_SPOTS])
        for t in team_names:
            proj_wins_sum[t] += sim_w[t]
            if t in playoff_set:
                playoff_counts[t] += 1

    results = {}
    for t in team_names:
        results[t] = {
            'proj_wins':   round(proj_wins_sum[t]  / n_sims, 1),
            'playoff_pct': round(playoff_counts[t] / n_sims * 100, 1),
        }

    print(f"  Simulation complete.")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def polite_get(url, retries=3):
    time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 200: return r
            elif r.status_code == 429:
                wait=(attempt+1)*10; print(f"  Rate limited — waiting {wait}s..."); time.sleep(wait)
            else:
                print(f"  HTTP {r.status_code} for {url}"); return None
        except requests.exceptions.RequestException as e:
            wait=(attempt+1)*5; print(f"  Error ({e}) — retry {attempt+1}/{retries} in {wait}s"); time.sleep(wait)
    print(f"  Gave up on {url}"); return None

def clean_val(val):
    if val is None: return 0.0
    if isinstance(val,(int,float)): return float(val)
    try: return float(str(val).replace('%','').strip())
    except: return 0.0

def parse_game_text(text):
    parts=[p.strip() for p in text.strip().split(',')]
    if len(parts)!=2: return None
    m1=re.search(r'^(@?)(.*?)\s+(\d+)$',parts[0]); m2=re.search(r'^(@?)(.*?)\s+(\d+)$',parts[1])
    if not m1 or not m2: return None
    at1,n1,s1=m1.group(1),m1.group(2).strip(),int(m1.group(3))
    at2,n2,s2=m2.group(1),m2.group(2).strip(),int(m2.group(3))
    if at1=='@': return n1,s1,n2,s2
    elif at2=='@': return n2,s2,n1,s1
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 1. TEAM PACE / EFFICIENCY
# ══════════════════════════════════════════════════════════════════════════════

def scrape_team_leaders():
    print("\n[1/4] Scraping team pace and efficiency stats...")
    url=f"{BASE_URL}teamleaders.htm"; team_stats={}
    r=polite_get(url)
    if not r:
        print("  Could not reach teamleaders.htm — using default pace 95.0."); return team_stats
    soup=BeautifulSoup(r.text,'html.parser')
    def extract_leaderboard(keywords):
        for td in soup.find_all('td',class_='tableheader'):
            if not any(kw in td.get_text().strip().lower() for kw in keywords): continue
            table=td.find_parent('table')
            if not table: continue
            result={}
            for row in table.find_all('tr'):
                if 'row' not in ' '.join(row.get('class',[])): continue
                cells=[c.get_text().strip().replace('\xa0','') for c in row.find_all('td',class_='main')]
                if len(cells)>=4:
                    name=cells[2].strip(); val=clean_val(cells[3])
                    if name: result[name]=val
            if result:
                print(f"  Found '{td.get_text().strip()}' — {len(result)} teams."); return result
        return {}
    pace_data=extract_leaderboard(['pace','possessions','pos/g'])
    oeff_data=extract_leaderboard(['off eff','oeff','offensive eff','ortg','off rating'])
    deff_data=extract_leaderboard(['def eff','deff','defensive eff','drtg','def rating'])
    all_teams=set(pace_data)|set(oeff_data)|set(deff_data)
    for t in all_teams:
        pace=pace_data.get(t,95.0); oeff=oeff_data.get(t,100.0); deff=deff_data.get(t,100.0)
        team_stats[t.lower()]={'name':t,'pace':pace if pace>50 else 95.0,'oeff':oeff,'deff':deff,'net_eff':oeff-deff}
    if not team_stats and pace_data:
        for t,pace in pace_data.items():
            team_stats[t.lower()]={'name':t,'pace':pace if pace>50 else 95.0,'oeff':100.0,'deff':100.0,'net_eff':0.0}
    print(f"  Found stats for {len(team_stats)} teams."); return team_stats


# ══════════════════════════════════════════════════════════════════════════════
# 2. SCHEDULE SCRAPER + RIDGE SOLVER
# ══════════════════════════════════════════════════════════════════════════════

def solve_team_ratings():
    print("\n[2/4] Fetching schedule and solving team ratings...")
    url=f"{BASE_URL}schedule.htm"
    r=polite_get(url)
    if not r: print("  Could not reach schedule.htm."); return {},0.0,6.0
    soup=BeautifulSoup(r.text,'html.parser')
    played_games=[]; unplayed_games=[]; in_regular_season=False; in_playoffs=False
    for td in soup.find_all('td'):
        td_class=' '.join(td.get('class',[])).lower(); text_lower=td.get_text(separator=' ').strip().lower()
        if 'tableheader' in td_class:
            if 'regular season' in text_lower:
                in_regular_season=True; in_playoffs=False; print("  Found Regular Season marker.")
            elif 'preseason' in text_lower or 'pre-season' in text_lower:
                in_regular_season=False; print("  Found Preseason marker.")
            elif any(kw in text_lower for kw in ('playoff','postseason','post season')):
                in_playoffs=True; print("  Found Playoffs marker — stopping.")
            continue
        if in_playoffs: break
        if not in_regular_season: continue
        if 'main' not in td_class: continue
        links=td.find_all('a',href=True)
        box_links=[l for l in links if 'box' in l['href'].lower()]
        ros_links=[l for l in links if 'roster' in l['href'].lower()]
        if box_links:
            parsed=parse_game_text(box_links[0].get_text(separator=' ').strip())
            if parsed:
                home,h_score,away,a_score=parsed
                # Skip East/West all-star exhibition game
                if home.lower() in ALLSTAR_TEAMS or away.lower() in ALLSTAR_TEAMS:
                    continue
                raw_href=box_links[0]['href'].replace('./','').replace('../','').lstrip('/')
                box_url=BASE_URL+raw_href
                played_games.append({'home':home,'away':away,'home_score':h_score,'away_score':a_score,'margin':h_score-a_score,'box_url':box_url})
        elif len(ros_links)==2:
            away_name=ros_links[0].get_text().strip(); home_name=ros_links[1].get_text().strip()
            if away_name.lower() in ALLSTAR_TEAMS or home_name.lower() in ALLSTAR_TEAMS:
                continue
            if away_name and home_name: unplayed_games.append({'home':home_name,'away':away_name})
    n_played=len(played_games); n_unplayed=len(unplayed_games); total=n_played+n_unplayed
    print(f"  {n_played} played, {n_unplayed} remaining.")
    if n_played==0: print("  No played games found."); return {},0.0,6.0
    pct_rem=n_unplayed/total if total>0 else 0.5
    current_lambda=round(1.0+9.0*pct_rem,2)
    print(f"  Dynamic λ = {current_lambda}  ({pct_rem*100:.1f}% remaining)")
    teams=sorted(set(g['home'] for g in played_games)|set(g['away'] for g in played_games))
    team_idx={t:i for i,t in enumerate(teams)}; n_teams=len(teams)
    X=np.zeros((n_played,n_teams+1)); y=np.zeros(n_played)
    for i,g in enumerate(played_games):
        X[i,team_idx[g['home']]]=1.0; X[i,team_idx[g['away']]]=-1.0; X[i,-1]=1.0; y[i]=g['margin']
    XtX=X.T@X; Xty=X.T@y; I_reg=np.eye(n_teams+1); I_reg[-1,-1]=0.0
    beta=np.linalg.solve(XtX+current_lambda*I_reg,Xty)
    hca=float(beta[-1]); raw_ratings=beta[:-1]; avg_rating=float(np.mean(raw_ratings))
    ratings={teams[i]:float(raw_ratings[i])-avg_rating for i in range(n_teams)}
    stats={t:{'w':0,'l':0,'mov_sum':0.0,'opps_played':[],'opps_rem':[]} for t in teams}
    for g in played_games:
        stats[g['home']]['mov_sum']+=g['margin']; stats[g['away']]['mov_sum']-=g['margin']
        stats[g['home']]['opps_played'].append(g['away']); stats[g['away']]['opps_played'].append(g['home'])
        if g['margin']>0: stats[g['home']]['w']+=1; stats[g['away']]['l']+=1
        elif g['margin']<0: stats[g['away']]['w']+=1; stats[g['home']]['l']+=1
    for g in unplayed_games:
        if g['home'] in stats and g['away'] in stats:
            stats[g['home']]['opps_rem'].append(g['away']); stats[g['away']]['opps_rem'].append(g['home'])
    team_output={}
    for t in teams:
        s=stats[t]; gp=s['w']+s['l']; mov=s['mov_sum']/gp if gp>0 else 0.0
        sos=float(np.mean([ratings[o] for o in s['opps_played']])) if s['opps_played'] else 0.0
        rsos=float(np.mean([ratings[o] for o in s['opps_rem']])) if s['opps_rem'] else 0.0
        team_output[t]={'team':t,'wins':s['w'],'losses':s['l'],'mov':round(mov,2),
                        'rating':round(ratings[t],2),'sos':round(sos,2),'rsos':round(rsos,2)}
    print(f"  HCA = {hca:+.2f} pts | {n_teams} teams solved."); return team_output,hca,current_lambda,played_games,unplayed_games



# ══════════════════════════════════════════════════════════════════════════════
# 2b. GAME LOG — single-game BPM, accumulated in game_log.json
# ══════════════════════════════════════════════════════════════════════════════

GAME_LOG_FILE = "game_log.json"

POS_MAP = {'pg':1.0,'g':1.5,'sg':2.0,'sf':3.0,'f':3.5,'pf':4.0,'fc':4.5,'c':5.0}

def load_game_log():
    if os.path.exists(GAME_LOG_FILE):
        try:
            with open(GAME_LOG_FILE,'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}   # keyed by box_url

def save_game_log(log):
    with open(GAME_LOG_FILE,'w') as f:
        json.dump(log, f, indent=2)

def parse_combined(val):
    """Parse '8-15' → (8, 15). Returns (0,0) on failure."""
    val = val.strip().replace('\xa0','')
    if '-' in val:
        parts = val.split('-')
        try: return int(parts[0]), int(parts[1])
        except: pass
    return 0, 0

def scrape_box_score(url, home_team, away_team, home_score, away_score):
    """
    Scrape one box score page and return a list of player-game dicts.
    Uses listed position (PG/C etc.) directly — more reliable for single games.
    """
    r = polite_get(url)
    if not r: return []

    try:
        soup = BeautifulSoup(r.text, 'html.parser')
        results = []

        # Game pace proxy: total possessions ≈ (home_score + away_score) / 2 / (48/5)
        # Simple estimate: just use total pts to normalise per-100
        total_pts = home_score + away_score
        est_pace  = max(total_pts / 2.0 * (100.0 / 110.0), 80.0)  # rough

        for side in ('away','home'):
            team_name = away_team if side=='away' else home_team
            hdr_class = 'awayheader' if side=='away' else 'homeheader'

            # Find the table whose header cells use this class
            target_table = None
            for tbl in soup.find_all('table'):
                if tbl.find('td', class_=hdr_class):
                    target_table = tbl
                    break
            if not target_table: continue

            rows = target_table.find_all('tr')

            # Build header from the hdr_class cells
            header_row = None
            data_start  = 0
            for ri, row in enumerate(rows):
                hdrs = [td.get_text().strip().lower().replace('\xa0','').replace(' ','')
                        for td in row.find_all('td', class_=hdr_class)]
                if 'min' in hdrs and 'pts' in hdrs:
                    header_row = hdrs
                    data_start  = ri + 1
                    break
            if not header_row: continue

            for row in rows[data_start:]:
                cls = ' '.join(row.get('class', []))
                if 'row' not in cls: continue
                cells = [td.get_text().strip().replace('\xa0','')
                         for td in row.find_all('td', class_='main')]
                if len(cells) < 5: continue

                p = dict(zip(header_row, cells))

                # Pull player name and link from the first td
                name_td = row.find('td', class_='main')
                link    = name_td.find('a') if name_td else None
                player_name = link.get_text().strip() if link else cells[0].strip()
                player_url  = (BASE_URL + link['href'].replace('../','').lstrip('/')) if link else ''

                # Minutes — sometimes listed as integer, sometimes "39"
                minutes = clean_val(p.get('min', 0))
                if minutes <= 0: continue

                # Listed position → numeric
                listed_pos_str = p.get('pos','').strip().lower().replace('\xa0','')
                listed_pos = POS_MAP.get(listed_pos_str, 3.0)

                # Split combined fields
                fgm, fga_ = parse_combined(p.get('fgm-a', p.get('fgma','0-0')))
                tpm, tpa_ = parse_combined(p.get('3pm-a', p.get('3pma','0-0')))
                ftm, fta_ = parse_combined(p.get('ftm-a', p.get('ftma','0-0')))

                orb = clean_val(p.get('off', p.get('oreb', p.get('orb', 0))))
                trb = clean_val(p.get('reb', p.get('trb', 0)))
                drb = max(trb - orb, 0)
                ast = clean_val(p.get('ast', 0))
                stl = clean_val(p.get('stl', 0))
                blk = clean_val(p.get('blk', 0))
                to  = clean_val(p.get('to',  0))
                pf  = clean_val(p.get('pf',  0))
                pts = clean_val(p.get('pts', 0))
                pm  = clean_val(p.get('+/-', p.get('+-', 0)))

                # Per-100 (single-game uses game pace)
                pos_factor = (48.0 / est_pace) * 100.0
                s100 = {
                    'pts_100':     pts  / minutes * pos_factor,
                    'ast_100':     ast  / minutes * pos_factor,
                    'to_100':      to   / minutes * pos_factor,
                    'orb_100':     orb  / minutes * pos_factor,
                    'drb_100':     drb  / minutes * pos_factor,
                    'stl_100':     stl  / minutes * pos_factor,
                    'blk_100':     blk  / minutes * pos_factor,
                    'pf_100':      pf   / minutes * pos_factor,
                    'fga_100':     fga_ / minutes * pos_factor,
                    'fta_100':     fta_ / minutes * pos_factor,
                    'threepm_100': tpm  / minutes * pos_factor,
                }

                # For single-game BPM: use listed position, role=3 (neutral),
                # no team adjustment (too noisy on one game), no shrinkage
                pos  = listed_pos
                role = 3.0   # neutral — can't reliably estimate role from one game

                # Points adjustment: assume league-average team environment
                LEAGUE_PTS_PER_TSA = 1.05
                tsa_p = s100['fga_100'] + 0.44 * s100['fta_100']
                pts_adj = s100['pts_100']   # skip team adjustment for single game

                raw_bpm  = _raw_score(s100, pos, role, pts_adj,
                                      BPM_COEFFS,  BPM_ROLE_COEFFS,
                                      _pos_constant, _role_constant)
                raw_obpm = _raw_score(s100, pos, role, pts_adj,
                                      OBPM_COEFFS, OBPM_ROLE_COEFFS,
                                      _obpm_pos_constant, _obpm_role_constant)
                raw_dbpm = raw_bpm - raw_obpm

                results.append({
                    'player':   player_name,
                    'player_url': player_url,
                    'team':     team_name,
                    'side':     side,
                    'min':      int(minutes),
                    'pts':      int(pts),
                    'orb':      int(orb), 'drb': int(drb), 'trb': int(trb),
                    'ast':      int(ast), 'stl': int(stl), 'blk': int(blk),
                    'to':       int(to),  'pf':  int(pf),
                    'fgm':      fgm, 'fga': fga_,
                    'tpm':      tpm, 'tpa': tpa_,
                    'ftm':      ftm, 'fta': fta_,
                    'pm':       int(pm),
                    'listed_pos': listed_pos_str.upper(),
                    'bpm':      round(raw_bpm,  2),
                    'obpm':     round(raw_obpm, 2),
                    'dbpm':     round(raw_dbpm, 2),
                })

        return results

    except Exception as e:
        print(f"  Error scraping box score {url}: {e}")
        return []


def update_game_log(played_games):
    """
    Check which box scores aren't in the log yet, fetch only the new ones.
    Returns the full updated log.
    """
    log = load_game_log()
    new_games = [g for g in played_games if g.get('box_url') and g['box_url'] not in log]

    if not new_games:
        print(f"  Game log up to date — {len(log)} games already stored.")
        return log

    print(f"  {len(log)} games cached · {len(new_games)} new games to fetch...")

    for i, g in enumerate(new_games, 1):
        url = g['box_url']
        print(f"  [{i}/{len(new_games)}] {g['away']} @ {g['home']} ({g['home_score']}-{g['away_score']})")
        player_lines = scrape_box_score(url, g['home'], g['away'], g['home_score'], g['away_score'])
        log[url] = {
            'home':       g['home'],
            'away':       g['away'],
            'home_score': g['home_score'],
            'away_score': g['away_score'],
            'players':    player_lines,
        }
        save_game_log(log)   # save after each game so progress isn't lost on crash

    print(f"  Game log updated — {len(log)} total games.")
    return log


# ══════════════════════════════════════════════════════════════════════════════
# 3. PLAYER SCRAPER + BPM 2.0
# ══════════════════════════════════════════════════════════════════════════════

def scrape_player_profile(url, team_name, player_name):
    r=polite_get(url)
    if not r: return None
    try:
        soup=BeautifulSoup(r.text,'html.parser')
        totals_table=None
        for td in soup.find_all('td',class_='tableheader'):
            if 'season totals' in td.get_text().strip().lower():
                totals_table=td.find_parent('table'); break
        if not totals_table: return None
        rows=totals_table.find_all('tr')
        header_row=None; data_start=0
        for ri,row in enumerate(rows):
            hdrs=[td.get_text().strip().lower().replace('\xa0','').replace(' ','') for td in row.find_all('td',class_='header')]
            if 'min' in hdrs and 'pts' in hdrs:
                header_row=hdrs; data_start=ri+1; break
        if not header_row: return None
        seen={}; fixed=[]
        for h in header_row:
            if h in seen: seen[h]+=1; fixed.append(f"{h}_{seen[h]}")
            else: seen[h]=0; fixed.append(h)
        header_row=fixed
        # Only accept a row whose season column matches CURRENT_SEASON
        # This prevents players with 0 games this season (but prior seasons)
        # from being included in the ratings.
        target_row=None
        for row in rows[data_start:]:
            cls=' '.join(row.get('class',[]))
            if 'row' not in cls: continue
            cells=[td.get_text().strip().replace('\xa0','') for td in row.find_all('td',class_='main')]
            if len(cells)!=len(header_row): continue
            # First cell is the season year — must match CURRENT_SEASON exactly
            row_year=cells[0].strip().replace('\xa0','')
            if row_year != CURRENT_SEASON: continue
            target_row=cells; break
        if not target_row: return None  # player has no stats this season — skip them
        p=dict(zip(header_row,target_row))
        minutes=clean_val(p.get('min',0))
        if minutes<100: return None
        gp=max(clean_val(p.get('g',1)),1.0)
        orb=clean_val(p.get('oreb',0)); trb=clean_val(p.get('reb',0)); drb=trb-orb
        stl=clean_val(p.get('stl',0)); blk=clean_val(p.get('stl_1',p.get('blk',0)))
        return {'name':player_name,'team':team_name,'min':minutes,'gp':gp,
                'pts':clean_val(p.get('pts',0)),'orb':orb,'drb':drb,'trb':trb,
                'ast':clean_val(p.get('ast',0)),'stl':stl,'blk':blk,
                'to':clean_val(p.get('to',0)),'pf':clean_val(p.get('pf',0)),
                'fga':clean_val(p.get('fga',0)),'fta':clean_val(p.get('fta',0)),
                'threepm':clean_val(p.get('3pm',0))}
    except Exception as e:
        print(f"  Error parsing {player_name}: {e}"); return None


def gather_player_stats(team_ratings, scraped_teams):
    print("\n[3/4] Gathering player stats (sequential — polite to the server)...")
    player_tasks=[]; seen_links=set()
    for team_id in range(1,45):
        r=polite_get(f"{BASE_URL}rosters/roster{team_id}.htm")
        if not r: continue
        soup=BeautifulSoup(r.text,'html.parser')
        title=soup.title.get_text().strip() if soup.title else f"Team {team_id}"
        team_name=re.sub(r'(?i)\s*(roster|team|data|profile|page)\b','',title).strip() or f"Team {team_id}"
        found=0
        for link in soup.find_all('a',href=True):
            href=link['href']
            if 'player' not in href.lower(): continue
            clean_href=href.replace('../','').lstrip('/')
            if clean_href in seen_links: continue
            seen_links.add(clean_href); player_tasks.append((BASE_URL+clean_href,team_name,link.get_text().strip())); found+=1
        if found: print(f"  Roster {team_id} ({team_name}): {found} players queued")
    if not player_tasks: print("  No player links found."); return []
    print(f"  {len(player_tasks)} profiles to fetch...")
    players_data=[]
    for i,(url,team,name) in enumerate(player_tasks,1):
        if i%20==0: print(f"  ...{i}/{len(player_tasks)} fetched")
        result=scrape_player_profile(url,team,name)
        if result: players_data.append(result)
    if not players_data: print("  No player data collected."); return []
    print(f"  {len(players_data)} qualifying players scraped.")
    df=pd.DataFrame(players_data)
    def get_pace(tn):
        tl=tn.lower()
        for key,val in scraped_teams.items():
            if key in tl or tl in key: return val['pace']
        return 95.0
    df['pace']=df['team'].apply(get_pace)
    team_agg=(df.groupby('team').agg(min=('min','sum'),pts=('pts','sum'),trb=('trb','sum'),
               orb=('orb','sum'),drb=('drb','sum'),ast=('ast','sum'),stl=('stl','sum'),
               blk=('blk','sum'),to=('to','sum'),pf=('pf','sum'),fga=('fga','sum'),fta=('fta','sum')).reset_index())
    team_agg['pace']=team_agg['team'].apply(get_pace)
    team_agg_dict={}
    for _,row in team_agg.iterrows():
        d=row.to_dict(); pf_=(48.0/row['pace'])*100.0; m=row['min']
        d.update({'pts_100':row['pts']/m*pf_ if m>0 else 0,'fga_100':row['fga']/m*pf_ if m>0 else 0,'fta_100':row['fta']/m*pf_ if m>0 else 0})
        team_agg_dict[row['team']]=d
    print("  Computing BPM 2.0 (position-interpolated)...")
    results=[]
    for p in players_data:
        team=p['team']; t_agg=team_agg_dict.get(team,{})
        t_100={'pts_100':t_agg.get('pts_100',100.0),'fga_100':t_agg.get('fga_100',85.0),'fta_100':t_agg.get('fta_100',25.0)}
        bpm_out=compute_bpm(p,t_agg,t_100)
        if bpm_out: results.append({**p,**bpm_out})
    if not results: return []
    bpm_df=pd.DataFrame(results); bpm_df=apply_team_adjustment(bpm_df,team_ratings)
    bpm_df['ppg']=(bpm_df['pts']/bpm_df['gp']).round(1)
    out=bpm_df[['name','team','gp','min','ppg','pos','role','obpm','dbpm','bpm','vorp']].copy()
    for c in ['obpm','dbpm','bpm','vorp']: out[c]=out[c].round(2)
    return out.sort_values('bpm',ascending=False).to_dict(orient='records')


# ══════════════════════════════════════════════════════════════════════════════
# 4. HTML DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# 5. HTML DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def write_dashboard_html(teams_data, players_data, hca, current_lambda, sim_results, game_log):
    import json as _json
    print("  Writing dashboard.html...")
    now      = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    sorted_teams  = sorted(teams_data.values(), key=lambda x: x["rating"], reverse=True)
    unique_teams  = sorted({p["team"] for p in players_data})
    pct_done      = round(100 * (1 - (current_lambda - 1) / 9))

    def signed(v, d=2):
        return ("+" if v >= 0 else "") + f"{v:.{d}f}"
    def rc(v):
        return " c-neg" if v < 0 else " c-pos"

    team_rows = ""
    for t in sorted_teams:
        sim = sim_results.get(t["team"], {})
        proj   = sim.get("proj_wins", "—")
        playoff = str(sim.get("playoff_pct", "—")) + "%" if "playoff_pct" in sim else "—"
        team_rows += (
            "<tr>"
            f'<td class="td-team">{t["team"]}</td>'
            f'<td class="td-num">{t["wins"]}-{t["losses"]}</td>'
            f'<td class="td-num">{signed(t["mov"],1)}</td>'
            f'<td class="td-num{rc(t["rating"])}">{signed(t["rating"])}</td>'
            f'<td class="td-num">{signed(t["sos"])}</td>'
            f'<td class="td-num" style="color:#006600">{signed(t["rsos"])}</td>'
            f'<td class="td-num" style="font-weight:700">{proj}</td>'
            f'<td class="td-num">{playoff}</td>'
            "</tr>\n"
        )

    def p_rows(data, cols, limit=25):
        html = ""
        for p in data[:limit]:
            html += "<tr>"
            html += f'<td class="td-team">{p["name"]}</td>'
            html += f'<td class="td-num" style="color:#555">{p["team"]}</td>'
            for c in cols:
                if c in ("obpm","dbpm","bpm","vorp"):
                    html += f'<td class="td-num{rc(p[c])}">{signed(p[c])}</td>'
                elif c in ("pos","role"):
                    html += f'<td class="td-num">{p[c]:.1f}</td>'
                else:
                    v = p[c]
                    html += f'<td class="td-num">{int(v) if isinstance(v,float) and v==int(v) else v}</td>'
            html += "</tr>\n"
        return html

    bpm_rows  = p_rows(players_data, ["pos","obpm","dbpm","bpm","vorp"])
    obpm_rows = p_rows(sorted(players_data, key=lambda x:x["obpm"], reverse=True), ["pos","obpm"])
    dbpm_rows = p_rows(sorted(players_data, key=lambda x:x["dbpm"], reverse=True), ["pos","dbpm"])
    vorp_rows = p_rows(sorted(players_data, key=lambda x:x["vorp"], reverse=True), ["min","bpm","vorp"])
    team_options = "\n".join(f'<option value="{t}">{t}</option>' for t in unique_teams)

    player_game_history = {}
    game_summaries      = []
    for url, gdata in game_log.items():
        game_id = url.split("/")[-1].replace(".htm","")
        label   = f'{gdata["away"]} @ {gdata["home"]} ({gdata["away_score"]}-{gdata["home_score"]})'
        game_summaries.append({"id":game_id,"label":label,"home":gdata["home"],"away":gdata["away"],
                               "home_score":gdata["home_score"],"away_score":gdata["away_score"]})
        for p in gdata.get("players",[]):
            name = p["player"]
            if name not in player_game_history:
                player_game_history[name] = []
            player_game_history[name].append({
                "game_id":game_id,"label":label,"team":p["team"],
                "opp":gdata["home"] if p["side"]=="away" else gdata["away"],
                "min":p["min"],"pts":p["pts"],"trb":p["trb"],"ast":p["ast"],"pm":p["pm"],
                "bpm":p["bpm"],"obpm":p["obpm"],"dbpm":p["dbpm"],
            })
    for name in player_game_history:
        player_game_history[name].sort(key=lambda x: x["game_id"])

    gl_json  = _json.dumps(player_game_history)
    gs_json  = _json.dumps(game_summaries)
    all_json = _json.dumps(players_data)

    analytics_out = {
        "updated":now,"hca":round(hca,3),"lambda":current_lambda,"pct_done":pct_done,
        "teams":list(teams_data.values()),"players":players_data,
        "sim_results":sim_results,"game_log":player_game_history,"game_summaries":game_summaries,
    }
    with open("analytics_data.json","w",encoding="utf-8") as f:
        _json.dump(analytics_out, f)
    print("  analytics_data.json written.")

    CSS = """
* { box-sizing:border-box; margin:0; padding:0; }
body { font-family: Arial, Helvetica, sans-serif; font-size:13px; background:#fff; color:#000; }
.topbar { background:#003366; color:#fff; padding:8px 16px; display:flex; justify-content:space-between; align-items:center; }
.topbar-title { font-size:16px; font-weight:bold; }
.topbar-ts { font-size:11px; color:#aac; }
.tabnav { background:#e8e8e8; border-bottom:2px solid #003366; display:flex; overflow-x:auto; }
.tab { padding:8px 16px; cursor:pointer; border:none; background:none; font-size:12px; font-weight:bold; color:#003366; white-space:nowrap; border-right:1px solid #ccc; }
.tab:hover { background:#d0d8e8; }
.tab.active { background:#003366; color:#fff; }
.page { display:none; padding:16px 20px; max-width:1300px; margin:0 auto; }
.page.active { display:block; }
h1 { font-size:18px; font-weight:bold; margin-bottom:4px; color:#003366; }
.page-sub { font-size:12px; color:#555; margin-bottom:14px; line-height:1.5; }
.section-title { font-size:14px; font-weight:bold; color:#003366; margin:14px 0 6px; border-bottom:1px solid #003366; padding-bottom:3px; }
table { border-collapse:collapse; width:100%; font-size:12px; }
th { background:#003366; color:#fff; padding:5px 8px; text-align:right; border:1px solid #225; white-space:nowrap; cursor:pointer; }
th.th-team { text-align:left; min-width:160px; }
th.th-num { min-width:50px; }
td { padding:4px 8px; border:1px solid #ddd; }
td.td-team { text-align:left; font-weight:bold; }
td.td-num { text-align:right; }
tr:nth-child(even) { background:#f2f2f2; }
tbody tr:hover { background:#e8f0ff; }
.c-pos { color:#006600; font-weight:bold; }
.c-neg { color:#cc0000; }
.filter-row { display:flex; gap:16px; align-items:center; margin-bottom:12px; flex-wrap:wrap; }
select, input[type=text] { border:1px solid #ccc; padding:4px 8px; font-size:12px; border-radius:3px; min-width:200px; }
select:focus, input[type=text]:focus { border-color:#003366; outline:none; }
input[type=range] { width:140px; accent-color:#003366; }
.trade-grid { display:grid; grid-template-columns:1fr auto 1fr; gap:20px; margin-bottom:16px; }
.trade-side h3 { font-size:12px; font-weight:bold; color:#003366; margin-bottom:8px; text-transform:uppercase; }
.trade-arrow { display:flex; align-items:center; justify-content:center; font-size:24px; color:#999; padding-top:24px; }
.player-chip { display:inline-flex; align-items:center; gap:6px; background:#e8e8e8; border:1px solid #ccc; border-radius:3px; padding:3px 8px; margin:3px; font-size:12px; cursor:pointer; }
.player-chip:hover { border-color:#003366; }
.trade-result-grid { display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:12px; }
.trade-result { background:#f9f9f9; border:1px solid #ccc; border-radius:4px; padding:14px; }
.trade-result h4 { font-size:11px; font-weight:bold; color:#555; text-transform:uppercase; margin-bottom:8px; }
.trade-delta { font-size:28px; font-weight:bold; margin-bottom:6px; }
.trade-delta.pos { color:#006600; }
.trade-delta.neg { color:#cc0000; }
.trade-delta.neu { color:#555; }
.trade-detail { font-size:12px; color:#555; line-height:1.6; }
#tradeBtn { background:#003366; border:none; color:#fff; font-size:13px; font-weight:bold; padding:8px 20px; border-radius:3px; cursor:pointer; }
#tradeBtn:hover { background:#004488; }
.suggestion-box { position:absolute; z-index:50; width:280px; border:1px solid #ccc; border-radius:0 0 3px 3px; max-height:200px; overflow-y:auto; background:#fff; }
.suggestion-item { padding:6px 10px; cursor:pointer; font-size:12px; border-bottom:1px solid #eee; }
.suggestion-item:hover { background:#e8f0ff; }
.avg-row td { font-weight:bold; color:#003366; border-top:2px solid #003366; }
.mode-btn { background:#e8e8e8; color:#003366; border:1px solid #ccc; padding:4px 12px; font-size:12px; font-weight:bold; cursor:pointer; border-radius:3px; }
.mode-btn.active { background:#003366; color:#fff; border-color:#003366; }
.notes { background:#f9f9f9; border:1px solid #ccc; border-radius:4px; padding:10px 14px; margin-top:14px; font-size:11px; color:#444; }
.notes ul { padding-left:18px; line-height:1.8; }
.foot { text-align:center; font-size:11px; color:#888; padding:20px; border-top:1px solid #ddd; margin-top:20px; }
th.sort-asc::after { content:" ↑"; color:#ffdd44; }
th.sort-desc::after { content:" ↓"; color:#ffdd44; }
"""

    HTML = (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
        "<meta charset=\"UTF-8\">\n<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n"
        "<title>ABL Analytics Dashboard</title>\n"
        f"<style>{CSS}</style>\n</head>\n<body>\n"
        f'<div class="topbar"><span class="topbar-title">ABL Analytics Dashboard</span>'
        f'<span class="topbar-ts">Updated: {now} &middot; HCA {signed(hca,2)} pts &middot; &lambda;={current_lambda} &middot; {pct_done}% complete</span></div>\n'
        '<div class="tabnav">\n'
        '  <button class="tab active" data-tab="standings" onclick="showTab(\'standings\')">Standings</button>\n'
        '  <button class="tab" data-tab="bpm" onclick="showTab(\'bpm\')">BPM Leaders</button>\n'
        '  <button class="tab" data-tab="obpm" onclick="showTab(\'obpm\')">OBPM Leaders</button>\n'
        '  <button class="tab" data-tab="dbpm" onclick="showTab(\'dbpm\')">DBPM Leaders</button>\n'
        '  <button class="tab" data-tab="vorp" onclick="showTab(\'vorp\')">VORP Leaders</button>\n'
        '  <button class="tab" data-tab="roster" onclick="showTab(\'roster\')">Roster Explorer</button>\n'
        '  <button class="tab" data-tab="trade" onclick="showTab(\'trade\')">Trade Analyzer</button>\n'
        '  <button class="tab" data-tab="gamelog" onclick="showTab(\'gamelog\')">Game Log</button>\n'
        '  <button class="tab" data-tab="matchup" onclick="showTab(\'matchup\')">Matchup Analyzer</button>\n'
        '  <button class="tab" data-tab="glossary" onclick="showTab(\'glossary\')">Glossary</button>\n'
        '</div>\n'
    )

    # Standings page
    HTML += (
        '<div class="page active" id="page-standings">\n'
        f'<h1>Power Ratings</h1>\n'
        f'<p class="page-sub">Ridge regression &middot; {pct_done}% complete &middot; {N_SIMS:,} simulations &middot; Click columns to sort</p>\n'
        '<table id="teamTable"><thead><tr>\n'
        '<th class="th-team" onclick="sortTable(\'teamTable\',0,\'str\')">Team</th>\n'
        '<th class="th-num" onclick="sortTable(\'teamTable\',1,\'rec\')">Record</th>\n'
        '<th class="th-num" onclick="sortTable(\'teamTable\',2,\'num\')">MOV</th>\n'
        '<th class="th-num" onclick="sortTable(\'teamTable\',3,\'num\')">Rating</th>\n'
        '<th class="th-num" onclick="sortTable(\'teamTable\',4,\'num\')">SOS</th>\n'
        '<th class="th-num" onclick="sortTable(\'teamTable\',5,\'num\')" style="color:#ffdd44">rSOS</th>\n'
        '<th class="th-num" onclick="sortTable(\'teamTable\',6,\'num\')" style="color:#ffdd44">Proj W</th>\n'
        '<th class="th-num" onclick="sortTable(\'teamTable\',7,\'num\')">Playoff %</th>\n'
        f'</tr></thead><tbody>{team_rows}</tbody></table>\n'
        '</div>\n'
    )

    # BPM leader pages
    # BPM leader pages
    leader_specs = [
        ("bpm",  "BPM Leaders",  "Box Plus/Minus - net points per 100 possessions above average",
         ["Pos","OBPM","DBPM","BPM","VORP"], bpm_rows),
        ("obpm", "OBPM Leaders", "Offensive Box Plus/Minus",
         ["Pos","OBPM"], obpm_rows),
        ("dbpm", "DBPM Leaders", "Defensive Box Plus/Minus",
         ["Pos","DBPM"], dbpm_rows),
        ("vorp", "VORP Leaders", "Value Over Replacement Player",
         ["Min","BPM","VORP"], vorp_rows),
    ]
    for tab_id, title, sub, cols, rows in leader_specs:
        q = "'"
        hdrs = "".join(
            '<th class="th-num" onclick="sortTable(' + q + tab_id + 'Table' + q + ',' + str(i+2) + ',' + q + 'num' + q + ')">' + c + '</th>'
            for i, c in enumerate(cols)
        )
        HTML += (
            '<div class="page" id="page-' + tab_id + '">'
            '<h1>' + title + '</h1>'
            '<p class="page-sub">' + sub + '</p>'
            '<table id="' + tab_id + 'Table"><thead><tr>'
            '<th class="th-team" onclick="sortTable(' + q + tab_id + 'Table' + q + ',0,' + q + 'str' + q + ')">Player</th>'
            '<th class="th-num" onclick="sortTable(' + q + tab_id + 'Table' + q + ',1,' + q + 'str' + q + ')">Team</th>'
            + hdrs +
            '</tr></thead><tbody>' + rows + '</tbody></table></div>'
        )

    # Roster Explorer
    HTML += (
        '<div class="page" id="page-roster">\n'
        '<h1>Roster Explorer</h1>\n'
        '<p class="page-sub">Browse BPM 2.0 by team &middot; Filter by minimum minutes</p>\n'
        '<div class="filter-row">\n'
        '<div><label>Team:</label><br>\n'
        f'<select id="teamSel" onchange="filterPlayers()"><option value="">— Select a team —</option>{team_options}</select></div>\n'
        '<div><label>Min minutes: <span id="minLabel">0</span></label><br>\n'
        '<input type="range" id="minSlider" min="0" max="1500" step="50" value="0"\n'
        '  oninput="document.getElementById(\'minLabel\').textContent=this.value; filterPlayers()"></div>\n'
        '</div>\n'
        '<table id="rosterTable"><thead><tr>\n'
        '<th class="th-team" onclick="sortTable(\'rosterTable\',0,\'str\')">Player</th>\n'
        '<th class="th-num" onclick="sortTable(\'rosterTable\',1,\'num\')">GP</th>\n'
        '<th class="th-num" onclick="sortTable(\'rosterTable\',2,\'num\')">Min</th>\n'
        '<th class="th-num" onclick="sortTable(\'rosterTable\',3,\'num\')">PPG</th>\n'
        '<th class="th-num" onclick="sortTable(\'rosterTable\',4,\'num\')">Pos</th>\n'
        '<th class="th-num" onclick="sortTable(\'rosterTable\',5,\'num\')">Role</th>\n'
        '<th class="th-num" onclick="sortTable(\'rosterTable\',6,\'num\')">OBPM</th>\n'
        '<th class="th-num" onclick="sortTable(\'rosterTable\',7,\'num\')">DBPM</th>\n'
        '<th class="th-num" onclick="sortTable(\'rosterTable\',8,\'num\')">BPM</th>\n'
        '<th class="th-num" onclick="sortTable(\'rosterTable\',9,\'num\')">VORP</th>\n'
        '</tr></thead>\n'
        '<tbody id="rosterBody"><tr><td colspan="10" style="text-align:center;padding:20px;color:#999;">Select a team above.</td></tr></tbody>\n'
        '</table></div>\n'
    )

    # Trade Analyzer
    HTML += (
        '<div class="page" id="page-trade">\n'
        '<h1>Trade Analyzer</h1><p class="page-sub">Build a trade and see net BPM impact</p>\n'
        '<div class="trade-grid">\n'
        '<div class="trade-side"><h3>Team A gives up</h3>\n'
        f'<select style="width:100%;margin-bottom:8px;" id="tradeTeamA" onchange="populateTradeDropdown(\'A\')"><option value="">— Select team —</option>{team_options}</select>\n'
        '<select style="width:100%;margin-bottom:8px;" id="tradePlayerA" onchange="addTradePlayer(\'A\')"><option value="">— Add player —</option></select>\n'
        '<div id="chipsA"></div></div>\n'
        '<div class="trade-arrow">&#8644;</div>\n'
        '<div class="trade-side"><h3>Team B gives up</h3>\n'
        f'<select style="width:100%;margin-bottom:8px;" id="tradeTeamB" onchange="populateTradeDropdown(\'B\')"><option value="">— Select team —</option>{team_options}</select>\n'
        '<select style="width:100%;margin-bottom:8px;" id="tradePlayerB" onchange="addTradePlayer(\'B\')"><option value="">— Add player —</option></select>\n'
        '<div id="chipsB"></div></div></div>\n'
        '<button id="tradeBtn" onclick="analyzeTrade()">Analyze Trade</button>\n'
        '<div id="tradeResult" style="display:none;">\n'
        '<div class="trade-result-grid">\n'
        '<div class="trade-result"><h4>Team A net change</h4><div class="trade-delta" id="deltaA">—</div><div class="trade-detail" id="detailA"></div></div>\n'
        '<div class="trade-result"><h4>Team B net change</h4><div class="trade-delta" id="deltaB">—</div><div class="trade-detail" id="detailB"></div></div>\n'
        '</div></div></div>\n'
    )

    # Game Log
    HTML += (
        '<div class="page" id="page-gamelog">\n'
        '<h1>Game Log</h1><p class="page-sub">Single-game BPM &middot; Uses listed position, no shrinkage</p>\n'
        '<div class="filter-row" style="margin-bottom:16px;align-items:flex-start;">\n'
        '<div><label style="display:block;font-size:11px;color:#555;margin-bottom:4px;">VIEW MODE</label>\n'
        '<div style="display:flex;gap:6px;">\n'
        '<button class="mode-btn active" data-mode="player" onclick="setLogMode(\'player\')">By Player</button>\n'
        '<button class="mode-btn" data-mode="team" onclick="setLogMode(\'team\')">By Team</button>\n'
        '<button class="mode-btn" data-mode="game" onclick="setLogMode(\'game\')">By Game</button>\n'
        '</div></div>\n'
        '<div id="playerPickWrap"><label style="display:block;font-size:11px;color:#555;margin-bottom:4px;">PLAYER</label>\n'
        '<div style="position:relative;"><input type="text" id="playerSearch" placeholder="Type player name..." oninput="searchPlayers()">\n'
        '<div id="playerSuggestions" class="suggestion-box"></div></div></div>\n'
        '<div id="teamPickWrap" style="display:none;"><label style="display:block;font-size:11px;color:#555;margin-bottom:4px;">TEAM</label>\n'
        '<select id="teamLogSel" onchange="showTeamLog()" style="min-width:200px;"><option value="">— Select a team —</option></select></div>\n'
        '<div id="gamePickWrap" style="display:none;"><label style="display:block;font-size:11px;color:#555;margin-bottom:4px;">GAME</label>\n'
        '<select id="gameSel" onchange="showGameLog()" style="min-width:320px;"><option value="">— Select a game —</option></select></div>\n'
        '</div>\n'

        '<div id="playerLogView">\n'
        '<canvas id="bpmSparkline" height="70" style="display:none;width:100%;margin-bottom:14px;border:1px solid #eee;"></canvas>\n'
        '<table id="playerLogTable"><thead><tr>\n'
        '<th class="th-team" onclick="sortTable(\'playerLogTable\',0,\'str\')">Game</th>\n'
        '<th class="th-num" onclick="sortTable(\'playerLogTable\',1,\'str\')">Opp</th>\n'
        '<th class="th-num" onclick="sortTable(\'playerLogTable\',2,\'num\')">Min</th>\n'
        '<th class="th-num" onclick="sortTable(\'playerLogTable\',3,\'num\')">Pts</th>\n'
        '<th class="th-num" onclick="sortTable(\'playerLogTable\',4,\'num\')">Reb</th>\n'
        '<th class="th-num" onclick="sortTable(\'playerLogTable\',5,\'num\')">Ast</th>\n'
        '<th class="th-num" onclick="sortTable(\'playerLogTable\',6,\'num\')">+/-</th>\n'
        '<th class="th-num" onclick="sortTable(\'playerLogTable\',7,\'num\')">OBPM</th>\n'
        '<th class="th-num" onclick="sortTable(\'playerLogTable\',8,\'num\')">DBPM</th>\n'
        '<th class="th-num" onclick="sortTable(\'playerLogTable\',9,\'num\')">BPM</th>\n'
        '</tr></thead>\n'
        '<tbody id="playerLogBody"><tr><td colspan="10" style="text-align:center;padding:20px;color:#999;">Search for a player above.</td></tr></tbody>\n'
        '</table></div>\n'

        '<div id="teamLogView" style="display:none;">\n'
        '<table id="teamLogTable"><thead><tr>\n'
        '<th class="th-team">Game</th><th class="th-num">Opp</th><th class="th-num">Result</th>\n'
        '<th class="th-num">Score</th><th class="th-team">Top BPM</th><th class="th-num">BPM</th><th class="th-num">Avg BPM</th>\n'
        '</tr></thead>\n'
        '<tbody id="teamLogBody"><tr><td colspan="7" style="text-align:center;padding:20px;color:#999;">Select a team above.</td></tr></tbody>\n'
        '</table></div>\n'

        '<div id="gameLogView" style="display:none;">\n'
        '<table id="gameLogTable"><thead><tr>\n'
        '<th class="th-team">Player</th><th class="th-num">Team</th>\n'
        '<th class="th-num">Min</th><th class="th-num">Pts</th><th class="th-num">Reb</th><th class="th-num">Ast</th>\n'
        '<th class="th-num">+/-</th><th class="th-num">OBPM</th><th class="th-num">DBPM</th><th class="th-num">BPM</th>\n'
        '</tr></thead>\n'
        '<tbody id="gameLogBody"><tr><td colspan="10" style="text-align:center;padding:20px;color:#999;">Select a game above.</td></tr></tbody>\n'
        '</table></div></div>\n'
    )

    # Matchup Analyzer
    HTML += (
        '<div class="page" id="page-matchup">\n'
        '<h1>Matchup Analyzer</h1><p class="page-sub">Head-to-head history between two teams</p>\n'
        '<div class="filter-row" style="align-items:flex-end;margin-bottom:16px;">\n'
        '<div><label style="display:block;font-size:11px;color:#555;margin-bottom:4px;">TEAM A</label>\n'
        '<select id="matchupTeamA" onchange="runMatchup()" style="min-width:180px;"><option value="">— Select team —</option></select></div>\n'
        '<div style="font-size:18px;color:#999;padding-bottom:4px;">vs</div>\n'
        '<div><label style="display:block;font-size:11px;color:#555;margin-bottom:4px;">TEAM B</label>\n'
        '<select id="matchupTeamB" onchange="runMatchup()" style="min-width:180px;"><option value="">— Select team —</option></select></div>\n'
        '</div>\n'
        '<div id="matchupResult" style="display:none;">\n'
        '<div id="matchupSummary" style="margin-bottom:16px;"></div>\n'
        '<canvas id="matchupChart" height="90" style="width:100%;margin-bottom:16px;border:1px solid #eee;"></canvas>\n'
        '<div class="section-title">Game Results <span id="matchupGameCount" style="font-weight:normal;font-size:12px;color:#555;"></span></div>\n'
        '<table id="matchupGamesTable" style="margin-bottom:20px;"><thead><tr>\n'
        '<th class="th-team">Game</th><th class="th-team">Home</th><th class="th-num">Score</th>\n'
        '<th class="th-team">Away</th><th class="th-num">Margin</th><th class="th-team">Winner</th>\n'
        '</tr></thead><tbody id="matchupGamesBody"></tbody></table>\n'
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;">\n'
        '<div><div class="section-title"><span id="matchupLabelA">Team A</span> — Top Performers</div>\n'
        '<table id="matchupPlayersA"><thead><tr>\n'
        '<th class="th-team">Player</th><th class="th-num">GP</th><th class="th-num">Avg Pts</th><th class="th-num">Avg BPM</th>\n'
        '</tr></thead><tbody id="matchupPlayersBodyA"></tbody></table></div>\n'
        '<div><div class="section-title"><span id="matchupLabelB">Team B</span> — Top Performers</div>\n'
        '<table id="matchupPlayersB"><thead><tr>\n'
        '<th class="th-team">Player</th><th class="th-num">GP</th><th class="th-num">Avg Pts</th><th class="th-num">Avg BPM</th>\n'
        '</tr></thead><tbody id="matchupPlayersBodyB"></tbody></table></div>\n'
        '</div></div>\n'
        '<div id="matchupEmpty" style="color:#999;padding:20px 0;">Select two teams above.</div>\n'
        '</div>\n'
    )

    # Glossary
    HTML += (
        '<div class="page" id="page-glossary">\n'
        '<h1>Glossary</h1><p class="page-sub">Stat definitions</p>\n'
        '<table><thead><tr>\n'
        '<th class="th-team" style="width:160px;">Term</th>\n'
        '<th class="th-team">Definition</th>\n'
        '<th class="th-team">Scale</th>\n'
        '</tr></thead><tbody>\n'
        '<tr><td class="td-team">Rating</td><td>Adjusted power rating from ridge regression. 0 = league average.</td><td>0 = avg</td></tr>\n'
        '<tr><td class="td-team">MOV</td><td>Raw average point differential per game.</td><td></td></tr>\n'
        '<tr><td class="td-team">SOS</td><td>Average adjusted rating of opponents played.</td><td>0 = avg</td></tr>\n'
        '<tr><td class="td-team">rSOS</td><td>Average adjusted rating of remaining opponents.</td><td>0 = avg</td></tr>\n'
        f'<tr><td class="td-team">Proj W</td><td>Average final wins across {N_SIMS:,} simulations of the real remaining schedule.</td><td></td></tr>\n'
        '<tr><td class="td-team">BPM</td><td>Box Plus/Minus. Net points per 100 possessions above average. BPM 2.0 with position and role interpolation.</td><td>+8 MVP &middot; 0 avg &middot; -2 replacement</td></tr>\n'
        '<tr><td class="td-team">OBPM</td><td>Offensive component of BPM.</td><td>+4 elite &middot; 0 avg</td></tr>\n'
        '<tr><td class="td-team">DBPM</td><td>Defensive component (BPM minus OBPM).</td><td>+2 elite &middot; 0 avg</td></tr>\n'
        '<tr><td class="td-team">VORP</td><td>Value Over Replacement Player. Total value above a -2.0 BPM player, scaled by minutes.</td><td>5+ MVP &middot; 3+ All-Star</td></tr>\n'
        '<tr><td class="td-team">Pos</td><td>Estimated position 1-5 from box score shares. 1=PG, 5=C.</td><td></td></tr>\n'
        '<tr><td class="td-team">Role</td><td>Offensive role 1-5. 1=creator, 5=receiver.</td><td></td></tr>\n'
        '</tbody></table></div>\n'
        f'<p class="foot">ABL Analytics &middot; BPM 2.0 &middot; {N_SIMS:,} simulations &middot; Updated {now}</p>\n'
    )

    # JavaScript
    HTML += (
        "<script>\n"
        "const ALL=" + all_json + ";\n"
        "const GAME_LOG=" + gl_json + ";\n"
        "const GAME_SUMMARIES=" + gs_json + ";\n"
    )

    HTML += """
function showTab(name){
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
  document.querySelector('.tab[data-tab="'+name+'"]').classList.add('active');
  document.getElementById('page-'+name).classList.add('active');
}
const sortState={};
function sortTable(id,col,type){
  const tbl=document.getElementById(id),body=tbl.querySelector('tbody');
  const rows=Array.from(body.querySelectorAll('tr'));
  const ths=tbl.querySelectorAll('thead th');
  const key=id+'_'+col,asc=sortState[key]!=='asc';
  sortState[key]=asc?'asc':'desc';
  ths.forEach(t=>t.classList.remove('sort-asc','sort-desc'));
  if(ths[col])ths[col].classList.add(asc?'sort-asc':'sort-desc');
  rows.sort((a,b)=>{
    const av=a.cells[col]?.textContent.trim()??'';
    const bv=b.cells[col]?.textContent.trim()??'';
    if(type==='num'){const an=parseFloat(av.replace(/[+%,]/g,''))||0,bn=parseFloat(bv.replace(/[+%,]/g,''))||0;return asc?an-bn:bn-an;}
    if(type==='rec'){return asc?(parseInt(av)||0)-(parseInt(bv)||0):(parseInt(bv)||0)-(parseInt(av)||0);}
    return asc?av.localeCompare(bv):bv.localeCompare(av);
  });
  rows.forEach(r=>body.appendChild(r));
}
function filterPlayers(){
  const team=document.getElementById('teamSel').value;
  const minMin=parseInt(document.getElementById('minSlider').value)||0;
  const body=document.getElementById('rosterBody');
  if(!team){body.innerHTML='<tr><td colspan="10" style="text-align:center;padding:20px;color:#999;">Select a team above.</td></tr>';return;}
  const rows=ALL.filter(p=>p.team.toLowerCase().includes(team.toLowerCase())&&p.min>=minMin).sort((a,b)=>b.bpm-a.bpm);
  if(!rows.length){body.innerHTML='<tr><td colspan="10" style="text-align:center;padding:20px;color:#999;">No players match.</td></tr>';return;}
  const s=(v,d=2)=>(v>=0?'+':'')+v.toFixed(d);
  const rc=v=>v<0?' c-neg':' c-pos';
  body.innerHTML=rows.map(p=>`<tr>
    <td class="td-team">${p.name}</td><td class="td-num">${p.gp}</td><td class="td-num">${Math.round(p.min)}</td>
    <td class="td-num">${p.ppg}</td><td class="td-num">${p.pos.toFixed(1)}</td><td class="td-num">${p.role.toFixed(1)}</td>
    <td class="td-num${rc(p.obpm)}">${s(p.obpm)}</td><td class="td-num${rc(p.dbpm)}">${s(p.dbpm)}</td>
    <td class="td-num${rc(p.bpm)}">${s(p.bpm)}</td><td class="td-num${rc(p.vorp)}">${s(p.vorp)}</td>
  </tr>`).join('');
}
const tradePicks={A:[],B:[]};
function populateTradeDropdown(side){
  const team=document.getElementById('tradeTeam'+side).value;
  const sel=document.getElementById('tradePlayer'+side);
  sel.innerHTML='<option value="">— Add player —</option>';
  if(!team)return;
  ALL.filter(p=>p.team.toLowerCase().includes(team.toLowerCase())).sort((a,b)=>b.bpm-a.bpm)
    .forEach(p=>{const o=document.createElement('option');o.value=p.name;
      o.textContent=`${p.name} (BPM ${p.bpm>=0?'+':''}${p.bpm.toFixed(2)})`;sel.appendChild(o);});
}
function addTradePlayer(side){const sel=document.getElementById('tradePlayer'+side);const name=sel.value;
  if(!name||tradePicks[side].includes(name)){sel.value='';return;}
  tradePicks[side].push(name);renderChips(side);sel.value='';}
function removePlayer(side,name){tradePicks[side]=tradePicks[side].filter(n=>n!==name);renderChips(side);}
function renderChips(side){document.getElementById('chips'+side).innerHTML=
  tradePicks[side].map(n=>`<span class="player-chip" onclick="removePlayer('${side}','${n}')">${n} ×</span>`).join('');}
function analyzeTrade(){
  const s=v=>(v>=0?'+':'')+v.toFixed(2);
  const delta=(gives,gets)=>gets.reduce((sum,n)=>{const p=ALL.find(x=>x.name===n);return sum+(p?p.bpm:0);},0)
    -gives.reduce((sum,n)=>{const p=ALL.find(x=>x.name===n);return sum+(p?p.bpm:0);},0);
  const dA=delta(tradePicks.A,tradePicks.B),dB=delta(tradePicks.B,tradePicks.A);
  const render=(elD,elDet,d,gives,gets)=>{
    elD.textContent=(d>=0?'+':'')+d.toFixed(2)+' BPM';
    elD.className='trade-delta '+(d>0.5?'pos':d<-0.5?'neg':'neu');
    const fmt=(names,lbl)=>names.length?`<b>${lbl}:</b> `+names.map(n=>{const p=ALL.find(x=>x.name===n);
      return `${n} (${p?s(p.bpm):'?'} BPM)`;}).join(', '):'';
    elDet.innerHTML=[fmt(gives,'Gives up'),fmt(gets,'Receives')].filter(Boolean).join('<br>');
  };
  render(document.getElementById('deltaA'),document.getElementById('detailA'),dA,tradePicks.A,tradePicks.B);
  render(document.getElementById('deltaB'),document.getElementById('detailB'),dB,tradePicks.B,tradePicks.A);
  document.getElementById('tradeResult').style.display='block';
}
(function(){
  const gSel=document.getElementById('gameSel');
  GAME_SUMMARIES.forEach(g=>{const o=document.createElement('option');o.value=g.id;o.textContent=g.label;gSel.appendChild(o);});
  const tSel=document.getElementById('teamLogSel');
  const teams=[...new Set(GAME_SUMMARIES.flatMap(g=>[g.home,g.away]))].sort();
  teams.forEach(t=>{const o=document.createElement('option');o.value=t;o.textContent=t;tSel.appendChild(o);});
  ['matchupTeamA','matchupTeamB'].forEach(id=>{const sel=document.getElementById(id);
    teams.forEach(t=>{const o=document.createElement('option');o.value=t;o.textContent=t;sel.appendChild(o);});});
})();
function setLogMode(mode){
  document.querySelectorAll('.mode-btn[data-mode]').forEach(b=>b.classList.toggle('active',b.dataset.mode===mode));
  document.getElementById('playerPickWrap').style.display=mode==='player'?'':'none';
  document.getElementById('teamPickWrap').style.display=mode==='team'?'':'none';
  document.getElementById('gamePickWrap').style.display=mode==='game'?'':'none';
  document.getElementById('playerLogView').style.display=mode==='player'?'':'none';
  document.getElementById('teamLogView').style.display=mode==='team'?'':'none';
  document.getElementById('gameLogView').style.display=mode==='game'?'':'none';
  document.getElementById('bpmSparkline').style.display='none';
}
function searchPlayers(){
  const q=document.getElementById('playerSearch').value.toLowerCase().trim();
  const box=document.getElementById('playerSuggestions');box.innerHTML='';
  if(q.length<2)return;
  Object.keys(GAME_LOG).filter(n=>n.toLowerCase().includes(q)).slice(0,8).forEach(name=>{
    const d=document.createElement('div');d.className='suggestion-item';d.textContent=name;
    d.onclick=()=>{selectPlayer(name);box.innerHTML='';};box.appendChild(d);});
}
function selectPlayer(name){
  document.getElementById('playerSearch').value=name;
  const games=GAME_LOG[name]||[];
  const s=(v,d=2)=>(v>=0?'+':'')+v.toFixed(d);
  const rc=v=>v<0?' c-neg':' c-pos';
  const body=document.getElementById('playerLogBody');
  if(!games.length){body.innerHTML='<tr><td colspan="10" style="text-align:center;padding:20px;color:#999;">No games found.</td></tr>';return;}
  const avg=key=>games.reduce((a,g)=>a+g[key],0)/games.length;
  body.innerHTML=games.map(g=>`<tr>
    <td class="td-team">${g.label}</td><td class="td-num">${g.opp}</td>
    <td class="td-num">${g.min}</td><td class="td-num">${g.pts}</td>
    <td class="td-num">${g.trb}</td><td class="td-num">${g.ast}</td>
    <td class="td-num">${g.pm>=0?'+':''}${g.pm}</td>
    <td class="td-num${rc(g.obpm)}">${s(g.obpm)}</td>
    <td class="td-num${rc(g.dbpm)}">${s(g.dbpm)}</td>
    <td class="td-num${rc(g.bpm)}">${s(g.bpm)}</td>
  </tr>`).join('')+`<tr class="avg-row">
    <td class="td-team">Season Avg</td><td class="td-num"></td>
    <td class="td-num">${avg('min').toFixed(0)}</td><td class="td-num">${avg('pts').toFixed(1)}</td>
    <td class="td-num">${avg('trb').toFixed(1)}</td><td class="td-num">${avg('ast').toFixed(1)}</td>
    <td class="td-num">${avg('pm')>=0?'+':''}${avg('pm').toFixed(1)}</td>
    <td class="td-num${rc(avg('obpm'))}">${s(avg('obpm'))}</td>
    <td class="td-num${rc(avg('dbpm'))}">${s(avg('dbpm'))}</td>
    <td class="td-num${rc(avg('bpm'))}">${s(avg('bpm'))}</td>
  </tr>`;
  drawSparkline(games.map(g=>g.bpm));
}
function showTeamLog(){
  const team=document.getElementById('teamLogSel').value;
  const body=document.getElementById('teamLogBody');
  if(!team){body.innerHTML='<tr><td colspan="7" style="text-align:center;padding:20px;color:#999;">Select a team above.</td></tr>';return;}
  const games=GAME_SUMMARIES.filter(g=>g.home===team||g.away===team);
  if(!games.length){body.innerHTML='<tr><td colspan="7" style="text-align:center;padding:20px;color:#999;">No games yet.</td></tr>';return;}
  const s=(v,d=2)=>(v>=0?'+':'')+v.toFixed(d);
  const rc=v=>v<0?' c-neg':' c-pos';
  body.innerHTML=games.map(g=>{
    const isHome=g.home===team,opp=isHome?g.away:g.home;
    const ts=isHome?g.home_score:g.away_score,os=isHome?g.away_score:g.home_score,won=ts>os;
    const pl=[];
    Object.entries(GAME_LOG).forEach(([pname,pGames])=>{pGames.forEach(pg=>{if(pg.game_id===g.id&&pg.team===team)pl.push({...pg,player:pname});});});
    const top=pl.sort((a,b)=>b.bpm-a.bpm)[0];
    const avg=pl.length?pl.reduce((a,p)=>a+p.bpm,0)/pl.length:null;
    return `<tr>
      <td class="td-team">${g.label}</td><td class="td-num">${opp}</td>
      <td class="td-num" style="color:${won?'#006600':'#cc0000'};font-weight:bold;">${won?'W':'L'}</td>
      <td class="td-num">${ts}-${os}</td>
      <td class="td-team">${top?top.player:'—'}</td>
      <td class="td-num${top?rc(top.bpm):''}">${top?s(top.bpm):'—'}</td>
      <td class="td-num${avg!==null?rc(avg):''}">${avg!==null?s(avg):'—'}</td>
    </tr>`;
  }).join('');
}
function showGameLog(){
  const gameId=document.getElementById('gameSel').value;
  const body=document.getElementById('gameLogBody');
  if(!gameId){body.innerHTML='<tr><td colspan="10" style="text-align:center;padding:20px;color:#999;">Select a game above.</td></tr>';return;}
  const lines=[],seen=new Set();
  Object.entries(GAME_LOG).forEach(([pname,pGames])=>{pGames.forEach(g=>{
    if(g.game_id===gameId&&!seen.has(pname)){seen.add(pname);lines.push({...g,player:pname});}});});
  lines.sort((a,b)=>b.bpm-a.bpm);
  const s=(v,d=2)=>(v>=0?'+':'')+v.toFixed(d);
  const rc=v=>v<0?' c-neg':' c-pos';
  if(!lines.length){body.innerHTML='<tr><td colspan="10" style="text-align:center;padding:20px;color:#999;">No data yet.</td></tr>';return;}
  body.innerHTML=lines.map(g=>`<tr>
    <td class="td-team">${g.player}</td><td class="td-num">${g.team}</td>
    <td class="td-num">${g.min}</td><td class="td-num">${g.pts}</td>
    <td class="td-num">${g.trb}</td><td class="td-num">${g.ast}</td>
    <td class="td-num">${g.pm>=0?'+':''}${g.pm}</td>
    <td class="td-num${rc(g.obpm)}">${s(g.obpm)}</td>
    <td class="td-num${rc(g.dbpm)}">${s(g.dbpm)}</td>
    <td class="td-num${rc(g.bpm)}">${s(g.bpm)}</td>
  </tr>`).join('');
}
function drawSparkline(values,canvasId='bpmSparkline'){
  const canvas=document.getElementById(canvasId);
  canvas.style.display='block';canvas.width=canvas.parentElement.offsetWidth||800;
  const ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.fillStyle='#f9f9f9';ctx.fillRect(0,0,canvas.width,canvas.height);
  const pad=20,w=canvas.width-pad*2,h=canvas.height-pad*2;
  const mn=Math.min(...values,-2),mx=Math.max(...values,2),rng=mx-mn||1;
  const x=i=>pad+(i/(values.length-1||1))*w;
  const y=v=>pad+h-((v-mn)/rng)*h;
  ctx.strokeStyle='#ddd';ctx.lineWidth=1;ctx.setLineDash([4,4]);
  ctx.beginPath();ctx.moveTo(pad,y(0));ctx.lineTo(pad+w,y(0));ctx.stroke();ctx.setLineDash([]);
  ctx.strokeStyle='#003366';ctx.lineWidth=2;ctx.beginPath();
  values.forEach((v,i)=>i===0?ctx.moveTo(x(i),y(v)):ctx.lineTo(x(i),y(v)));ctx.stroke();
  values.forEach((v,i)=>{ctx.fillStyle=v>=0?'#006600':'#cc0000';
    ctx.beginPath();ctx.arc(x(i),y(v),4,0,Math.PI*2);ctx.fill();});
}
function runMatchup(){
  const teamA=document.getElementById('matchupTeamA').value;
  const teamB=document.getElementById('matchupTeamB').value;
  const result=document.getElementById('matchupResult'),empty=document.getElementById('matchupEmpty');
  if(!teamA||!teamB||teamA===teamB){result.style.display='none';empty.style.display='block';
    empty.textContent=teamA===teamB?'Select two different teams.':'Select two teams above.';return;}
  const h2h=GAME_SUMMARIES.filter(g=>(g.home===teamA&&g.away===teamB)||(g.home===teamB&&g.away===teamA));
  if(!h2h.length){result.style.display='none';empty.style.display='block';
    empty.textContent=`No games between ${teamA} and ${teamB} yet.`;return;}
  result.style.display='block';empty.style.display='none';
  document.getElementById('matchupLabelA').textContent=teamA;
  document.getElementById('matchupLabelB').textContent=teamB;
  document.getElementById('matchupGameCount').textContent=`(${h2h.length} game${h2h.length>1?'s':''})`;
  const s=(v,d=2)=>(v>=0?'+':'')+v.toFixed(d);
  const rc=v=>v<0?' c-neg':' c-pos';
  let wA=0,wB=0;const margins=[];
  const gameRows=h2h.map(g=>{
    const aScore=g.home===teamA?g.home_score:g.away_score;
    const bScore=g.home===teamA?g.away_score:g.home_score;
    const margin=aScore-bScore;margins.push(margin);
    const winner=margin>0?teamA:teamB;
    if(margin>0)wA++;else wB++;
    return{g,aScore,bScore,margin,winner};
  });
  const pctA=Math.round(wA/h2h.length*100);
  document.getElementById('matchupSummary').innerHTML=
    `<strong>${teamA} ${wA} — ${wB} ${teamB}</strong> &nbsp; series &nbsp;
     <span style="color:#555;font-size:12px;">Avg margin: ${teamA} ${s(margins.reduce((a,m)=>a+m,0)/margins.length,1)} per game</span>
     <div style="background:#e8e8e8;border-radius:3px;height:8px;overflow:hidden;max-width:300px;margin-top:6px;">
       <div style="background:#003366;height:100%;width:${pctA}%;"></div></div>`;
  drawMatchupChart(margins,teamA,teamB);
  document.getElementById('matchupGamesBody').innerHTML=gameRows.map(({g,aScore,bScore,margin,winner})=>`<tr>
    <td class="td-team">${g.label}</td><td class="td-team">${g.home}</td>
    <td class="td-num">${g.home_score}–${g.away_score}</td><td class="td-team">${g.away}</td>
    <td class="td-num" style="color:${margin>0?'#006600':'#cc0000'};font-weight:bold;">${s(margin,0)}</td>
    <td class="td-team">${winner}</td>
  </tr>`).join('');
  const psA={},psB={};
  h2h.forEach(g=>{Object.entries(GAME_LOG).forEach(([pname,pGames])=>{pGames.forEach(pg=>{
    if(pg.game_id!==g.id)return;
    const bucket=pg.team===teamA?psA:pg.team===teamB?psB:null;
    if(!bucket)return;
    if(!bucket[pname])bucket[pname]={gp:0,pts:0,bpm:0};
    bucket[pname].gp++;bucket[pname].pts+=pg.pts;bucket[pname].bpm+=pg.bpm;
  });});});
  const renderP=(stats,bodyId)=>{
    const rows=Object.entries(stats).map(([name,d])=>({name,gp:d.gp,avgPts:d.pts/d.gp,avgBpm:d.bpm/d.gp}))
      .sort((a,b)=>b.avgBpm-a.avgBpm).slice(0,8);
    document.getElementById(bodyId).innerHTML=rows.map(r=>`<tr>
      <td class="td-team">${r.name}</td><td class="td-num">${r.gp}</td>
      <td class="td-num">${r.avgPts.toFixed(1)}</td>
      <td class="td-num${rc(r.avgBpm)}">${s(r.avgBpm)}</td>
    </tr>`).join('')||'<tr><td colspan="4" style="text-align:center;padding:10px;color:#999;">No data</td></tr>';
  };
  renderP(psA,'matchupPlayersBodyA');renderP(psB,'matchupPlayersBodyB');
}
function drawMatchupChart(margins,teamA,teamB){
  const canvas=document.getElementById('matchupChart');
  canvas.width=canvas.parentElement.offsetWidth||800;
  const ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.fillStyle='#f9f9f9';ctx.fillRect(0,0,canvas.width,canvas.height);
  const pad=24,w=canvas.width-pad*2,h=canvas.height-pad*2;
  const mx=Math.max(...margins.map(Math.abs),5);
  const x=i=>pad+(i/(margins.length-1||1))*w;
  const y=v=>pad+h/2-(v/mx)*(h/2);
  ctx.strokeStyle='#ddd';ctx.lineWidth=1;ctx.setLineDash([4,4]);
  ctx.beginPath();ctx.moveTo(pad,y(0));ctx.lineTo(pad+w,y(0));ctx.stroke();ctx.setLineDash([]);
  ctx.font='10px Arial';ctx.fillStyle='#999';
  ctx.fillText(teamA+' ahead',pad+4,y(0)-4);
  ctx.fillText(teamB+' ahead',pad+4,y(0)+12);
  ctx.beginPath();ctx.moveTo(x(0),y(0));
  margins.forEach((v,i)=>ctx.lineTo(x(i),y(v)));
  ctx.lineTo(x(margins.length-1),y(0));ctx.closePath();
  ctx.fillStyle='rgba(0,51,102,0.08)';ctx.fill();
  ctx.strokeStyle='#003366';ctx.lineWidth=2;ctx.beginPath();
  margins.forEach((v,i)=>i===0?ctx.moveTo(x(i),y(v)):ctx.lineTo(x(i),y(v)));ctx.stroke();
  margins.forEach((v,i)=>{ctx.fillStyle=v>0?'#006600':'#cc0000';
    ctx.beginPath();ctx.arc(x(i),y(v),4,0,Math.PI*2);ctx.fill();});
}
</script>
</body></html>
"""

    with open("dashboard.html", "w", encoding="utf-8") as f:
        f.write(HTML)
    print("  dashboard.html written.")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t0 = time.time()
    print("ABL Analytics — BPM 2.0 + Full Suite")
    print("=" * 50)
    scraped_teams            = scrape_team_leaders()
    teams_map, hca, lam, played, unplayed = solve_team_ratings()
    print("\n[3/5] Updating game log...")
    game_log                 = update_game_log(played)
    players_list             = gather_player_stats(teams_map, scraped_teams)
    sim_results              = simulate_season(teams_map, hca, unplayed) if teams_map else {}
    if teams_map:
        write_dashboard_html(teams_map, players_list, hca, lam, sim_results, game_log)
    else:
        print("\nNo team data — dashboard not written.")
    print(f"\nDone in {time.time()-t0:.1f}s")
