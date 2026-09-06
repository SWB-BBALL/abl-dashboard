"""
gold_standings.py — Gold Drafting standings for the ABL

Based on Adam Gold's system (via Micah Blake McCurdy / HockeyViz):
  - Teams accumulate Gold Points ONLY after they are eliminated from playoff
    contention (or declare themselves eliminated)
  - Gold Points = regular standings points earned after elimination
  - Draft order: most Gold Points picks first (no lottery)
  - Weaker teams benefit because they are eliminated earlier = longer runway

How it works with this league:
  - Reads game results from game_log.json (produced by abl_analytics.py)
  - Also accepts the schedule from ascension-basketball.com to model
    remaining games and compute playoff elimination
  - Tracks which teams have declared themselves eliminated (manual override)
  - Saves state to gold_state.json so declarations persist across runs
  - Outputs a dashboard HTML: gold_dashboard.html

Standings points (standard):
  Win  = 2 pts
  Loss = 0 pts
  (No OT/overtime in basketball so no 1-pt consolation)

Playoff spots: top 16 of 30 teams make playoffs (same cutoff used in sim)
"""

import os
import re
import json
import time
import random
import requests
import numpy as np
from datetime import datetime
from bs4 import BeautifulSoup

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL       = "https://ascension-basketball.com/"
HEADERS        = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept":     "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}
GAME_LOG_FILE  = "game_log.json"
GOLD_STATE_FILE = "gold_state.json"
TOTAL_GAMES    = 82
PLAYOFF_SPOTS  = 16          # top 16 of 30
ALLOW_DECLARATIONS = False       # set True to let teams voluntarily declare elimination early
ELIM_THRESHOLD   = 1e-6    # hard mathematical elimination for Gold clock purposes
DISPLAY_ELIM_PCT = 0.005   # show X in playoff picture below this (0.5%) — not actual elimination
SIM_N          = 50_000      # sims for elimination check
SPREAD         = 11.0        # game-level std dev for sim
ALLSTAR_TEAMS  = {'east', 'west'}    # team names to treat as exhibition — skip entirely
MIN_DELAY      = 1.5

# ── League structure ──────────────────────────────────────────────────────────
# 2 conferences × 3 divisions × 5 teams = 30 teams
# Playoff format:
#   - Each conference sends 8 teams to playoffs (16 total)
#   - The 3 division winners get seeds 1-3 within their conference
#     (ordered by record among themselves)
#   - Remaining 5 spots per conference go to the next-best records
#     regardless of division
#   - Tiebreaker: win % (same as wins in equal-GP league), then head-to-head
#     (approximated here as wins since we don't track H2H separately)

CONFERENCES = {
    'East': {
        'Atlantic':  ['Boston','New Jersey','New York','Philadelphia','Toronto'],
        'Southeast': ['Atlanta','Washington','Miami','Orlando','Charlotte'],
        'Central':   ['Chicago','Cleveland','Detroit','Indiana','Milwaukee'],
    },
    'West': {
        'Midwest':   ['Houston','San Antonio','Dallas','New Orleans','Memphis'],
        'Northwest': ['Denver','Seattle','Portland','Utah','Minnesota'],
        'Pacific':   ['Golden State','Los Angeles (LAC)','Los Angeles (LAL)','Phoenix','Sacramento'],
    },
}

# Flat lookups built from CONFERENCES
TEAM_CONFERENCE = {}
TEAM_DIVISION   = {}
for conf, divs in CONFERENCES.items():
    for div, teams in divs.items():
        for t in teams:
            TEAM_CONFERENCE[t] = conf
            TEAM_DIVISION[t]   = div

PLAYOFF_SPOTS_PER_CONF = 8   # 8 per conference = 16 total
DIV_WINNERS_PER_CONF   = 3   # one per division, seeds 1-3
WILDCARD_SPOTS         = PLAYOFF_SPOTS_PER_CONF - DIV_WINNERS_PER_CONF  # 5 wildcard spots
MAX_DELAY      = 3.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def polite_get(url, retries=3):
    time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 200:
                return r
            elif r.status_code == 429:
                wait = (attempt + 1) * 10
                print(f"  Rate limited — waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  HTTP {r.status_code} for {url}")
                return None
        except requests.exceptions.RequestException as e:
            wait = (attempt + 1) * 5
            print(f"  Request error: {e} — retry {attempt+1}/{retries} in {wait}s")
            time.sleep(wait)
    return None


def parse_game_text(text):
    """@ = home team."""
    parts = [p.strip() for p in text.strip().split(',')]
    if len(parts) != 2:
        return None
    m1 = re.search(r'^(@?)(.*?)\s+(\d+)$', parts[0])
    m2 = re.search(r'^(@?)(.*?)\s+(\d+)$', parts[1])
    if not m1 or not m2:
        return None
    at1, n1, s1 = m1.group(1), m1.group(2).strip(), int(m1.group(3))
    at2, n2, s2 = m2.group(1), m2.group(2).strip(), int(m2.group(3))
    if at1 == '@':
        return n1, s1, n2, s2
    elif at2 == '@':
        return n2, s2, n1, s1
    return None



# ── Load / save Gold state ────────────────────────────────────────────────────

def load_gold_state():
    """
    State file stores manual declarations and elimination dates.
    {
      "declarations": ["TeamA", "TeamB"],   # teams that declared themselves out
      "eliminated_after": {"TeamA": "box15-3"}  # box_id after which they were eliminated
    }
    """
    if os.path.exists(GOLD_STATE_FILE):
        with open(GOLD_STATE_FILE) as f:
            return json.load(f)
    return {"declarations": [], "eliminated_after": {}}


def save_gold_state(state):
    with open(GOLD_STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)
    print(f"  Gold state saved to {GOLD_STATE_FILE}")


# ── Fetch schedule ────────────────────────────────────────────────────────────

def fetch_schedule():
    """Returns (played_games, unplayed_games) lists."""
    print("Fetching schedule...")
    r = polite_get(f"{BASE_URL}schedule.htm")
    if not r:
        print("  Could not reach schedule.htm")
        return [], []

    soup = BeautifulSoup(r.text, 'html.parser')
    played, unplayed = [], []
    in_reg = False
    current_date = ''

    for td in soup.find_all('td'):
        td_class   = ' '.join(td.get('class', [])).lower()
        text_lower = td.get_text(separator=' ').strip().lower()
        text_raw   = td.get_text(separator=' ').strip()

        if 'tableheader' in td_class:
            if 'regular season' in text_lower:
                in_reg = True
            elif 'preseason' in text_lower:
                # If we've already seen regular season, a new preseason means
                # we've looped into next season — stop entirely
                if in_reg:
                    break
                in_reg = False
            elif any(k in text_lower for k in ('playoff','postseason','first round','semifinals','finals')):
                # Stop collecting regular season games once playoffs start
                in_reg = False
                break
            continue

        # Date header rows — class=header, contains YYYY-MM-DD
        if 'header' in td_class and 'tableheader' not in td_class and 'main' not in td_class:
            date_match = re.search(r'\d{4}-\d{2}-\d{2}', text_raw)
            if date_match:
                try:
                    from datetime import datetime as _dt
                    current_date = _dt.strptime(date_match.group(), '%Y-%m-%d').strftime('%B %d, %Y')
                except Exception:
                    current_date = date_match.group()
            continue

        if not in_reg or 'main' not in td_class:
            continue

        links     = td.find_all('a', href=True)
        box_links = [l for l in links if 'box'    in l['href'].lower()]
        ros_links = [l for l in links if 'roster' in l['href'].lower()]

        if box_links:
            parsed = parse_game_text(box_links[0].get_text(separator=' ').strip())
            if parsed:
                home, h_score, away, a_score = parsed
                # Skip all-star / exhibition games (East vs West)
                if home.lower() in ALLSTAR_TEAMS or away.lower() in ALLSTAR_TEAMS:
                    continue
                raw_href = box_links[0]['href'].replace('./','').replace('../','').lstrip('/')
                played.append({
                    'home': home, 'away': away,
                    'home_score': h_score, 'away_score': a_score,
                    'margin': h_score - a_score,
                    'box_id': raw_href.replace('boxes/','').replace('.htm',''),
                    'date':   current_date,
                })
        elif len(ros_links) == 2:
            away_name = ros_links[0].get_text().strip()
            home_name = ros_links[1].get_text().strip()
            # Skip all-star / exhibition games
            if away_name.lower() in ALLSTAR_TEAMS or home_name.lower() in ALLSTAR_TEAMS:
                continue
            if away_name and home_name:
                unplayed.append({'home': home_name, 'away': away_name})

    print(f"  {len(played)} played, {len(unplayed)} remaining.")
    return played, unplayed


# ── Build standings from played games ─────────────────────────────────────────

def build_standings(played_games):
    """
    Returns dict: team -> {wins, losses, gp, pts, mov_sum, opponents}
    """
    standings = {}
    for g in played_games:
        for t in (g['home'], g['away']):
            if t not in standings:
                standings[t] = {'wins':0,'losses':0,'gp':0,'pts':0,'mov_sum':0.0,'opponents':[]}

        if g['margin'] > 0:
            standings[g['home']]['wins'] += 1
            standings[g['away']]['losses'] += 1
        else:
            standings[g['away']]['wins'] += 1
            standings[g['home']]['losses'] += 1

        standings[g['home']]['mov_sum'] += g['margin']
        standings[g['away']]['mov_sum'] -= g['margin']
        standings[g['home']]['opponents'].append(g['away'])
        standings[g['away']]['opponents'].append(g['home'])

    for t, s in standings.items():
        s['gp']  = s['wins'] + s['losses']
        s['pts'] = s['wins'] * 2
        s['mov'] = round(s['mov_sum'] / s['gp'], 2) if s['gp'] > 0 else 0.0
        s['games_remaining'] = TOTAL_GAMES - s['gp']

    # Compute games back from division leader
    for conf, divs in CONFERENCES.items():
        for div, div_teams in divs.items():
            in_div = [t for t in div_teams if t in standings]
            if not in_div:
                continue
            leader_wins   = max(standings[t]['wins']   for t in in_div)
            leader_losses = min(standings[t]['losses'] for t in in_div)
            for t in in_div:
                gb = ((leader_wins - standings[t]['wins']) +
                      (standings[t]['losses'] - leader_losses)) / 2.0
                standings[t]['gb'] = 0.0 if gb <= 0 else gb

    return standings


# ── Ridge solver for team ratings (needed for elimination sim) ────────────────

def solve_ratings(played_games):
    teams    = sorted(set(g['home'] for g in played_games) | set(g['away'] for g in played_games))
    team_idx = {t: i for i, t in enumerate(teams)}
    n        = len(teams)
    n_played = len(played_games)
    if n_played == 0:
        return {t: 0.0 for t in teams}, 0.0

    X = np.zeros((n_played, n + 1))
    y = np.zeros(n_played)
    for i, g in enumerate(played_games):
        X[i, team_idx[g['home']]] =  1.0
        X[i, team_idx[g['away']]] = -1.0
        X[i, -1]                  =  1.0
        y[i]                      =  g['margin']

    lam   = 3.0
    XtX   = X.T @ X
    I_reg = np.eye(n + 1); I_reg[-1,-1] = 0.0
    beta  = np.linalg.solve(XtX + lam * I_reg, X.T @ y)
    hca   = float(beta[-1])
    raw   = beta[:-1]
    avg   = float(np.mean(raw))
    return {teams[i]: float(raw[i]) - avg for i in range(n)}, hca


# ── Clinching & Elimination Scenarios ─────────────────────────────────────────

# Probability thresholds
CLINCH_THRESHOLD = 0.9999   # effectively clinched


def compute_seed_probs(standings, unplayed_games, ratings, hca, n_sims=SIM_N):
    """
    Simulate seasons and compute per-team:
      - P(each seed 1-8 in conference) — for the playoff picture grid
      - P(no playoffs)
      - P(div winner), top4, top2
    """
    team_names = list(standings.keys())
    current_w  = {t: standings[t]['wins'] for t in team_names}
    known      = set(team_names)

    sched = [(g['home'], g['away']) for g in unplayed_games
             if g['home'] in known and g['away'] in known]

    probs_list = []
    for home, away in sched:
        diff   = ratings.get(home, 0) - ratings.get(away, 0) + hca
        p_home = 1.0 / (1.0 + np.exp(-diff / SPREAD))
        probs_list.append((home, away, p_home))

    # Per-seed counts: seed_counts[team][seed] where seed 1-8 = playoff seeds, 0 = no playoffs
    seed_counts    = {t: {s: 0 for s in range(9)} for t in team_names}
    div_win_counts = {t: 0 for t in team_names}
    sim_win_totals = {t: 0.0 for t in team_names}

    rng = np.random.default_rng(99)
    for _ in range(n_sims):
        sim_w = dict(current_w)
        rolls = rng.random(len(probs_list))
        for (home, away, p_home), roll in zip(probs_list, rolls):
            if roll < p_home:
                sim_w[home] += 1
            else:
                sim_w[away] += 1

        for conf, divs in CONFERENCES.items():
            conf_teams = [t for div_teams in divs.values() for t in div_teams if t in sim_w]

            # Division winners
            div_winners = []
            for div, div_teams in divs.items():
                eligible = [t for t in div_teams if t in sim_w]
                if eligible:
                    div_winners.append(max(eligible, key=lambda t: sim_w[t]))
            div_winner_set = set(div_winners)

            # Seed div winners 1-3 by record
            div_winners_sorted = sorted(div_winners, key=lambda t: sim_w[t], reverse=True)

            # Remaining conf teams sorted by record for wildcard spots
            remaining = sorted([t for t in conf_teams if t not in div_winner_set],
                               key=lambda t: sim_w[t], reverse=True)
            wildcards = remaining[:WILDCARD_SPOTS]

            # Final seeded list: div winners 1-3 then wildcards 4-8 all by record
            playoff_teams_conf = sorted(div_winners_sorted + wildcards,
                                        key=lambda t: sim_w[t], reverse=True)

            playoff_set_conf = set(playoff_teams_conf)

            for t in conf_teams:
                sim_win_totals[t] += sim_w[t]
                if t in div_winner_set:
                    div_win_counts[t] += 1
                if t in playoff_set_conf:
                    seed = playoff_teams_conf.index(t) + 1
                    seed_counts[t][seed] += 1
                else:
                    seed_counts[t][0] += 1  # no playoffs

    result = {}
    for t in team_names:
        result[t] = {
            'playoff':    (n_sims - seed_counts[t][0]) / n_sims,
            'div_winner': div_win_counts[t] / n_sims,
            'top2_conf':  sum(seed_counts[t][s] for s in [1,2]) / n_sims,
            'top4_conf':  sum(seed_counts[t][s] for s in range(1,5)) / n_sims,
            'proj_wins':  sim_win_totals[t] / n_sims,
        }
        # Per-seed probabilities
        for s in range(9):
            result[t][f'seed_{s}'] = seed_counts[t][s] / n_sims

    return result


def compute_magic_numbers(standings, unplayed_games, seed_probs):
    """
    Conference-aware magic and elimination numbers.

    For each team, the relevant bubble is within their own conference:
      - Division winner: hardest path (must beat best team in division)
      - Playoff spot: must finish top 8 in conference

    Magic number (MN): wins needed to clinch a playoff spot in conference
    Elimination number (EN): losses until playoff is impossible in conference

    Both use the max-points formula which is exact in a 2-outcome league.
    """
    pts = {t: standings[t]['pts'] for t in standings}
    rem = {t: standings[t]['games_remaining'] for t in standings}

    magic       = {}
    elim_number = {}

    for conf, divs in CONFERENCES.items():
        conf_teams = [t for div_teams in divs.values() for t in div_teams if t in standings]

        # Sort conference teams by current points
        conf_sorted = sorted(conf_teams, key=lambda t: -pts[t])

        # The 8th-place team in conference is the bubble
        if len(conf_sorted) >= PLAYOFF_SPOTS_PER_CONF:
            eighth_team = conf_sorted[PLAYOFF_SPOTS_PER_CONF - 1]
            ninth_team  = conf_sorted[PLAYOFF_SPOTS_PER_CONF] if len(conf_sorted) > PLAYOFF_SPOTS_PER_CONF else None
        else:
            eighth_team = conf_sorted[-1] if conf_sorted else None
            ninth_team  = None

        for t in conf_teams:
            pp = seed_probs.get(t, {}).get('playoff', 0)

            # Magic number
            if pp >= CLINCH_THRESHOLD:
                magic[t] = 0
            elif pp <= ELIM_THRESHOLD:
                magic[t] = None
            else:
                if ninth_team and ninth_team != t:
                    # Need to stay ahead of the first team outside the playoff line
                    ninth_max = pts[ninth_team] + 2 * rem[ninth_team]
                    mn = max(0, int(np.ceil((ninth_max - pts[t] + 1) / 2)))
                    magic[t] = min(mn, rem[t])
                elif eighth_team and eighth_team != t:
                    eighth_max = pts[eighth_team] + 2 * rem[eighth_team]
                    mn = max(0, int(np.ceil((eighth_max - pts[t] + 1) / 2)))
                    magic[t] = min(mn, rem[t])
                else:
                    magic[t] = max(0, rem[t])

            # Elimination number
            if pp >= CLINCH_THRESHOLD:
                elim_number[t] = None
            elif pp <= ELIM_THRESHOLD:
                elim_number[t] = 0
            else:
                # Eliminated when max_pts < 8th place current pts
                eighth_pts  = pts[eighth_team] if eighth_team else 0
                max_possible = pts[t] + 2 * rem[t]
                gap = max_possible - eighth_pts
                en  = int(np.floor(gap / 2)) + 1 if gap >= 0 else 0
                elim_number[t] = max(0, min(en, rem[t]))

    return magic, elim_number


def find_retroactive_elimination(played_games, all_unplayed, teams_to_check, hca):
    """
    Replay the season game-by-game in chronological order.
    For each team in teams_to_check, find the box_id of the game AFTER WHICH
    they could no longer mathematically make the playoffs, even if they won
    every remaining game.

    Uses the "maximum points" elimination test:
      A team is eliminated at game G if:
        team_pts_after_G + 2 * (games_remaining_after_G) < 16th_place_current_min_pts

    This is the simplest exact test — no simulation needed.
    Returns dict: team -> box_id (the last game before gold clock starts,
                                  i.e. gold counts from the NEXT game onward)
    """
    def box_sort_key(box_id):
        parts = re.findall(r'\d+', str(box_id))
        return tuple(int(p) for p in parts) if parts else (0,)

    sorted_games = sorted(played_games, key=lambda g: box_sort_key(g.get('box_id','')))
    all_teams    = sorted(set(g['home'] for g in played_games) | set(g['away'] for g in played_games))
    total_games  = TOTAL_GAMES

    result = {}   # team -> dict with box_id, date, reason, eighth_team, eighth_pts, team_max

    # Walk through games one at a time, maintaining running standings
    running = {t: {'wins':0,'losses':0,'pts':0,'gp':0} for t in all_teams}

    for _, game in enumerate(sorted_games):
        # Update standings with this game's result
        home, away = game['home'], game['away']
        if game['margin'] > 0:
            running[home]['wins'] += 1; running[away]['losses'] += 1
        else:
            running[away]['wins'] += 1; running[home]['losses'] += 1
        for t in (home, away):
            running[t]['gp'] += 1
            running[t]['pts'] = running[t]['wins'] * 2

        # After this game, check each unchecked team
        for team in list(teams_to_check):
            if team in result:
                continue

            s       = running[team]
            gp_so_far = s['gp']
            rem       = total_games - gp_so_far
            max_pts   = s['pts'] + 2 * rem   # best case: win every remaining game

            # Find the minimum points the 16th-place team currently has
            # (the lowest pts among the top-PLAYOFF_SPOTS teams by current pts)
            sorted_by_pts = sorted(
                [t2 for t2 in all_teams if running[t2]['gp'] > 0],
                key=lambda t2: -running[t2]['pts']
            )

            if len(sorted_by_pts) < PLAYOFF_SPOTS:
                continue   # not enough teams have played yet

            # Conference-aware elimination check
            # Team is eliminated if their max possible points can't reach
            # the current 8th-place team in their own conference
            conf = TEAM_CONFERENCE.get(team)
            eliminated_here = False
            if conf:
                conf_divs = CONFERENCES[conf]
                conf_teams_now = [t2 for div_t in conf_divs.values()
                                  for t2 in div_t if running[t2]['gp'] > 0]
                conf_sorted = sorted(conf_teams_now, key=lambda t2: -running[t2]['pts'])
                if len(conf_sorted) >= PLAYOFF_SPOTS_PER_CONF:
                    eighth_pts = running[conf_sorted[PLAYOFF_SPOTS_PER_CONF - 1]]['pts']
                    if max_pts < eighth_pts:
                        eliminated_here = True
            else:
                # Fallback: league-wide if team not in structure
                league_sorted = sorted([t2 for t2 in all_teams if running[t2]['gp'] > 0],
                                       key=lambda t2: -running[t2]['pts'])
                if len(league_sorted) >= PLAYOFF_SPOTS:
                    if max_pts <= running[league_sorted[PLAYOFF_SPOTS - 1]]['pts']:
                        eliminated_here = True

            if eliminated_here:
                eighth_team = conf_sorted[PLAYOFF_SPOTS_PER_CONF - 1] if conf else None
                result[team] = {
                    'box_id':      game['box_id'],
                    'date':        game.get('date', ''),
                    'eighth_team': eighth_team,
                    'eighth_pts':  eighth_pts if conf else 0,
                    'team_max':    max_pts,
                    'team_pts':    s['pts'],
                    'reason':      f"Cannot reach {eighth_team} ({eighth_pts} pts) even winning all remaining games (max {max_pts} pts)" if eighth_team else "Mathematically eliminated",
                }

    # Any team still not found: season ended before elimination was detected
    for team in teams_to_check:
        if team not in result:
            team_games = [g for g in sorted_games if team in (g['home'], g['away'])]
            if team_games:
                last = team_games[-1]
                result[team] = {
                    'box_id':      last['box_id'],
                    'date':        last.get('date', ''),
                    'eighth_team': None,
                    'eighth_pts':  0,
                    'team_max':    0,
                    'team_pts':    running.get(team, {}).get('pts', 0),
                    'reason':      'Season ended',
                }

    return result


def compute_gold_points(played_games, standings, elimination_map, gold_state):
    """
    For each team that is eliminated (by math or by declaration),
    count standings points earned AFTER their elimination game.

    elimination_map: dict team -> box_id of the last game BEFORE gold clock starts
                     (i.e., the first game counted is the one AFTER this box_id)
    declarations:    teams that declared themselves out (start from their declaration)

    Returns: dict team -> {gold_pts, gold_wins, gold_losses, gold_gp, elim_type}
    """
    declarations = set(gold_state.get('declarations', []))
    elim_after   = gold_state.get('eliminated_after', {})

    def box_sort_key(box_id):
        # Normalize: strip path and extension, keep just the 'box123-4' part
        bid = str(box_id).split('/')[-1].replace('.htm','').replace('boxes/','')
        parts = re.findall(r'\d+', bid)
        return tuple(int(p) for p in parts) if parts else (0,)

    def norm_box(box_id):
        return str(box_id).split('/')[-1].replace('.htm','').replace('boxes/','')

    sorted_games = sorted(played_games, key=lambda g: box_sort_key(g.get('box_id','')))

    gold = {}
    for team, s in standings.items():
        raw_elim  = elim_after.get(team)
        in_elim_after = norm_box(raw_elim) if raw_elim else None
        is_declared   = team in declarations

        if not in_elim_after and not is_declared:
            gold[team] = {'gold_pts':0, 'gold_wins':0, 'gold_losses':0,
                          'gold_gp':0, 'eligible':False, 'elim_type': None}
            continue

        # Debug: verify the elim_after box_id exists in the game list
        team_box_ids = set(norm_box(g['box_id']) for g in sorted_games
                           if team in (g['home'], g['away']))
        if in_elim_after and in_elim_after not in team_box_ids:
            # Elim marker doesn't match any game — count from earliest possible
            # This can happen if gold_state.json has a stale box_id
            print(f"  Warning: {team} elim_after={in_elim_after} not found in game list. "
                  f"Closest: {sorted(team_box_ids, key=box_sort_key)[-3:] if team_box_ids else 'none'}")

        counting = is_declared and not in_elim_after
        gw = gl = 0
        found_elim_game = False

        for g in sorted_games:
            if team not in (g['home'], g['away']):
                continue

            g_box = norm_box(g['box_id'])

            if in_elim_after and not counting:
                if g_box == in_elim_after:
                    counting = True
                    found_elim_game = True
                continue   # don't count the elimination game itself

            if not counting:
                continue

            won = (g['home'] == team and g['margin'] > 0) or \
                  (g['away'] == team and g['margin'] < 0)
            if won:
                gw += 1
            else:
                gl += 1

        if in_elim_after and not found_elim_game:
            # Fallback: count all games after the elim_after sort key
            elim_key = box_sort_key(in_elim_after)
            gw = gl = 0
            for g in sorted_games:
                if team not in (g['home'], g['away']):
                    continue
                if box_sort_key(g.get('box_id','')) <= elim_key:
                    continue
                won = (g['home'] == team and g['margin'] > 0) or \
                      (g['away'] == team and g['margin'] < 0)
                if won: gw += 1
                else:   gl += 1

        elim_type = 'declared' if is_declared and not in_elim_after else \
                    'declared+math' if is_declared else 'math'

        gold[team] = {
            'gold_pts':    gw * 2,
            'gold_wins':   gw,
            'gold_losses': gl,
            'gold_gp':     gw + gl,
            'eligible':    True,
            'elim_type':   elim_type,
        }

    return gold


# ── Draft order ───────────────────────────────────────────────────────────────

def compute_draft_order(standings, gold, playoff_probs):
    """
    Draft order among eliminated teams:
      1. Eligible (eliminated) teams sorted by Gold Points desc
      2. Ties broken by fewer regular standings points (weaker team picks first)
      3. Non-eliminated teams (playoff teams) don't get picks in this list
    """
    teams = list(standings.keys())

    # Playoff teams (still alive or made it)
    playoff_teams = sorted(
        [t for t in teams if playoff_probs.get(t, 0) > ELIM_THRESHOLD],
        key=lambda t: standings[t]['pts'],
        reverse=True
    )

    # Eliminated teams with gold eligibility — sorted by gold pts desc, then reg pts asc
    elim_teams = sorted(
        [t for t in teams if gold.get(t, {}).get('eligible', False)],
        key=lambda t: (-gold[t]['gold_pts'], standings[t]['pts'])
    )

    # Eliminated teams with no gold yet (eliminated but 0 gold games played)
    elim_no_gold = sorted(
        [t for t in teams if not gold.get(t, {}).get('eligible', False)
         and playoff_probs.get(t, 1) <= ELIM_THRESHOLD],
        key=lambda t: standings[t]['pts']
    )

    return elim_teams + elim_no_gold, playoff_teams


# ── CLI: declare elimination ──────────────────────────────────────────────────

def declare_elimination(team_name):
    """Called when a team wants to declare themselves eliminated."""
    if not ALLOW_DECLARATIONS:
        print("  Declarations are disabled in this league (ALLOW_DECLARATIONS = False).")
        print("  Teams are only eliminated mathematically.")
        return
    state = load_gold_state()
    if team_name in state['declarations']:
        print(f"  {team_name} already declared.")
        return
    state['declarations'].append(team_name)
    save_gold_state(state)
    print(f"  {team_name} has declared themselves eliminated. Gold clock starts now.")


def mark_math_eliminated(team_name, after_box_id):
    """Mark a team as mathematically eliminated after a specific game."""
    state = load_gold_state()
    state['eliminated_after'][team_name] = after_box_id
    save_gold_state(state)
    print(f"  {team_name} marked as math-eliminated after {after_box_id}.")


# ── HTML Dashboard ────────────────────────────────────────────────────────────

def write_gold_dashboard(standings, gold, playoff_probs, seed_probs, magic, elim_number,
                         draft_order, playoff_teams, gold_state, elim_after, elim_info):
    print("\n  Writing gold_dashboard.html...")
    now          = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    declarations = set(gold_state.get('declarations', []))

    def pct_fmt(v):
        pv = v * 100
        return f"{pv:.0f}%"

    n_elim   = len(draft_order)
    n_clinch = sum(1 for t, p in playoff_probs.items() if p >= CLINCH_THRESHOLD)

    # ── Per-seed probability cell ─────────────────────────────────────────────
    def po_cell(t, seed):
        p   = seed_probs.get(t, {}).get(f'seed_{seed}', 0)
        pp  = playoff_probs.get(t, 0)
        mn  = magic.get(t)
        en  = elim_number.get(t)

        # Only show as truly clinched/eliminated if mathematically confirmed
        is_math_elim   = pp <= ELIM_THRESHOLD or en == 0
        is_math_clinch = mn == 0

        if seed == 0:   # No Playoffs column
            if is_math_elim:
                return '<td class="c-certain">100%</td>'
            if is_math_clinch:
                return '<td class="c-x">X</td>'
            if p <= DISPLAY_ELIM_PCT:
                return '<td class="c-x">X</td>'
            return f'<td class="c-pct-bad">{pct_fmt(p)}</td>'
        else:
            if is_math_elim:
                return '<td class="c-x">X</td>'
            if is_math_clinch and seed == 1:
                # Only show 100% for seed 1 if they've clinched the top spot
                return f'<td class="c-certain">100%</td>'
            if p >= CLINCH_THRESHOLD and is_math_clinch:
                return f'<td class="c-certain">100%</td>'
            if p <= DISPLAY_ELIM_PCT:
                better = sum(seed_probs.get(t,{}).get(f'seed_{s}',0) for s in range(1,seed))
                if better > DISPLAY_ELIM_PCT:
                    return '<td class="c-caret">^</td>'
                return '<td class="c-x">X</td>'
            return f'<td class="c-pct">{pct_fmt(p)}</td>'

    # ── Per-seed magic number cell ────────────────────────────────────────────
    # ── Conference playoff picture table ──────────────────────────────────────
    def po_table(conf, cell_fn, col_header_fn):
        divs = CONFERENCES[conf]
        rows = ""
        for div, div_teams in divs.items():
            in_div = [t for t in div_teams if t in standings]
            div_sorted = sorted(in_div, key=lambda t: -standings[t]['wins'])
            rows += f'<tr class="div-hdr"><td colspan="13"><strong>{div} Division</strong></td></tr>'
            for t in div_sorted:
                s = standings[t]
                pp = playoff_probs.get(t, 0)
                is_elim = pp <= ELIM_THRESHOLD
                row_cls = 'tr-elim' if is_elim else ('tr-clinch' if pp >= CLINCH_THRESHOLD else '')
                cells = "".join(cell_fn(t, s_) for s_ in range(1, 9)) + cell_fn(t, 0)
                rows += f'''<tr class="{row_cls}">
                  <td class="td-team">{t}</td>
                  <td class="td-num">{s['wins']}</td>
                  <td class="td-num">{s['losses']}</td>
                  <td class="td-num">{s['gp']}</td>
                  {cells}
                </tr>'''
        return rows

    east_po_rows = po_table('East', po_cell, None)
    west_po_rows = po_table('West', po_cell, None)
    # ── Magic number summary table (one MN per team for PO spot) ─────────────
    def magic_summary_table(conf):
        divs = CONFERENCES[conf]
        rows = ""
        for div, div_teams in divs.items():
            in_div = [t for t in div_teams if t in standings]
            div_sorted = sorted(in_div, key=lambda t: -standings[t]['wins'])
            rows += f'<tr class="div-hdr"><td colspan="8"><strong>{div} Division</strong></td></tr>'
            for t in div_sorted:
                s = standings[t]
                pp = playoff_probs.get(t, 0)
                mn = magic.get(t)
                en = elim_number.get(t)
                dv = seed_probs.get(t, {}).get('div_winner', 0)
                is_elim   = pp <= ELIM_THRESHOLD or t in declarations
                is_clinch = pp >= CLINCH_THRESHOLD
                gb_val    = s.get('gb', 0)
                gb_str    = "—" if gb_val == 0 else (int(gb_val) if gb_val == int(gb_val) else gb_val)

                if is_clinch:
                    mn_td = '<td class="td-num c-certain">0</td>'
                    en_td = '<td class="td-num">—</td>'
                    pp_td = f'<td class="td-num c-certain">100%</td>'
                elif is_elim:
                    mn_td = '<td class="td-num c-x">X</td>'
                    en_td = '<td class="td-num c-x">0</td>'
                    pp_td = '<td class="td-num c-x">—</td>'
                else:
                    mn_td = f'<td class="td-num c-mn">{mn if mn is not None else "?"}</td>'
                    en_td = f'<td class="td-num c-en">{en if en is not None else "?"}</td>'
                    pp_td = f'<td class="td-num c-pct">{pct_fmt(pp)}</td>'

                dv_td = f'<td class="td-num">{pct_fmt(dv) if dv > 0.001 and not is_elim else "—"}</td>'
                row_cls = 'tr-elim' if is_elim else ('tr-clinch' if is_clinch else '')
                rows += f'''<tr class="{row_cls}">
                  <td class="td-team">{t}</td>
                  <td class="td-num">{s['wins']}-{s['losses']}</td>
                  <td class="td-num">{gb_str}</td>
                  <td class="td-num">{s['games_remaining']}</td>
                  {mn_td}{en_td}{pp_td}{dv_td}
                </tr>'''
        return rows

    east_magic_rows = magic_summary_table('East')
    west_magic_rows = magic_summary_table('West')

    # ── Gold draft tracker rows ───────────────────────────────────────────────
    draft_rows = ""
    if draft_order:
        for pick, t in enumerate(draft_order, 1):
            s  = standings[t]
            g  = gold.get(t, {})
            et = g.get('elim_type', 'math')
            info       = elim_info.get(t, {})
            elim_date_d  = info.get('date', '') if isinstance(info, dict) else ''
            elim_eighth  = info.get('eighth_team', '') if isinstance(info, dict) else ''
            elim_8pts    = info.get('eighth_pts', 0) if isinstance(info, dict) else 0
            elim_max_d   = info.get('team_max', 0) if isinstance(info, dict) else 0
            if elim_date_d:
                elim_display = elim_date_d
                elim_tip     = (f"Could not reach {elim_eighth} ({elim_8pts} pts, max possible {elim_max_d})"
                                if elim_eighth else info.get('reason', ''))
            else:
                elim_display = elim_after.get(t, '—')
                elim_tip     = 'Run: python gold_standings.py reset to get dates'
            elim_type_str = 'Declared' if 'declared' in (et or '') else 'Math'
            gb_val = s.get('gb', 0)
            gb_str = "—" if gb_val == 0 else (int(gb_val) if gb_val == int(gb_val) else gb_val)
            draft_rows += f'''<tr>
              <td class="td-num" style="font-weight:700;color:#1a6b1a;">#{pick}</td>
              <td class="td-team">{t}</td>
              <td class="td-num">{s['wins']}-{s['losses']}</td>
              <td class="td-num">{gb_str}</td>
              <td class="td-num" style="font-weight:700;color:#1a6b1a;">{g.get('gold_pts',0)}</td>
              <td class="td-num">{g.get('gold_wins',0)}-{g.get('gold_losses',0)}</td>
              <td class="td-num">{g.get('gold_gp',0)}</td>
              <td class="td-num">{elim_type_str}</td>
              <td class="td-num" title="{elim_tip}" style="cursor:help;">{elim_display}</td>
            </tr>'''
    else:
        draft_rows = '<tr><td colspan="9" style="text-align:center;padding:20px;color:#666;">No teams eliminated yet.</td></tr>'

    # ── Full standings rows ───────────────────────────────────────────────────
    def team_sort_key(t):
        return (TEAM_CONFERENCE.get(t,'ZZ'), TEAM_DIVISION.get(t,'ZZ'), -standings[t]['pts'])
    all_teams_sorted = sorted(standings.keys(), key=team_sort_key)

    standing_rows = ""
    cur_conf = cur_div = None
    rank = 0
    for t in all_teams_sorted:
        tc  = TEAM_CONFERENCE.get(t,'')
        td2 = TEAM_DIVISION.get(t,'')
        if tc != cur_conf:
            cur_conf = tc; cur_div = None
            standing_rows += f'<tr class="conf-hdr"><td colspan="11"><strong>{tc.upper()} CONFERENCE</strong></td></tr>'
        if td2 != cur_div:
            cur_div = td2
            standing_rows += f'<tr class="div-hdr"><td colspan="11"><strong>&nbsp;&nbsp;{td2} Division</strong></td></tr>'
        rank += 1
        s   = standings[t]; g = gold.get(t,{}); pp = playoff_probs.get(t,0)
        mn  = magic.get(t);  en = elim_number.get(t)
        is_elim   = pp <= ELIM_THRESHOLD or t in declarations
        is_clinch = pp >= CLINCH_THRESHOLD
        mn_str = '0' if is_clinch else '—' if is_elim else (str(mn) if mn is not None else '?')
        en_str = '—' if is_clinch else '0' if (is_elim or en==0) else (str(en) if en is not None else '?')
        mn_cls = 'c-certain' if is_clinch else 'c-x' if is_elim else 'c-mn'
        en_cls = 'c-x' if (is_elim and not is_clinch) or en==0 else 'c-en'
        pp_cls = 'c-certain' if is_clinch else 'c-x' if is_elim else 'c-pct'
        pp_str = '100%' if is_clinch else '—' if is_elim else pct_fmt(pp)
        info2  = elim_info.get(t,{})
        if info2 and info2.get('date'):
            edisplay = info2['date']; etip = info2.get('reason','')
        elif t in declarations:
            edisplay = 'declared'; etip = 'Voluntary declaration'
        elif elim_after.get(t):
            edisplay = elim_after[t]; etip = 'Run reset to get dates'
        else:
            edisplay = ''; etip = ''
        gp_str = str(g.get('gold_pts','—')) if g.get('eligible') else '—'
        gb_val = s.get('gb',0)
        gb_str = "—" if gb_val==0 else (int(gb_val) if gb_val==int(gb_val) else gb_val)
        row_cls = 'tr-elim' if is_elim else ('tr-clinch' if is_clinch else ('tr-po' if pp>0.5 else ''))
        standing_rows += f'''<tr class="{row_cls}">
          <td class="td-num" style="color:#999;">{rank}</td>
          <td class="td-team">{t}</td>
          <td class="td-num">{s['wins']}-{s['losses']}</td>
          <td class="td-num">{gb_str}</td>
          <td class="td-num">{s['games_remaining']}</td>
          <td class="td-num {mn_cls}">{mn_str}</td>
          <td class="td-num {en_cls}">{en_str}</td>
          <td class="td-num {pp_cls}">{pp_str}</td>
          <td class="td-num" style="color:#1a6b1a;font-weight:700;">{gp_str}</td>
          <td class="td-num">{g.get('gold_wins',0) if g.get('eligible') else "—"}-{g.get('gold_losses',0) if g.get('eligible') else ""}</td>
          <td class="td-num" title="{etip}" style="cursor:help;">{edisplay}</td>
        </tr>'''

    gold_leader     = draft_order[0] if draft_order else '—'
    gold_leader_pts = gold.get(gold_leader,{}).get('gold_pts',0) if draft_order else 0

    MAGIC_HDR = """<tr>
          <th class="th-team">Team</th><th class="th-num">Rec</th><th class="th-num">GB</th><th class="th-num">Rem</th>
          <th class="th-seed" style="color:#1a6b1a;">Win MN</th>
          <th class="th-seed" style="color:#c00;">Lose EN</th>
          <th class="th-seed">PO%</th><th class="th-seed">Div%</th>
        </tr>"""

    HTML = []
    HTML.append("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ABL Gold Standings</title>
<link href="https://fonts.googleapis.com/css2?family=Arial" rel="stylesheet">
<style>
  * { box-sizing:border-box; margin:0; padding:0; }
  body { font-family: Arial, Helvetica, sans-serif; font-size: 13px;
         background:#fff; color:#000; }

  /* Top bar */
  .topbar { background:#003366; color:#fff; padding:8px 16px;
            display:flex; justify-content:space-between; align-items:center; }
  .topbar-title { font-size:16px; font-weight:bold; letter-spacing:0.5px; }
  .topbar-ts { font-size:11px; color:#aac; }

  /* Tab nav — mimics the site nav style */
  .tabnav { background:#e8e8e8; border-bottom:2px solid #003366;
            display:flex; overflow-x:auto; }
  .tab { padding:8px 16px; cursor:pointer; border:none; background:none;
         font-size:12px; font-weight:bold; color:#003366; white-space:nowrap;
         border-right:1px solid #ccc; }
  .tab:hover { background:#d0d8e8; }
  .tab.active { background:#003366; color:#fff; }

  /* Page content */
  .page { display:none; padding:16px 20px; max-width:1300px; margin:0 auto; }
  .page.active { display:block; }
  h1 { font-size:18px; font-weight:bold; margin-bottom:4px; color:#003366; }
  .page-sub { font-size:12px; color:#555; margin-bottom:14px; line-height:1.5; }

  /* Section header */
  .section-title { font-size:15px; font-weight:bold; color:#003366;
                   margin:16px 0 6px; border-bottom:2px solid #003366; padding-bottom:4px; }

  /* Two-column layout */
  .conf-grid { display:grid; grid-template-columns:1fr 1fr; gap:24px; }
  @media(max-width:1050px) { .conf-grid { grid-template-columns:1fr; } }
  .conf-block h2 { font-size:14px; font-weight:bold; color:#003366; margin-bottom:6px;
                   border-bottom:1px solid #003366; padding-bottom:3px; }

  /* Tables */
  table { border-collapse:collapse; width:100%; font-size:12px; }
  th { background:#003366; color:#fff; padding:5px 8px; text-align:center;
       border:1px solid #225; white-space:nowrap; }
  th.th-team { text-align:left; min-width:130px; }
  th.th-num  { min-width:36px; }
  th.th-seed { min-width:44px; }
  th.th-nopo { min-width:52px; background:#550000; }
  td { padding:4px 8px; border:1px solid #ccc; text-align:center; }
  td.td-team { text-align:left; font-weight:bold; white-space:nowrap; }
  td.td-num  { text-align:right; }
  tr:nth-child(even) { background:#f2f2f2; }
  tr:nth-child(even).tr-clinch,
  tr:nth-child(odd).tr-clinch  { background:#e8f5e8; }
  tr:nth-child(even).tr-elim,
  tr:nth-child(odd).tr-elim    { background:#f5f5f5; color:#999; }
  tr:nth-child(even).tr-po,
  tr:nth-child(odd).tr-po      { background:#fffae8; }
  tr.div-hdr td { background:#dde4ee; font-weight:bold; color:#003366;
                  font-size:11px; padding:3px 8px; }
  tr.conf-hdr td { background:#003366; color:#fff; font-size:12px;
                   font-weight:bold; padding:4px 8px; }

  /* Cell states */
  .c-certain { color:#006600; font-weight:bold; text-align:center; }
  .c-x       { color:#cc0000; font-weight:bold; text-align:center; }
  .c-caret   { color:#888; text-align:center; }
  .c-pct     { color:#006600; text-align:center; }
  .c-pct-bad { color:#cc0000; text-align:center; }
  .c-mn      { color:#006600; font-weight:bold; text-align:center; }
  .c-en      { color:#cc0000; font-weight:bold; text-align:center; }

  /* Notes box */
  .notes { background:#f9f9f9; border:1px solid #ccc; border-radius:4px;
           padding:10px 14px; margin-top:14px; font-size:11px; color:#444; }
  .notes ul { padding-left:18px; line-height:1.8; }

  /* Gold tracker */
  .gold-leader-box { background:#fffae0; border:2px solid #cc9900;
                     border-radius:4px; padding:10px 16px; margin-bottom:14px;
                     display:inline-block; }
  .gold-leader-box .label { font-size:11px; color:#666; text-transform:uppercase; }
  .gold-leader-box .value { font-size:20px; font-weight:bold; color:#1a6b1a; }

  /* Commands box */
  .cmd-box { background:#f4f4f4; border:1px solid #ccc; border-radius:4px;
             padding:12px 16px; margin-top:14px; }
  .cmd-box h3 { font-size:12px; font-weight:bold; color:#003366; margin-bottom:6px; }
  code { background:#e0e0e0; padding:2px 6px; border-radius:3px;
         font-family:monospace; font-size:12px; }
  pre  { background:#222; color:#8bc34a; padding:10px; border-radius:4px;
         font-size:12px; margin:6px 0 12px; overflow-x:auto; }

  .foot { text-align:center; font-size:11px; color:#888;
          padding:20px; border-top:1px solid #ddd; margin-top:20px; }
</style>
</head>
<body>
""")

    HTML.append(f"""
<div class="topbar">
  <span class="topbar-title">ABL Gold Drafting Standings</span>
  <span class="topbar-ts">Updated: {now}</span>
</div>
<div class="tabnav">
  <button class="tab active" data-tab="east-po"    onclick="showTab('east-po')">East Playoff Picture</button>
  <button class="tab"        data-tab="west-po"    onclick="showTab('west-po')">West Playoff Picture</button>
  <button class="tab"        data-tab="east-magic" onclick="showTab('east-magic')">East Magic Numbers</button>
  <button class="tab"        data-tab="west-magic" onclick="showTab('west-magic')">West Magic Numbers</button>
  <button class="tab"        data-tab="gold"       onclick="showTab('gold')">Gold Draft Tracker</button>
  <button class="tab"        data-tab="standings"  onclick="showTab('standings')">Full Standings</button>
  <button class="tab"        data-tab="info"       onclick="showTab('info')">How It Works</button>
</div>
""")

    def po_page(conf, rows, tab_id):
        east_clinch = sum(1 for t in [x for d in CONFERENCES[conf].values() for x in d]
                          if playoff_probs.get(t,0) >= CLINCH_THRESHOLD)
        active = 'active' if tab_id == 'east-po' else ''
        return f"""
<div class="page {active}" id="page-{tab_id}">
  <h1>{conf}ern Conference Playoff Picture</h1>
  <p class="page-sub">
    Probability each team wins each conference playoff seed (1–8) or misses playoffs entirely (No PO).<br>
    Seeds 1–3 are reserved for division winners (ordered by record among winners). Seeds 4–8 go to the best remaining records.<br>
    <strong style="color:#006600">100%</strong> = clinched this spot &nbsp;
    <strong style="color:#cc0000">X</strong> = eliminated from this spot &nbsp;
    <strong style="color:#888">^</strong> = guaranteed to finish better than this spot &nbsp;
    {east_clinch} of {PLAYOFF_SPOTS_PER_CONF} spots clinched.
  </p>
  <table>
    <thead>
      <tr>
        <th class="th-team" rowspan="2">Team</th>
        <th class="th-num" rowspan="2">W</th>
        <th class="th-num" rowspan="2">L</th>
        <th class="th-num" rowspan="2">GP</th>
        <th colspan="3" style="background:#004488;">Division Winners</th>
        <th colspan="5">Wildcards</th>
        <th class="th-nopo" rowspan="2">No PO</th>
      </tr>
      <tr>
        <th class="th-seed" style="background:#004488;">1*</th>
        <th class="th-seed" style="background:#004488;">2*</th>
        <th class="th-seed" style="background:#004488;">3*</th>
        <th class="th-seed">4</th><th class="th-seed">5</th>
        <th class="th-seed">6</th><th class="th-seed">7</th><th class="th-seed">8</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
  <div class="notes"><ul>
    <li>^ means the team will finish better than this playoff spot</li>
    <li>X means the team cannot win this playoff spot</li>
    <li><span style="color:#006600">50%</span> — probability team wins this spot (controls own destiny)</li>
    <li>* Seeds 1–3 reserved for division winners. ** Wildcard seeds 4–8 by record regardless of division.</li>
  </ul></div>
</div>"""

    HTML.append(po_page('East', east_po_rows, 'east-po'))
    HTML.append(po_page('West', west_po_rows, 'west-po'))

    def magic_page(conf, sum_rows, tab_id):
        return f"""
<div class="page" id="page-{tab_id}">
  <h1>{conf}ern Conference Magic Numbers</h1>
  <p class="page-sub">
    <strong style="color:#006600">Win Magic Number (MN)</strong> — wins needed to clinch a playoff spot in the {conf}ern Conference.<br>
    <strong style="color:#cc0000">Lose Elimination Number (EN)</strong> — losses until mathematically eliminated from the {conf}ern Conference playoff race.<br>
    <strong style="color:#888">^</strong> = already better than this spot &nbsp;
    <strong style="color:#cc0000">X</strong> = eliminated &nbsp;
    <strong style="color:#006600">0</strong> = clinched
  </p>
  <table>
    <thead>{MAGIC_HDR}</thead>
    <tbody>{sum_rows}</tbody>
  </table>
  <div class="notes"><ul>
    <li>Win MN = 0 means team has clinched a playoff spot</li>
    <li>Lose EN = 0 means team is mathematically eliminated</li>
    <li>Magic numbers computed within each conference independently</li>
    <li>Div% = probability of winning the division (earning a top-3 seed)</li>
  </ul></div>
</div>"""

    HTML.append(magic_page('East', east_magic_rows, 'east-magic'))
    HTML.append(magic_page('West', west_magic_rows, 'west-magic'))

    HTML.append(f"""
<div class="page" id="page-gold">
  <h1>Gold Draft Tracker</h1>
  <p class="page-sub">
    Eliminated teams earn Gold Points for every win after their clock starts.
    The team with the most Gold Points earns the #1 overall draft pick — no lottery, no tanking incentive.
  </p>
  <div class="gold-leader-box">
    <div class="label">Current #1 Overall Pick</div>
    <div class="value">{gold_leader} &nbsp; {gold_leader_pts} Gold Pts</div>
  </div>
  <p style="font-size:12px;color:#555;margin-bottom:10px;">
    {n_elim} teams eliminated (gold clock running) &nbsp;·&nbsp;
    {n_clinch} teams clinched playoffs &nbsp;·&nbsp;
    Declarations: <strong>{'ON' if ALLOW_DECLARATIONS else 'OFF'}</strong>
  </p>

  <div class="section-title">Draft Order</div>
  <table>
    <thead><tr>
      <th class="th-num">Pick</th>
      <th class="th-team">Team</th>
      <th class="th-num">Record</th>
      <th class="th-num">GB</th>
      <th class="th-seed" style="color:#ffdd44;">Gold Pts</th>
      <th class="th-seed">Gold Rec</th>
      <th class="th-seed">Gold GP</th>
      <th class="th-seed">Elim Type</th>
      <th class="th-seed">Eliminated</th>
    </tr></thead>
    <tbody>{draft_rows}</tbody>
  </table>

  <div class="section-title" style="margin-top:20px;">Still In Playoff Race</div>
  <table>
    <thead><tr>
      <th class="th-team">Team</th>
      <th class="th-num">Record</th>
      <th class="th-num">GB</th>
      <th class="th-seed">PO%</th>
    </tr></thead>
    <tbody>{"".join(
        f'<tr><td class="td-team">{t}</td><td class="td-num">{standings[t]["wins"]}-{standings[t]["losses"]}</td>'
        f'<td class="td-num">{standings[t].get("gb",0) if standings[t].get("gb",0)!=0 else "—"}</td>'
        f'<td class="td-num c-pct">{pct_fmt(playoff_probs.get(t,0))}</td></tr>'
        for t in sorted(playoff_teams, key=lambda t: -standings[t]["pts"])
    )}</tbody>
  </table>

  <div class="cmd-box">
    <h3>Commands</h3>
    <p style="margin-bottom:8px;font-size:12px;">New season (clears all saved data):</p>
    <pre>python gold_standings.py fullreset</pre>
    <p style="margin-bottom:8px;font-size:12px;">Re-detect elimination dates only:</p>
    <pre>python gold_standings.py reset</pre>
    {('<p style="margin-bottom:8px;font-size:12px;">Declare a team eliminated:</p>'
       '<pre>python gold_standings.py declare "Team Name"</pre>') if ALLOW_DECLARATIONS else ''}
    <p style="margin-bottom:8px;font-size:12px;">Manually set elimination point:</p>
    <pre>python gold_standings.py elim "Team Name" box43-7</pre>
  </div>
</div>
""")

    HTML.append(f"""
<div class="page" id="page-standings">
  <h1>Full Standings</h1>
  <p class="page-sub">All 30 teams grouped by conference and division. Magic # and Elim # are conference-specific.</p>
  <table>
    <thead><tr>
      <th class="th-num">#</th>
      <th class="th-team">Team</th>
      <th class="th-num">Record</th>
      <th class="th-num">GB</th>
      <th class="th-num">Rem</th>
      <th class="th-seed" style="color:#ffdd44;">Win MN</th>
      <th class="th-seed" style="color:#ff8888;">Lose EN</th>
      <th class="th-seed">PO%</th>
      <th class="th-seed" style="color:#ffdd44;">Gold Pts</th>
      <th class="th-seed">Gold Rec</th>
      <th class="th-seed">Eliminated</th>
    </tr></thead>
    <tbody>{standing_rows}</tbody>
  </table>
  <div class="notes"><ul>
    <li>GB = games back from division leader</li>
    <li>Win MN = wins needed to clinch a playoff spot in conference</li>
    <li>Lose EN = losses until mathematically eliminated from conference playoff race</li>
    <li>Gold Pts = points earned after elimination (wins × 2), used for draft order</li>
    <li>Hover over "Eliminated" date for explanation</li>
  </ul></div>
</div>
""")

    HTML.append(f"""
<div class="page" id="page-info">
  <h1>How It Works</h1>

  <div class="section-title">Gold Drafting System</div>
  <p style="font-size:13px;line-height:1.8;margin-bottom:12px;">
    Devised by Adam Gold (2012 Sloan Sports Analytics Conference) and popularized by
    Micah Blake McCurdy (<a href="https://hockeyviz.com/txt/gold" target="_blank">HockeyViz</a>).
    The traditional draft lottery incentivizes tanking — losing on purpose to get a better pick.
    Gold fixes this: <strong>only wins after elimination earn draft value</strong>.
  </p>
  <ul style="font-size:13px;line-height:2;padding-left:20px;margin-bottom:14px;">
    <li>When a team is <strong>mathematically eliminated</strong> from the conference playoff race, their Gold clock starts</li>
    <li>Every <strong>win</strong> after that earns <strong>2 Gold Points</strong></li>
    <li>Draft order = most Gold Points picks first</li>
    <li>Ties in Gold Points broken by fewer regular-season wins (weaker team picks higher)</li>
    <li>Playoff teams pick after all eliminated teams</li>
  </ul>

  <div class="section-title">Magic Numbers</div>
  <ul style="font-size:13px;line-height:2;padding-left:20px;margin-bottom:14px;">
    <li><strong style="color:#006600">Win Magic Number</strong> — exact number of wins to guarantee a playoff spot, even if all other teams win out. Uses the maximum-points formula: exact in a 2-outcome league.</li>
    <li><strong style="color:#cc0000">Lose Elimination Number</strong> — number of losses until you cannot reach the 8th seed in your conference even winning every remaining game.</li>
  </ul>

  <div class="section-title">Simulation</div>
  <p style="font-size:13px;line-height:1.8;">
    {SIM_N:,} Monte Carlo simulations of the real remaining schedule.
    Each game: P(home wins) = logistic((home_rating − away_rating + HCA) / {SPREAD}).
    Ratings from ridge regression on all regular-season results.
    Seeding: 3 division winners per conference (seeds by record among winners),
    then 5 wildcard spots by best remaining conference record.
  </p>
</div>

<p class="foot">ABL Gold Drafting Standings &nbsp;·&nbsp; Based on Adam Gold (2012 Sloan) via Micah Blake McCurdy
&nbsp;·&nbsp; {SIM_N:,} simulations &nbsp;·&nbsp; Updated {now}</p>

<script>
function showTab(name) {{
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelector('.tab[data-tab="' + name + '"]').classList.add('active');
  document.getElementById('page-' + name).classList.add('active');
}}
</script>
</body></html>
""")

    with open("gold_dashboard.html", "w", encoding="utf-8") as f:
        f.write("".join(HTML))
    print("  gold_dashboard.html written.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import sys

    # CLI: declare or elim commands
    if len(sys.argv) >= 2:
        cmd = sys.argv[1].lower()

        if cmd == 'reset':
            # Delete gold_state.json so retroactive elimination detection reruns from scratch
            if os.path.exists(GOLD_STATE_FILE):
                os.remove(GOLD_STATE_FILE)
                print(f"Deleted {GOLD_STATE_FILE} — elimination points will be redetected on next run.")
            else:
                print(f"{GOLD_STATE_FILE} not found — nothing to reset.")
            return

        if cmd == 'fullreset':
            # New season: delete both state files
            deleted = []
            for f_path in [GOLD_STATE_FILE, GAME_LOG_FILE]:
                if os.path.exists(f_path):
                    os.remove(f_path)
                    deleted.append(f_path)
            if deleted:
                print(f"Deleted: {', '.join(deleted)}")
                print("Ready for new season — run normally to start fresh.")
            else:
                print("No state files found — already clean.")
            return

        if len(sys.argv) >= 3:
            team = sys.argv[2]
            if cmd == 'declare':
                declare_elimination(team)
                return
            elif cmd == 'elim' and len(sys.argv) >= 4:
                mark_math_eliminated(team, sys.argv[3])
                return
            else:
                print(f"Unknown command: {cmd}")

        print("Usage:")
        print("  python gold_standings.py                              # run normally")
        print("  python gold_standings.py reset                        # clear saved state & redetect")
        print("  python gold_standings.py declare 'Team Name'          # team declares elimination")
        print("  python gold_standings.py elim 'Team Name' box15-3    # manually set math elim point")
        return

    print("=" * 55)
    print("ABL Gold Drafting Standings")
    print("=" * 55)

    # 1. Fetch schedule
    played_games, unplayed_games = fetch_schedule()

    # Pre-season / offseason mode: no games played yet
    # Build empty standings from known league structure and write dashboard anyway
    if not played_games:
        print("\nNo games played yet — generating pre-season dashboard.")
        all_teams = [t for conf in CONFERENCES.values() for div in conf.values() for t in div]
        standings = {t: {'wins':0,'losses':0,'gp':0,'pts':0,'mov':0.0,
                         'mov_sum':0.0,'opponents':[],'games_remaining':TOTAL_GAMES,'gb':0.0}
                     for t in all_teams}
        ratings      = {t: 0.0 for t in all_teams}
        hca          = 0.0
        # Everyone at 50% in pre-season — pure simulation from empty slate
        seed_probs   = {t: {'playoff': PLAYOFF_SPOTS_PER_CONF/15,
                             'div_winner': 1/5,
                             'top2_conf': 2/15,
                             'top4_conf': 4/15,
                             'proj_wins': TOTAL_GAMES * PLAYOFF_SPOTS_PER_CONF / 15,
                             **{f'seed_{s}': (1/15 if 1<=s<=8 else 7/15) for s in range(9)}}
                         for t in all_teams}
        playoff_probs = {t: PLAYOFF_SPOTS_PER_CONF/15 for t in all_teams}
        magic         = {t: 1 for t in all_teams}
        elim_number   = {t: TOTAL_GAMES for t in all_teams}
        gold_state    = load_gold_state()
        elim_after    = gold_state.get('eliminated_after', {})
        gold          = {t: {'gold_pts':0,'gold_wins':0,'gold_losses':0,
                              'gold_gp':0,'eligible':False,'elim_type':None}
                         for t in all_teams}
        draft_order, playoff_teams = [], list(all_teams)
        write_gold_dashboard(standings, gold, playoff_probs, seed_probs, magic, elim_number,
                             draft_order, playoff_teams, gold_state, elim_after,
                             gold_state.get('elim_info', {}))
        print("\nPre-season dashboard written. Run again once games start.")
        return

    # 2. Build standings
    print("\nBuilding standings...")
    standings = build_standings(played_games)
    print(f"  {len(standings)} teams, {len(played_games)} games processed.")

    # 3. Solve ratings for elimination sim
    print("\nSolving team ratings...")
    ratings, hca = solve_ratings(played_games)
    print(f"  HCA = {hca:+.2f} pts")

    # 4. Compute playoff probabilities + seed probs + magic numbers
    print(f"\nRunning {SIM_N:,} simulations for elimination & clinching check...")
    seed_probs    = compute_seed_probs(standings, unplayed_games, ratings, hca)
    playoff_probs = {t: seed_probs[t]['playoff'] for t in seed_probs}
    magic, elim_number = compute_magic_numbers(standings, unplayed_games, seed_probs)
    n_elim   = sum(1 for t, p in playoff_probs.items() if p <= ELIM_THRESHOLD)
    n_clinch = sum(1 for t, p in playoff_probs.items() if p >= CLINCH_THRESHOLD)
    print(f"  {n_elim} teams eliminated, {n_clinch} teams clinched.")

    # 5. Load gold state and retroactively find elimination points
    gold_state = load_gold_state()
    elim_after = gold_state.get('eliminated_after', {})
    declarations = set(gold_state.get('declarations', []))

    # For any currently-eliminated team not yet marked, replay the season
    # game-by-game to find the EARLIEST point they became eliminated.
    # We do this by sorting all played games in order, then at each step
    # rebuilding standings-so-far + remaining schedule and checking if the
    # team's max possible points can still reach the playoff cutoff.
    teams_needing_retroactive = [
        t for t, prob in playoff_probs.items()
        if prob <= ELIM_THRESHOLD
        and t not in elim_after
        and t not in declarations
    ]

    if teams_needing_retroactive:
        print(f"\n  Retroactively finding elimination points for: {', '.join(sorted(teams_needing_retroactive))}")
        newly_found = find_retroactive_elimination(
            played_games, unplayed_games, teams_needing_retroactive, hca
        )
        for team, info in newly_found.items():
            box_id   = info['box_id'] if isinstance(info, dict) else info
            date_str = info.get('date', '') if isinstance(info, dict) else ''
            reason   = info.get('reason', '') if isinstance(info, dict) else ''
            elim_after[team] = box_id
            print(f"    {team}: eliminated {date_str} — {reason}")

        if newly_found:
            gold_state['eliminated_after'] = elim_after
            # Store human-readable elimination info separately
            if 'elim_info' not in gold_state:
                gold_state['elim_info'] = {}
            for team, info in newly_found.items():
                if isinstance(info, dict):
                    gold_state['elim_info'][team] = info
            save_gold_state(gold_state)
            print(f"  Gold state updated — {len(newly_found)} teams retroactively marked.")

    # 6. Compute gold points
    print("\nComputing Gold Points...")
    gold = compute_gold_points(played_games, standings, elim_after, gold_state)

    # 7. Draft order
    draft_order, playoff_teams = compute_draft_order(standings, gold, playoff_probs)

    # 8. Print summary to console
    print("\n" + "─" * 55)
    print(f"{'GOLD DRAFT ORDER':^55}")
    print("─" * 55)
    if draft_order:
        for pick, t in enumerate(draft_order, 1):
            g  = gold.get(t, {})
            gp = g.get('gold_pts', 0)
            gr = f"{g.get('gold_wins',0)}-{g.get('gold_losses',0)}"
            et = f"[{g.get('elim_type','?')}]"
            print(f"  #{pick:<3} {t:<22} {gp:>3} Gold Pts  ({gr}) {et}")
    else:
        print("  No teams eliminated yet.")

    print("\n" + "─" * 55)
    print(f"{'PLAYOFF RACE':^55}")
    print("─" * 55)
    for t in sorted(playoff_teams, key=lambda t: -standings[t]['pts'])[:10]:
        s   = standings[t]
        pp  = playoff_probs.get(t, 0)
        mn  = magic.get(t)
        en  = elim_number.get(t)
        status = 'CLINCHED' if pp >= CLINCH_THRESHOLD else f'MN:{mn}' if mn else '?'
        print(f"  {t:<25} {s['pts']:>3} pts  {pp*100:>5.1f}%  {status}")
    if len(playoff_teams) > 10:
        print(f"  ... and {len(playoff_teams)-10} more")

    # 9. Write dashboard
    print("\nWriting gold_dashboard.html...")
    write_gold_dashboard(standings, gold, playoff_probs, seed_probs, magic, elim_number, draft_order, playoff_teams, gold_state, elim_after, gold_state.get('elim_info', {}))

    print(f"\nDone.")


if __name__ == "__main__":
    main()
