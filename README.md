# ABL Analytics & Gold Standings

Automated dashboards for the Ascension Basketball League (ABL), a sim league running Fast Break Pro Basketball 3.

## Dashboards

Live at: `https://YOUR_USERNAME.github.io/YOUR_REPO/`

| Dashboard | URL | Description |
|-----------|-----|-------------|
| Landing page | `/` | Links to both dashboards + update status |
| Gold Standings | `/gold.html` | Playoff picture, magic numbers, Gold draft tracker |
| Analytics | `/analytics.html` | BPM 2.0, power ratings, game log, trade analyzer |

## Setup

### 1. Clone and configure

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
pip install -r requirements.txt
```

### 2. Update season config

In `abl_analytics.py`, update:
```python
CURRENT_SEASON = "2005"   # change to current season year
```

In `gold_standings.py`, update `CONFERENCES` if team/division structure changes.

### 3. Enable GitHub Pages

1. Go to your repo → Settings → Pages
2. Set source to **Deploy from branch**
3. Select branch: `main`, folder: `/docs`
4. Save — your site will be live at `https://YOUR_USERNAME.github.io/YOUR_REPO/`

### 4. Run manually (local)

```bash
python push.py        # run both scripts and push to GitHub
python push.py --dry  # test run without pushing
```

### 5. Automatic updates (GitHub Actions)

The workflow in `.github/workflows/update.yml` runs every 15 minutes automatically.

**Note:** If the ABL site blocks GitHub's servers, the Action will fail silently
(`continue-on-error: true`) and the last successful update stays live. In that case,
run `push.py` manually from your local machine where the site allows access.

## State files

| File | Purpose |
|------|---------|
| `gold_state.json` | Elimination markers and declarations — persists across runs |
| `game_log.json` | Box score BPM data — accumulates all season |

### New season reset

```bash
python gold_standings.py fullreset   # clears both gold_state.json and game_log.json
```

## CLI commands (gold_standings.py)

```bash
python gold_standings.py                          # normal run
python gold_standings.py reset                    # re-detect elimination dates
python gold_standings.py fullreset                # new season — clear all state
python gold_standings.py declare "Team Name"      # voluntary elimination (if enabled)
python gold_standings.py elim "Team Name" box43-7 # manually set elimination point
```
