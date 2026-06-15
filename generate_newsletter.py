#!/usr/bin/env python3
"""Daily ML/DS newsletter generator. Fetches recent arXiv papers and ranks by impact heuristics."""

import datetime
import os
import re
import sys
import xml.etree.ElementTree as ET
from urllib.request import urlopen
from urllib.error import URLError

# ─── Config ───────────────────────────────────────────────────────────────────

CATEGORIES = ['cs.AI', 'cs.LG', 'cs.CL', 'cs.CV', 'stat.ML', 'cs.RO', 'cs.NE']
MAX_FETCH = 300
TOP_N = 15
LOOKBACK_DAYS = 3  # how far back to search if yesterday has no papers

SCORE_RULES = [
    # (regex, score, title_only)
    (r'\blarge language model\b',         4, False),
    (r'\bfoundation model\b',             4, False),
    (r'\bvision.language model\b|\bvlm\b',3, False),
    (r'\bmultimodal\b',                   3, False),
    (r'\bdiffusion model\b',              3, False),
    (r'\btext.to.image\b|\btext to image\b', 3, False),
    (r'\bgpt\b|\bllm\b',                  3, False),
    (r'\bsurvey\b',                       3, True),   # survey in title = high value
    (r'\balignment\b',                    3, False),
    (r'\bstate.of.the.art\b|\bsota\b',   2, False),
    (r'\boutperform\b|\bsurpass\b',       2, False),
    (r'\bbenchmark\b',                    2, False),
    (r'\bsafety\b',                       2, False),
    (r'\bhallucination\b',                2, False),
    (r'\breasoning\b',                    2, False),
    (r'\bagent\b|\bagentic\b',            2, False),
    (r'\bopen.source\b|\bopen source\b',  2, False),
    (r'\bzero.shot\b|\bfew.shot\b',       2, False),
    (r'\bin.context learning\b',          2, False),
    (r'\bchain.of.thought\b|\bcot\b',     2, False),
    (r'\binstruction.tuning\b|\binstruction tuning\b', 2, False),
    (r'\blora\b|\bqlora\b|\bpeft\b',      2, False),
    (r'\bretrieval.augmented\b|\b\brag\b',2, False),
    (r'\bcode generation\b|\bcode synthesis\b', 2, False),
    (r'\bscaling law\b|\bscaling\b',      2, False),
    (r'\bfine.tuning\b|\bfinetuning\b',   1, False),
    (r'\btransformer\b',                  1, False),
    (r'\bquantization\b|\bpruning\b|\bcompression\b', 1, False),
    (r'\bmath\b|\btheorem\b|\bformal\b',  1, False),
    (r'\brobustness\b|\badversarial\b',   1, False),
    (r'\bautonomous\b|\brobotics\b',      1, False),
    (r'\bmedical\b|\bclinical\b|\bhealthcare\b', 1, False),
]

CATEGORY_SCORES = {
    'cs.CL': 3, 'cs.AI': 3, 'cs.LG': 2,
    'cs.CV': 2, 'cs.RO': 2, 'stat.ML': 1, 'cs.NE': 1,
}

THEMES = {
    r'\blarge language model\b|\bllm\b|\bgpt\b':       'Large Language Models',
    r'\bdiffusion model\b|\btext.to.image\b':           'Diffusion / Generative',
    r'\bagent\b|\bagentic\b':                           'AI Agents',
    r'\breasoning\b|\bchain.of.thought\b':              'Reasoning',
    r'\balignment\b|\bsafety\b|\bhallucination\b':      'Alignment & Safety',
    r'\bmultimodal\b|\bvision.language\b|\bvlm\b':      'Multimodal / VLM',
    r'\bcode generation\b|\bcode synthesis\b':          'Code Generation',
    r'\bscaling law\b|\bscaling\b':                     'Scaling',
    r'\bfine.tuning\b|\blora\b|\bpeft\b':               'Fine-Tuning / PEFT',
    r'\bretrieval.augmented\b|\brag\b':                 'RAG / Retrieval',
    r'\brobotics\b|\bmanipulation\b':                   'Robotics',
    r'\bmedical\b|\bclinical\b':                        'Medical AI',
    r'\bmath\b|\btheorem\b':                            'Math & Formal Reasoning',
    r'\brobustness\b|\badversarial\b':                  'Robustness',
}

NS = {'a': 'http://www.w3.org/2005/Atom'}

# ─── Fetch ────────────────────────────────────────────────────────────────────

def fetch_xml():
    query = '+OR+'.join(f'cat:{c}' for c in CATEGORIES)
    url = (
        'http://export.arxiv.org/api/query?'
        f'search_query={query}'
        f'&sortBy=submittedDate&sortOrder=descending'
        f'&max_results={MAX_FETCH}'
    )
    print(f'Fetching {url}', file=sys.stderr)
    try:
        with urlopen(url, timeout=45) as r:
            return r.read()
    except URLError as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)


def parse_papers(xml_bytes):
    root = ET.fromstring(xml_bytes)
    papers = []
    for entry in root.findall('a:entry', NS):
        published = (entry.findtext('a:published', '', NS) or '')[:10]
        title = (entry.findtext('a:title', '', NS) or '').replace('\n', ' ').strip()
        abstract = (entry.findtext('a:summary', '', NS) or '').replace('\n', ' ').strip()

        url = entry.findtext('a:id', '', NS) or ''
        for link in entry.findall('a:link', NS):
            if link.get('type') == 'text/html':
                url = link.get('href', url)
                break

        authors = [a.findtext('a:name', '', NS) for a in entry.findall('a:author', NS)]
        categories = [t.get('term', '') for t in entry.findall('a:category', NS)]

        papers.append({
            'title': title, 'abstract': abstract, 'url': url,
            'authors': authors, 'categories': categories, 'published': published,
        })
    return papers


def find_papers_for_date(papers, target_date):
    d = target_date.isoformat()
    return [p for p in papers if p['published'] == d]


def get_papers():
    """Return (papers, date_used). Falls back up to LOOKBACK_DAYS if no papers found."""
    raw = fetch_xml()
    all_papers = parse_papers(raw)
    today = datetime.date.today()
    for offset in range(1, LOOKBACK_DAYS + 1):
        d = today - datetime.timedelta(days=offset)
        found = find_papers_for_date(all_papers, d)
        if found:
            print(f'Found {len(found)} papers for {d}', file=sys.stderr)
            return found, d
    print('No papers found in lookback window.', file=sys.stderr)
    return [], today - datetime.timedelta(days=1)

# ─── Rank ─────────────────────────────────────────────────────────────────────

def score(paper):
    title = paper['title'].lower()
    abstract = paper['abstract'].lower()
    s = 0
    for pattern, weight, title_only in SCORE_RULES:
        if re.search(pattern, title):
            s += weight + 1
        elif not title_only and re.search(pattern, abstract):
            s += weight
    primary = paper['categories'][0] if paper['categories'] else ''
    s += CATEGORY_SCORES.get(primary, 0)
    known = sum(1 for c in paper['categories'] if c in CATEGORY_SCORES)
    s += max(0, known - 1)
    return s


def top_themes(papers, n=6):
    counts = {}
    for label_re, label in THEMES.items():
        c = sum(1 for p in papers
                if re.search(label_re, (p['title'] + ' ' + p['abstract']).lower()))
        if c:
            counts[label] = c
    return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:n]

# ─── Render ───────────────────────────────────────────────────────────────────

CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
     background:#0d1117;color:#e6edf3;line-height:1.6}
a{color:inherit}
header{background:#161b22;border-bottom:1px solid #30363d;padding:28px 16px;text-align:center}
header h1{font-size:1.8rem;font-weight:700;color:#58a6ff;letter-spacing:-.5px}
header .sub{color:#8b949e;margin-top:6px;font-size:.9rem}
.wrap{max-width:820px;margin:0 auto;padding:28px 16px 56px}
.themes{background:#161b22;border:1px solid #30363d;border-radius:8px;
        padding:16px 20px;margin-bottom:28px}
.themes h3{font-size:.72rem;text-transform:uppercase;letter-spacing:1px;
           color:#8b949e;margin-bottom:10px}
.theme-list{display:flex;flex-wrap:wrap;gap:8px}
.tag{background:#1f6feb22;border:1px solid #1f6feb55;color:#79c0ff;
     border-radius:20px;padding:3px 12px;font-size:.78rem}
.meta-bar{color:#8b949e;font-size:.82rem;margin-bottom:20px}
.paper{display:flex;gap:16px;padding:22px 0;border-bottom:1px solid #21262d}
.paper:last-child{border-bottom:none}
.rank{color:#30363d;font-size:1.5rem;font-weight:700;min-width:40px;
      padding-top:2px;font-variant-numeric:tabular-nums}
.body{flex:1;min-width:0}
.body h2{font-size:1rem;font-weight:600;line-height:1.45;margin-bottom:6px}
.body h2 a{text-decoration:none;color:#e6edf3}
.body h2 a:hover{color:#58a6ff}
.pmeta{font-size:.78rem;color:#8b949e;margin-bottom:8px}
.cats{display:inline-flex;gap:4px;flex-wrap:wrap;margin-top:3px}
.cat{background:#1f6feb18;color:#58a6ff;border:1px solid #1f6feb44;
     border-radius:4px;padding:1px 6px;font-size:.72rem}
.abs{font-size:.84rem;color:#8b949e;margin-bottom:8px;line-height:1.55}
.link{font-size:.78rem;color:#58a6ff;text-decoration:none}
.link:hover{text-decoration:underline}
footer{text-align:center;padding:24px 16px;color:#484f58;
       font-size:.78rem;border-top:1px solid #21262d;margin-top:16px}
@media(max-width:600px){
  .rank{font-size:1.1rem;min-width:28px}
  .body h2{font-size:.93rem}
}
"""

def esc(s):
    return s.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;').replace('"','&quot;')

def trunc(s, n=380):
    return s[:n].rsplit(' ', 1)[0] + '…' if len(s) > n else s


def render(papers, date):
    date_label = date.strftime('%B %d, %Y')
    ranked = sorted(papers, key=score, reverse=True)[:TOP_N]
    themes = top_themes(papers)

    theme_html = ''.join(
        f'<span class="tag">{t} <span style="opacity:.5">·{c}</span></span>'
        for t, c in themes
    )

    cards = ''
    for i, p in enumerate(ranked, 1):
        authors = esc(', '.join(p['authors'][:3]) + (' et al.' if len(p['authors']) > 3 else ''))
        cats = ''.join(f'<span class="cat">{esc(c)}</span>' for c in p['categories'][:3])
        cards += f'''
  <div class="paper">
    <div class="rank">#{i}</div>
    <div class="body">
      <h2><a href="{esc(p['url'])}" target="_blank" rel="noopener">{esc(p['title'])}</a></h2>
      <div class="pmeta">{authors}<br><span class="cats">{cats}</span></div>
      <p class="abs">{trunc(esc(p['abstract']))}</p>
      <a class="link" href="{esc(p['url'])}" target="_blank" rel="noopener">arXiv &rarr;</a>
    </div>
  </div>'''

    total = len(papers)
    shown = min(TOP_N, len(ranked))
    sources = ', '.join(CATEGORIES)
    generated = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>DS/ML Daily — {date_label}</title>
  <style>{CSS}</style>
</head>
<body>
  <header>
    <h1>DS/ML Daily</h1>
    <div class="sub">{date_label} &mdash; Top arXiv papers</div>
  </header>
  <div class="wrap">
    <div class="themes">
      <h3>Today&rsquo;s themes</h3>
      <div class="theme-list">{theme_html}</div>
    </div>
    <div class="meta-bar">{total} papers &mdash; showing top {shown} by estimated impact</div>
    {cards}
  </div>
  <footer>Sources: {sources} &middot; Generated {generated}</footer>
</body>
</html>'''

# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    papers, date = get_papers()
    html = render(papers, date)
    os.makedirs('dist', exist_ok=True)
    path = os.path.join('dist', 'index.html')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'Written {path} ({len(papers)} papers, date={date})')
