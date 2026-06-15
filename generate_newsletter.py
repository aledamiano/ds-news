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
LOOKBACK_DAYS = 3

# General ML/AI impact rules: (regex, score, title_only)
SCORE_RULES = [
    (r'\blarge language model\b',                       4, False),
    (r'\bfoundation model\b',                           4, False),
    (r'\bvision.language model\b|\bvlm\b',              3, False),
    (r'\bmultimodal\b',                                 3, False),
    (r'\bdiffusion model\b',                            3, False),
    (r'\btext.to.image\b|\btext to image\b',            3, False),
    (r'\bgpt\b|\bllm\b',                                3, False),
    (r'\bsurvey\b',                                     3, True),
    (r'\balignment\b',                                  3, False),
    (r'\bstate.of.the.art\b|\bsota\b',                 2, False),
    (r'\boutperform\b|\bsurpass\b',                     2, False),
    (r'\bbenchmark\b',                                  2, False),
    (r'\bsafety\b',                                     2, False),
    (r'\bhallucination\b',                              2, False),
    (r'\breasoning\b',                                  2, False),
    (r'\bagent\b|\bagentic\b',                          2, False),
    (r'\bopen.source\b|\bopen source\b',                2, False),
    (r'\bzero.shot\b|\bfew.shot\b',                     2, False),
    (r'\bin.context learning\b',                        2, False),
    (r'\bchain.of.thought\b|\bcot\b',                   2, False),
    (r'\binstruction.tuning\b|\binstruction tuning\b',  2, False),
    (r'\blora\b|\bqlora\b|\bpeft\b',                    2, False),
    (r'\bretrieval.augmented\b|\brag\b',                2, False),
    (r'\bcode generation\b|\bcode synthesis\b',         2, False),
    (r'\bscaling law\b|\bscaling\b',                    2, False),
    (r'\bfine.tuning\b|\bfinetuning\b',                 1, False),
    (r'\btransformer\b',                                1, False),
    (r'\bquantization\b|\bpruning\b|\bcompression\b',   1, False),
    (r'\bmath\b|\btheorem\b|\bformal\b',                1, False),
    (r'\brobustness\b|\badversarial\b',                 1, False),
    (r'\bautonomous\b|\brobotics\b',                    1, False),
    (r'\bmedical\b|\bclinical\b|\bhealthcare\b',        1, False),
]

# Domain-specific rules
DOMAIN_RULES = [
    # Credit & risk — core business
    (r'\bcredit\b|\bcredit.risk\b|\bcredit.scor',       4, False),
    (r'\bdefault.predict\b|\bprobability of default\b', 4, False),
    (r'\bfraud.detect\b|\bfraud.prevent\b',             4, False),
    (r'\banomaly.detect\b|\boutlier.detect\b',          3, False),
    (r'\brisk.model\b|\brisk.assess\b|\brisk.score\b',  3, False),
    # Explainability — regulatory requirement in finance
    (r'\bexplainab\b|\binterpretab\b|\bxai\b',          4, False),
    (r'\bshap\b|\blime\b|\bfeature.import\b',           3, False),
    (r'\bcounterfactual\b',                             3, False),
    # Fairness & bias — compliance / GDPR
    (r'\bfairness\b|\bbias.detect\b|\balgorithmic.bias\b', 4, False),
    (r'\bdiscrimination\b|\bequity\b|\bequal.opportun\b',  3, False),
    # Tabular / structured data — primary data type at Experian
    (r'\btabular.data\b|\bstructured.data\b',           4, False),
    (r'\bgradient.boost\b|\bxgboost\b|\blightgbm\b|\bcatboost\b', 3, False),
    (r'\brandom.forest\b|\bdecision.tree\b',            2, False),
    # Time series & forecasting
    (r'\btime.series\b|\btemporal.model\b',             3, False),
    (r'\bforecasting\b|\bpredictive.model\b',           2, False),
    # Causal inference — important for business decisions
    (r'\bcausal.inference\b|\bcausal.ml\b|\bcausality\b', 4, False),
    (r'\btreatment.effect\b|\bcounterfactual\b|\bdo.calculus\b', 3, False),
    # Privacy — financial data sensitivity
    (r'\bdifferential.privacy\b|\bfederated.learn\b',   3, False),
    (r'\bprivacy.preserv\b|\bsecure.aggreg\b',          3, False),
    # MLOps & production
    (r'\bmodel.monitor\b|\bdata.drift\b|\bconceptdrift\b|\bconcept.drift\b', 3, False),
    (r'\bmlops\b|\bml.pipeline\b|\bmodel.deploy\b',     2, False),
    # Class imbalance — fraud/default datasets are heavily skewed
    (r'\bclass.imbalanc\b|\bimbalanc\b|\bsmote\b|\boversampl\b', 3, False),
    # Graph ML — fraud/credit network analysis
    (r'\bgraph.neural\b|\bgnn\b|\bgraph.learn\b',       2, False),
    # Financial NLP
    (r'\bfinancial.text\b|\bfinancial.nlp\b|\bearnings.call\b', 3, False),
    (r'\bsentiment.analy\b',                            2, False),
    # AutoML / feature engineering
    (r'\bautoml\b|\bauto.ml\b|\bfeature.engineer\b|\bfeature.select\b', 2, False),
    # Customer / behavioural modelling
    (r'\bcustomer.segment\b|\bchurn.predict\b|\bcustomer.lifetime\b', 3, False),
    (r'\bbehavioral.model\b|\buser.model\b',            2, False),
]

CATEGORY_SCORES = {
    'cs.CL': 3, 'cs.AI': 3, 'cs.LG': 2,
    'cs.CV': 2, 'cs.RO': 2, 'stat.ML': 1, 'cs.NE': 1,
}

THEMES = {
    r'\blarge language model\b|\bllm\b|\bgpt\b':           'Large Language Models',
    r'\bdiffusion model\b|\btext.to.image\b':               'Diffusion / Generative',
    r'\bagent\b|\bagentic\b':                               'AI Agents',
    r'\breasoning\b|\bchain.of.thought\b':                  'Reasoning',
    r'\balignment\b|\bsafety\b|\bhallucination\b':          'Alignment & Safety',
    r'\bmultimodal\b|\bvision.language\b|\bvlm\b':          'Multimodal / VLM',
    r'\bcode generation\b|\bcode synthesis\b':              'Code Generation',
    r'\bscaling law\b|\bscaling\b':                         'Scaling',
    r'\bfine.tuning\b|\blora\b|\bpeft\b':                   'Fine-Tuning / PEFT',
    r'\bretrieval.augmented\b|\brag\b':                     'RAG / Retrieval',
    r'\bexplainab\b|\binterpretab\b|\bxai\b|\bshap\b':      'Explainability / XAI',
    r'\bcredit\b|\bfraud.detect\b|\brisk.model\b':          'Credit & Risk',
    r'\bcausal.inference\b|\bcausality\b|\btreatment.effect\b': 'Causal Inference',
    r'\btabular.data\b|\bstructured.data\b|\bxgboost\b|\blightgbm\b': 'Tabular / Structured Data',
    r'\bfairness\b|\bbias.detect\b|\bdiscrimination\b':     'Fairness & Bias',
    r'\bdifferential.privacy\b|\bfederated.learn\b':        'Privacy / Federated',
    r'\btime.series\b|\bforecasting\b':                     'Time Series',
    r'\broboticsb|\bmanipulation\b':                        'Robotics',
    r'\bmedical\b|\bclinical\b':                            'Medical AI',
    r'\bmath\b|\btheorem\b':                                'Math & Formal Reasoning',
    r'\brobustness\b|\badversarial\b':                      'Robustness',
}

NS = {'a': 'http://www.w3.org/2005/Atom'}

# ─── Fetch ────────────────────────────────────────────────────────────────────

def fetch_xml():
    query = '+OR+'.join(f'cat:{c}' for c in CATEGORIES)
    url = (
        'https://export.arxiv.org/api/query?'
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
    for pattern, weight, title_only in SCORE_RULES + DOMAIN_RULES:
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

# ─── Goal / finding extraction ────────────────────────────────────────────────

# Sentence splitter: split on ". " followed by a capital letter
_SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

def _sentences(text):
    return [s.strip() for s in _SENT_RE.split(text) if len(s.strip()) > 20]

_GOAL_RE = re.compile(
    r'\b(we|this paper|this work|this study)\b.{0,20}'
    r'\b(propose|present|introduce|develop|design|investigate|study|examine|explore|address|tackle|aim)\b',
    re.IGNORECASE,
)
_RESULT_RE = re.compile(
    r'\b(we|our (method|approach|model|framework|system|results?))\b.{0,20}'
    r'\b(show|demonstrate|achieve|find|outperform|surpass|improve|establish|yield)\b'
    r'|'
    r'\b(state.of.the.art|sota|significant(ly)?|outperform|surpass)\b',
    re.IGNORECASE,
)


def extract_goal(abstract):
    for sent in _sentences(abstract):
        if _GOAL_RE.search(sent):
            return _trunc(sent, 240)
    # fallback: first sentence
    sents = _sentences(abstract)
    return _trunc(sents[0], 240) if sents else ''


def extract_result(abstract):
    sents = _sentences(abstract)
    # prefer later sentences for results (conclusions tend to appear at the end)
    for sent in reversed(sents):
        if _RESULT_RE.search(sent):
            return _trunc(sent, 240)
    return ''

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
.pmeta{font-size:.78rem;color:#8b949e;margin-bottom:10px}
.cats{display:inline-flex;gap:4px;flex-wrap:wrap;margin-top:3px}
.cat{background:#1f6feb18;color:#58a6ff;border:1px solid #1f6feb44;
     border-radius:4px;padding:1px 6px;font-size:.72rem}
.gist{margin-bottom:10px;display:flex;flex-direction:column;gap:5px}
.gist-row{display:flex;gap:8px;font-size:.83rem;line-height:1.45}
.gist-label{color:#3fb950;font-size:.7rem;font-weight:700;text-transform:uppercase;
            letter-spacing:.6px;min-width:56px;padding-top:2px;flex-shrink:0}
.gist-label.res{color:#d29922}
.gist-text{color:#c9d1d9}
details.absbox{margin-bottom:8px}
details.absbox summary{font-size:.78rem;color:#484f58;cursor:pointer;
                        list-style:none;display:flex;align-items:center;gap:4px}
details.absbox summary::-webkit-details-marker{display:none}
details.absbox summary::before{content:'▶';font-size:.6rem;transition:transform .15s}
details.absbox[open] summary::before{transform:rotate(90deg)}
details.absbox summary:hover{color:#8b949e}
.abs{font-size:.83rem;color:#8b949e;margin-top:6px;line-height:1.55;
     padding-left:14px;border-left:2px solid #21262d}
.link{font-size:.78rem;color:#58a6ff;text-decoration:none}
.link:hover{text-decoration:underline}
footer{text-align:center;padding:24px 16px;color:#484f58;
       font-size:.78rem;border-top:1px solid #21262d;margin-top:16px}
@media(max-width:600px){
  .rank{font-size:1.1rem;min-width:28px}
  .body h2{font-size:.93rem}
  .gist-label{min-width:46px}
}
"""

def esc(s):
    return s.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;').replace('"','&quot;')

def safe_url(url):
    """Only allow http/https URLs to prevent javascript: injection in href attributes."""
    return url if url.startswith(('https://', 'http://')) else '#'

def _trunc(s, n=240):
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

        goal = esc(extract_goal(p['abstract']))
        result = esc(extract_result(p['abstract']))
        full_abs = esc(p['abstract'])

        gist_rows = ''
        if goal:
            gist_rows += f'<div class="gist-row"><span class="gist-label">Goal</span><span class="gist-text">{goal}</span></div>'
        if result:
            gist_rows += f'<div class="gist-row"><span class="gist-label res">Finding</span><span class="gist-text">{result}</span></div>'

        cards += f'''
  <div class="paper">
    <div class="rank">#{i}</div>
    <div class="body">
      <h2><a href="{esc(safe_url(p['url']))}" target="_blank" rel="noopener">{esc(p['title'])}</a></h2>
      <div class="pmeta">{authors}<br><span class="cats">{cats}</span></div>
      <div class="gist">{gist_rows}</div>
      <details class="absbox">
        <summary>Full abstract</summary>
        <p class="abs">{full_abs}</p>
      </details>
      <a class="link" href="{esc(safe_url(p['url']))}" target="_blank" rel="noopener">arXiv &rarr;</a>
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
