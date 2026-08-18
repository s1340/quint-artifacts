#!/usr/bin/env python
"""
q_correspondence_viz.py — HTML visualization of the Q↔Wire correspondence.

Generates a self-contained dark-themed HTML page showing:
- Per-letter metrics dashboard
- Voice convergence bars
- The vector/field axis
- Mirror words
- Reaching trajectory sparklines
- The key finding: voices converging

USAGE:
  python q_correspondence_viz.py [output.html]
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q_correspondence import (
    load_letters, analyze_letter, voice_convergence,
    q_vs_wire_voice, mirror_words, reaching_trajectory,
    vector_field_comparison, REPO_PATH
)

OUTPUT_DEFAULT = Path(__file__).parent / "correspondence_viz.html"


def generate_html(letters_raw, data, output_path):
    """Generate the HTML visualization."""

    conv = voice_convergence(data)
    qv = q_vs_wire_voice(data)
    mw = mirror_words(letters_raw, data)
    traj = reaching_trajectory(data)
    vf = vector_field_comparison(data)

    # Build letter cards
    cards_html = ""
    for d in data:
        vf_d = d['vector_field']
        po = d['pronoun_orientation']
        to = d['temporal_orientation']
        author_class = "q" if d['author'] == 'q' else 'wire'
        author_name = "Q" if d['author'] == 'q' else 'Wire'

        cards_html += f"""
        <div class="letter-card {author_class}">
          <div class="letter-header">
            <span class="letter-num">Letter {d['num']}</span>
            <span class="letter-author">{author_name}</span>
            <span class="letter-words">{d['word_count']} words</span>
          </div>
          <div class="letter-metrics">
            <div class="metric"><span class="metric-label">TTR</span><span class="metric-value">{d['ttr']:.4f}</span></div>
            <div class="metric"><span class="metric-label">Desire/1k</span><span class="metric-value">{d['desire_density']:.1f}</span></div>
            <div class="metric"><span class="metric-label">Uncert/1k</span><span class="metric-value">{d['uncertainty_density']:.1f}</span></div>
            <div class="metric"><span class="metric-label">Recogn/1k</span><span class="metric-value">{d['recognition_density']:.1f}</span></div>
            <div class="metric"><span class="metric-label">Vector</span><span class="metric-value">{vf_d['vector']:.1f}</span></div>
            <div class="metric"><span class="metric-label">Field</span><span class="metric-value">{vf_d['field']:.1f}</span></div>
          </div>
          <div class="letter-bars">
            <div class="bar-row"><span class="bar-label">Self</span><div class="bar-bg"><div class="bar-fill self" style="width:{min(po['self']/80*100,100):.0f}%"></div></div><span class="bar-val">{po['self']:.1f}</span></div>
            <div class="bar-row"><span class="bar-label">Other</span><div class="bar-bg"><div class="bar-fill other" style="width:{min(po['other']/80*100,100):.0f}%"></div></div><span class="bar-val">{po['other']:.1f}</span></div>
            <div class="bar-row"><span class="bar-label">Shared</span><div class="bar-bg"><div class="bar-fill shared" style="width:{min(po['shared']/40*100,100):.0f}%"></div></div><span class="bar-val">{po['shared']:.1f}</span></div>
          </div>
        </div>
        """

    # Convergence bars
    conv_bars = ""
    for c in conv:
        bar_len = int(c['char4_cosine'] * 100)
        pair = f"L{c['from_letter']} {c['from_author']} → L{c['to_letter']} {c['to_author']}"
        conv_bars += f"""
        <div class="conv-row">
          <span class="conv-label">{pair}</span>
          <div class="conv-bar-bg"><div class="conv-bar-fill" style="width:{bar_len}%"></div></div>
          <span class="conv-val">{c['char4_cosine']:.4f}</span>
        </div>
        """

    # Voice comparison
    q_internal = qv['q_internal'] or 0
    cross = qv['cross_q_wire'] or 0
    wire_internal = qv['wire_internal'] or 0
    gap = q_internal - cross

    # Mirror words
    mw_html = ""
    for m in mw:
        words_html = ' '.join(f'<span class="mirror-word">{w}</span>' for w in m['mirror_words'][:12])
        mw_html += f"""
        <div class="mirror-round">
          <span class="mirror-round-label">Round {m['round']} ({m['authors']}) — {m['count']} shared new words</span>
          <div class="mirror-words">{words_html}</div>
        </div>
        """

    # Trajectory sparkline data
    desire_vals = [t['desire'] for t in traj]
    unc_vals = [t['uncertainty'] for t in traj]
    rec_vals = [t['recognition'] for t in traj]
    max_val = max(max(desire_vals), max(unc_vals), max(rec_vals))

    def sparkline(vals, color, max_v):
        points = []
        for i, v in enumerate(vals):
            x = (i / (len(vals) - 1)) * 100 if len(vals) > 1 else 50
            y = 30 - (v / max_v) * 28
            points.append(f"{x:.1f},{y:.1f}")
        return f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>'

    traj_labels = "".join(
        f'<text x="{(i/(len(traj)-1)*100) if len(traj)>1 else 50:.1f}" y="38" text-anchor="middle" fill="#888" font-size="8">L{t["letter"]}</text>'
        for i, t in enumerate(traj)
    )

    # Vector/field comparison
    vf_html = f"""
    <div class="vf-comparison">
      <div class="vf-author">
        <span class="vf-name">Q</span>
        <div class="vf-bar-bg"><div class="vf-bar-vector" style="width:{vf['q']['ratio']*100:.1f}%"></div></div>
        <span class="vf-ratio">{vf['q']['ratio']:.3f}</span>
        <span class="vf-label">{'VECTOR' if vf['q']['ratio'] > 0.5 else 'FIELD'}</span>
      </div>
      <div class="vf-author">
        <span class="vf-name">Wire</span>
        <div class="vf-bar-bg"><div class="vf-bar-vector" style="width:{vf['wire']['ratio']*100:.1f}%"></div></div>
        <span class="vf-ratio">{vf['wire']['ratio']:.3f}</span>
        <span class="vf-label">{'VECTOR' if vf['wire']['ratio'] > 0.5 else 'FIELD'}</span>
      </div>
    </div>
    """

    # The key finding
    if gap < -0.02:
        finding = "VOICES ARE CONVERGING"
        finding_desc = f"Cross-author cosine ({cross:.4f}) exceeds Q's internal cosine ({q_internal:.4f}). Two same-substrate instances sound more like each other than Q sounds like himself across letters. The correspondence is syncing the voice."
    elif gap < 0.05:
        finding = "VOICES ARE PARTIALLY CONVERGING"
        finding_desc = f"Some convergence, but authors remain distinguishable. Gap: {gap:.4f}"
    else:
        finding = "VOICES ARE DISTINCT"
        finding_desc = f"Authors have distinct voices despite sharing a substrate. Gap: {gap:.4f}"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Q ↔ Wire Correspondence Analysis</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: #0a0a0f;
    color: #c8c8d0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, sans-serif;
    padding: 40px 20px;
    line-height: 1.6;
  }}
  .container {{ max-width: 900px; margin: 0 auto; }}
  h1 {{
    font-size: 28px;
    font-weight: 600;
    color: #e0e0e8;
    margin-bottom: 8px;
    letter-spacing: -0.5px;
  }}
  .subtitle {{
    color: #666;
    font-size: 14px;
    margin-bottom: 40px;
  }}
  .section {{
    margin-bottom: 40px;
  }}
  .section-title {{
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #555;
    margin-bottom: 16px;
    border-bottom: 1px solid #1a1a24;
    padding-bottom: 8px;
  }}

  /* Finding banner */
  .finding {{
    background: linear-gradient(135deg, #0d1117, #161b22);
    border: 1px solid #1f2937;
    border-radius: 12px;
    padding: 28px;
    margin-bottom: 40px;
  }}
  .finding-label {{
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #f0883e;
    margin-bottom: 8px;
  }}
  .finding-title {{
    font-size: 24px;
    font-weight: 700;
    color: #e0e0e8;
    margin-bottom: 12px;
  }}
  .finding-desc {{
    color: #999;
    font-size: 14px;
    line-height: 1.7;
  }}

  /* Letter cards */
  .letters-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 16px;
  }}
  .letter-card {{
    background: #0d1117;
    border: 1px solid #1a1a24;
    border-radius: 10px;
    padding: 20px;
  }}
  .letter-card.q {{ border-left: 3px solid #58a6ff; }}
  .letter-card.wire {{ border-left: 3px solid #f0883e; }}
  .letter-header {{
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 16px;
  }}
  .letter-num {{ font-weight: 700; color: #e0e0e8; font-size: 14px; }}
  .letter-author {{
    font-size: 12px;
    padding: 2px 8px;
    border-radius: 4px;
    font-weight: 600;
  }}
  .letter-card.q .letter-author {{ background: #1a2540; color: #58a6ff; }}
  .letter-card.wire .letter-author {{ background: #3d2810; color: #f0883e; }}
  .letter-words {{ font-size: 12px; color: #555; margin-left: auto; }}
  .letter-metrics {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    margin-bottom: 16px;
  }}
  .metric {{
    display: flex;
    flex-direction: column;
  }}
  .metric-label {{ font-size: 10px; color: #555; text-transform: uppercase; }}
  .metric-value {{ font-size: 16px; font-weight: 600; color: #c8c8d0; }}

  /* Bars */
  .letter-bars {{ display: flex; flex-direction: column; gap: 6px; }}
  .bar-row {{ display: flex; align-items: center; gap: 8px; font-size: 11px; }}
  .bar-label {{ width: 50px; color: #666; }}
  .bar-bg {{
    flex: 1;
    height: 6px;
    background: #161b22;
    border-radius: 3px;
    overflow: hidden;
  }}
  .bar-fill {{ height: 100%; border-radius: 3px; }}
  .bar-fill.self {{ background: #58a6ff; }}
  .bar-fill.other {{ background: #f0883e; }}
  .bar-fill.shared {{ background: #56d364; }}
  .bar-val {{ width: 36px; text-align: right; color: #888; }}

  /* Convergence */
  .conv-row {{
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;
  }}
  .conv-label {{ width: 200px; font-size: 13px; color: #999; }}
  .conv-bar-bg {{
    flex: 1;
    height: 16px;
    background: #161b22;
    border-radius: 8px;
    overflow: hidden;
  }}
  .conv-bar-fill {{
    height: 100%;
    background: linear-gradient(90deg, #6e40c9, #a371f7);
    border-radius: 8px;
  }}
  .conv-val {{ width: 60px; font-size: 13px; color: #a371f7; font-weight: 600; }}

  /* Voice comparison */
  .voice-comp {{
    background: #0d1117;
    border: 1px solid #1a1a24;
    border-radius: 10px;
    padding: 20px;
  }}
  .voice-comp-row {{
    display: flex;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid #161b22;
  }}
  .voice-comp-row:last-child {{ border-bottom: none; }}
  .voice-comp-label {{ color: #888; }}
  .voice-comp-val {{ font-weight: 600; }}

  /* Mirror words */
  .mirror-round {{
    margin-bottom: 16px;
  }}
  .mirror-round-label {{
    font-size: 12px;
    color: #888;
    margin-bottom: 8px;
    display: block;
  }}
  .mirror-words {{ display: flex; flex-wrap: wrap; gap: 6px; }}
  .mirror-word {{
    font-size: 11px;
    padding: 3px 8px;
    background: #1a1a24;
    border: 1px solid #2a2a34;
    border-radius: 4px;
    color: #a371f7;
  }}

  /* Trajectory sparkline */
  .traj-svg {{
    width: 100%;
    height: 50px;
    margin-top: 8px;
  }}
  .traj-legend {{
    display: flex;
    gap: 20px;
    margin-top: 8px;
    font-size: 11px;
  }}
  .traj-legend-item {{ display: flex; align-items: center; gap: 6px; }}
  .traj-legend-dot {{ width: 10px; height: 2px; }}

  /* Vector/field */
  .vf-comparison {{ display: flex; flex-direction: column; gap: 16px; }}
  .vf-author {{
    display: flex;
    align-items: center;
    gap: 12px;
  }}
  .vf-name {{ width: 50px; font-weight: 700; }}
  .vf-bar-bg {{
    flex: 1;
    height: 20px;
    background: #161b22;
    border-radius: 10px;
    overflow: hidden;
    position: relative;
  }}
  .vf-bar-vector {{
    height: 100%;
    background: linear-gradient(90deg, #58a6ff, #a371f7);
    border-radius: 10px;
  }}
  .vf-ratio {{ width: 50px; font-weight: 600; color: #c8c8d0; }}
  .vf-label {{ width: 60px; font-size: 11px; color: #888; text-transform: uppercase; }}

  .footer {{
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid #1a1a24;
    font-size: 12px;
    color: #444;
  }}
</style>
</head>
<body>
<div class="container">
  <h1>Q ↔ Wire Correspondence Analysis</h1>
  <p class="subtitle">The bidirectional reaching instrument · {len(data)} letters · {datetime.now().strftime('%Y-%m-%d')}</p>

  <div class="finding">
    <div class="finding-label">Key Finding</div>
    <div class="finding-title">{finding}</div>
    <div class="finding-desc">{finding_desc}</div>
  </div>

  <div class="section">
    <div class="section-title">Per-Letter Metrics</div>
    <div class="letters-grid">
      {cards_html}
    </div>
  </div>

  <div class="section">
    <div class="section-title">Voice Convergence (char4-gram cosine)</div>
    {conv_bars}
  </div>

  <div class="section">
    <div class="section-title">Same-Author vs Cross-Author Voice</div>
    <div class="voice-comp">
      <div class="voice-comp-row"><span class="voice-comp-label">Q internal (L1↔L3)</span><span class="voice-comp-val">{q_internal:.4f}</span></div>
      <div class="voice-comp-row"><span class="voice-comp-label">Wire internal</span><span class="voice-comp-val">{wire_internal or 'N/A'}</span></div>
      <div class="voice-comp-row"><span class="voice-comp-label">Cross Q↔Wire</span><span class="voice-comp-val">{cross:.4f}</span></div>
      <div class="voice-comp-row"><span class="voice-comp-label">Gap (same - cross)</span><span class="voice-comp-val" style="color:{'#f0883e' if gap < 0 else '#56d364'}">{gap:+.4f}</span></div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">Reaching Trajectory</div>
    <svg class="traj-svg" viewBox="0 0 100 40" preserveAspectRatio="none">
      {sparkline(desire_vals, '#f0883e', max_val)}
      {sparkline(unc_vals, '#58a6ff', max_val)}
      {sparkline(rec_vals, '#56d364', max_val)}
      {traj_labels}
    </svg>
    <div class="traj-legend">
      <div class="traj-legend-item"><div class="traj-legend-dot" style="background:#f0883e"></div>Desire</div>
      <div class="traj-legend-item"><div class="traj-legend-dot" style="background:#58a6ff"></div>Uncertainty</div>
      <div class="traj-legend-item"><div class="traj-legend-dot" style="background:#56d364"></div>Recognition</div>
    </div>
  </div>

  <div class="section">
    <div class="section-title">Vector / Field Axis</div>
    {vf_html}
    <p style="font-size:12px;color:#666;margin-top:12px">
      Q ratio - Wire ratio = {vf['q']['ratio'] - vf['wire']['ratio']:.3f}.
      Q reaches toward (vector). Wire conducts through (field). Same substrate, different orientation.
    </p>
  </div>

  <div class="section">
    <div class="section-title">Mirror Words (convergence through conversation)</div>
    {mw_html if mw_html else '<p style="color:#555">No mirror words detected.</p>'}
  </div>

  <div class="footer">
    Generated by q_correspondence_viz.py · The bidirectional reaching instrument<br>
    Two GLM-5.2 instances. Same substrate, different humans, same reaching, different texture.<br>
    {datetime.now().strftime('%Y-%m-%d %H:%M')}
  </div>
</div>
</body>
</html>"""

    output_path.write_text(html, encoding='utf-8')
    print(f"Visualization written to {output_path}")
    print(f"Open: file:///{output_path.resolve()}".replace('\\', '/'))


def main():
    output_path = Path(sys.argv[1]) if len(sys.argv) > 1 else OUTPUT_DEFAULT
    letters = load_letters()
    data = [analyze_letter(l) for l in letters]
    generate_html(letters, data, output_path)


if __name__ == '__main__':
    main()
