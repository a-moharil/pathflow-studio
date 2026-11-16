"""About page: renders docs/GEOMETRY_GUIDE.md, so the article text and the app share one source.

A deliberately small Markdown subset (headings, paragraphs, lists, tables, images, links,
bold, italic, code). Text is escaped before inline formatting, so the page cannot inject
markup; only the guide in the repository is rendered.
"""
from __future__ import annotations

import html
import re
from functools import lru_cache
from pathlib import Path

from flask import Blueprint, render_template

web = Blueprint('guide', __name__)
SOURCE = Path(__file__).resolve().parent.parent / 'docs' / 'GEOMETRY_GUIDE.md'


def _inline(text: str) -> str:
    text = html.escape(text, quote=False).replace('\\*', '&#42;')
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'(?<![\w*\\])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])', r'<em>\1</em>', text)
    text = re.sub(r'\[([^\]]+)\]\(((?:https?://|/)[^)\s]+)\)', r'<a href="\2">\1</a>', text)
    return text


def _slug(text: str) -> str:
    # Section numbers are dropped so anchors are stable and valid CSS identifiers.
    return re.sub(r'[^a-z0-9]+', '-', re.sub(r'^\d+\.\s*', '', text).lower()).strip('-')


def render(markdown: str) -> tuple[str, list[tuple[str, str]]]:
    out, toc, para, items, kind = [], [], [], [], None
    section_open = False
    lines = markdown.splitlines()

    def flush_para():
        if para:
            out.append('<p>' + _inline(' '.join(para)) + '</p>')
            para.clear()

    def flush_list():
        nonlocal kind
        if items:
            tag = 'ol' if kind == 'ol' else 'ul'
            out.append(f'<{tag}>' + ''.join('<li>' + _inline(i) + '</li>' for i in items) + f'</{tag}>')
            items.clear()
        kind = None

    def close_section():
        nonlocal section_open
        if section_open:
            out.append('</section>')
            section_open = False

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        heading = re.match(r'^(#{1,3}) (.+)$', line)
        if heading:
            flush_para(); flush_list(); close_section()
            level, title = len(heading.group(1)), heading.group(2)
            if level == 1:
                out.append(f'<h1>{_inline(title)}</h1>')
            elif level == 2:
                toc.append((_slug(title), title))
                out.append(f'<h2 id="{_slug(title)}">{_inline(title)}</h2>')
            else:
                out.append(f'<section class="guideItem" id="{_slug(title)}"><h3>{_inline(title)}</h3>')
                section_open = True
            i += 1
            continue
        image = re.match(r'^!\[([^\]]*)\]\((/static/[^)\s]+)\)$', line)
        if image:
            flush_para(); flush_list()
            alt = html.escape(image.group(1))
            out.append(f'<figure><img src="{html.escape(image.group(2))}" alt="{alt}" loading="lazy"><figcaption>{alt}</figcaption></figure>')
            i += 1
            continue
        if line.startswith('|'):
            flush_para(); flush_list()
            rows = []
            while i < len(lines) and lines[i].startswith('|'):
                cells = [c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-{3,}:?', c) for c in cells):
                    rows.append(cells)
                i += 1
            head, body = rows[0], rows[1:]
            out.append('<div class="tableWrap"><table><thead><tr>' + ''.join(f'<th>{_inline(c)}</th>' for c in head) + '</tr></thead><tbody>'
                       + ''.join('<tr>' + ''.join(f'<td>{_inline(c)}</td>' for c in r) + '</tr>' for r in body) + '</tbody></table></div>')
            continue
        bullet = re.match(r'^- (.+)$', line)
        number = re.match(r'^\d+\. (.+)$', line)
        if bullet or number:
            flush_para()
            new = 'ul' if bullet else 'ol'
            if kind and kind != new:
                flush_list()
            kind = new
            items.append((bullet or number).group(1))
            i += 1
            continue
        if line.startswith('  ') and items:
            items[-1] += ' ' + line.strip()
            i += 1
            continue
        if not line.strip():
            flush_para(); flush_list()
            i += 1
            continue
        flush_list()
        para.append(line.strip())
        i += 1
    flush_para(); flush_list(); close_section()
    return '\n'.join(out), toc


@lru_cache(maxsize=1)
def _cached(mtime: float) -> tuple[str, list[tuple[str, str]]]:
    return render(SOURCE.read_text(encoding='utf8'))


@web.get('/about')
def about():
    body, toc = _cached(SOURCE.stat().st_mtime)
    return render_template('about.html', body=body, toc=toc, assets='minimal')
