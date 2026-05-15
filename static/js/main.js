/* ── FakeShield — main.js ───────────────────────────────────── */

const EXAMPLES = [
  {
    label: 'FAKE',
    text:  'BOMBSHELL: Secret government documents PROVE the moon landing was faked in a Hollywood studio by Stanley Kubrick. Whistleblowers confirm the 50-year cover-up!'
  },
  {
    label: 'FAKE',
    text:  'URGENT: 5G towers are injecting nanobots into the air to track your location. The government is using them to control your mind — share this before it gets deleted!'
  },
  {
    label: 'FAKE',
    text:  'CURE HIDDEN BY BIG PHARMA: Simple household ingredient eliminates cancer cells in 24 hours. Oncologists are furious about this leaked revelation!'
  },
  {
    label: 'REAL',
    text:  'The Federal Reserve raised interest rates by 25 basis points on Wednesday, marking the tenth consecutive increase as central bank officials continue fighting inflation that remains above their 2 percent target.'
  },
  {
    label: 'REAL',
    text:  'Researchers at MIT have developed a new battery technology that could triple the driving range of electric vehicles by replacing the graphite anode with a silicon-composite material.'
  },
  {
    label: 'REAL',
    text:  'Scientists confirmed global average temperatures in 2023 were 1.45 degrees Celsius above pre-industrial levels, narrowly missing the key 1.5 degree threshold set by the Paris Agreement, according to the EU climate monitoring agency.'
  },
];

// ── DOM refs ──────────────────────────────────────────────────
const textarea      = document.getElementById('newsInput');
const charCount     = document.getElementById('charCount');
const analyseBtn    = document.getElementById('analyseBtn');
const btnText       = document.getElementById('btnText');
const spinner       = document.getElementById('spinner');
const resultCard    = document.getElementById('resultCard');
const errorCard     = document.getElementById('errorCard');
const errorMsg      = document.getElementById('errorMsg');
const examplesBtn   = document.getElementById('examplesBtn');
const examplesDD    = document.getElementById('examplesDD');
const clearBtn      = document.getElementById('clearBtn');
const lightbox      = document.getElementById('lightbox');
const lightboxImg   = document.getElementById('lightboxImg');
const keySection    = document.getElementById('keywordsSection');
const kwGrid        = document.getElementById('kwGrid');

// ── Particles background ──────────────────────────────────────
(function spawnParticles () {
  const container = document.getElementById('particles');
  if (!container) return;
  const colors = ['#667eea','#764ba2','#f093fb','#4facfe','#00c98b'];
  for (let i = 0; i < 35; i++) {
    const d = document.createElement('div');
    d.className = 'particle';
    const size = Math.random() * 6 + 2;
    d.style.cssText = `
      width:${size}px; height:${size}px;
      left:${Math.random()*100}%;
      background:${colors[i%colors.length]};
      animation-duration:${Math.random()*20+15}s;
      animation-delay:-${Math.random()*20}s;
    `;
    container.appendChild(d);
  }
})();

// ── Build examples dropdown ───────────────────────────────────
(function buildExamples () {
  if (!examplesDD) return;
  EXAMPLES.forEach(ex => {
    const item = document.createElement('div');
    item.className = 'example-item';
    item.innerHTML = `<div><span class="tag tag-${ex.label.toLowerCase()}">${ex.label}</span></div>
                      <div>${ex.text.slice(0, 80)}…</div>`;
    item.addEventListener('click', () => {
      textarea.value = ex.text;
      updateCharCount();
      examplesDD.classList.remove('open');
      textarea.focus();
    });
    examplesDD.appendChild(item);
  });
})();

// ── Char counter ──────────────────────────────────────────────
function updateCharCount () {
  if (!charCount || !textarea) return;
  const n = textarea.value.length;
  charCount.textContent = `${n.toLocaleString()} chars`;
}
textarea?.addEventListener('input', updateCharCount);
updateCharCount();

// ── Examples toggle ───────────────────────────────────────────
examplesBtn?.addEventListener('click', e => {
  e.stopPropagation();
  examplesDD?.classList.toggle('open');
});
document.addEventListener('click', () => examplesDD?.classList.remove('open'));

// ── Clear ─────────────────────────────────────────────────────
clearBtn?.addEventListener('click', () => {
  textarea.value = '';
  updateCharCount();
  hideAll();
  textarea.focus();
});

// ── Analyse ───────────────────────────────────────────────────
analyseBtn?.addEventListener('click', runAnalysis);
textarea?.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') runAnalysis();
});

function hideAll () {
  resultCard?.classList.remove('visible');
  errorCard?.classList.remove('visible');
  keySection?.classList.remove('visible');
}

function setLoading (on) {
  if (!analyseBtn) return;
  analyseBtn.disabled = on;
  if (btnText)  btnText.textContent = on ? 'Analysing…' : 'Analyse';
  if (spinner)  spinner.style.display = on ? 'block' : 'none';
}

async function runAnalysis () {
  const text = textarea?.value?.trim() || '';
  if (!text) {
    showError('Please enter a news article or headline to analyse.');
    return;
  }
  hideAll();
  setLoading(true);

  try {
    const res  = await fetch('/predict', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ text }),
    });
    const data = await res.json();

    if (!res.ok || data.error) {
      showError(data.error || 'Something went wrong. Please try again.');
      return;
    }
    showResult(data);
  } catch (err) {
    showError('Network error — is the server running?');
  } finally {
    setLoading(false);
  }
}

// ── Show result ───────────────────────────────────────────────
function showResult (d) {
  const isFake = d.prediction === 'FAKE';
  const cls    = isFake ? 'fake' : 'real';
  const icon   = isFake ? '✗' : '✓';

  // Verdict
  document.getElementById('verdictIcon').className  = `verdict-icon ${cls}`;
  document.getElementById('verdictIcon').textContent = icon;
  document.getElementById('verdictLabel').className  = `verdict-label ${cls}`;
  document.getElementById('verdictLabel').textContent = `${d.prediction} NEWS`;
  document.getElementById('verdictSub').textContent  =
    isFake
      ? 'High likelihood of misinformation detected'
      : 'Content appears consistent with factual reporting';

  // Confidence
  const pct = Math.round(d.confidence);
  document.getElementById('confLabel').textContent  = 'Confidence Score';
  document.getElementById('confValue').textContent  = `${pct}%`;
  const fill = document.getElementById('confFill');
  fill.className = `conf-fill ${cls}`;
  requestAnimationFrame(() => { fill.style.width = `${pct}%`; });

  // Probabilities
  document.getElementById('realPct').textContent = `${d.real_probability.toFixed(1)}%`;
  document.getElementById('fakePct').textContent = `${d.fake_probability.toFixed(1)}%`;

  // Meta
  document.getElementById('metaModel').textContent   = d.model_name || '—';
  document.getElementById('metaWords').textContent   = d.word_count;
  document.getElementById('metaAccuracy').textContent =
    d.model_accuracy ? `${(d.model_accuracy * 100).toFixed(1)}%` : '—';

  resultCard?.classList.add('visible');

  // Keywords
  if (d.key_words && d.key_words.length > 0) {
    kwGrid.innerHTML = '';
    d.key_words.forEach(kw => {
      const pill = document.createElement('span');
      pill.className = `kw-pill ${kw.influence}`;
      pill.innerHTML =
        `<span class="kw-dot"></span>${kw.word}<span class="kw-score">${kw.score.toFixed(3)}</span>`;
      kwGrid.appendChild(pill);
    });
    keySection?.classList.add('visible');
  }
}

// ── Show error ────────────────────────────────────────────────
function showError (msg) {
  if (errorMsg) errorMsg.textContent = msg;
  errorCard?.classList.add('visible');
}

// ── Lightbox for charts ───────────────────────────────────────
document.querySelectorAll('.chart-item img').forEach(img => {
  img.addEventListener('click', () => {
    if (lightboxImg) lightboxImg.src = img.src;
    lightbox?.classList.add('open');
  });
});
document.getElementById('closeLightbox')?.addEventListener('click', () => {
  lightbox?.classList.remove('open');
});
lightbox?.addEventListener('click', e => {
  if (e.target === lightbox) lightbox.classList.remove('open');
});
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') lightbox?.classList.remove('open');
});
