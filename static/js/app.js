/**
 * AURA-Term™ Term Deposit Marketing Campaign Propensity Portal
 * Client Application Controller
 */

document.addEventListener('DOMContentLoaded', () => {
  // App State
  let currentProfileData = null;
  let currentBatchData = null;
  let allProfiles = [];

  // DOM Elements
  const tabButtons = document.querySelectorAll('.nav-tab');
  const tabPanes = document.querySelectorAll('.tab-pane');
  const form = document.getElementById('underwriterForm');
  const inputAge = document.getElementById('inputAge');
  const inputBalance = document.getElementById('inputBalance');
  const balanceDisplay = document.getElementById('balanceDisplay');
  const derivedAgeGroup = document.getElementById('derivedAgeGroup');
  const derivedBalanceStatus = document.getElementById('derivedBalanceStatus');
  const gaugeCanvas = document.getElementById('gaugeCanvas');
  const scorePercentageDisplay = document.getElementById('scorePercentageDisplay');
  const decisionTierPill = document.getElementById('decisionTierPill');
  const latencyDisplay = document.getElementById('latencyDisplay');
  const confidenceDisplay = document.getElementById('confidenceDisplay');
  const projectedDepositDisplay = document.getElementById('projectedDepositDisplay');
  const requestIdBadge = document.getElementById('requestIdBadge');
  const recTitle = document.getElementById('recTitle');
  const recDescription = document.getElementById('recDescription');
  const attributionList = document.getElementById('attributionList');
  const liveClock = document.getElementById('liveClock');

  // Sensitivity Elements
  const stressBalanceSlider = document.getElementById('stressBalanceSlider');
  const stressBalanceLabel = document.getElementById('stressBalanceLabel');
  const stressAgeSlider = document.getElementById('stressAgeSlider');
  const stressAgeLabel = document.getElementById('stressAgeLabel');
  const btnRefreshCurves = document.getElementById('btnRefreshCurves');
  const balanceChartCanvas = document.getElementById('balanceChartCanvas');
  const ageChartCanvas = document.getElementById('ageChartCanvas');
  const monthChartCanvas = document.getElementById('monthChartCanvas');

  // Batch Portfolio Elements
  const batchTotalCount = document.getElementById('batchTotalCount');
  const batchAvgPropensity = document.getElementById('batchAvgPropensity');
  const batchInflowDisplay = document.getElementById('batchInflowDisplay');
  const batchTier1Count = document.getElementById('batchTier1Count');
  const portfolioTableBody = document.getElementById('portfolioTableBody');
  const cohortButtons = document.querySelectorAll('.btn-cohort[data-cohort]');
  const btnTriggerUpload = document.getElementById('btnTriggerUpload');
  const csvFileInput = document.getElementById('csvFileInput');
  const btnExportAuditCsv = document.getElementById('btnExportAuditCsv');

  // Governance Elements
  const specModelSha256 = document.getElementById('specModelSha256');
  const fairnessTableBody = document.getElementById('fairnessTableBody');
  const auditLogContainer = document.getElementById('auditLogContainer');

  // 1. Live UTC Clock
  function updateClock() {
    const now = new Date();
    if (liveClock) {
      liveClock.textContent = now.toISOString().slice(11, 19) + ' UTC';
    }
  }
  setInterval(updateClock, 1000);
  updateClock();

  // 2. Navigation Tab Switching
  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetPaneId = btn.getAttribute('data-tab');
      tabButtons.forEach(b => b.classList.remove('active'));
      tabPanes.forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const targetPane = document.getElementById(targetPaneId);
      if (targetPane) {
        targetPane.classList.add('active');
      }

      // Trigger redraw of charts on tab display
      if (targetPaneId === 'sensitivityPane') {
        loadSensitivityCurves();
      } else if (targetPaneId === 'governancePane') {
        loadGovernanceData();
      }
    });
  });

  // 3. Real-Time Feature Engineering Derivation
  function updateDerivedFeatures() {
    const age = parseFloat(inputAge.value) || 40;
    const balance = parseFloat(inputBalance.value) || 0;

    // Format balance helper
    if (balanceDisplay) {
      balanceDisplay.textContent = `€${balance.toLocaleString()}`;
    }

    let ageGroup = 'Mid_Career';
    if (age < 25) ageGroup = 'Young';
    else if (age < 35) ageGroup = 'Early_Career';
    else if (age < 50) ageGroup = 'Mid_Career';
    else if (age < 65) ageGroup = 'Older';
    else ageGroup = 'Senior';

    let balStatus = 'Positive';
    if (balance < 0) balStatus = 'Negative';
    else if (balance === 0) balStatus = 'Zero';

    if (derivedAgeGroup) derivedAgeGroup.textContent = `Age Cohort: ${ageGroup}`;
    if (derivedBalanceStatus) derivedBalanceStatus.textContent = `Liquidity: ${balStatus}`;
  }

  inputAge.addEventListener('input', updateDerivedFeatures);
  inputBalance.addEventListener('input', updateDerivedFeatures);

  // 4. Form Serialization
  function getFormData() {
    const formData = new FormData(form);
    const data = {};
    formData.forEach((value, key) => {
      if (['age', 'balance', 'pdays', 'previous'].includes(key)) {
        data[key] = parseFloat(value);
      } else {
        data[key] = value;
      }
    });
    return data;
  }

  function setFormData(data) {
    for (const key in data) {
      const el = form.elements[key];
      if (el) {
        el.value = data[key];
      }
    }
    updateDerivedFeatures();
    if (stressBalanceSlider && data.balance !== undefined) {
      stressBalanceSlider.value = Math.min(100000, Math.max(0, data.balance));
      stressBalanceLabel.textContent = `€${parseFloat(data.balance).toLocaleString()}`;
    }
    if (stressAgeSlider && data.age !== undefined) {
      stressAgeSlider.value = data.age;
      stressAgeLabel.textContent = `${data.age} Years`;
    }
  }

  // 5. Predict Client Execution
  async function executePrediction() {
    const payload = getFormData();
    currentProfileData = payload;

    try {
      if (requestIdBadge) requestIdBadge.textContent = 'SCORING...';
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        throw new Error(`Server returned ${res.status}`);
      }

      const result = await res.json();
      renderPredictionResult(result);
    } catch (err) {
      console.error('Prediction failed:', err);
      alert('Propensity scoring failed: ' + err.message);
    }
  }

  function renderPredictionResult(result) {
    const pct = result.propensity_percentage;
    const dec = result.decision;

    // Update Request & Latency Badges
    if (requestIdBadge) requestIdBadge.textContent = result.request_id;
    if (latencyDisplay) latencyDisplay.textContent = `${result.inference_time_ms} ms`;
    if (confidenceDisplay) confidenceDisplay.textContent = result.confidence_rating;
    if (projectedDepositDisplay) {
      projectedDepositDisplay.textContent = `€${dec.estimated_deposit_volume.toLocaleString()}`;
    }

    // Animate Percentage
    if (scorePercentageDisplay) {
      scorePercentageDisplay.textContent = `${pct}%`;
    }

    // Tier Badge & Styling
    if (decisionTierPill) {
      decisionTierPill.textContent = dec.tier_name.toUpperCase();
      decisionTierPill.className = 'decision-tier-pill';
      if (pct >= 70) {
        decisionTierPill.classList.add('tier-badge-high');
      } else if (pct >= 45) {
        decisionTierPill.classList.add('tier-badge-mod');
      } else {
        decisionTierPill.classList.add('tier-badge-low');
      }
    }

    // Recommendation Box
    if (recTitle) recTitle.textContent = dec.action_title;
    if (recDescription) recDescription.textContent = dec.action_description;

    // Draw Radial Gauge Canvas
    BankCharts.drawRadialGauge(gaugeCanvas, pct, dec.tier_color);

    // Render XAI Feature Attribution Waterfall
    renderAttributionList(result.feature_attributions);
  }

  function renderAttributionList(attributions) {
    if (!attributionList) return;
    attributionList.innerHTML = '';

    if (!attributions || attributions.length === 0) {
      const empty = document.createElement('div');
      empty.style.cssText = 'color: var(--text-muted); font-size: 0.75rem;';
      empty.textContent = 'Attribution computing...';
      attributionList.appendChild(empty);
      return;
    }

    attributions.slice(0, 6).forEach(attr => {
      const isPos = attr.direction === 'positive';
      const sign = isPos ? '+' : '-';
      const impactFormatted = `${sign}${attr.magnitude}%`;
      const barWidth = Math.min(100, attr.magnitude * 3.5);

      const item = document.createElement('div');
      item.className = 'attribution-item';

      // Info row
      const infoRow = document.createElement('div');
      infoRow.className = 'attr-info-row';

      const nameSpan = document.createElement('span');
      nameSpan.className = 'attr-name';
      nameSpan.textContent = attr.display_name + ' ';
      const valueTag = document.createElement('span');
      valueTag.style.cssText = 'color: var(--text-faint); font-family: var(--font-mono); font-size: 0.68rem;';
      valueTag.textContent = `[${attr.current_value}]`;
      nameSpan.appendChild(valueTag);

      const impactSpan = document.createElement('span');
      impactSpan.className = `attr-impact-val ${isPos ? 'pos' : 'neg'}`;
      impactSpan.textContent = impactFormatted;

      infoRow.appendChild(nameSpan);
      infoRow.appendChild(impactSpan);

      // Bar track
      const track = document.createElement('div');
      track.className = 'attr-bar-track';
      const fill = document.createElement('div');
      fill.className = `attr-bar-fill ${isPos ? 'pos' : 'neg'}`;
      fill.style.width = `${barWidth}%`;
      track.appendChild(fill);

      item.appendChild(infoRow);
      item.appendChild(track);
      attributionList.appendChild(item);
    });
  }

  // Form Submit Handler
  form.addEventListener('submit', (e) => {
    e.preventDefault();
    executePrediction();
  });

  // 6. Pre-load Benchmark Profiles
  async function loadProfiles() {
    try {
      const res = await fetch('/api/profiles');
      if (res.ok) {
        allProfiles = await res.json();
      }
    } catch (e) {
      console.warn('Could not load profiles from API', e);
    }
  }

  document.querySelectorAll('.preset-chip').forEach(btn => {
    btn.addEventListener('click', () => {
      const pid = btn.getAttribute('data-profile');
      const found = allProfiles.find(p => p.id === pid);
      if (found) {
        setFormData(found.data);
        executePrediction();
      }
    });
  });

  // 7. Sensitivity Simulator Logic
  async function loadSensitivityCurves() {
    const payload = getFormData();
    try {
      const res = await fetch('/api/sensitivity', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (!res.ok) throw new Error('Sensitivity fetch error');
      const curves = await res.json();

      // Draw Curves
      BankCharts.drawLineChart(
        balanceChartCanvas, 
        curves.balance_curve.x, 
        curves.balance_curve.y,
        { formatX: (val) => val >= 1000 ? `€${val/1000}k` : `€${val}` }
      );

      BankCharts.drawLineChart(
        ageChartCanvas, 
        curves.age_curve.x, 
        curves.age_curve.y,
        { formatX: (val) => `${val}y` }
      );

      BankCharts.drawMonthlyBarChart(
        monthChartCanvas,
        curves.month_curve.x,
        curves.month_curve.y
      );
    } catch (e) {
      console.error('Sensitivity error:', e);
    }
  }

  if (stressBalanceSlider) {
    stressBalanceSlider.addEventListener('input', () => {
      const val = parseFloat(stressBalanceSlider.value);
      stressBalanceLabel.textContent = `€${val.toLocaleString()}`;
      inputBalance.value = val;
      updateDerivedFeatures();
    });
    stressBalanceSlider.addEventListener('change', () => {
      executePrediction();
      loadSensitivityCurves();
    });
  }

  if (stressAgeSlider) {
    stressAgeSlider.addEventListener('input', () => {
      const val = parseInt(stressAgeSlider.value);
      stressAgeLabel.textContent = `${val} Years`;
      inputAge.value = val;
      updateDerivedFeatures();
    });
    stressAgeSlider.addEventListener('change', () => {
      executePrediction();
      loadSensitivityCurves();
    });
  }

  if (btnRefreshCurves) {
    btnRefreshCurves.addEventListener('click', loadSensitivityCurves);
  }

  // 8. Institutional Portfolio Batch Audit
  async function loadPortfolioCohort(cohortId = 'sovereign_hnw') {
    if (portfolioTableBody) {
      portfolioTableBody.innerHTML = '';
      const loadingTr = document.createElement('tr');
      const loadingTd = document.createElement('td');
      loadingTd.colSpan = 11;
      loadingTd.style.cssText = 'text-align: center; padding: 2.5rem; color: var(--text-muted);';
      loadingTd.textContent = 'Scoring Portfolio Cohort via Vectorized Neural Forward Pass...';
      loadingTr.appendChild(loadingTd);
      portfolioTableBody.appendChild(loadingTr);
    }

    try {
      const res = await fetch(`/api/cohorts/${cohortId}`);
      if (!res.ok) throw new Error('Failed to fetch cohort');
      const data = await res.json();
      currentBatchData = data;
      renderPortfolioResults(data);
    } catch (e) {
      console.error('Cohort error:', e);
    }
  }

  function renderPortfolioResults(data) {
    if (batchTotalCount) batchTotalCount.textContent = data.total_accounts_scored.toLocaleString();
    if (batchAvgPropensity) batchAvgPropensity.textContent = `${data.average_propensity}%`;
    if (batchInflowDisplay) batchInflowDisplay.textContent = `€${(data.total_projected_inflow_eur / 1000000).toFixed(2)}M`;
    if (batchTier1Count) batchTier1Count.textContent = data.high_propensity_accounts;

    if (!portfolioTableBody) return;
    portfolioTableBody.innerHTML = '';

    data.accounts.forEach(acc => {
      const tr = document.createElement('tr');
      const score = acc.propensity_score;
      let badgeClass = 'score-low';
      if (score >= 70) badgeClass = 'score-high';
      else if (score >= 45) badgeClass = 'score-mid';

      // Helper to create a safe <td>
      function safeTd(text, style) {
        const td = document.createElement('td');
        if (style) td.style.cssText = style;
        td.textContent = text;
        return td;
      }

      tr.appendChild(safeTd(acc.account_id, 'font-family: var(--font-mono); font-size: 0.72rem; color: #818cf8;'));
      tr.appendChild(safeTd(String(acc.age)));
      tr.appendChild(safeTd(acc.job, 'text-transform: capitalize;'));
      tr.appendChild(safeTd(`€${acc.balance.toLocaleString()}`, 'font-family: var(--font-mono);'));

      // Housing — icon + safe text
      const tdHousing = document.createElement('td');
      tdHousing.textContent = acc.housing === 'yes' ? '🔴 Yes' : '⚪ None';
      tr.appendChild(tdHousing);

      const tdLoan = document.createElement('td');
      tdLoan.textContent = acc.loan === 'yes' ? '🔴 Yes' : '⚪ None';
      tr.appendChild(tdLoan);

      tr.appendChild(safeTd(acc.month, 'text-transform: uppercase; font-family: var(--font-mono);'));
      tr.appendChild(safeTd(acc.poutcome, 'text-transform: capitalize;'));

      const tdScore = document.createElement('td');
      const badge = document.createElement('span');
      badge.className = `score-badge-cell ${badgeClass}`;
      badge.textContent = `${score}%`;
      tdScore.appendChild(badge);
      tr.appendChild(tdScore);

      tr.appendChild(safeTd(acc.tier));
      tr.appendChild(safeTd(`€${acc.expected_deposit.toLocaleString()}`, 'font-family: var(--font-mono); color: #34d399;'));

      portfolioTableBody.appendChild(tr);
    });
  }

  cohortButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      cohortButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const cid = btn.getAttribute('data-cohort');
      loadPortfolioCohort(cid);
    });
  });

  // CSV File Upload Trigger
  if (btnTriggerUpload && csvFileInput) {
    btnTriggerUpload.addEventListener('click', () => csvFileInput.click());
    csvFileInput.addEventListener('change', async (e) => {
      const file = e.target.files[0];
      if (!file) return;

      const formData = new FormData();
      formData.append('file', file);

      if (portfolioTableBody) {
        portfolioTableBody.innerHTML = '';
        const loadingTr = document.createElement('tr');
        const loadingTd = document.createElement('td');
        loadingTd.colSpan = 11;
        loadingTd.style.cssText = 'text-align: center; padding: 2.5rem; color: #38bdf8;';
        loadingTd.textContent = 'Uploading & Scoring Portfolio CSV via Keras Engine...';
        loadingTr.appendChild(loadingTd);
        portfolioTableBody.appendChild(loadingTr);
      }

      try {
        const res = await fetch('/api/batch-upload', {
          method: 'POST',
          body: formData
        });
        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || 'Upload failed');
        }
        const data = await res.json();
        currentBatchData = data;
        renderPortfolioResults(data);
      } catch (err) {
        alert('CSV Batch Scoring Error: ' + err.message);
        loadPortfolioCohort('sovereign_hnw');
      }
    });
  }

  // Export Basel Regulatory Audit Report
  if (btnExportAuditCsv) {
    btnExportAuditCsv.addEventListener('click', () => {
      if (!currentBatchData || !currentBatchData.accounts) {
        alert('No batch portfolio data available to export.');
        return;
      }
      const headers = ['Account_ID', 'Age', 'Job', 'Balance_EUR', 'Housing_Mortgage', 'Consumer_Loan', 'Month', 'Prior_Outcome', 'Propensity_Score_Pct', 'Subscribed_Pred', 'Tier', 'Est_Deposit_Volume_EUR'];
      const rows = currentBatchData.accounts.map(a => [
        a.account_id,
        a.age,
        a.job,
        a.balance,
        a.housing,
        a.loan,
        a.month,
        a.poutcome,
        a.propensity_score,
        a.subscribed_pred ? 'YES' : 'NO',
        `"${a.tier}"`,
        a.expected_deposit
      ]);

      const csvContent = 'data:text/csv;charset=utf-8,' + [headers.join(','), ...rows.map(e => e.join(','))].join('\n');
      const encodedUri = encodeURI(csvContent);
      const link = document.createElement('a');
      link.setAttribute('href', encodedUri);
      link.setAttribute('download', `AURA_Portfolio_Report_${Date.now()}.csv`);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    });
  }

  // 9. Model Governance & Audit Registry
  async function loadGovernanceData() {
    try {
      const res = await fetch('/api/governance');
      if (!res.ok) throw new Error('Governance fetch error');
      const gov = await res.json();

      if (specModelSha256) specModelSha256.textContent = gov.sha256_checksum;

      // Populate Fairness Table
      if (fairnessTableBody) {
        fairnessTableBody.innerHTML = '';
        gov.demographic_fairness_parity.forEach(p => {
          const tr = document.createElement('tr');

          function safeTd(text, style) {
            const td = document.createElement('td');
            if (style) td.style.cssText = style;
            td.textContent = text;
            return td;
          }

          tr.appendChild(safeTd(p.cohort));
          tr.appendChild(safeTd(p.sample_representation, 'font-family: var(--font-mono);'));
          tr.appendChild(safeTd(p.disparate_impact_ratio, 'font-family: var(--font-mono);'));

          const tdStatus = document.createElement('td');
          const statusSpan = document.createElement('span');
          statusSpan.style.cssText = 'color: #f59e0b; font-weight: 600;';
          statusSpan.textContent = '⚠ ' + p.status;
          tdStatus.appendChild(statusSpan);
          tr.appendChild(tdStatus);

          fairnessTableBody.appendChild(tr);
        });
      }

      // Populate Audit Log
      if (auditLogContainer) {
        auditLogContainer.innerHTML = '';
        if (!gov.recent_audit_log || gov.recent_audit_log.length === 0) {
          const empty = document.createElement('div');
          empty.style.cssText = 'color: var(--text-muted); font-size: 0.75rem; padding: 1rem; text-align: center;';
          empty.textContent = 'No audit events recorded yet. Perform a propensity prediction to register entries.';
          auditLogContainer.appendChild(empty);
        } else {
          gov.recent_audit_log.forEach(log => {
            const div = document.createElement('div');
            div.className = 'audit-entry';

            // Row 1: request ID + demographic info
            const row1 = document.createElement('div');
            const reqId = document.createElement('span');
            reqId.className = 'audit-req-id';
            reqId.textContent = log.request_id;
            const demog = document.createElement('span');
            demog.style.cssText = 'color: var(--text-muted); margin-left: 0.5rem;';
            demog.textContent = `Age ${log.client_age} | ${log.client_job} | €${log.balance.toLocaleString()}`;
            row1.appendChild(reqId);
            row1.appendChild(demog);

            // Row 2: score + tier + timestamp
            const row2 = document.createElement('div');
            const scoreSpan = document.createElement('span');
            scoreSpan.className = 'audit-score';
            scoreSpan.textContent = `${(log.propensity_score * 100).toFixed(1)}%`;
            const tierSpan = document.createElement('span');
            tierSpan.style.cssText = 'color: var(--text-faint); margin-left: 0.5rem;';
            tierSpan.textContent = log.decision_tier.split(':')[0];
            const timeSpan = document.createElement('span');
            timeSpan.className = 'audit-time';
            timeSpan.style.marginLeft = '0.75rem';
            timeSpan.textContent = log.timestamp;
            row2.appendChild(scoreSpan);
            row2.appendChild(tierSpan);
            row2.appendChild(timeSpan);

            div.appendChild(row1);
            div.appendChild(row2);
            auditLogContainer.appendChild(div);
          });
        }
      }
    } catch (e) {
      console.error('Governance error:', e);
    }
  }

  // Initial Boot
  async function boot() {
    updateDerivedFeatures();
    await loadProfiles();
    // Run initial prediction on default values
    await executePrediction();
    // Load initial cohort in background
    loadPortfolioCohort('sovereign_hnw');
  }

  boot();
});
