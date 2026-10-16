/**
 * AURA-Term™ Institutional Banking Chart & Gauge Rendering Engine
 * Pure Canvas zero-dependency rendering for sub-millisecond redraws.
 */

const BankCharts = {
  // 1. High-Precision Radial Propensity Gauge
  drawRadialGauge(canvas, percentage, colorHex = "#10B981") {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const width = canvas.clientWidth || 220;
    const height = canvas.clientHeight || 220;

    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    const centerX = width / 2;
    const centerY = height / 2;
    const radius = width * 0.38;
    const lineWidth = 16;

    ctx.clearRect(0, 0, width, height);

    // Track arc (220 degrees sweep, starting from -200 deg to 20 deg)
    const startAngle = (140 * Math.PI) / 180;
    const endAngle = (400 * Math.PI) / 180;
    const totalSweep = endAngle - startAngle;

    // Background track
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, startAngle, endAngle);
    ctx.strokeStyle = 'rgba(51, 65, 85, 0.35)';
    ctx.lineWidth = lineWidth;
    ctx.lineCap = 'round';
    ctx.stroke();

    // Subtle tick marks
    const ticksCount = 10;
    for (let i = 0; i <= ticksCount; i++) {
      const angle = startAngle + (i / ticksCount) * totalSweep;
      const innerR = radius - 14;
      const outerR = radius - 8;
      const x1 = centerX + Math.cos(angle) * innerR;
      const y1 = centerY + Math.sin(angle) * innerR;
      const x2 = centerX + Math.cos(angle) * outerR;
      const y2 = centerY + Math.sin(angle) * outerR;

      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.strokeStyle = i === 5 ? 'rgba(251, 191, 36, 0.7)' : 'rgba(148, 163, 184, 0.25)';
      ctx.lineWidth = i === 5 ? 2 : 1;
      ctx.stroke();
    }

    // Active Value Arc
    const valRatio = Math.max(0, Math.min(100, percentage)) / 100;
    const currentAngle = startAngle + valRatio * totalSweep;

    if (valRatio > 0.01) {
      // Glow
      ctx.save();
      ctx.shadowColor = colorHex;
      ctx.shadowBlur = 18;
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius, startAngle, currentAngle);
      ctx.strokeStyle = colorHex;
      ctx.lineWidth = lineWidth;
      ctx.lineCap = 'round';
      ctx.stroke();
      ctx.restore();
    }

    // Baseline marker at 50%
    const midAngle = startAngle + 0.5 * totalSweep;
    const markerX = centerX + Math.cos(midAngle) * (radius + 12);
    const markerY = centerY + Math.sin(midAngle) * (radius + 12);
    ctx.fillStyle = '#fbbf24';
    ctx.beginPath();
    ctx.arc(markerX, markerY, 3, 0, Math.PI * 2);
    ctx.fill();
  },

  // 2. Smooth Sensitivity Line Chart (Balance / Age)
  drawLineChart(canvas, xLabels, yValues, options = {}) {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const width = canvas.clientWidth || 400;
    const height = canvas.clientHeight || 260;

    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    const padding = { top: 30, right: 25, bottom: 40, left: 45 };
    const chartW = width - padding.left - padding.right;
    const chartH = height - padding.top - padding.bottom;

    const minY = 30;
    const maxY = 100;

    // Draw Grid Lines & Y-labels
    ctx.strokeStyle = 'rgba(51, 65, 85, 0.3)';
    ctx.lineWidth = 1;
    ctx.fillStyle = '#64748b';
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.textAlign = 'right';

    const ySteps = 4;
    for (let i = 0; i <= ySteps; i++) {
      const val = minY + (i / ySteps) * (maxY - minY);
      const y = padding.top + chartH - (i / ySteps) * chartH;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.stroke();
      ctx.fillText(`${Math.round(val)}%`, padding.left - 8, y + 3);
    }

    // 50% Threshold Baseline
    const y50 = padding.top + chartH - ((50 - minY) / (maxY - minY)) * chartH;
    ctx.save();
    ctx.strokeStyle = 'rgba(245, 158, 11, 0.4)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(padding.left, y50);
    ctx.lineTo(width - padding.right, y50);
    ctx.stroke();
    ctx.fillStyle = '#fbbf24';
    ctx.fillText('50% Threshold', width - padding.right, y50 - 5);
    ctx.restore();

    // Map Points
    const points = yValues.map((val, idx) => {
      const x = padding.left + (idx / (yValues.length - 1)) * chartW;
      const normalizedY = Math.max(minY, Math.min(maxY, val));
      const y = padding.top + chartH - ((normalizedY - minY) / (maxY - minY)) * chartH;
      return { x, y, val, label: xLabels[idx] };
    });

    if (points.length < 2) return;

    // Draw Gradient Area Under Curve
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 0; i < points.length - 1; i++) {
      const xc = (points[i].x + points[i + 1].x) / 2;
      const yc = (points[i].y + points[i + 1].y) / 2;
      ctx.quadraticCurveTo(points[i].x, points[i].y, xc, yc);
    }
    ctx.lineTo(points[points.length - 1].x, points[points.length - 1].y);
    ctx.lineTo(points[points.length - 1].x, padding.top + chartH);
    ctx.lineTo(points[0].x, padding.top + chartH);
    ctx.closePath();

    const gradient = ctx.createLinearGradient(0, padding.top, 0, padding.top + chartH);
    gradient.addColorStop(0, 'rgba(56, 189, 248, 0.28)');
    gradient.addColorStop(1, 'rgba(56, 189, 248, 0.0)');
    ctx.fillStyle = gradient;
    ctx.fill();
    ctx.restore();

    // Draw Smooth Line
    ctx.save();
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 3;
    ctx.shadowColor = 'rgba(56, 189, 248, 0.5)';
    ctx.shadowBlur = 10;
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 0; i < points.length - 1; i++) {
      const xc = (points[i].x + points[i + 1].x) / 2;
      const yc = (points[i].y + points[i + 1].y) / 2;
      ctx.quadraticCurveTo(points[i].x, points[i].y, xc, yc);
    }
    ctx.lineTo(points[points.length - 1].x, points[points.length - 1].y);
    ctx.stroke();
    ctx.restore();

    // Data Points & X-Labels
    ctx.fillStyle = '#ffffff';
    ctx.textAlign = 'center';
    points.forEach((pt, idx) => {
      // Circle on point
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 4, 0, Math.PI * 2);
      ctx.fillStyle = '#38bdf8';
      ctx.fill();
      ctx.strokeStyle = '#0f172a';
      ctx.lineWidth = 2;
      ctx.stroke();

      // X-Label (sample every few points if crowded)
      if (idx % 2 === 0 || idx === points.length - 1) {
        ctx.fillStyle = '#94a3b8';
        ctx.font = '10px JetBrains Mono, monospace';
        const lbl = options.formatX ? options.formatX(pt.label) : String(pt.label);
        ctx.fillText(lbl, pt.x, height - padding.bottom + 16);
      }
    });
  },

  // 3. Monthly Telemarketing Seasonality Bar Chart
  drawMonthlyBarChart(canvas, months, scores) {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const width = canvas.clientWidth || 400;
    const height = canvas.clientHeight || 260;

    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    const padding = { top: 30, right: 20, bottom: 40, left: 45 };
    const chartW = width - padding.left - padding.right;
    const chartH = height - padding.top - padding.bottom;

    const minY = 30;
    const maxY = 100;

    // Grid lines
    ctx.strokeStyle = 'rgba(51, 65, 85, 0.3)';
    ctx.lineWidth = 1;
    ctx.fillStyle = '#64748b';
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.textAlign = 'right';

    for (let i = 0; i <= 3; i++) {
      const val = minY + (i / 3) * (maxY - minY);
      const y = padding.top + chartH - (i / 3) * chartH;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.stroke();
      ctx.fillText(`${Math.round(val)}%`, padding.left - 8, y + 3);
    }

    const barWidth = Math.max(12, (chartW / months.length) * 0.65);
    const step = chartW / months.length;

    months.forEach((m, idx) => {
      const score = scores[idx] || 50;
      const normScore = Math.max(minY, Math.min(maxY, score));
      const barH = ((normScore - minY) / (maxY - minY)) * chartH;
      const x = padding.left + idx * step + (step - barWidth) / 2;
      const y = padding.top + chartH - barH;

      // Color based on performance
      let barGrad = ctx.createLinearGradient(0, y, 0, y + barH);
      if (score >= 65) {
        barGrad.addColorStop(0, '#10b981');
        barGrad.addColorStop(1, 'rgba(16, 185, 129, 0.4)');
      } else if (score >= 48) {
        barGrad.addColorStop(0, '#3b82f6');
        barGrad.addColorStop(1, 'rgba(59, 130, 246, 0.4)');
      } else {
        barGrad.addColorStop(0, '#64748b');
        barGrad.addColorStop(1, 'rgba(100, 116, 139, 0.3)');
      }

      ctx.fillStyle = barGrad;
      ctx.beginPath();
      ctx.roundRect(x, y, barWidth, barH, [4, 4, 0, 0]);
      ctx.fill();

      // Top label for standout months
      if (score >= 60) {
        ctx.fillStyle = '#34d399';
        ctx.font = '9px JetBrains Mono, monospace';
        ctx.textAlign = 'center';
        ctx.fillText(`${Math.round(score)}%`, x + barWidth / 2, y - 5);
      }

      // Month name
      ctx.fillStyle = '#94a3b8';
      ctx.font = '10px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      ctx.fillText(m.toUpperCase(), x + barWidth / 2, height - padding.bottom + 16);
    });
  }
};
