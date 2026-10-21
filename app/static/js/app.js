/**
 * RecoSphere AI — Advanced Interactive Web Application JavaScript
 * Connects directly to FastAPI backend for live predictions, similar item retrieval,
 * dynamic multi-objective ranking, power analysis, and A/B simulations.
 */

// Global State
let activeTab = 'executive';
let funnelChart = null;
let benchmarkChart = null;
let lossChart = null;
let eventDistChart = null;
let currentRecommendations = [];
let currentSimulationData = null;

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initCharts();
    checkSystemHealth();
    bindEvents();
});

// Navigation Handling
function initNavigation() {
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const target = link.getAttribute('data-target');
            switchTab(target);
        });
    });
}

function switchTab(tabId) {
    activeTab = tabId;
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.toggle('active', link.getAttribute('data-target') === tabId);
    });
    document.querySelectorAll('.view-section').forEach(sec => {
        sec.style.display = sec.id === `view-${tabId}` ? 'block' : 'none';
    });

    // Resize charts if becoming visible
    if (tabId === 'executive') {
        if (funnelChart) funnelChart.resize();
        if (benchmarkChart) benchmarkChart.resize();
    } else if (tabId === 'model') {
        if (lossChart) lossChart.resize();
    } else if (tabId === 'data') {
        if (eventDistChart) eventDistChart.resize();
    }
}

// Chart Initializations (Chart.js)
function initCharts() {
    // 1. Funnel Chart
    const ctxFunnel = document.getElementById('funnelChart');
    if (ctxFunnel) {
        funnelChart = new Chart(ctxFunnel, {
            type: 'bar',
            data: {
                labels: ['Product Views', 'Add to Cart', 'Purchases'],
                datasets: [{
                    label: 'Interactions',
                    data: [2425368, 248049, 82684],
                    backgroundColor: ['#6366f1', '#06b6d4', '#10b981'],
                    borderRadius: 6,
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
                    y: { grid: { display: false }, ticks: { color: '#f8fafc', font: { weight: '600' } } }
                }
            }
        });
    }

    // 2. Model Benchmark Chart
    const ctxBench = document.getElementById('benchmarkChart');
    if (ctxBench) {
        benchmarkChart = new Chart(ctxBench, {
            type: 'bar',
            data: {
                labels: ['Popularity Baseline', 'Category Affinity', 'Two-Tower Neural'],
                datasets: [
                    {
                        label: 'HitRate@10 (%)',
                        data: [0.60, 0.80, 0.60],
                        backgroundColor: '#6366f1',
                        borderRadius: 6,
                    },
                    {
                        label: 'NDCG@10',
                        data: [0.24, 0.31, 0.24],
                        backgroundColor: '#10b981',
                        borderRadius: 6,
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#94a3b8' } }
                },
                scales: {
                    x: { grid: { display: false }, ticks: { color: '#f8fafc' } },
                    y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } }
                }
            }
        });
    }

    // 3. Loss History Chart
    const ctxLoss = document.getElementById('lossChart');
    if (ctxLoss) {
        lossChart = new Chart(ctxLoss, {
            type: 'line',
            data: {
                labels: ['Epoch 1', 'Epoch 2', 'Epoch 3'],
                datasets: [
                    {
                        label: 'Training Loss',
                        data: [5.5491, 5.4912, 5.2192],
                        borderColor: '#6366f1',
                        backgroundColor: 'rgba(99, 102, 241, 0.1)',
                        tension: 0.3,
                        fill: true,
                    },
                    {
                        label: 'Validation Loss',
                        data: [5.4616, 5.5042, 5.5071],
                        borderColor: '#f59e0b',
                        tension: 0.3,
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { labels: { color: '#94a3b8' } } },
                scales: {
                    x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } },
                    y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } }
                }
            }
        });
    }

    // 4. Event Distribution Doughnut
    const ctxEvent = document.getElementById('eventDistChart');
    if (ctxEvent) {
        eventDistChart = new Chart(ctxEvent, {
            type: 'doughnut',
            data: {
                labels: ['Views (88%)', 'Add-to-Cart (9%)', 'Transactions (3%)'],
                datasets: [{
                    data: [88, 9, 3],
                    backgroundColor: ['#6366f1', '#f59e0b', '#10b981'],
                    borderColor: '#0a0e17',
                    borderWidth: 3,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { position: 'bottom', labels: { color: '#94a3b8' } } }
            }
        });
    }
}

// System Health Check
async function checkSystemHealth() {
    try {
        const res = await fetch('/health');
        if (res.ok) {
            const data = await res.json();
            document.getElementById('sys-status-text').textContent = 'API & Model Online';
            document.getElementById('health-model-status').textContent = data.model_loaded ? 'Active' : 'Standby';
            document.getElementById('health-index-status').textContent = data.index_loaded ? 'Ready (20,980 items)' : 'Offline';
        }
    } catch (e) {
        console.warn('Health check deferred:', e);
        document.getElementById('sys-status-text').textContent = 'Standalone Mode';
    }
}

// Event Bindings
function bindEvents() {
    // Recommendation button
    const recBtn = document.getElementById('get-recs-btn');
    if (recBtn) recBtn.addEventListener('click', fetchRecommendations);

    // Quick visitor pickers
    document.querySelectorAll('.quick-pick-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.getElementById('visitor-id-input').value = btn.getAttribute('data-id');
            fetchRecommendations();
        });
    });

    // Similar items button
    const simItemBtn = document.getElementById('get-similar-btn');
    if (simItemBtn) simItemBtn.addEventListener('click', fetchSimilarItems);

    document.querySelectorAll('.similar-pick-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.getElementById('similar-item-input').value = btn.getAttribute('data-id');
            fetchSimilarItems();
        });
    });

    // A/B Simulation Button
    const simBtn = document.getElementById('run-sim-btn');
    if (simBtn) simBtn.addEventListener('click', runSimulation);

    // Sample Size Calculator Button
    const calcBtn = document.getElementById('calc-sample-btn');
    if (calcBtn) calcBtn.addEventListener('click', calculateSampleSize);

    // CSV Exports
    const exportRecBtn = document.getElementById('export-recs-csv');
    if (exportRecBtn) exportRecBtn.addEventListener('click', exportRecommendationsCSV);

    const exportSimBtn = document.getElementById('export-sim-csv');
    if (exportSimBtn) exportSimBtn.addEventListener('click', exportSimulationCSV);
}

// Live Recommendation Query
async function fetchRecommendations() {
    const visitorId = parseInt(document.getElementById('visitor-id-input').value, 10);
    const topN = parseInt(document.getElementById('top-n-slider').value, 10);
    const excludeSeen = document.getElementById('exclude-seen-check').checked;
    const enableDiv = document.getElementById('enable-div-check').checked;

    // Advanced dynamic weights
    const wRel = parseFloat(document.getElementById('weight-rel').value);
    const wPop = parseFloat(document.getElementById('weight-pop').value);
    const wCat = parseFloat(document.getElementById('weight-cat').value);
    const lmbda = parseFloat(document.getElementById('weight-lmbda').value);

    const container = document.getElementById('recs-container');
    const noticeEl = document.getElementById('rec-notice');
    const latencyBadge = document.getElementById('rec-latency-badge');
    const exportBtn = document.getElementById('export-recs-csv');

    container.innerHTML = '<div style="padding: 24px; color: #94a3b8; grid-column: 1 / -1; text-align: center;">Computing neural embeddings & ranking...</div>';

    try {
        const response = await fetch('/recommend', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                visitor_id: visitorId,
                top_n: topN,
                exclude_seen: excludeSeen,
                enable_diversification: enableDiv,
                relevance_weight: wRel,
                popularity_weight: wPop,
                category_weight: wCat,
                mmr_lambda: lmbda,
            })
        });

        if (!response.ok) throw new Error('API request failed');

        const data = await response.json();
        currentRecommendations = data.recommendations;

        // Render Notice & Latency
        if (data.is_cold_start) {
            noticeEl.innerHTML = `<span style="color: #f59e0b;">⚠️ <b>Cold-Start Fallback Active</b>: ${data.explanation}</span>`;
        } else {
            noticeEl.innerHTML = `<span style="color: #10b981;">⚡ <b>Two-Tower Neural Retrieval Active</b>: ${data.explanation}</span>`;
        }
        noticeEl.style.display = 'block';

        if (data.latency_ms !== undefined) {
            latencyBadge.textContent = `⚡ ${data.latency_ms} ms latency`;
            latencyBadge.style.display = 'inline-block';
        }

        exportBtn.style.display = 'inline-flex';

        // Render Product Cards
        container.innerHTML = '';
        data.recommendations.forEach(item => {
            const exp = item.explanation;
            const card = document.createElement('div');
            card.className = 'rec-card';
            card.innerHTML = `
                <div class="rec-rank">#${item.rank}</div>
                <div class="rec-item-id">Product ${item.item_id}</div>
                <div class="rec-cat-tag">Category ${item.category_id || 'Catalog'}</div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #10b981; margin-bottom: 8px;">
                    $${item.price.toFixed(2)}
                </div>
                <div class="score-bar-bg">
                    <div class="score-bar-fill" style="width: ${Math.min(100, Math.max(10, item.score * 100))}%;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #94a3b8;">
                    <span>Match: ${(item.score * 100).toFixed(1)}%</span>
                    <span>Sim: ${item.similarity.toFixed(3)}</span>
                </div>
                <div class="attribution-tags">
                    <span class="attr-tag">Vec: ${exp.vector_similarity_contrib}</span>
                    <span class="attr-tag">Pop: ${exp.catalog_popularity_contrib}</span>
                    <span class="attr-tag">Cat: ${exp.category_affinity_contrib}</span>
                </div>
            `;
            container.appendChild(card);
        });

    } catch (e) {
        container.innerHTML = `<div style="color: #f43f5e; padding: 20px; grid-column: 1 / -1; text-align: center;">Failed to fetch recommendations: ${e.message}</div>`;
    }
}

// Item-to-Item Similar Products Query
async function fetchSimilarItems() {
    const itemId = parseInt(document.getElementById('similar-item-input').value, 10);
    const topK = parseInt(document.getElementById('similar-top-k').value, 10);

    const container = document.getElementById('similar-container');
    const latencyEl = document.getElementById('similar-latency');

    container.innerHTML = '<div style="padding: 24px; color: #94a3b8; grid-column: 1 / -1; text-align: center;">Searching vector space for nearest product neighbors...</div>';

    try {
        const response = await fetch('/recommend/similar', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ item_id: itemId, top_n: topK })
        });

        if (!response.ok) throw new Error('API request failed');

        const data = await response.json();

        if (data.latency_ms !== undefined) {
            latencyEl.textContent = `⚡ Nearest Neighbors retrieved in ${data.latency_ms} ms`;
            latencyEl.style.display = 'inline-block';
        }

        container.innerHTML = '';
        if (data.similar_items.length === 0) {
            container.innerHTML = `<div style="padding: 24px; color: #f59e0b; grid-column: 1 / -1; text-align: center;">Item ${itemId} not found in the indexed active catalog.</div>`;
            return;
        }

        data.similar_items.forEach(item => {
            const card = document.createElement('div');
            card.className = 'rec-card';
            card.innerHTML = `
                <div class="rec-rank">#${item.rank}</div>
                <div class="rec-item-id">Product ${item.item_id}</div>
                <div class="rec-cat-tag">Category ${item.category_id || 'Catalog'}</div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #10b981; margin-bottom: 8px;">
                    $${item.price.toFixed(2)}
                </div>
                <div class="score-bar-bg">
                    <div class="score-bar-fill" style="width: ${Math.min(100, Math.max(10, item.similarity * 100))}%;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #94a3b8;">
                    <span>Cosine Similarity:</span>
                    <span style="font-weight: 700; color: #818cf8;">${item.similarity.toFixed(4)}</span>
                </div>
            `;
            container.appendChild(card);
        });

    } catch (e) {
        container.innerHTML = `<div style="color: #f43f5e; padding: 20px; grid-column: 1 / -1; text-align: center;">Error searching similar products: ${e.message}</div>`;
    }
}

// Statistical Power & Sample Size Calculator
async function calculateSampleSize() {
    const baseRate = parseFloat(document.getElementById('calc-base-rate').value);
    const mde = parseFloat(document.getElementById('calc-mde').value) / 100.0;
    const power = parseFloat(document.getElementById('calc-power').value);
    const alpha = parseFloat(document.getElementById('calc-alpha').value);
    const resultEl = document.getElementById('calc-result');

    try {
        const response = await fetch('/experiment/sample-size', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                baseline_rate: baseRate,
                mde_relative: mde,
                alpha: alpha,
                power: power,
            })
        });

        if (!response.ok) throw new Error('Calculation failed');
        const data = await response.json();

        resultEl.innerHTML = `
            Required Traffic: <span style="color: #10b981;">${data.required_sample_size_per_variant.toLocaleString()} users/variant</span><br>
            Total Traffic: <span style="color: #f8fafc;">${data.total_sample_size.toLocaleString()} visitors</span> for ${(power * 100).toFixed(0)}% Statistical Power.
        `;
    } catch (e) {
        resultEl.textContent = 'Error: ' + e.message;
    }
}

// A/B Simulation Query
async function runSimulation() {
    const sampleSize = parseInt(document.getElementById('sim-sample-size').value, 10);
    const baselineCvr = parseFloat(document.getElementById('sim-baseline-cvr').value);
    const liftCvr = parseFloat(document.getElementById('sim-lift-cvr').value) / 100.0;
    const baselineCtr = parseFloat(document.getElementById('sim-baseline-ctr').value);
    const liftCtr = parseFloat(document.getElementById('sim-lift-ctr').value) / 100.0;
    const alpha = parseFloat(document.getElementById('sim-alpha').value);

    const btn = document.getElementById('run-sim-btn');
    btn.disabled = true;
    btn.textContent = 'Simulating Experiment...';

    try {
        const response = await fetch('/experiment/simulate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                sample_size_per_variant: sampleSize,
                baseline_ctr: baselineCtr,
                baseline_cvr: baselineCvr,
                treatment_lift_ctr: liftCtr,
                treatment_lift_cvr: liftCvr,
                alpha: alpha,
            })
        });

        if (!response.ok) throw new Error('Simulation API error');

        const res = await response.json();
        currentSimulationData = res;

        const cvr = res.metrics.conversion_rate;
        const rev = res.metrics.revenue_per_user;

        document.getElementById('sim-ctrl-cvr').textContent = `${(cvr.control_rate * 100).toFixed(2)}%`;
        document.getElementById('sim-treat-cvr').textContent = `${(cvr.treatment_rate * 100).toFixed(2)}%`;
        document.getElementById('sim-lift-stat').textContent = `${cvr.relative_lift_pct > 0 ? '+' : ''}${cvr.relative_lift_pct}%`;
        document.getElementById('sim-p-val').textContent = `p = ${cvr.p_value.toFixed(4)}`;
        document.getElementById('sim-z-val').textContent = `Z = ${cvr.z_statistic.toFixed(2)}`;

        const decisionEl = document.getElementById('sim-decision');
        if (cvr.is_statistically_significant) {
            decisionEl.innerHTML = `<span style="color: #10b981; font-weight: 700;">Statistically Significant (p < ${alpha}) — Reject H0</span>`;
        } else {
            decisionEl.innerHTML = `<span style="color: #94a3b8; font-weight: 700;">Inconclusive / Not Significant — Fail to Reject H0</span>`;
        }

        document.getElementById('sim-ci-range').textContent = `[${(cvr.ci_lower * 100).toFixed(2)}%, ${(cvr.ci_upper * 100).toFixed(2)}%]`;
        document.getElementById('sim-rev-ci').textContent = `Control: $${rev.control_mean} | Treatment: $${rev.treatment_mean} | Bootstrap 95% CI: [$${rev.ci_lower}, $${rev.ci_upper}]`;

        document.getElementById('sim-results-panel').style.display = 'block';

    } catch (e) {
        alert('Simulation execution error: ' + e.message);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Run Monte-Carlo Experiment';
    }
}

// Export Recommendations to CSV
function exportRecommendationsCSV() {
    if (!currentRecommendations || currentRecommendations.length === 0) return;

    let csv = "Rank,ItemID,Category,Price,CompositeScore,Similarity\n";
    currentRecommendations.forEach(r => {
        csv += `${r.rank},${r.item_id},${r.category_id || ''},${r.price},${r.score},${r.similarity}\n`;
    });

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `recosphere_recommendations_${Date.now()}.csv`;
    a.click();
}

// Export Simulation to CSV
function exportSimulationCSV() {
    if (!currentSimulationData) return;

    const m = currentSimulationData.metrics.conversion_rate;
    const r = currentSimulationData.metrics.revenue_per_user;
    let csv = "Metric,Control,Treatment,RelativeLiftPct,PValue,ZStatistic,CILower,CIUpper\n";
    csv += `ConversionRate,${m.control_rate},${m.treatment_rate},${m.relative_lift_pct},${m.p_value},${m.z_statistic},${m.ci_lower},${m.ci_upper}\n`;
    csv += `RevenuePerUser,${r.control_mean},${r.treatment_mean},${r.relative_lift_pct},${r.p_value_approx},N/A,${r.ci_lower},${r.ci_upper}\n`;

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `recosphere_ab_simulation_${Date.now()}.csv`;
    a.click();
}
