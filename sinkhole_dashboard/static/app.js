let map;
let polygonLayer;
let eventLayer;
let riskChart;
let eventChart;
let monthlyChart;
let bootstrapData = null;

const preferredMetrics = ['risk_prob', 'population', 'building_count', 'slope_deg', 'rain_sum', 'sw_old_rt'];

function initMap() {
  map = L.map('map').setView([37.45, 127.0], 10);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; OpenStreetMap contributors'
  }).addTo(map);
}

function setStatusMessage(message) {
  document.getElementById('statusMessage').textContent = message || '';
}

function metricLabel(metric) {
  const labels = {
    risk_prob: '종합 위험도',
    population: '인구',
    building_count: '건물 수',
    slope_deg: '경사도',
    rain_sum: '누적 강수',
    sw_old_rt: '노후 하수도 비율'
  };
  return labels[metric] || metric;
}

function buildMetricList(features) {
  const wrap = document.getElementById('metricList');
  wrap.innerHTML = '';
  const metrics = ['risk_prob', ...preferredMetrics.filter(m => m !== 'risk_prob' && features.includes(m))];

  metrics.forEach((metric, idx) => {
    const label = document.createElement('label');
    label.innerHTML = `<input type="radio" name="metricMode" value="${metric}" ${idx === 0 ? 'checked' : ''}> ${metricLabel(metric)}`;
    wrap.appendChild(label);
  });
}

function updateInfoCards(data, metric='risk_prob') {
  document.getElementById('modelName').textContent = data.model_name || '-';
  document.getElementById('thresholdValue').textContent = data.threshold ?? '-';
  document.getElementById('currentMetricName').textContent = metricLabel(metric);
  document.getElementById('totalGridCount').textContent = data.grid_summary.total_grid_count ?? 0;
  document.getElementById('highRiskGridCount').textContent = data.grid_summary.high_risk_grid_count ?? 0;
  document.getElementById('avgRiskProb').textContent = data.grid_summary.avg_risk_prob ?? 0;
  document.getElementById('totalEvents').textContent = data.event_summary.total_events ?? 0;
  document.getElementById('recent3m').textContent = data.event_summary.recent_3m_events ?? 0;
  document.getElementById('recent6m').textContent = data.event_summary.recent_6m_events ?? 0;
}

function renderFeatureList(features) {
  const wrap = document.getElementById('featureList');
  wrap.innerHTML = '';
  features.forEach(f => {
    const span = document.createElement('span');
    span.className = 'feature-chip';
    span.textContent = f;
    wrap.appendChild(span);
  });
}

function clearLayers() {
  if (polygonLayer) { map.removeLayer(polygonLayer); polygonLayer = null; }
  if (eventLayer) { map.removeLayer(eventLayer); eventLayer = null; }
}

function renderMap(mapPayload, eventSummary, metric='risk_prob') {
  clearLayers();
  if (!mapPayload?.map_ready || !mapPayload.geojson) {
    setStatusMessage(mapPayload?.message || '지도에 표시할 데이터가 없습니다.');
    return;
  }

  polygonLayer = L.geoJSON(mapPayload.geojson, {
    style: feat => ({
      color: '#4b5563',
      weight: 0.25,
      fillColor: feat.properties.color || '#7BC96F',
      fillOpacity: 0.74
    }),
    onEachFeature: (feature, layer) => {
      const p = feature.properties || {};
      layer.bindPopup(`
        <div>
          <strong>${p.display_name || p.SGG_NM || '-'}</strong><br>
          id: ${p.id || '-'}<br>
          위험확률: ${Number(p.risk_prob || 0).toFixed(4)}<br>
          위험등급: ${p.risk_label || '-'}<br>
          ${metricLabel(metric)}: ${p.display_value ?? '-'}
        </div>
      `);
    }
  }).addTo(map);

  const bounds = polygonLayer.getBounds();
  if (bounds.isValid()) map.fitBounds(bounds.pad(0.02));

  if (eventSummary?.event_points?.length) {
    eventLayer = L.layerGroup();
    eventSummary.event_points.forEach(item => {
      if (item.위도 && item.경도) {
        const marker = L.circleMarker([item.위도, item.경도], {
          radius: 3.5,
          fillColor: '#111827',
          color: '#111827',
          weight: 0.4,
          fillOpacity: 0.5,
        }).bindPopup(`
          <div>
            <strong>${item.SGG_NM || ''}</strong><br>
            주소: ${item.주소 || '-'}<br>
            발생일: ${item.event_date || '-'}<br>
            원인: ${item.최초발생원인 || '-'}
          </div>
        `);
        eventLayer.addLayer(marker);
      }
    });
    eventLayer.addTo(map);
  }
}

function buildOrUpdateChart(chartRef, canvasId, type, labels, data, datasetLabel) {
  if (chartRef) chartRef.destroy();
  const ctx = document.getElementById(canvasId);
  return new Chart(ctx, {
    type,
    data: {
      labels,
      datasets: [{ label: datasetLabel, data }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: type !== 'bar' } },
      scales: type === 'bar' || type === 'line' ? { y: { beginAtZero: true } } : {}
    }
  });
}

function updateCharts(data) {
  const dist = data.grid_summary.risk_distribution || [];
  riskChart = buildOrUpdateChart(riskChart, 'riskChart', 'doughnut', dist.map(d => d.label), dist.map(d => d.count), '위험도 분포');

  const sgg = data.event_summary.sgg_event_counts || [];
  eventChart = buildOrUpdateChart(eventChart, 'eventChart', 'bar', sgg.map(d => d.SGG_NM || '미상'), sgg.map(d => d.event_count), '발생건수');

  const monthly = data.event_summary.monthly_counts || [];
  monthlyChart = buildOrUpdateChart(monthlyChart, 'monthlyChart', 'line', monthly.map(d => d.period), monthly.map(d => d.count), '월별 발생');
}

async function loadBootstrap(metric = 'risk_prob') {
  const res = await fetch(`/api/bootstrap?selected_metric=${encodeURIComponent(metric)}`);
  const data = await res.json();
  bootstrapData = data;
  buildMetricList(data.features || []);
  renderFeatureList(data.features || []);
  document.getElementById('mapTitle').textContent = metricLabel(metric);
  updateInfoCards(data, metric);
  updateCharts(data);
  renderMap(data.map_payload, data.event_summary, metric);
  setStatusMessage(data.message || data.map_payload?.message || '');
  bindMetricEvents();
}

async function refreshMapOnly(metric) {
  const res = await fetch(`/api/map-data?metric=${encodeURIComponent(metric)}`);
  const mapPayload = await res.json();
  document.getElementById('mapTitle').textContent = metricLabel(metric);
  document.getElementById('currentMetricName').textContent = metricLabel(metric);
  renderMap(mapPayload, bootstrapData?.event_summary, metric);
}

async function uploadFile(endpoint, fileInputId) {
  const input = document.getElementById(fileInputId);
  if (!input.files.length) {
    alert('파일을 선택해줘.');
    return;
  }
  const form = new FormData();
  form.append('file', input.files[0]);
  const res = await fetch(endpoint, { method: 'POST', body: form });
  const result = await res.json();
  alert(result.message || (result.ok ? '업로드 완료' : '업로드 실패'));
  if (result.ok) {
    const selected = document.querySelector('input[name="metricMode"]:checked')?.value || 'risk_prob';
    await loadBootstrap(selected);
  }
}

function bindMetricEvents() {
  document.querySelectorAll('input[name="metricMode"]').forEach(el => {
    el.onchange = async (e) => {
      await refreshMapOnly(e.target.value);
    };
  });
}

function bindDropzone() {
  const zone = document.getElementById('gridDropzone');
  const input = document.getElementById('gridUpload');

  zone.addEventListener('click', () => input.click());
  zone.addEventListener('dragover', (e) => {
    e.preventDefault();
    zone.classList.add('dragover');
  });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    zone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
      input.files = e.dataTransfer.files;
      zone.querySelector('.dropzone-text strong').textContent = e.dataTransfer.files[0].name;
    }
  });
  input.addEventListener('change', () => {
    if (input.files.length) {
      zone.querySelector('.dropzone-text strong').textContent = input.files[0].name;
    }
  });
}

document.addEventListener('DOMContentLoaded', async () => {
  initMap();
  bindDropzone();
  await loadBootstrap('risk_prob');
  document.getElementById('gridUploadBtn').addEventListener('click', async () => {
    await uploadFile('/api/upload-grid', 'gridUpload');
  });
});
