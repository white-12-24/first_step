let map = null;
let geoLayer = null;
let stateData = null;
let selectedFile = null;

let riskPieChart = null;
let cityPieChart = null;
let monthBarChart = null;

/* =========================================================
   [JS 조절 포인트 1]
   위험도 단계별 지도 색상
========================================================= */
const riskColors = {
    1: "#2fbf71",
    2: "#8bdc65",
    3: "#f1d54b",
    4: "#f39c34",
    5: "#e74c3c"
};

const layerTitles = {
    risk: "종합 위험도",
    population: "인구",
    building_count: "건물 수",
    slope_deg: "경사도",
    rain_sum: "누적 강수",
    sw_old_rt: "노후 하수도 비율"
};

/* =========================================================
   [JS 조절 포인트 2]
   위험도 클릭 on/off 상태
   처음엔 1~5 전부 표시
========================================================= */
let activeRiskLevels = new Set([1, 2, 3, 4, 5]);

/* =========================================================
   [JS 조절 포인트 3]
   현재 선택된 지역
   null이면 전체
========================================================= */
let activeRegion = null;

document.addEventListener("DOMContentLoaded", async () => {
    map = L.map("map", { zoomControl: true }).setView([37.45, 127.1], 10);

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        maxZoom: 19,
        attribution: "&copy; OpenStreetMap contributors"
    }).addTo(map);

    await loadState(true, false);

    document.querySelectorAll('input[name="layerMode"]').forEach(radio => {
        radio.addEventListener("change", () => {
            document.getElementById("current-layer-label").textContent = layerTitles[radio.value] || radio.value;

            // 레이어만 바꾸는 건 현재 확대 상태 유지
            renderGeoLayer(false, false);
        });
    });

    document.getElementById("reset-region-btn").addEventListener("click", () => {
        activeRegion = null;
        renderRegionChips();

        // 전체 보기로 돌아갈 때는 전체 범위에 맞춰 다시 fit
        renderGeoLayer(true, false);
        updateCurrentRegionLabel();
    });

    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const uploadBtn = document.getElementById("upload-btn");
    const fileNameText = document.getElementById("upload-file-name");
    const uploadMessage = document.getElementById("upload-message");

    dropZone.addEventListener("click", () => fileInput.click());

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");

        if (e.dataTransfer.files.length > 0) {
            selectedFile = e.dataTransfer.files[0];
            fileNameText.textContent = selectedFile.name;
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            selectedFile = e.target.files[0];
            fileNameText.textContent = selectedFile.name;
        }
    });

    uploadBtn.addEventListener("click", async () => {
        if (!selectedFile) {
            uploadMessage.textContent = "먼저 업로드할 파일을 선택해줘.";
            return;
        }

        uploadMessage.textContent = "업로드 중...";

        const formData = new FormData();
        formData.append("file", selectedFile);

        try {
            const response = await fetch("/api/upload", {
                method: "POST",
                body: formData
            });

            const result = await response.json();

            if (!response.ok || !result.ok) {
                uploadMessage.textContent = result.message || "업로드 실패";
                return;
            }

            uploadMessage.textContent = result.message || "업로드 완료";

            // 업로드 후에는 현재 보고 있던 지역/확대 상태 최대한 유지
            await loadState(false, true);
        } catch (err) {
            uploadMessage.textContent = "업로드 중 오류가 발생했어.";
        }
    });
});

/* =========================================================
   [JS 조절 포인트 4]
   loadState(fitToRegion, preserveView)
   - fitToRegion = true  : 현재 필터 범위에 맞춰 지도 확대
   - preserveView = true : 현재 확대/이동 상태 유지
========================================================= */
async function loadState(fitToRegion = false, preserveView = false) {
    const response = await fetch("/api/state");
    stateData = await response.json();

    document.getElementById("card-total-grid").textContent = stateData.summary_cards.total_grid ?? 0;
    document.getElementById("card-high-risk").textContent = stateData.summary_cards.high_risk_grid ?? 0;
    document.getElementById("card-avg-risk").textContent = stateData.summary_cards.avg_risk ?? 0;
    document.getElementById("card-event-total").textContent = stateData.summary_cards.event_total ?? 0;
    document.getElementById("card-event-3m").textContent = stateData.summary_cards.event_recent_3m ?? 0;
    document.getElementById("card-event-6m").textContent = stateData.summary_cards.event_recent_6m ?? 0;

    renderRegionChips();
    updateCurrentRegionLabel();
    renderGeoLayer(fitToRegion, preserveView);
    renderCharts();
}

function renderRegionChips() {
    const wrap = document.getElementById("region-chip-wrap");
    wrap.innerHTML = "";

    if (!stateData || !stateData.regions) return;

    stateData.regions.forEach(regionName => {
        const chip = document.createElement("span");
        chip.className = "feature-chip";
        chip.textContent = regionName;

        if (activeRegion === regionName) {
            chip.classList.add("active");
        }

        chip.addEventListener("click", () => {
            if (activeRegion === regionName) {
                activeRegion = null;
            } else {
                activeRegion = regionName;
            }

            renderRegionChips();
            updateCurrentRegionLabel();

            // 지역 선택은 선택된 지역 범위에 맞춰 확대
            renderGeoLayer(true, false);
        });

        wrap.appendChild(chip);
    });
}

function updateCurrentRegionLabel() {
    const labelEl = document.getElementById("current-region-label");
    labelEl.textContent = activeRegion ? activeRegion : "전체";
}

/* =========================================================
   [JS 조절 포인트 5]
   renderGeoLayer(fitToLayer, preserveView)
   - fitToLayer=true 이면 현재 필터된 레이어 bounds에 맞춰 확대
   - preserveView=true 이면 현재 확대 상태 유지
   위험도 차트 클릭 시에는 preserveView=true로 호출
========================================================= */
function renderGeoLayer(fitToLayer = false, preserveView = false) {
    if (!stateData || !stateData.geojson) return;

    const currentCenter = map.getCenter();
    const currentZoom = map.getZoom();

    if (geoLayer) {
        map.removeLayer(geoLayer);
    }

    const selectedLayer = document.querySelector('input[name="layerMode"]:checked').value;
    const values = [];

    stateData.geojson.features.forEach(ft => {
        if (activeRegion && ft.properties.SGG_NM !== activeRegion) {
            return;
        }

        const v = ft.properties[selectedLayer];
        if (selectedLayer !== "risk" && v !== null && v !== undefined && !isNaN(v)) {
            values.push(Number(v));
        }
    });

    let minVal = null;
    let maxVal = null;

    if (values.length > 0) {
        minVal = Math.min(...values);
        maxVal = Math.max(...values);
    }

    geoLayer = L.geoJSON(stateData.geojson, {
        filter: function(feature) {
            if (activeRegion && feature.properties.SGG_NM !== activeRegion) {
                return false;
            }

            if (selectedLayer === "risk") {
                const level = feature.properties.risk_level;
                return activeRiskLevels.has(level);
            }

            return true;
        },
        style: function(feature) {
            let fill = "#d1d5db";

            if (selectedLayer === "risk") {
                const level = feature.properties.risk_level;
                fill = riskColors[level] || "#d1d5db";
            } else {
                const val = feature.properties[selectedLayer];

                if (val !== null && val !== undefined && !isNaN(val) && minVal !== null && maxVal !== null) {
                    const num = Number(val);
                    const span = (maxVal - minVal) || 1;
                    const p = (num - minVal) / span;

                    if (p <= 0.2) fill = "#2fbf71";
                    else if (p <= 0.4) fill = "#8bdc65";
                    else if (p <= 0.6) fill = "#f1d54b";
                    else if (p <= 0.8) fill = "#f39c34";
                    else fill = "#e74c3c";
                }
            }

            return {
                color: "#7f8aa3",
                weight: 0.22,
                fillColor: fill,
                fillOpacity: 0.70
            };
        },
        onEachFeature: function(feature, layer) {
            const p = feature.properties;

            const selectedValue = selectedLayer === "risk"
                ? `위험확률: ${p.risk_prob ?? "-"} / 단계: ${p.risk_level ?? "-"}`
                : `${layerTitles[selectedLayer] || selectedLayer}: ${p[selectedLayer] ?? "-"}`;

            layer.bindPopup(`
                <div>
                    <strong>${p.SGG_NM || "-"} ${p.DONG || ""}</strong><br>
                    ID: ${p.id || "-"}<br>
                    ${selectedValue}<br>
                    인구: ${p.population ?? "-"}<br>
                    건물 수: ${p.building_count ?? "-"}<br>
                    경사도: ${p.slope_deg ?? "-"}<br>
                    누적 강수: ${p.rain_sum ?? "-"}<br>
                    노후 하수도 비율: ${p.sw_old_rt ?? "-"}
                </div>
            `);
        }
    }).addTo(map);

    try {
        // 현재 확대 상태 유지가 우선일 때
        if (preserveView) {
            map.setView(currentCenter, currentZoom);
            return;
        }

        // 현재 필터 결과에 맞춰 확대할 때
        if (fitToLayer) {
            const bounds = geoLayer.getBounds();
            if (bounds.isValid()) {
                map.fitBounds(bounds, { padding: [20, 20] });
            }
        }
    } catch (e) {
    }
}

function renderCharts() {
    if (riskPieChart) riskPieChart.destroy();
    if (cityPieChart) cityPieChart.destroy();
    if (monthBarChart) monthBarChart.destroy();

    const pieCtx = document.getElementById("riskPieChart");
    const cityCtx = document.getElementById("cityPieChart");
    const monthCtx = document.getElementById("monthBarChart");

    /* =========================================================
       [JS 조절 포인트 6]
       위험도 분포 원형차트 색상
       활성 상태면 원래 색 / 비활성 상태면 회색
    ========================================================= */
    const riskBaseColors = ["#2fbf71", "#8bdc65", "#f1d54b", "#f39c34", "#e74c3c"];

    const riskDisplayColors = stateData.risk_distribution.levels.map((lv, idx) => {
        if (activeRiskLevels.has(lv)) {
            return riskBaseColors[idx];
        }
        return "#d1d5db";
    });

    riskPieChart = new Chart(pieCtx, {
        type: "pie",
        data: {
            labels: stateData.risk_distribution.labels,
            datasets: [{
                data: stateData.risk_distribution.values,
                backgroundColor: riskDisplayColors
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { position: "bottom" }
            },
            onClick: function(evt, elements) {
                if (!elements.length) return;

                const idx = elements[0].index;
                const clickedLevel = stateData.risk_distribution.levels[idx];

                if (activeRiskLevels.has(clickedLevel)) {
                    activeRiskLevels.delete(clickedLevel);
                } else {
                    activeRiskLevels.add(clickedLevel);
                }

                // 위험도 on/off는 현재 확대 상태 유지
                renderGeoLayer(false, true);
                renderCharts();
            }
        }
    });

    /* =========================================================
       [JS 조절 포인트 7]
       지역별 발생건수 원형차트 색상
       backgroundColor 배열 수정 가능
    ========================================================= */
    cityPieChart = new Chart(cityCtx, {
        type: "pie",
        data: {
            labels: stateData.city_top10.labels,
            datasets: [{
                label: "발생건수",
                data: stateData.city_top10.values,
                backgroundColor: [
                    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
                    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ab"
                ]
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: "bottom"
                }
            },
            onClick: function(evt, elements) {
                if (!elements.length) return;

                const idx = elements[0].index;
                const clickedRegion = stateData.city_top10.labels[idx];

                if (activeRegion === clickedRegion) {
                    activeRegion = null;
                } else {
                    activeRegion = clickedRegion;
                }

                renderRegionChips();
                updateCurrentRegionLabel();

                // 특정 지역 클릭 시 그 지역 전체가 지도에 담기도록 fit
                renderGeoLayer(true, false);
                renderCharts();
            }
        }
    });

    /* =========================================================
       [JS 조절 포인트 8]
       월별 발생 추이 막대그래프 색상 / 테두리색
    ========================================================= */
    monthBarChart = new Chart(monthCtx, {
        type: "bar",
        data: {
            labels: stateData.month_series.labels,
            datasets: [{
                label: "월별 발생건수",
                data: stateData.month_series.values,
                backgroundColor: "#5aa9e6",
                borderColor: "#2f6fed",
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                x: {
                    ticks: {
                        autoSkip: false
                    }
                },
                y: {
                    beginAtZero: true
                }
            },
            plugins: {
                legend: {
                    display: true,
                    position: "top"
                }
            }
        }
    });
}