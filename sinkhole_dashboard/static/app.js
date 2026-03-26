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
   1~5단계 색상 직접 바꿀 수 있음
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
   위험도 분포 차트 클릭 필터
   처음엔 1~5 전체 표시
   클릭하면 해당 위험도 숨김 / 다시 클릭하면 다시 표시
========================================================= */
let activeRiskLevels = new Set([1, 2, 3, 4, 5]);

document.addEventListener("DOMContentLoaded", async () => {
    map = L.map("map", { zoomControl: true }).setView([37.45, 127.1], 10);

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        maxZoom: 19,
        attribution: "&copy; OpenStreetMap contributors"
    }).addTo(map);

    await loadState();

    document.querySelectorAll('input[name="layerMode"]').forEach(radio => {
        radio.addEventListener("change", () => {
            document.getElementById("current-layer-label").textContent = layerTitles[radio.value] || radio.value;
            renderGeoLayer();
        });
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
            await loadState();
        } catch (err) {
            uploadMessage.textContent = "업로드 중 오류가 발생했어.";
        }
    });
});

async function loadState() {
    const response = await fetch("/api/state");
    stateData = await response.json();

    document.getElementById("card-total-grid").textContent = stateData.summary_cards.total_grid ?? 0;
    document.getElementById("card-high-risk").textContent = stateData.summary_cards.high_risk_grid ?? 0;
    document.getElementById("card-avg-risk").textContent = stateData.summary_cards.avg_risk ?? 0;
    document.getElementById("card-event-total").textContent = stateData.summary_cards.event_total ?? 0;
    document.getElementById("card-event-3m").textContent = stateData.summary_cards.event_recent_3m ?? 0;
    document.getElementById("card-event-6m").textContent = stateData.summary_cards.event_recent_6m ?? 0;

    renderGeoLayer();
    renderCharts();
}

function renderGeoLayer() {
    if (!stateData || !stateData.geojson) return;

    if (geoLayer) {
        map.removeLayer(geoLayer);
    }

    const selectedLayer = document.querySelector('input[name="layerMode"]:checked').value;
    const values = [];

    stateData.geojson.features.forEach(ft => {
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
        map.fitBounds(geoLayer.getBounds(), { padding: [15, 15] });
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
       [JS 조절 포인트 3]
       위험도 분포 원형차트 색상
       순서: 매우 낮음, 낮음, 보통, 높음, 매우 높음
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

                renderGeoLayer();
                renderCharts();
            }
        }
    });

    /* =========================================================
       [JS 조절 포인트 4]
       지역별 발생건수 원형차트 색상
       색상 배열 수정 가능
    ========================================================= */
    cityPieChart = new Chart(cityCtx, {
        type: "pie",
        data: {
            labels: stateData.city_top15.labels,
            datasets: [{
                label: "발생건수",
                data: stateData.city_top15.values,
                backgroundColor: [
                    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
                    "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ab",
                    "#6baed6", "#fd8d3c", "#74c476", "#9e9ac8", "#e377c2"
                ]
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: "bottom"
                }
            }
        }
    });

    /* =========================================================
       [JS 조절 포인트 5]
       월별 발생 추이 막대그래프 색상 / 테두리색 조절
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