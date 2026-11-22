const percent = (x) => (x === null ? "–" : `${(x * 100).toFixed(1)}%`);

function cell(text) {
  const td = document.createElement("td");
  td.textContent = text;
  return td;
}

async function refresh() {
  const response = await fetch("/api/stats");
  const stats = await response.json();

  document.getElementById("total").textContent = stats.total;
  document.getElementById("confidence").textContent = percent(stats.average_confidence);
  document.getElementById("low").textContent = percent(stats.low_confidence_rate);
  document.getElementById("latency").textContent =
    stats.average_latency_ms === null ? "–" : `${stats.average_latency_ms} ms`;

  const classes = document.getElementById("classes");
  classes.replaceChildren();
  const max = Math.max(1, ...Object.values(stats.by_class));
  for (const [name, count] of Object.entries(stats.by_class)) {
    const row = document.createElement("div");
    row.className = "bar-row";
    const label = document.createElement("span");
    label.textContent = name;
    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = `${(count / max) * 100}%`;
    track.append(fill);
    const value = document.createElement("span");
    value.className = "bar-value";
    value.textContent = count;
    row.append(label, track, value);
    classes.append(row);
  }

  const recent = document.getElementById("recent");
  recent.replaceChildren();
  for (const e of stats.recent) {
    const tr = document.createElement("tr");
    tr.append(
      cell(e.time.replace("T", " ").replace("+00:00", "")),
      cell(e.source),
      cell(e.label + (e.low_confidence ? " (low)" : "")),
      cell(percent(e.confidence)),
      cell(`${e.latency_ms} ms`),
    );
    recent.append(tr);
  }
}

refresh();
setInterval(refresh, 3000);
