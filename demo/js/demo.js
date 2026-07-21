(() => {
  const scenes = [...document.querySelectorAll(".scene")];
  const labels = ["Title", "Problem", "Scan", "Key", "Chain"];
  const progressEl = document.getElementById("progress");
  const timeline = document.getElementById("timeline");
  const chapters = document.getElementById("chapters");
  const sceneLabel = document.getElementById("sceneLabel");
  const timeLabel = document.getElementById("timeLabel");
  const toggleBtn = document.getElementById("togglePlay");
  const playIcon = document.getElementById("playIcon");
  const playMain = document.getElementById("playMain");

  const durations = scenes.map((s) => Number(s.dataset.duration) || 8000);
  const total = durations.reduce((a, b) => a + b, 0);

  let index = 0;
  let playing = false;
  let raf = 0;
  let sceneStarted = 0;
  let elapsedInScene = 0;

  // Chapter marks
  durations.forEach((d, i) => {
    const mark = document.createElement("div");
    mark.className = "chapter-mark";
    mark.style.flex = String(d);
    mark.title = labels[i];
    mark.addEventListener("click", (e) => {
      e.stopPropagation();
      seekToScene(i);
    });
    chapters.appendChild(mark);
  });

  function format(ms) {
    const s = Math.floor(ms / 1000);
    return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  }

  function absoluteTime() {
    return durations.slice(0, index).reduce((a, b) => a + b, 0) + elapsedInScene;
  }

  function setPlayUI(isPlaying) {
    playIcon.className = isPlaying ? "icon-pause" : "icon-play";
    toggleBtn.setAttribute("aria-label", isPlaying ? "Pause" : "Play");
  }

  function activateScene(i, restartAnims = true) {
    index = Math.max(0, Math.min(scenes.length - 1, i));
    scenes.forEach((s, n) => s.classList.toggle("is-active", n === index));
    document.querySelectorAll(".nav-link").forEach((btn) => {
      btn.classList.toggle("is-current", Number(btn.dataset.seek) === index);
    });
    sceneLabel.textContent = labels[index];
    if (restartAnims) {
      if (index === 2) buildConstellation();
      if (index === 3) animateBits();
    }
    sceneStarted = performance.now() - elapsedInScene;
  }

  function seekToScene(i) {
    elapsedInScene = 0;
    activateScene(i, true);
    updateProgress();
    if (playing) {
      cancelAnimationFrame(raf);
      tick();
    }
  }

  function seekToRatio(ratio) {
    const t = Math.max(0, Math.min(1, ratio)) * total;
    let acc = 0;
    for (let i = 0; i < durations.length; i++) {
      if (t <= acc + durations[i]) {
        elapsedInScene = t - acc;
        activateScene(i, true);
        updateProgress();
        return;
      }
      acc += durations[i];
    }
    elapsedInScene = durations[durations.length - 1];
    activateScene(scenes.length - 1, true);
    updateProgress();
  }

  function updateProgress() {
    const t = absoluteTime();
    const pct = (t / total) * 100;
    progressEl.style.width = `${pct}%`;
    timeline.setAttribute("aria-valuenow", String(Math.round(pct)));
    timeLabel.textContent = format(t);
  }

  function tick(now = performance.now()) {
    if (!playing) return;
    elapsedInScene = now - sceneStarted;
    if (elapsedInScene >= durations[index]) {
      if (index >= scenes.length - 1) {
        playing = false;
        elapsedInScene = durations[index];
        setPlayUI(false);
        updateProgress();
        return;
      }
      elapsedInScene = 0;
      activateScene(index + 1, true);
      sceneStarted = performance.now();
    }
    updateProgress();
    raf = requestAnimationFrame(tick);
  }

  function play() {
    if (playing) return;
    if (index === scenes.length - 1 && elapsedInScene >= durations[index] - 30) {
      elapsedInScene = 0;
      activateScene(0, true);
    }
    playing = true;
    sceneStarted = performance.now() - elapsedInScene;
    setPlayUI(true);
    tick();
  }

  function pause() {
    playing = false;
    setPlayUI(false);
    cancelAnimationFrame(raf);
  }

  function toggle() {
    playing ? pause() : play();
  }

  toggleBtn.addEventListener("click", toggle);
  playMain?.addEventListener("click", () => {
    seekToScene(1);
    play();
  });

  document.querySelectorAll("[data-seek]").forEach((el) => {
    el.addEventListener("click", () => {
      seekToScene(Number(el.dataset.seek));
      if (el.dataset.seek !== "0") play();
      else pause();
    });
  });

  timeline.addEventListener("click", (e) => {
    const rect = timeline.getBoundingClientRect();
    seekToRatio((e.clientX - rect.left) / rect.width);
    if (!playing) play();
  });

  timeline.addEventListener("keydown", (e) => {
    if (e.key === "ArrowRight") seekToRatio((absoluteTime() + 2000) / total);
    if (e.key === "ArrowLeft") seekToRatio((absoluteTime() - 2000) / total);
    if (e.key === " ") {
      e.preventDefault();
      toggle();
    }
  });

  window.addEventListener("keydown", (e) => {
    if (e.code === "Space" && e.target === document.body) {
      e.preventDefault();
      toggle();
    }
  });

  // --- Crystal canvas ---
  const canvas = document.getElementById("crystal");
  const ctx = canvas?.getContext("2d");
  let crystalAngle = 0;

  function drawCrystal(t) {
    if (!ctx || !canvas) return;
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.save();
    ctx.translate(w / 2, h / 2);
    ctx.rotate(crystalAngle);

    const scale = 3.2;
    const facets = [
      { pts: [[0, -110], [70, -20], [0, 20], [-70, -20]], fill: "#f7fbfc" },
      { pts: [[0, -110], [70, -20], [95, -55]], fill: "#c5ddd6" },
      { pts: [[0, -110], [-70, -20], [-95, -55]], fill: "#e7f2ef" },
      { pts: [[70, -20], [95, -55], [78, 35], [0, 20]], fill: "#2a7f6d" },
      { pts: [[-70, -20], [-95, -55], [-78, 35], [0, 20]], fill: "#1a5a4d" },
      { pts: [[0, 20], [78, 35], [0, 120], [-78, 35]], fill: "#133f37" },
      { pts: [[0, -110], [40, -70], [0, -40], [-40, -70]], fill: "rgba(255,255,255,0.85)" },
    ];

    facets.forEach((f, i) => {
      ctx.beginPath();
      f.pts.forEach((p, n) => {
        const x = p[0] * scale;
        const y = p[1] * scale;
        n === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.closePath();
      ctx.fillStyle = f.fill;
      ctx.globalAlpha = 0.92 - i * 0.02;
      ctx.fill();
      ctx.strokeStyle = "rgba(16,20,26,0.08)";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    });

    // Inclusion points inside
    const seeds = [
      [-18, -10], [22, 8], [-8, 35], [12, -28], [-28, 22], [30, 40], [0, 55], [-35, -35],
    ];
    seeds.forEach((p, i) => {
      const pulse = 0.5 + 0.5 * Math.sin(t / 400 + i);
      ctx.beginPath();
      ctx.arc(p[0] * scale * 0.55, p[1] * scale * 0.55, 3.5 + pulse * 2, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(16,20,26,${0.35 + pulse * 0.35})`;
      ctx.fill();
    });

    ctx.restore();
    crystalAngle += 0.0035;
  }

  function crystalLoop(t) {
    drawCrystal(t);
    requestAnimationFrame(crystalLoop);
  }
  requestAnimationFrame(crystalLoop);

  // --- Constellation ---
  const cloud = document.querySelector(".inclusion-cloud");
  const points = [
    [200, 95], [255, 130], [290, 185], [270, 245], [210, 285],
    [145, 270], [115, 210], [130, 150], [175, 175], [230, 200],
    [190, 230], [160, 120], [245, 165],
  ];

  function buildConstellation() {
    if (!cloud) return;
    cloud.innerHTML = "";
    const links = [
      [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 0],
      [8, 9], [9, 10], [10, 8], [0, 8], [2, 9], [5, 10], [11, 7], [12, 1],
    ];
    links.forEach(([a, b], i) => {
      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("x1", points[a][0]);
      line.setAttribute("y1", points[a][1]);
      line.setAttribute("x2", points[b][0]);
      line.setAttribute("y2", points[b][1]);
      line.setAttribute("class", "inc-link");
      line.style.animationDelay = `${0.15 + i * 0.05}s`;
      cloud.appendChild(line);
    });
    points.forEach((p, i) => {
      const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      c.setAttribute("cx", p[0]);
      c.setAttribute("cy", p[1]);
      c.setAttribute("r", i < 8 ? 4.2 : 3);
      c.setAttribute("class", "inc-point");
      c.style.animationDelay = `${0.05 + i * 0.06}s`;
      cloud.appendChild(c);
    });
  }

  // --- Key bits ---
  const keyBits = document.getElementById("keyBits");
  function animateBits() {
    if (!keyBits) return;
    keyBits.innerHTML = "";
    const n = 48;
    for (let i = 0; i < n; i++) {
      const bit = document.createElement("div");
      bit.className = "bit";
      bit.textContent = "0";
      keyBits.appendChild(bit);
      const delay = 80 + i * 35;
      setTimeout(() => {
        const on = Math.random() > 0.42;
        bit.textContent = on ? "1" : "0";
        bit.classList.toggle("on", on);
      }, delay);
    }
  }

  // Init
  activateScene(0, true);
  updateProgress();
  buildConstellation();
})();
