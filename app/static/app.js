/* =====================================================================
   agentplatform2api · Production Console Frontend (100% Offline, Zero CDN)
   ===================================================================== */
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  let currentTab = "overview";

  const THEMES = [
    { id: "sky-day", name: "晴空汽水", swatch: "swatch-sky-day", mode: "day" },
    { id: "paper-day", name: "宣纸茶寮", swatch: "swatch-paper-day", mode: "day" },
    { id: "pop-day", name: "潮玩波普", swatch: "swatch-pop-day", mode: "day" },
    { id: "star-night", name: "极夜星图", swatch: "swatch-star-night", mode: "night" },
    { id: "lacquer-night", name: "黑漆绯金", swatch: "swatch-lacquer-night", mode: "night" },
    { id: "vinyl-night", name: "午夜黑胶", swatch: "swatch-vinyl-night", mode: "night" }
  ];

  const COMMON_ARS = [
    "1:1", "3:2", "2:3", "3:4", "4:3", "4:5", "5:4",
    "9:16", "16:9", "21:9", "1:4", "4:1", "1:8", "8:1", "9:21"
  ];

  const PER_MODEL_KEYS = [
    "thinking_g3_level", "native_thinking_mode", "synthetic_sandbox_enabled",
    "prefill_mode", "prefill_suppress_thinking",
    "inject_system_instruction", "inject_prefill", "prefill_instruction",
    "image_size", "image_aspect_ratio", "image_system_instruction", "inject_prefill_for_image"
  ];

  const GLOBAL_KEYS = [
    "express_location", "retry_max", "retry_backoff_seconds", "cross_channel_fallback",
    "img_compress_enabled", "img_compress_max_dim", "img_compress_max_mb", "img_compress_quality",
    "fake_streaming", "fake_streaming_interval",
    "resumable_enabled", "resumable_ttl_seconds", "resumable_max_streams",
    "roundrobin", "safety_score", "debug_outbound"
  ];

  const GLOBAL_NUM_KEYS = new Set([
    "retry_max", "retry_backoff_seconds",
    "img_compress_max_dim", "img_compress_max_mb", "img_compress_quality",
    "fake_streaming_interval", "resumable_ttl_seconds", "resumable_max_streams"
  ]);

  /* ── Shared Utilities ── */
  function escHtml(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function fmtNum(n) {
    return Number(n || 0).toLocaleString("zh-CN");
  }

  function fmtTok(n) {
    const v = Number(n || 0);
    if (v >= 1e6) return (v / 1e6).toFixed(2) + "M";
    if (v >= 1e3) return (v / 1e3).toFixed(1) + "K";
    return String(v);
  }

  function showToast(msg) {
    const host = $("toastHost");
    if (!host) return;
    const el = document.createElement("div");
    el.className = "toast";
    el.innerHTML = `<span class="sparkle">✦</span><span>${escHtml(msg)}</span>`;
    host.appendChild(el);
    setTimeout(() => {
      el.style.opacity = "0";
      el.style.transform = "translateY(-4px)";
      el.style.transition = "all 0.2s";
      setTimeout(() => el.remove(), 220);
    }, 2400);
  }

  async function apiFetch(url, opts) {
    const r = await fetch(url, opts || {});
    if (r.status === 401) {
      location.href = "/";
      throw new Error("unauthorized");
    }
    return r;
  }

  /* =====================================================================
     1. THEME SYSTEM (6 ART SKINS + CLICK-ORIGIN RIPPLE VIEW TRANSITION)
     ===================================================================== */
  const LEGACY_THEME_MAP = {
    "sakura-day": "paper-day",
    "sky-night": "star-night",
    "sakura-night": "lacquer-night",
    "hud-night": "lacquer-night"
  };
  let currentTheme = document.documentElement.getAttribute("data-theme") || "sky-day";
  if (LEGACY_THEME_MAP[currentTheme]) currentTheme = LEGACY_THEME_MAP[currentTheme];
  if (!THEMES.some((t) => t.id === currentTheme)) currentTheme = "sky-day";
  document.documentElement.setAttribute("data-theme", currentTheme);

  function renderThemeFx(id) {
    const bg = $("themeFxBg");
    const fg = $("themeFx");
    if (!bg || !fg) return;
    if (id === "sky-day") {
      bg.innerHTML = `
        <div class="fx-sunbeam"></div>
        <svg class="fx-bg-svg fx-clouds" viewBox="0 0 1200 280" preserveAspectRatio="xMidYMax slice" fill="none">
          <defs>
            <linearGradient id="skyCloudHi" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stop-color="#ffffff" stop-opacity="0.88"/>
              <stop offset="68%" stop-color="#e0f4ff" stop-opacity="0.58"/>
              <stop offset="100%" stop-color="#bae6fd" stop-opacity="0.18"/>
            </linearGradient>
            <linearGradient id="skyCloudMid" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stop-color="#ffffff" stop-opacity="0.72"/>
              <stop offset="100%" stop-color="#7dd3fc" stop-opacity="0.34"/>
            </linearGradient>
            <radialGradient id="ramuneOrb" cx="34%" cy="30%" r="68%">
              <stop offset="0%" stop-color="#ffffff" stop-opacity="0.72"/>
              <stop offset="48%" stop-color="#38bdf8" stop-opacity="0.16"/>
              <stop offset="100%" stop-color="#0db896" stop-opacity="0.24"/>
            </radialGradient>
          </defs>
          <!-- Distant anime cumulus peaks -->
          <path d="M0 280V176C48 150 96 154 134 176C174 118 258 110 318 158C354 108 432 94 494 146C548 116 616 120 662 162C718 106 814 96 882 148C930 104 1022 98 1088 148C1132 126 1174 132 1200 148V280H0Z" fill="url(#skyCloudHi)"/>
          <!-- Mid-layer cloud shadow belly -->
          <path d="M0 280V212C82 184 164 188 232 216C312 162 426 160 510 206C596 168 708 170 792 210C886 160 1002 162 1094 202C1138 184 1176 188 1200 200V280H0Z" fill="url(#skyCloudMid)"/>
          <!-- Soft sunlit rim stroke -->
          <path d="M134 176C174 118 258 110 318 158C354 108 432 94 494 146" stroke="rgba(255,255,255,0.85)" stroke-width="2" stroke-linecap="round"/>
          <path d="M662 162C718 106 814 96 882 148C930 104 1022 98 1088 148" stroke="rgba(255,255,255,0.80)" stroke-width="2" stroke-linecap="round"/>
          <!-- Ramune glass marble crest -->
          <circle cx="984" cy="84" r="46" fill="url(#ramuneOrb)" stroke="rgba(0,119,204,0.22)" stroke-width="1.5"/>
          <circle cx="984" cy="84" r="66" stroke="rgba(13,184,150,0.18)" stroke-width="1.2" stroke-dasharray="4 7"/>
          <path d="M962 66 A28 28 0 0 1 996 56" stroke="rgba(255,255,255,0.88)" stroke-width="3" stroke-linecap="round"/>
          <path d="M910 54 L913 62 L921 65 L913 68 L910 76 L907 68 L899 65 L907 62 Z" fill="rgba(0,159,227,0.34)"/>
          <path d="M1060 36 L1062 42 L1068 44 L1062 46 L1060 52 L1058 46 L1052 44 L1058 42 Z" fill="rgba(13,184,150,0.36)"/>
        </svg>
      `;
      fg.innerHTML = `
        <span class="fx-bubble" style="--bx:8%;--bs:24px;--bd:11.5s;--bdelay:-1.5s;--bdrift:16px"></span>
        <span class="fx-bubble" style="--bx:22%;--bs:14px;--bd:9.2s;--bdelay:-5.4s;--bdrift:-12px"></span>
        <span class="fx-bubble" style="--bx:44%;--bs:18px;--bd:12.4s;--bdelay:-8.2s;--bdrift:14px"></span>
        <span class="fx-bubble" style="--bx:66%;--bs:22px;--bd:10.6s;--bdelay:-3.1s;--bdrift:-15px"></span>
        <span class="fx-bubble" style="--bx:82%;--bs:15px;--bd:8.8s;--bdelay:-6.8s;--bdrift:12px"></span>
        <span class="fx-bubble" style="--bx:92%;--bs:28px;--bd:13.2s;--bdelay:-2.2s;--bdrift:-18px"></span>
      `;
    } else if (id === "paper-day") {
      bg.innerHTML = `
        <div class="fx-seal"><span>茶</span><span>寮</span></div>
        <svg class="fx-bg-svg fx-bamboo" viewBox="0 0 340 340" fill="none">
          <!-- Primary & secondary bamboo stalks with authentic nodes (竹节) -->
          <path d="M26 -8 C40 72, 48 158, 34 308" stroke="rgba(39,115,92,0.20)" stroke-width="3.6" stroke-linecap="round"/>
          <path d="M58 -12 C68 62, 72 138, 60 250" stroke="rgba(110,96,80,0.13)" stroke-width="2.2" stroke-linecap="round"/>
          <line x1="29" y1="68" x2="43" y2="66" stroke="rgba(39,115,92,0.32)" stroke-width="2.4" stroke-linecap="round"/>
          <line x1="36" y1="142" x2="50" y2="140" stroke="rgba(39,115,92,0.30)" stroke-width="2.4" stroke-linecap="round"/>
          <line x1="34" y1="218" x2="47" y2="216" stroke="rgba(39,115,92,0.26)" stroke-width="2.2" stroke-linecap="round"/>
          <line x1="60" y1="98" x2="71" y2="96" stroke="rgba(110,96,80,0.24)" stroke-width="1.8" stroke-linecap="round"/>
          <!-- Fine side twigs -->
          <path d="M37 68 C84 52, 134 48, 184 62" stroke="rgba(39,115,92,0.16)" stroke-width="1.6" stroke-linecap="round"/>
          <path d="M43 141 C98 118, 152 116, 212 136" stroke="rgba(39,115,92,0.15)" stroke-width="1.5" stroke-linecap="round"/>
          <path d="M66 98 C104 82, 136 80, 168 92" stroke="rgba(184,124,34,0.16)" stroke-width="1.2" stroke-linecap="round"/>
          <!-- Calligraphic 个/介 bamboo leaf sprays -->
          <path d="M98 56 Q132 24 166 42 Q132 58 98 56Z" fill="rgba(39,115,92,0.20)"/>
          <path d="M126 57 Q166 38 202 60 Q164 68 126 57Z" fill="rgba(39,115,92,0.16)"/>
          <path d="M108 58 Q138 68 164 88 Q130 78 108 58Z" fill="rgba(39,115,92,0.14)"/>
          <path d="M114 126 Q154 94 192 112 Q152 130 114 126Z" fill="rgba(39,115,92,0.18)"/>
          <path d="M150 128 Q194 112 228 138 Q188 144 150 128Z" fill="rgba(184,124,34,0.17)"/>
          <path d="M132 132 Q162 146 186 168 Q154 156 132 132Z" fill="rgba(39,115,92,0.13)"/>
          <path d="M122 86 Q152 68 178 82 Q150 92 122 86Z" fill="rgba(184,72,36,0.12)"/>
        </svg>
        <svg class="fx-bg-svg fx-mountains" viewBox="0 0 1200 280" preserveAspectRatio="xMidYMax slice" fill="none">
          <defs>
            <linearGradient id="inkFar" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stop-color="#6e6050" stop-opacity="0.16"/>
              <stop offset="78%" stop-color="#6e6050" stop-opacity="0.04"/>
              <stop offset="100%" stop-color="#f5efe4" stop-opacity="0"/>
            </linearGradient>
            <linearGradient id="inkMid" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stop-color="#27735c" stop-opacity="0.16"/>
              <stop offset="82%" stop-color="#27735c" stop-opacity="0.04"/>
              <stop offset="100%" stop-color="#f5efe4" stop-opacity="0"/>
            </linearGradient>
            <linearGradient id="inkNear" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stop-color="#7c5836" stop-opacity="0.13"/>
              <stop offset="85%" stop-color="#7c5836" stop-opacity="0.03"/>
              <stop offset="100%" stop-color="#f5efe4" stop-opacity="0"/>
            </linearGradient>
          </defs>
          <!-- 墨分五色 misty ink-wash ridges -->
          <path d="M0 280V172 C120 112 220 74 340 138 C460 202 580 96 730 132 C870 166 970 86 1090 126 C1140 142 1176 146 1200 156 V280Z" fill="url(#inkFar)"/>
          <!-- Floating valley mist veil -->
          <path d="M0 204 Q280 182 560 198 T1200 186 V226 Q900 236 600 220 T0 232 Z" fill="rgba(248,243,233,0.48)"/>
          <path d="M0 280V202 C160 138 290 132 430 188 C580 248 730 144 890 176 C1020 202 1120 166 1200 188 V280Z" fill="url(#inkMid)"/>
          <path d="M0 280V234 C220 184 410 196 620 228 C820 258 1010 202 1200 224 V280Z" fill="url(#inkNear)"/>
        </svg>
      `;
      fg.innerHTML = `
        <span class="fx-leaf" style="--lx:22%;--ls:16px;--ld:14s;--ldelay:-2s"></span>
        <span class="fx-leaf" style="--lx:58%;--ls:13px;--ld:17s;--ldelay:-8s"></span>
        <span class="fx-leaf" style="--lx:86%;--ls:15px;--ld:15s;--ldelay:-12s"></span>
      `;
    } else if (id === "pop-day") {
      bg.innerHTML = `
        <svg class="fx-bg-svg fx-pop-art" viewBox="0 0 1200 800" preserveAspectRatio="xMidYMid slice" fill="none">
          <!-- Top-right Comic Starburst & Offset Sun Disc -->
          <polygon points="1116,12 1134,56 1180,44 1152,82 1192,108 1144,114 1152,162 1114,132 1084,168 1084,120 1038,122 1072,88 1042,50 1090,62" fill="rgba(255,212,59,0.34)" stroke="rgba(28,25,43,0.22)" stroke-width="2.5" stroke-linejoin="round"/>
          <circle cx="1104" cy="96" r="88" stroke="rgba(28,25,43,0.16)" stroke-width="2.2" stroke-dasharray="10 8"/>
          <!-- Edge-anchored Memphis Offset Zigzag Ribbons (shadow + vibrant ink) -->
          <path d="M-16 704 L52 662 L116 704 L180 662 L244 704 L308 662 L372 704" stroke="rgba(28,25,43,0.14)" stroke-width="6.5" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M-20 698 L48 656 L112 698 L176 656 L240 698 L304 656 L368 698" stroke="rgba(45,92,242,0.26)" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M846 748 L901 712 L956 748 L1011 712 L1066 748 L1121 712 L1176 748" stroke="rgba(28,25,43,0.14)" stroke-width="6.5" stroke-linecap="round" stroke-linejoin="round"/>
          <path d="M840 742 L895 706 L950 742 L1005 706 L1060 742 L1115 706 L1170 742" stroke="rgba(255,71,108,0.26)" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/>
          <!-- Left-edge Pop Candy Capsule with comic offset shadow -->
          <rect x="44" y="128" width="54" height="54" rx="12" transform="rotate(-14 44 128)" fill="rgba(28,25,43,0.12)"/>
          <rect x="40" y="124" width="54" height="54" rx="12" transform="rotate(-14 40 124)" fill="rgba(18,184,116,0.20)" stroke="rgba(28,25,43,0.22)" stroke-width="2.5"/>
          <!-- Right-margin Isometric Triangle & 4-point Spark -->
          <polygon points="1116,520 1144,572 1088,572" transform="rotate(16 1116 548)" fill="rgba(255,212,59,0.36)" stroke="rgba(28,25,43,0.20)" stroke-width="2.5" stroke-linejoin="round"/>
          <path d="M98 468 L104 484 L120 490 L104 496 L98 512 L92 496 L76 490 L92 484 Z" fill="rgba(255,71,108,0.24)" stroke="rgba(28,25,43,0.18)" stroke-width="1.8" stroke-linejoin="round"/>
        </svg>
        <span class="fx-pop-badge fx-pop-b1">✦ POP</span>
        <span class="fx-pop-badge fx-pop-b2">★ 429</span>
      `;
      fg.innerHTML = "";
    } else if (id === "vinyl-night") {
      bg.innerHTML = `
        <div class="fx-vinyl-disc"></div>
        <svg class="fx-bg-svg fx-vinyl-arm" viewBox="0 0 160 160" fill="none">
          <!-- Hi-Fi Turntable Gimbal Pivot & Counterweight -->
          <circle cx="128" cy="24" r="16" fill="#1a1714" stroke="rgba(229,169,60,0.48)" stroke-width="1.6"/>
          <circle cx="128" cy="24" r="7" fill="rgba(229,169,60,0.34)" stroke="rgba(245,202,118,0.65)" stroke-width="1.2"/>
          <rect x="133" y="4" width="12" height="14" rx="3" transform="rotate(32 133 4)" fill="#2a241d" stroke="rgba(229,169,60,0.42)" stroke-width="1.2"/>
          <!-- S-Curve Brushed Brass Tonearm Tube -->
          <path d="M124 34 C112 66, 98 84, 72 108 L56 122" stroke="rgba(245,202,118,0.62)" stroke-width="3.2" stroke-linecap="round"/>
          <path d="M124 34 C112 66, 98 84, 72 108 L56 122" stroke="rgba(255,244,214,0.28)" stroke-width="1" stroke-linecap="round"/>
          <!-- Angled Headshell & Glowing Amber Stylus Cartridge -->
          <rect x="40" y="118" width="22" height="10" rx="2.5" transform="rotate(-38 40 118)" fill="#29231c" stroke="rgba(229,169,60,0.68)" stroke-width="1.4"/>
          <circle cx="44" cy="130" r="2.6" fill="#de6b53"/>
          <circle cx="44" cy="130" r="6" fill="rgba(222,107,83,0.24)"/>
        </svg>
        <svg class="fx-bg-svg fx-vinyl-wave" viewBox="0 0 1200 210" preserveAspectRatio="xMidYMax slice" fill="none">
          <!-- Analog dB Reference Scale -->
          <line x1="0" y1="142" x2="1200" y2="142" stroke="rgba(229,169,60,0.10)" stroke-width="1" stroke-dasharray="2 10"/>
          <line x1="0" y1="172" x2="1200" y2="172" stroke="rgba(88,184,159,0.09)" stroke-width="1" stroke-dasharray="2 14"/>
          <!-- Warm Tube Harmonic Waves -->
          <path d="M0 132 Q120 68 240 126 T480 116 T720 130 T960 92 T1200 124" stroke="rgba(229,169,60,0.26)" stroke-width="1.8"/>
          <path d="M0 152 Q150 96 300 144 T600 132 T900 150 T1200 136" stroke="rgba(88,184,159,0.20)" stroke-width="1.4"/>
          <path d="M0 172 Q180 136 360 168 T720 156 T1080 174 T1200 162" stroke="rgba(222,107,83,0.18)" stroke-width="1.2" stroke-dasharray="4 6"/>
        </svg>
      `;
      fg.innerHTML = "";
    } else if (id === "lacquer-night") {
      bg.innerHTML = `
        <div class="fx-lacquer-seal"><span>漆</span><span>绘</span></div>
        <svg class="fx-bg-svg fx-lacquer-branch" viewBox="0 0 360 340" fill="none">
          <!-- Nocturnal Urushi Maki-e Ume Branch & Scattered Kinpaku Gold Dust -->
          <path d="M-10 18 C72 62, 134 88, 224 92" stroke="rgba(204,168,110,0.24)" stroke-width="2.6" stroke-linecap="round"/>
          <path d="M92 66 C134 36, 178 24, 232 34" stroke="rgba(204,168,110,0.18)" stroke-width="1.5" stroke-linecap="round"/>
          <path d="M138 82 C176 106, 210 124, 254 128" stroke="rgba(204,168,110,0.14)" stroke-width="1.2" stroke-linecap="round"/>
          <!-- Delicate 5-petal dusty crimson ume blossoms -->
          <g fill="rgba(217,104,122,0.26)">
            <circle cx="152" cy="40" r="4.5"/><circle cx="159" cy="44" r="4.5"/><circle cx="156" cy="51" r="4.5"/><circle cx="148" cy="51" r="4.5"/><circle cx="145" cy="44" r="4.5"/>
            <circle cx="204" cy="84" r="5"/><circle cx="212" cy="89" r="5"/><circle cx="209" cy="97" r="5"/><circle cx="199" cy="97" r="5"/><circle cx="196" cy="89" r="5"/>
          </g>
          <circle cx="152" cy="46" r="2.2" fill="rgba(204,168,110,0.68)"/>
          <circle cx="204" cy="91" r="2.5" fill="rgba(204,168,110,0.72)"/>
          <circle cx="232" cy="36" r="4" fill="rgba(217,104,122,0.24)"/>
          <circle cx="252" cy="126" r="3.5" fill="rgba(204,168,110,0.34)"/>
          <rect x="266" y="48" width="5" height="3.5" rx="0.8" transform="rotate(24 266 48)" fill="rgba(204,168,110,0.36)"/>
          <rect x="292" y="88" width="4" height="4" rx="0.6" transform="rotate(-18 292 88)" fill="rgba(217,104,122,0.32)"/>
        </svg>
        <svg class="fx-bg-svg fx-seigaiha" viewBox="0 0 640 280" preserveAspectRatio="xMaxYMax slice" fill="none">
          <!-- Traditional Seigaiha (青海波) Lacquer Filigree Waves -->
          <g stroke="rgba(204,168,110,0.20)" stroke-width="1.15">
            <path d="M240 280 A90 90 0 0 1 420 280"/><path d="M268 280 A62 62 0 0 1 392 280"/><path d="M296 280 A34 34 0 0 1 364 280"/>
            <path d="M390 280 A90 90 0 0 1 570 280"/><path d="M418 280 A62 62 0 0 1 542 280"/><path d="M446 280 A34 34 0 0 1 514 280"/>
            <path d="M510 280 A90 90 0 0 1 690 280"/><path d="M538 280 A62 62 0 0 1 662 280"/>
            <path d="M315 218 A90 90 0 0 1 495 218"/><path d="M343 218 A62 62 0 0 1 467 218"/><path d="M371 218 A34 34 0 0 1 439 218"/>
            <path d="M465 218 A90 90 0 0 1 645 218"/><path d="M493 218 A62 62 0 0 1 617 218"/>
          </g>
          <g stroke="rgba(217,104,122,0.17)" stroke-width="1">
            <path d="M90 280 A90 90 0 0 1 270 280"/><path d="M118 280 A62 62 0 0 1 242 280"/><path d="M146 280 A34 34 0 0 1 214 280"/>
            <path d="M165 218 A90 90 0 0 1 345 218"/><path d="M193 218 A62 62 0 0 1 317 218"/>
          </g>
        </svg>
      `;
      fg.innerHTML = `
        <span class="fx-gold-fleck" style="--gx:14%;--gs:6px;--gd:15s;--gdelay:-2s"></span>
        <span class="fx-gold-fleck" style="--gx:34%;--gs:4.5px;--gd:18s;--gdelay:-9s"></span>
        <span class="fx-gold-fleck" style="--gx:58%;--gs:5.5px;--gd:16s;--gdelay:-5s"></span>
        <span class="fx-gold-fleck" style="--gx:76%;--gs:7px;--gd:14s;--gdelay:-11s"></span>
        <span class="fx-gold-fleck" style="--gx:90%;--gs:5px;--gd:17s;--gdelay:-3.5s"></span>
      `;
    } else if (id === "star-night") {
      bg.innerHTML = `
        <div class="fx-aurora-veil"></div>
        <svg class="fx-bg-svg fx-astrolabe" viewBox="0 0 360 360" fill="none">
          <!-- Outer Graduated Brass Degree Ring -->
          <circle cx="180" cy="180" r="168" stroke="#e8d1a0" stroke-width="1.1" stroke-dasharray="2 7"/>
          <circle cx="180" cy="180" r="158" stroke="#e8d1a0" stroke-width="0.6" stroke-opacity="0.45"/>
          <circle cx="180" cy="180" r="132" stroke="#7cc4fa" stroke-width="1.1" stroke-opacity="0.78"/>
          <circle cx="180" cy="180" r="94" stroke="#e8d1a0" stroke-width="0.9" stroke-dasharray="1.5 5"/>
          <circle cx="180" cy="180" r="54" stroke="#7cc4fa" stroke-width="0.8" stroke-opacity="0.65"/>
          <!-- Tilted Ecliptic & Meridian Armillary Rings -->
          <ellipse cx="180" cy="180" rx="158" ry="62" transform="rotate(-26 180 180)" stroke="#e8d1a0" stroke-width="0.9" stroke-opacity="0.62"/>
          <ellipse cx="180" cy="180" rx="132" ry="44" transform="rotate(34 180 180)" stroke="#7cc4fa" stroke-width="0.8" stroke-opacity="0.48" stroke-dasharray="4 4"/>
          <!-- Celestial Axes & Cardinal Compass Rose -->
          <line x1="180" y1="8" x2="180" y2="352" stroke="#e8d1a0" stroke-width="0.6" stroke-opacity="0.48"/>
          <line x1="8" y1="180" x2="352" y2="180" stroke="#7cc4fa" stroke-width="0.6" stroke-opacity="0.48"/>
          <line x1="58" y1="58" x2="302" y2="302" stroke="#e8d1a0" stroke-width="0.5" stroke-opacity="0.32"/>
          <line x1="302" y1="58" x2="58" y2="302" stroke="#7cc4fa" stroke-width="0.5" stroke-opacity="0.32"/>
          <path d="M180 156 L186 174 L204 180 L186 186 L180 204 L174 186 L156 180 L174 174 Z" fill="rgba(232,209,160,0.42)" stroke="#e8d1a0" stroke-width="0.8"/>
          <!-- Crescent Orbit Marker -->
          <path d="M276 86 A9 9 0 1 1 264 98 A7 7 0 1 0 276 86 Z" fill="#e8d1a0" fill-opacity="0.72"/>
        </svg>
        <svg class="fx-bg-svg fx-constellation" viewBox="0 0 560 270" fill="none">
          <!-- Big Dipper & Celestial Arc Constellation -->
          <polyline points="36,216 114,162 204,176 284,114 378,130 468,68 514,116" stroke="rgba(232,209,160,0.34)" stroke-width="1.1" stroke-dasharray="3 4"/>
          <polyline points="204,176 248,228 354,208 378,130" stroke="rgba(124,196,250,0.26)" stroke-width="1" stroke-dasharray="2 4"/>
          <!-- Double-halo Glowing Star Nodes -->
          <circle cx="36" cy="216" r="6" fill="rgba(232,209,160,0.16)"/><circle cx="36" cy="216" r="2.6" fill="#e8d1a0" fill-opacity="0.82"/>
          <circle cx="114" cy="162" r="7" fill="rgba(124,196,250,0.18)"/><circle cx="114" cy="162" r="3" fill="#7cc4fa" fill-opacity="0.86"/>
          <circle cx="204" cy="176" r="9" fill="rgba(232,209,160,0.20)"/><circle cx="204" cy="176" r="3.4" fill="#e8d1a0" fill-opacity="0.90"/>
          <path d="M204 166v20M194 176h20" stroke="rgba(232,209,160,0.45)" stroke-width="0.8"/>
          <circle cx="284" cy="114" r="7" fill="rgba(124,196,250,0.16)"/><circle cx="284" cy="114" r="2.8" fill="#7cc4fa" fill-opacity="0.82"/>
          <circle cx="378" cy="130" r="6" fill="rgba(232,209,160,0.15)"/><circle cx="378" cy="130" r="2.6" fill="#e8d1a0" fill-opacity="0.78"/>
          <circle cx="468" cy="68" r="10" fill="rgba(232,209,160,0.22)"/><circle cx="468" cy="68" r="3.6" fill="#e8d1a0" fill-opacity="0.94"/>
          <path d="M468 56v24M456 68h24" stroke="rgba(232,209,160,0.52)" stroke-width="0.9"/>
          <circle cx="514" cy="116" r="5" fill="rgba(124,196,250,0.14)"/><circle cx="514" cy="116" r="2.2" fill="#7cc4fa" fill-opacity="0.72"/>
          <circle cx="248" cy="228" r="2.2" fill="#7cc4fa" fill-opacity="0.68"/>
          <circle cx="354" cy="208" r="2.4" fill="#e8d1a0" fill-opacity="0.72"/>
        </svg>
      `;
      fg.innerHTML = `
        <span class="fx-meteor" style="--my:11%;--mx:-8%;--md:7.2s;--mdelay:0.6s"></span>
        <span class="fx-meteor" style="--my:28%;--mx:-12%;--md:10.4s;--mdelay:4.2s"></span>
        <span class="fx-star" style="--sx:11%;--sy:16%;--ss:12px;--sc:#e8d1a0;--sd:3.2s;--sdelay:-0.8s">✦</span>
        <span class="fx-star" style="--sx:86%;--sy:14%;--ss:10px;--sc:#7cc4fa;--sd:3.8s;--sdelay:-2.1s">✦</span>
        <span class="fx-star" style="--sx:91%;--sy:44%;--ss:13px;--sc:#e8d1a0;--sd:4.2s;--sdelay:-1.4s">✦</span>
        <span class="fx-star" style="--sx:7%;--sy:62%;--ss:10px;--sc:#7cc4fa;--sd:3.6s;--sdelay:-2.9s">✦</span>
        <span class="fx-star" style="--sx:82%;--sy:78%;--ss:11px;--sc:#e8d1a0;--sd:3.4s;--sdelay:-0.4s">✦</span>
      `;
    } else {
      bg.innerHTML = "";
      fg.innerHTML = "";
    }
  }

  function renderThemePickers() {
    const chip = (t) => `
      <button class="theme-chip ${t.id === currentTheme ? "active" : ""}" onclick="setTheme('${t.id}', event)">
        <span class="swatch ${t.swatch}"></span>
        <span class="truncate">${t.name}</span>
      </button>
    `;
    const dayChips = THEMES.filter((t) => t.mode === "day").map(chip).join("");
    const nightChips = THEMES.filter((t) => t.mode === "night").map(chip).join("");
    const dayIcon = `<svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M4.93 4.93l1.41 1.41m11.32 11.32 1.41 1.41M2 12h2m16 0h2M4.93 19.07l1.41-1.41m11.32-11.32 1.41-1.41"/></svg>`;
    const nightIcon = `<svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>`;
    const html = `
      <div class="theme-col">
        <div class="theme-col-head">${dayIcon}<span>白天</span></div>
        ${dayChips}
      </div>
      <div class="theme-col">
        <div class="theme-col-head">${nightIcon}<span>黑夜</span></div>
        ${nightChips}
      </div>
    `;
    ["sideThemeGrid", "mobThemeGrid"].forEach((id) => {
      const el = $(id);
      if (el) el.innerHTML = html;
    });
    const active = THEMES.find((t) => t.id === currentTheme);
    if ($("mobThemeName") && active) $("mobThemeName").textContent = active.name;
    renderThemeFx(currentTheme);
  }

  const SUPPORTS_POPOVER = typeof HTMLElement !== "undefined" && Object.hasOwn(HTMLElement.prototype, "popover");

  function closeMobileThemePopover() {
    const pop = $("mobileThemePopover");
    if (!pop) return;
    if (SUPPORTS_POPOVER && typeof pop.hidePopover === "function") {
      try { if (pop.matches(":popover-open")) pop.hidePopover(); } catch (e) {}
    }
    pop.classList.remove("open");
  }

  const VT_STYLE_MAP = {
    "sky-day": "ripple",
    "star-night": "ripple",
    "paper-day": "scroll",
    "lacquer-night": "scroll",
    "pop-day": "slash",
    "vinyl-night": "iris"
  };

  /* ── Optional Zero-Asset Web Audio Theme Micro-SFX (Default OFF) ── */
  let sfxEnabled = false;
  try { sfxEnabled = localStorage.getItem("ap2a_sfx") === "1"; } catch (e) {}
  let audioCtx = null;

  function syncSfxUi() {
    const onIcon = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 0 1 0 7.07"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14"/></svg>`;
    const offIcon = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><line x1="23" y1="9" x2="17" y2="15"/><line x1="17" y1="9" x2="23" y2="15"/></svg>`;
    const sBtn = $("sideSfxBtn");
    if (sBtn) {
      sBtn.classList.toggle("sfx-on", sfxEnabled);
      sBtn.innerHTML = `${sfxEnabled ? onIcon : offIcon}<span id="sideSfxText">触控音效：${sfxEnabled ? "开" : "关"}</span>`;
    }
    const mBtn = $("mobSfxBtn");
    if (mBtn) {
      mBtn.classList.toggle("sfx-on", sfxEnabled);
      mBtn.title = `触控音效：${sfxEnabled ? "开" : "关"}`;
      mBtn.innerHTML = sfxEnabled ? onIcon : offIcon;
    }
  }

  function playThemeSfx(themeId, kind) {
    if (!sfxEnabled) return;
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return;
    try {
      if (!audioCtx) audioCtx = new AC();
      const emit = () => {
        const now = audioCtx.currentTime;
        const th = themeId || currentTheme;
        const scale = kind === "light" ? 0.72 : 1;
        const tone = (type, f0, f1, dur, vol, delay) => {
          const osc = audioCtx.createOscillator();
          const gain = audioCtx.createGain();
          const t0 = now + (delay || 0);
          osc.type = type;
          osc.frequency.setValueAtTime(f0, t0);
          if (f1 !== f0) osc.frequency.exponentialRampToValueAtTime(Math.max(40, f1), t0 + dur);
          gain.gain.setValueAtTime(0.0001, t0);
          gain.gain.linearRampToValueAtTime(vol * scale, t0 + 0.006);
          gain.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
          osc.connect(gain);
          gain.connect(audioCtx.destination);
          osc.start(t0);
          osc.stop(t0 + dur + 0.012);
        };
        if (th === "sky-day") {
          tone("sine", 540, 1040, 0.062, 0.055);
        } else if (th === "star-night") {
          tone("sine", 880, 880, 0.105, 0.04);
          tone("sine", 1318.5, 1318.5, 0.115, 0.026, 0.018);
        } else if (th === "paper-day") {
          tone("triangle", 360, 195, 0.05, 0.058);
        } else if (th === "lacquer-night") {
          tone("triangle", 270, 145, 0.058, 0.055);
        } else if (th === "pop-day") {
          tone("square", 660, 260, 0.036, 0.034);
        } else if (th === "vinyl-night") {
          tone("triangle", 190, 78, 0.048, 0.062);
          tone("sine", 1420, 960, 0.018, 0.024, 0.006);
        }
      };
      if (audioCtx.state === "suspended") audioCtx.resume().then(emit).catch(() => {});
      else emit();
    } catch (e) {}
  }

  window.toggleSfx = function () {
    sfxEnabled = !sfxEnabled;
    try { localStorage.setItem("ap2a_sfx", sfxEnabled ? "1" : "0"); } catch (e) {}
    syncSfxUi();
    if (sfxEnabled) {
      playThemeSfx(currentTheme, "main");
      showToast("已开启主题触控音效（纯前端离线合成）");
    } else {
      showToast("已静音触控音效");
    }
  };

  /* ── Theme-Tailored Click Particle Micro-Burst (Zero Persistent Overhead) ── */
  let burstHost = null;
  function spawnThemeBurst(x, y, themeId) {
    if (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    if (typeof x !== "number" || typeof y !== "number" || (!x && !y)) return;
    if (!burstHost) {
      burstHost = document.createElement("div");
      burstHost.className = "fx-burst-host";
      burstHost.setAttribute("aria-hidden", "true");
      document.body.appendChild(burstHost);
    }
    const th = themeId || currentTheme;
    const count = 5;
    for (let i = 0; i < count; i++) {
      const el = document.createElement("span");
      const angle = ((Math.PI * 2) / count) * i + (Math.random() * 0.55 - 0.27);
      const dist = 18 + Math.random() * 20;
      const dx = Math.cos(angle) * dist;
      const dy = Math.sin(angle) * dist - 6;
      if (th === "sky-day") {
        el.className = "fx-burst-p dot-p";
        el.style.setProperty("--bc", i % 2 ? "#0db896" : "#38bdf8");
        el.style.setProperty("--bs", (5 + Math.random() * 4).toFixed(1) + "px");
      } else if (th === "star-night") {
        el.className = "fx-burst-p";
        el.textContent = "✦";
        el.style.setProperty("--bc", i % 2 ? "#7cc4fa" : "#e8d1a0");
        el.style.setProperty("--bs", (9 + Math.random() * 4).toFixed(1) + "px");
      } else if (th === "paper-day") {
        el.className = "fx-burst-p leaf-p";
        el.style.setProperty("--bc", i % 2 ? "#b84824" : "#27735c");
        el.style.setProperty("--bs", (8 + Math.random() * 4).toFixed(1) + "px");
      } else if (th === "lacquer-night") {
        el.className = "fx-burst-p fleck-p";
        el.style.setProperty("--bc", i % 2 ? "#d9687a" : "#cca86e");
        el.style.setProperty("--bs", (6 + Math.random() * 3.5).toFixed(1) + "px");
      } else if (th === "pop-day") {
        el.className = "fx-burst-p";
        el.textContent = i % 2 ? "★" : "✦";
        el.style.setProperty("--bc", i % 3 === 0 ? "#ffd43b" : (i % 3 === 1 ? "#ff476c" : "#2d5cf2"));
        el.style.setProperty("--bs", (10 + Math.random() * 4).toFixed(1) + "px");
      } else {
        el.className = "fx-burst-p dot-p";
        el.style.setProperty("--bc", i % 2 ? "#58b89f" : "#e5a93c");
        el.style.setProperty("--bs", (5 + Math.random() * 3).toFixed(1) + "px");
      }
      el.style.setProperty("--bx0", x.toFixed(1) + "px");
      el.style.setProperty("--by0", y.toFixed(1) + "px");
      el.style.setProperty("--bdx", dx.toFixed(1) + "px");
      el.style.setProperty("--bdy", dy.toFixed(1) + "px");
      el.style.setProperty("--brot", Math.round((Math.random() - 0.5) * 180) + "deg");
      burstHost.appendChild(el);
      setTimeout(() => el.remove(), 480);
    }
  }

  const SFX_CLICK_SEL = "button, summary.record-head, .channel-choice, .log-line, [role='button'], [role='tab']";
  document.addEventListener("click", (e) => {
    const el = e.target && e.target.closest ? e.target.closest(SFX_CLICK_SEL) : null;
    if (!el || el.disabled) return;
    if (el.id === "sideSfxBtn" || el.id === "mobSfxBtn" || el.classList.contains("theme-chip")) return;
    const isPrimary = el.classList.contains("btn-primary");
    if (isPrimary) spawnThemeBurst(e.clientX, e.clientY, currentTheme);
    playThemeSfx(currentTheme, isPrimary ? "main" : "light");
  });
  document.addEventListener("change", (e) => {
    if (e.target && e.target.matches && e.target.matches(".switch input[type='checkbox'], select.field-select")) {
      playThemeSfx(currentTheme, "light");
    }
  });

  let activeThemeVT = null;

  window.setTheme = function (id, ev) {
    if (LEGACY_THEME_MAP[id]) id = LEGACY_THEME_MAP[id];
    if (id === currentTheme && document.documentElement.getAttribute("data-theme") === id) {
      closeMobileThemePopover();
      return;
    }
    const applyTheme = () => {
      currentTheme = id;
      document.documentElement.setAttribute("data-theme", id);
      try { localStorage.setItem("ap2a_theme", id); } catch (e) {}
      renderThemePickers();
      if (lastStatsPayload) renderOverview(lastStatsPayload);
      closeMobileThemePopover();
    };

    const reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let x = window.innerWidth / 2;
    let y = window.innerHeight / 2;
    if (ev && typeof ev.clientX === "number" && (ev.clientX || ev.clientY)) {
      x = ev.clientX;
      y = ev.clientY;
    } else if (ev && ev.currentTarget && ev.currentTarget.getBoundingClientRect) {
      const rect = ev.currentTarget.getBoundingClientRect();
      x = rect.left + rect.width / 2;
      y = rect.top + rect.height / 2;
    }

    playThemeSfx(id);
    if (document.startViewTransition && !reduceMotion) {
      const endRadius = Math.hypot(Math.max(x, window.innerWidth - x), Math.max(y, window.innerHeight - y));
      const root = document.documentElement;
      if (activeThemeVT) {
        try { activeThemeVT.skipTransition(); } catch (e) {}
        activeThemeVT = null;
      }
      root.style.setProperty("--vt-x", `${x}px`);
      root.style.setProperty("--vt-y", `${y}px`);
      root.style.setProperty("--vt-r", `${endRadius}px`);
      root.setAttribute("data-vt-style", VT_STYLE_MAP[id] || "ripple");
      root.classList.add("theme-vt-active");
      try {
        const vt = document.startViewTransition(() => {
          applyTheme();
        });
        activeThemeVT = vt;
        vt.finished.finally(() => {
          if (activeThemeVT === vt) activeThemeVT = null;
          root.classList.remove("theme-vt-active");
          root.removeAttribute("data-vt-style");
          spawnThemeBurst(x, y, id);
        }).catch(() => {});
        return;
      } catch (e) {
        root.classList.remove("theme-vt-active");
        root.removeAttribute("data-vt-style");
      }
    }

    applyTheme();
    spawnThemeBurst(x, y, id);
    if (!reduceMotion) {
      const wave = document.createElement("div");
      wave.className = "theme-ripple-fallback";
      wave.style.setProperty("--vt-x", `${x}px`);
      wave.style.setProperty("--vt-y", `${y}px`);
      document.body.appendChild(wave);
      setTimeout(() => wave.remove(), 500);
    }
  };

  // Fallback click/light-dismiss handler ONLY when native Popover API is unavailable
  if (!SUPPORTS_POPOVER) {
    if ($("mobThemeBtn")) {
      $("mobThemeBtn").addEventListener("click", (e) => {
        e.stopPropagation();
        $("mobileThemePopover").classList.toggle("open");
      });
    }
    document.addEventListener("click", (e) => {
      const pop = $("mobileThemePopover");
      const btn = $("mobThemeBtn");
      if (pop && pop.classList.contains("open") && !pop.contains(e.target) && (!btn || !btn.contains(e.target))) {
        pop.classList.remove("open");
      }
    });
  }

  /* =====================================================================
     2. OVERVIEW & LIVE STREAM X-RAY PULSE
     ===================================================================== */
  let currentRange = "today";
  let modelSort = "calls";
  let showAllModels = false;
  let lastStatsPayload = null;
  let lastStatsSig = "";
  let overviewDirty = false;
  let lastRecordsVersion = -1;
  let lastLiveCount = 0;

  window.setRange = function (range) {
    currentRange = range;
    lastStatsSig = "";
    document.querySelectorAll("[data-range]").forEach((b) => b.classList.toggle("active", b.dataset.range === range));
    fetchStats();
  };

  window.setModelSort = function (sort) {
    modelSort = sort;
    document.querySelectorAll("[data-sort]").forEach((b) => b.classList.toggle("active", b.dataset.sort === sort));
    if (lastStatsPayload) renderModelStats(lastStatsPayload.models || []);
  };

  window.toggleShowAllModels = function () {
    showAllModels = !showAllModels;
    if (lastStatsPayload) renderModelStats(lastStatsPayload.models || []);
  };

  async function fetchStats() {
    try {
      const r = await apiFetch("/api/stats?range=" + encodeURIComponent(currentRange));
      if (!r.ok) return;
      const d = await r.json();
      lastStatsPayload = d;
      const uptimeH = (Number(d.uptime || 0) / 3600).toFixed(1);
      if ($("sideUptimeText")) $("sideUptimeText").textContent = `已运行 ${uptimeH} h`;
      if ($("mobUptimeText")) $("mobUptimeText").textContent = `已运行 ${uptimeH} h`;

      const sig = JSON.stringify([
        currentRange, d.total, d.success, d.retries, d.error,
        d.prompt_tokens, d.cached_tokens, d.completion_tokens,
        d.thought_tokens, d.output_tokens, d.records_version,
        d.models ? d.models.length : 0
      ]);
      if (sig !== lastStatsSig) {
        lastStatsSig = sig;
        if (currentTab === "overview") {
          renderOverview(d);
          overviewDirty = false;
        } else {
          overviewDirty = true;
        }
      }
      if (typeof d.records_version === "number" && d.records_version !== lastRecordsVersion) {
        fetchRecords();
      }
    } catch (e) {}
  }

  function renderOverview(d) {
    const total = Number(d.total || 0);
    const ok = Number(d.success || 0);
    const retry = Number(d.retries || 0);
    const err = Number(d.error || 0);
    const exReq = Number(d.express_requests || 0);
    const ckReq = Number(d.cookie_requests || 0);
    const uptimeH = (Number(d.uptime || 0) / 3600).toFixed(1);

    if ($("sideUptimeText")) $("sideUptimeText").textContent = `已运行 ${uptimeH} h`;
    if ($("mobUptimeText")) $("mobUptimeText").textContent = `已运行 ${uptimeH} h`;

    const doneTotal = ok + err;
    const okRateNum = doneTotal > 0 ? (ok / doneTotal) * 100 : null;
    const okRateStr = okRateNum != null ? okRateNum.toFixed(1) : "—";

    $("kpiTotal").textContent = fmtNum(total);
    if (total > 0) {
      const exPct = Math.round((exReq / total) * 100);
      const ckPct = Math.max(0, 100 - exPct);
      $("kpiTotalSub").textContent = `Express ${exPct}% · Cookie ${ckPct}%`;
    } else {
      $("kpiTotalSub").textContent = "等待发起请求";
    }
    $("kpiOk").textContent = fmtNum(ok);
    $("kpiOkSub").textContent = okRateNum != null ? `成功率 ${okRateStr}%` : "待命";
    $("kpiRetry").textContent = fmtNum(retry);
    $("kpiErr").textContent = fmtNum(err);

    // Health Donut SVG (clean 4px separated arcs, calm standby ring when empty)
    const cx = 64, cy = 64, r = 50, circ = 2 * Math.PI * r;
    const rawSegs = [
      { val: ok, color: "var(--success)" },
      { val: retry, color: "var(--warning)" },
      { val: err, color: "var(--danger)" }
    ].filter((s) => s.val > 0);

    let arcsHtml = "";
    if (rawSegs.length > 0) {
      const sumVal = rawSegs.reduce((a, b) => a + b.val, 0);
      const gap = rawSegs.length > 1 ? 4 : 0;
      const usable = Math.max(10, circ - gap * rawSegs.length);
      const minArc = 10;
      let rawLens = rawSegs.map((s) => Math.max(minArc, (s.val / sumVal) * usable));
      const lenTotal = rawLens.reduce((a, b) => a + b, 0);
      rawLens = rawLens.map((l) => (l / lenTotal) * usable);

      let offset = 0;
      arcsHtml = rawSegs.map((s, idx) => {
        const len = rawLens[idx];
        const dashOffset = -(offset + gap / 2);
        offset += len + gap;
        return `<circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="${s.color}" stroke-width="10.5"
          stroke-dasharray="${len.toFixed(2)} ${circ.toFixed(2)}" stroke-dashoffset="${dashOffset.toFixed(2)}"
          stroke-linecap="butt" transform="rotate(-90 ${cx} ${cy})"/>`;
      }).join("");
    }

    let donutDecor = `<circle cx="${cx}" cy="${cy}" r="39.5" fill="none" stroke="var(--border)" stroke-width="1" stroke-dasharray="2.5 3.5" opacity="0.75"/>`;
    let donutOuter = "";
    if (currentTheme === "star-night") {
      donutDecor = `
        <circle cx="${cx}" cy="${cy}" r="60" fill="none" stroke="rgba(232,209,160,0.42)" stroke-width="1" stroke-dasharray="2 5"/>
        <path d="M64 1v5M64 122v5M1 64h5M122 64h5" stroke="#e8d1a0" stroke-width="1.2" opacity="0.72"/>
        <circle cx="${cx}" cy="${cy}" r="39" fill="none" stroke="rgba(124,196,250,0.4)" stroke-width="0.9" stroke-dasharray="1.5 4"/>
      `;
    } else if (currentTheme === "pop-day") {
      donutOuter = `
        <circle cx="${cx}" cy="${cy}" r="56" fill="none" stroke="#1c192b" stroke-width="2"/>
        <circle cx="${cx}" cy="${cy}" r="44" fill="none" stroke="#1c192b" stroke-width="2"/>
      `;
      donutDecor = "";
    } else if (currentTheme === "vinyl-night") {
      donutDecor = `
        <circle cx="${cx}" cy="${cy}" r="59" fill="none" stroke="rgba(229,169,60,0.28)" stroke-width="1"/>
        <circle cx="${cx}" cy="${cy}" r="56.5" fill="none" stroke="rgba(229,169,60,0.14)" stroke-width="0.8"/>
        <circle cx="${cx}" cy="${cy}" r="41" fill="none" stroke="rgba(229,169,60,0.22)" stroke-width="0.9"/>
        <circle cx="${cx}" cy="${cy}" r="38" fill="none" stroke="rgba(229,169,60,0.34)" stroke-width="0.8" stroke-dasharray="2 3"/>
      `;
    } else if (currentTheme === "paper-day" || currentTheme === "lacquer-night") {
      const ringCol = currentTheme === "lacquer-night" ? "rgba(204,168,110,0.45)" : "rgba(184,72,36,0.35)";
      donutDecor = `
        <circle cx="${cx}" cy="${cy}" r="58.5" fill="none" stroke="${ringCol}" stroke-width="1"/>
        <circle cx="${cx}" cy="${cy}" r="39.5" fill="none" stroke="${ringCol}" stroke-width="1"/>
        <circle cx="${cx}" cy="${cy}" r="36.5" fill="none" stroke="${ringCol}" stroke-width="0.6" stroke-dasharray="2 4"/>
      `;
    } else if (currentTheme === "sky-day") {
      donutDecor = `
        <circle cx="${cx}" cy="${cy}" r="58" fill="none" stroke="rgba(56,189,248,0.26)" stroke-width="1.5"/>
        <circle cx="${cx}" cy="${cy}" r="39.5" fill="none" stroke="rgba(0,119,204,0.22)" stroke-width="1" stroke-dasharray="3 4"/>
      `;
    }

    $("healthDonutSvg").innerHTML = `
      <circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="var(--track-bg)" stroke-width="10.5"/>
      ${donutDecor}
      ${arcsHtml}
      ${donutOuter}
    `;

    const hp = $("healthStatusPill");
    if (okRateNum == null) {
      $("donutPct").textContent = "—";
      $("donutSub").textContent = "待命";
      if (hp) { hp.textContent = "待命"; hp.className = "pill primary num"; }
    } else {
      $("donutPct").textContent = `${okRateStr}%`;
      $("donutSub").textContent = "成功率";
      if (hp) {
        hp.textContent = (okRateNum >= 95 ? "健康 " : "波动 ") + okRateStr + "%";
        hp.className = okRateNum >= 95 ? "pill success num" : "pill warning num";
      }
    }

    $("donutLegend").innerHTML = `
      <div class="donut-stat-cell">
        <span class="donut-stat-lbl"><i class="dot" style="background:var(--success)"></i>正常</span>
        <strong class="donut-stat-val num">${fmtNum(ok)}</strong>
      </div>
      <div class="donut-stat-cell">
        <span class="donut-stat-lbl"><i class="dot" style="background:var(--warning)"></i>重试</span>
        <strong class="donut-stat-val num">${fmtNum(retry)}</strong>
      </div>
      <div class="donut-stat-cell">
        <span class="donut-stat-lbl"><i class="dot" style="background:var(--danger)"></i>异常</span>
        <strong class="donut-stat-val num">${fmtNum(err)}</strong>
      </div>
    `;

    // 7-Day Smooth Bezier Sparkline SVG (natural 320x64 aspect ratio)
    const spArr = Array.isArray(d.sparkline_7d) && d.sparkline_7d.length === 7
      ? d.sparkline_7d
      : [0, 0, 0, 0, 0, 0, 0].map((v, i) => ({ label: `D-${6 - i}`, total: v }));
    const pts = spArr.map((x) => Number(x.total || 0));
    const peak = Math.max(...pts);
    const padX = 12, w = 320, padTop = 9, padBot = 8, h = 64;

    let coords;
    if (peak <= 0) {
      coords = pts.map((_, i) => [padX + (i / (pts.length - 1)) * (w - padX * 2), h - padBot - 4]);
    } else {
      const minP = Math.min(...pts) * 0.72;
      const maxP = peak * 1.12;
      const spanP = Math.max(1, maxP - minP);
      coords = pts.map((v, i) => [
        padX + (i / (pts.length - 1)) * (w - padX * 2),
        (h - padBot) - ((v - minP) / spanP) * (h - padTop - padBot)
      ]);
    }

    function smoothPath(p) {
      let dStr = `M${p[0][0].toFixed(1)},${p[0][1].toFixed(1)}`;
      for (let i = 0; i < p.length - 1; i++) {
        const p0 = p[Math.max(0, i - 1)];
        const p1 = p[i];
        const p2 = p[i + 1];
        const p3 = p[Math.min(p.length - 1, i + 2)];
        const cp1x = p1[0] + (p2[0] - p0[0]) / 6;
        const cp1y = Math.min(h - padBot, Math.max(padTop, p1[1] + (p2[1] - p0[1]) / 6));
        const cp2x = p2[0] - (p3[0] - p1[0]) / 6;
        const cp2y = Math.min(h - padBot, Math.max(padTop, p2[1] - (p3[1] - p1[1]) / 6));
        dStr += ` C${cp1x.toFixed(1)},${cp1y.toFixed(1)} ${cp2x.toFixed(1)},${cp2y.toFixed(1)} ${p2[0].toFixed(1)},${p2[1].toFixed(1)}`;
      }
      return dStr;
    }

    const linePath = smoothPath(coords);
    const areaPath = `${linePath} L${coords[coords.length - 1][0].toFixed(1)},${h - 2} L${coords[0][0].toFixed(1)},${h - 2} Z`;
    $("sparkSummary").textContent = `峰值 ${fmtNum(peak)} / 日`;
    if (currentTheme === "vinyl-night") {
      // Hi-Fi Analog Equalizer LED Brick Columns
      const colW = 24, gapX = (w - padX * 2 - colW * pts.length) / (pts.length - 1);
      const maxBricks = 9, brickH = 4.2, brickGap = 1.8;
      let eqSvg = `<line x1="${padX}" y1="${h - 4}" x2="${w - padX}" y2="${h - 4}" stroke="rgba(229,169,60,0.24)" stroke-width="1"/>`;
      pts.forEach((v, i) => {
        const bx = padX + i * (colW + gapX);
        const lit = peak > 0 ? Math.max(1, Math.round((v / peak) * maxBricks)) : 0;
        const title = `${escHtml(spArr[i].label || "")}：${fmtNum(v)} 次请求`;
        eqSvg += `<g><title>${title}</title>`;
        for (let b = 0; b < maxBricks; b++) {
          const by = h - 6 - (b + 1) * (brickH + brickGap);
          const isLit = b < lit;
          const fill = !isLit ? "rgba(229,169,60,0.08)" : (b >= maxBricks - 2 ? "#de6b53" : (i === pts.length - 1 ? "#f4c15d" : "#e5a93c"));
          eqSvg += `<rect x="${bx.toFixed(1)}" y="${by.toFixed(1)}" width="${colW}" height="${brickH}" rx="1.2" fill="${fill}"/>`;
        }
        eqSvg += `</g>`;
      });
      $("sparkSvg").innerHTML = eqSvg;
    } else if (currentTheme === "pop-day") {
      // Comic Hard-Outline Bar Chart with Offset Shadow
      const barW = 22, gapX = (w - padX * 2 - barW * pts.length) / (pts.length - 1);
      let popSvg = `<line x1="${padX}" y1="${h - 5}" x2="${w - padX}" y2="${h - 5}" stroke="#1c192b" stroke-width="2"/>`;
      pts.forEach((v, i) => {
        const bx = padX + i * (barW + gapX);
        const bh = peak > 0 ? Math.max(9, (v / peak) * (h - 16)) : 7;
        const by = h - 5 - bh;
        const isLast = i === pts.length - 1;
        const fill = isLast ? "#ffd43b" : (i % 2 === 0 ? "#2d5cf2" : "#12b874");
        const title = `${escHtml(spArr[i].label || "")}：${fmtNum(v)} 次请求`;
        popSvg += `<g><title>${title}</title>
          <rect x="${(bx + 2.5).toFixed(1)}" y="${(by + 2).toFixed(1)}" width="${barW}" height="${bh.toFixed(1)}" rx="4" fill="#1c192b"/>
          <rect x="${bx.toFixed(1)}" y="${by.toFixed(1)}" width="${barW}" height="${bh.toFixed(1)}" rx="4" fill="${fill}" stroke="#1c192b" stroke-width="2"/>
        </g>`;
      });
      $("sparkSvg").innerHTML = popSvg;
    } else if (currentTheme === "paper-day" || currentTheme === "lacquer-night") {
      // Bamboo-Slip / Kinpaku Scroll Vertical Pillars
      const barW = 14, gapX = (w - padX * 2 - barW * pts.length) / (pts.length - 1);
      const isLacq = currentTheme === "lacquer-night";
      const capCol = isLacq ? "#cca86e" : "#b84824";
      let scSvg = `<defs><linearGradient id="slipGrad" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="var(--primary)" stop-opacity="0.88"/><stop offset="100%" stop-color="var(--primary)" stop-opacity="0.22"/></linearGradient></defs>
        <line x1="${padX}" y1="${h - 5}" x2="${w - padX}" y2="${h - 5}" stroke="var(--border-strong)" stroke-width="1"/>`;
      pts.forEach((v, i) => {
        const bx = padX + i * (barW + gapX);
        const bh = peak > 0 ? Math.max(8, (v / peak) * (h - 15)) : 6;
        const by = h - 5 - bh;
        const title = `${escHtml(spArr[i].label || "")}：${fmtNum(v)} 次请求`;
        scSvg += `<g><title>${title}</title>
          <rect x="${bx.toFixed(1)}" y="${by.toFixed(1)}" width="${barW}" height="${bh.toFixed(1)}" rx="2.5" fill="url(#slipGrad)"/>
          <rect x="${bx.toFixed(1)}" y="${by.toFixed(1)}" width="${barW}" height="3" rx="1.5" fill="${i === pts.length - 1 ? capCol : "var(--accent)"}"/>
        </g>`;
      });
      $("sparkSvg").innerHTML = scSvg;
    } else if (currentTheme === "star-night") {
      // Celestial Constellation Star-Chart Polyline
      const polyPts = coords.map((p) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ");
      $("sparkSvg").innerHTML = `
        <line x1="${padX}" y1="${(h / 2).toFixed(1)}" x2="${w - padX}" y2="${(h / 2).toFixed(1)}" stroke="rgba(232,209,160,0.14)" stroke-width="1" stroke-dasharray="2 5"/>
        <polyline points="${polyPts}" fill="none" stroke="#e8d1a0" stroke-width="1.4" stroke-dasharray="3 3" opacity="0.72"/>
        ${coords.map((p, i) => {
          const isLast = i === coords.length - 1;
          const title = `${escHtml(spArr[i].label || "")}：${fmtNum(pts[i])} 次请求`;
          const col = i % 2 ? "#7cc4fa" : "#e8d1a0";
          return isLast
            ? `<circle cx="${p[0].toFixed(1)}" cy="${p[1].toFixed(1)}" r="8" fill="rgba(232,209,160,0.22)"/>
               <path d="M${p[0].toFixed(1)} ${(p[1] - 7).toFixed(1)}v14M${(p[0] - 7).toFixed(1)} ${p[1].toFixed(1)}h14" stroke="#e8d1a0" stroke-width="1.1"/>
               <circle cx="${p[0].toFixed(1)}" cy="${p[1].toFixed(1)}" r="3.2" fill="#fff7e0"><title>${title}</title></circle>`
            : `<circle cx="${p[0].toFixed(1)}" cy="${p[1].toFixed(1)}" r="5" fill="${col}" opacity="0.16"/>
               <circle cx="${p[0].toFixed(1)}" cy="${p[1].toFixed(1)}" r="2.5" fill="${col}"><title>${title}</title></circle>`;
        }).join("")}
      `;
    } else {
      // Sky-day: Glossy Soda Wave + Specular Bubble Nodes
      $("sparkSvg").innerHTML = `
        <defs>
          <linearGradient id="sparkGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="#009fe3" stop-opacity="0.32"/>
            <stop offset="100%" stop-color="#0db896" stop-opacity="0.02"/>
          </linearGradient>
        </defs>
        <line x1="${padX}" y1="${(h / 2).toFixed(1)}" x2="${w - padX}" y2="${(h / 2).toFixed(1)}" stroke="var(--border)" stroke-width="1" stroke-dasharray="3 4"/>
        <path d="${areaPath}" fill="url(#sparkGrad)"/>
        <path d="${linePath}" fill="none" stroke="var(--primary)" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>
        ${coords.map((p, i) => {
          const isLast = i === coords.length - 1;
          const title = `${escHtml(spArr[i].label || "")}：${fmtNum(pts[i])} 次请求`;
          return isLast
            ? `<circle cx="${p[0].toFixed(1)}" cy="${p[1].toFixed(1)}" r="6.5" fill="rgba(0,159,227,0.22)"/>
               <circle cx="${p[0].toFixed(1)}" cy="${p[1].toFixed(1)}" r="3.8" fill="#0077cc" stroke="#ffffff" stroke-width="1.6"><title>${title}</title></circle>
               <circle cx="${(p[0] - 1.2).toFixed(1)}" cy="${(p[1] - 1.2).toFixed(1)}" r="1" fill="#ffffff"/>`
            : `<circle cx="${p[0].toFixed(1)}" cy="${p[1].toFixed(1)}" r="3.2" fill="#e6f7ff" stroke="#0077cc" stroke-width="1.6"><title>${title}</title></circle>`;
        }).join("")}
      `;
    }

    // Token & Implicit Prefix Cache Radar
    const promptTk = Number(d.prompt_tokens || 0);
    const cachedTk = Number(d.cached_tokens || 0);
    const thoughtTk = Number(d.thought_tokens || 0);
    const outputTk = Number(d.output_tokens != null ? d.output_tokens : Math.max(0, Number(d.completion_tokens || 0) - thoughtTk));
    const grandTotal = promptTk + thoughtTk + outputTk;
    const cacheRate = promptTk > 0 ? ((cachedTk / promptTk) * 100).toFixed(1) : "0.0";

    $("cacheHitBadge").innerHTML = `
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
      隐式缓存命中 ${cacheRate}%
    `;
    $("tokenGrandTotal").textContent = fmtTok(grandTotal);
    $("tokenCacheSaveHint").textContent = grandTotal > 0
      ? (cachedTk > 0 ? `隐式前缀缓存已命中 ${fmtTok(cachedTk)} 输入算力（占总输入 ${cacheRate}%）` : "当前周期暂未命中隐式前缀缓存")
      : "暂无算力消耗记录";

    const meters = [
      { name: "输入 Prompt 总量", val: fmtTok(promptTk), pct: grandTotal > 0 ? (promptTk / grandTotal) * 100 : 0, color: "var(--primary)", extra: grandTotal > 0 ? `占比 ${((promptTk / grandTotal) * 100).toFixed(1)}%` : "0%" },
      { name: "隐式前缀缓存命中", val: fmtTok(cachedTk), pct: promptTk > 0 ? (cachedTk / promptTk) * 100 : 0, color: "linear-gradient(90deg, var(--cache), var(--cache-glow))", extra: `命中输入 ${cacheRate}%`, badge: true },
      { name: "思考消耗（含隐性思考）", val: fmtTok(thoughtTk), pct: grandTotal > 0 ? (thoughtTk / grandTotal) * 100 : 0, color: "var(--think)", extra: grandTotal > 0 ? `占比 ${((thoughtTk / grandTotal) * 100).toFixed(1)}%` : "0%" },
      { name: "可见正文输出", val: fmtTok(outputTk), pct: grandTotal > 0 ? (outputTk / grandTotal) * 100 : 0, color: "var(--accent)", extra: grandTotal > 0 ? `占比 ${((outputTk / grandTotal) * 100).toFixed(1)}%` : "0%" }
    ];
    $("tokenRadarList").innerHTML = meters.map((m) => `
      <div class="meter-item">
        <div class="meter-top">
          <span class="meter-name">
            <i class="dot" style="background:${m.badge ? "var(--cache)" : m.color}"></i>
            ${m.name}
          </span>
          <span class="meter-val num">
            <span>${m.val}</span>
            <span style="font-size:11.5px;font-weight:550;color:var(--text-muted);">${m.extra}</span>
          </span>
        </div>
        <div class="meter-track">
          <div class="meter-fill" style="width:${m.pct > 0 ? Math.max(3, m.pct).toFixed(1) : "0"}%;background:${m.color};"></div>
        </div>
      </div>
    `).join("");

    renderModelStats(d.models || []);
  }

  function renderModelStats(rawList) {
    const list = Array.isArray(rawList) ? rawList.slice() : [];
    const box = $("modelStatsList");
    const expandWrap = $("modelExpandWrap");
    if (!list.length) {
      box.innerHTML = `<div class="empty-state">当前时间范围内暂无模型调用记录。发起请求后，此处将自动展示各模型调用量、成功率、Token 与缓存占比及平均速度。</div>`;
      expandWrap.innerHTML = "";
      return;
    }
    list.sort((a, b) => {
      if (modelSort === "calls") return (b.calls - a.calls) || (b.total_tk - a.total_tk);
      return (b.total_tk - a.total_tk) || (b.calls - a.calls);
    });
    const visible = showAllModels ? list : list.slice(0, 6);
    box.innerHTML = visible.map((m) => {
      const pTk = Number(m.prompt_tk || 0);
      const cTk = Number(m.cached_tk || 0);
      const thTk = Number(m.thought_tk || 0);
      const oTk = Number(m.output_tk || 0);
      const total = Math.max(0, Number(m.total_tk || (pTk + thTk + oTk)));
      const rateNum = m.calls > 0 ? (m.ok / m.calls) * 100 : 0;
      const rate = rateNum.toFixed(1);
      const rateCls = rateNum >= 97 ? "success" : (rateNum >= 90 ? "warning" : "danger");
      const inPct = total > 0 ? (pTk / total) * 100 : 0;
      const cacheInPromptPct = pTk > 0 ? (cTk / pTk) * 100 : 0;
      const thinkPct = total > 0 ? (thTk / total) * 100 : 0;
      const outPct = total > 0 ? (oTk / total) * 100 : 0;
      const ttftTxt = m.avg_ttft != null ? `${Number(m.avg_ttft).toFixed(2)}s` : "—";
      const tpsTxt = m.avg_tps != null ? `${Number(m.avg_tps).toFixed(1)} tok/s` : "—";
      return `
        <div class="model-row">
          <div class="model-id">
            <div class="model-name mono" title="${escHtml(m.model)}">${escHtml(m.model)}</div>
            <div class="model-sub">${escHtml(m.channel || "Express")} 通道 · 缓存 ${cacheInPromptPct.toFixed(0)}%</div>
          </div>
          <div class="model-calls num">
            <strong>${fmtNum(m.calls)} 次</strong>
            <span class="pill ${rateCls}">${rate}%</span>
          </div>
          <div class="stacked-bar-wrap">
            <div class="stacked-bar" title="输入 ${fmtTok(pTk)} (含缓存 ${fmtTok(cTk)}) / 思考 ${fmtTok(thTk)} / 正文 ${fmtTok(oTk)}">
              <div class="seg-in" style="width:${inPct.toFixed(1)}%">
                <div class="seg-cache-overlay" style="width:${cacheInPromptPct.toFixed(1)}%"></div>
              </div>
              <div class="seg-think" style="width:${thinkPct.toFixed(1)}%"></div>
              <div class="seg-out" style="width:${outPct.toFixed(1)}%"></div>
            </div>
            <div class="stacked-caption num">
              <span>入 ${fmtTok(pTk)}</span>
              <span style="color:var(--primary-text);">缓 ${fmtTok(cTk)}</span>
              <span>思 ${fmtTok(thTk)}</span>
              <span>出 ${fmtTok(oTk)}</span>
            </div>
          </div>
          <div class="model-perf num">
            <span class="perf-chip">首字 <strong>${ttftTxt}</strong></span>
            <span class="perf-chip">速度 <strong>${tpsTxt}</strong></span>
          </div>
          <div class="model-total num">${total > 0 ? fmtTok(total) : "—"}</div>
        </div>
      `;
    }).join("");

    if (list.length > 6) {
      expandWrap.innerHTML = `<button class="btn btn-ghost btn-sm" onclick="toggleShowAllModels()">${showAllModels ? "收起至前 6 项" : `展开全部 ${list.length} 个已调用模型`}</button>`;
    } else {
      expandWrap.innerHTML = "";
    }
  }

  /* ── Real Live Stream X-Ray Polling & Abort (with 30s post-finish linger) ── */
  function updateAvatarMood(activeList) {
    const list = Array.isArray(activeList) ? activeList : [];
    const hasWarnOrErr = list.some((it) => (Number(it.retries || 0) > 0) || it.status === "error" || it.status === "trunc");
    const hasRunning = list.some((it) => !it.finished);
    let mood = "idle";
    let label = "429 娘 · 待命守护中";
    if (hasWarnOrErr) {
      mood = "warn";
      label = "429 娘 · 检测到重试或异常波动";
    } else if (hasRunning) {
      mood = "streaming";
      label = "429 娘 · 正在实时流式推理中";
    }
    if (document.body) {
      document.body.setAttribute("data-avatar-mood", mood);
    }
    document.querySelectorAll(".avatar-ring").forEach((el) => {
      el.setAttribute("title", label);
    });
  }

  async function fetchLive() {
    try {
      const r = await apiFetch("/api/live");
      if (!r.ok) return;
      const d = await r.json();
      const active = Array.isArray(d.active) ? d.active : [];
      updateAvatarMood(active);
      renderLivePulse(active);
      const runningNow = active.filter((it) => !it.finished).length;
      if (lastLiveCount > 0 && runningNow === 0) {
        // 请求刚结束（此时脉冲进入 30s 停留展示态），立刻刷新统计、记录与 Key 命中数
        fetchStats();
        fetchRecords();
        loadExpressKeysQuiet();
        loadRuntime();
      }
      lastLiveCount = runningNow;
    } catch (e) {}
  }

  function renderLivePulse(active) {
    const wrap = $("pulseWrapper");
    const stack = $("pulseStack");
    if (!wrap || !stack) return;
    if (!active.length) {
      wrap.classList.remove("open");
      return;
    }
    stack.innerHTML = active.map((it) => {
      const isFin = Boolean(it.finished);
      const st = it.status || (isFin ? "ok" : "running");
      const ph = it.phase || "connecting";

      let s0Cls, s1Cls, s2Cls, s3Cls;
      if (isFin) {
        s0Cls = "done";
        s1Cls = it.think_sec != null ? "done" : "";
        s2Cls = (st === "ok" || st === "trunc" || it.chars > 0) ? "done" : "";
        s3Cls = st === "error" ? "err" : (st === "trunc" || st === "cancelled" ? "warn" : "done");
      } else {
        s0Cls = ph === "connecting" || ph === "retrying" ? "active" : "done";
        s1Cls = ph === "thinking" ? "active" : (ph === "streaming" || ph === "finishing" ? "done" : "");
        s2Cls = ph === "streaming" ? "active" : (ph === "finishing" ? "done" : "");
        s3Cls = ph === "finishing" ? "active" : "";
      }

      const s0Txt = (!isFin && ph === "retrying")
        ? escHtml(it.note || `重试第 ${it.retries} 次`)
        : (it.conn_sec != null ? `${Number(it.conn_sec).toFixed(2)}s 完成` : (isFin ? "已完成握手" : `连接中 · ${escHtml(it.cred_label)}`));
      const s1Txt = it.think_sec != null
        ? `${(!isFin && ph === "thinking") ? "推理中" : "完成"} ${Number(it.think_sec).toFixed(1)}s`
        : (isFin || ph === "streaming" ? "无显式思考" : "等待中");
      const s2Txt = it.chars > 0
        ? `${fmtNum(it.chars)} 字${it.tps_est ? ` · ≈${it.tps_est} tok/s` : ""}`
        : (isFin ? (st === "error" ? "未产出正文" : "已结束") : (it.is_stream ? "等待正文首字" : "整包生成中"));

      let s3Title = "4. 通道守护";
      let s3Txt = it.sandbox_active ? "compose_reply 尾窗守护" : "原生直出";
      if (isFin) {
        s3Title = "4. 完成状态";
        const frTxt = escHtml(it.finish_reason || (st === "ok" ? "STOP" : st.toUpperCase()));
        if (st === "error") {
          s3Txt = it.note ? `${frTxt} · ${escHtml(it.note)}` : frTxt;
        } else if (st === "cancelled") {
          s3Txt = escHtml(it.note || "请求已中止");
        } else {
          const extras = [];
          if (it.cache_pct != null && Number(it.cache_pct) > 0) extras.push(`缓存 ${Number(it.cache_pct).toFixed(0)}%`);
          else if (it.sandbox_active) extras.push("尾窗守护");
          s3Txt = extras.length ? `${frTxt} · ${extras.join(" · ")}` : frTxt;
        }
      }

      const credText = it.project_id ? `${it.cred_label} · ${it.project_id}` : it.cred_label;
      let cardCls = "pulse-card";
      let headTitle = it.is_stream ? "正在处理流式请求" : "正在处理非流式请求";
      let statusBadge = it.sandbox_active ? `<span class="pill success">抗截断守护中</span>` : "";
      if (isFin) {
        cardCls += " finished";
        if (st === "error") {
          cardCls += " finished-err";
          headTitle = "请求失败";
          statusBadge = `<span class="pill danger">${escHtml(it.finish_reason || "失败")}</span>`;
        } else if (st === "cancelled") {
          cardCls += " finished-warn";
          headTitle = "请求已中止";
          statusBadge = `<span class="pill warning">已中止</span>`;
        } else if (st === "trunc") {
          cardCls += " finished-warn";
          headTitle = "请求已结束";
          statusBadge = `<span class="pill warning">${escHtml(it.finish_reason || "达上限/截断")}</span>`;
        } else {
          headTitle = "请求已完成";
          statusBadge = `<span class="pill success">${escHtml(it.finish_reason || "STOP")}</span>`;
        }
      }

      const rightCtrl = isFin
        ? `<span class="num" style="font-size:12.5px;color:var(--text-secondary);">总耗时 ${Number(it.elapsed_sec || 0).toFixed(1)}s</span>
           <span class="pill num" title="30 秒内若有新请求会立即切换，无新请求将自动收起">将在 ${Number(it.linger_remain || 30)}s 后收起</span>`
        : `<span class="num" style="font-size:12.5px;color:var(--text-secondary);">已耗时 ${Number(it.elapsed_sec || 0).toFixed(1)}s</span>
           <button class="btn btn-danger-soft btn-sm" ${it.aborted ? "disabled" : ""} onclick="abortLiveReq('${escHtml(it.req_id)}')">
             <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>
             ${it.aborted ? "中止中…" : "中止请求"}
           </button>`;

      return `
        <div class="${cardCls}">
          <div class="pulse-top">
            <div class="pulse-id">
              <span class="live-dot"></span>
              <span>${headTitle}</span>
              <span class="pill primary mono">${escHtml(it.model)}</span>
              <span class="pill">${escHtml(credText)}</span>
              ${statusBadge}
              ${it.retries > 0 ? `<span class="pill warning">重试 ×${it.retries}</span>` : ""}
            </div>
            <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
              ${rightCtrl}
            </div>
          </div>
          <div class="pulse-steps">
            <div class="pulse-step ${s0Cls}">
              <div class="pulse-step-title">1. 选钥建连</div>
              <div class="pulse-step-meta num">${s0Txt}</div>
            </div>
            <div class="pulse-step ${s1Cls}">
              <div class="pulse-step-title">2. 深度思考</div>
              <div class="pulse-step-meta num">${s1Txt}</div>
            </div>
            <div class="pulse-step ${s2Cls}">
              <div class="pulse-step-title">3. 正文吐字</div>
              <div class="pulse-step-meta num">${s2Txt}</div>
            </div>
            <div class="pulse-step ${s3Cls}">
              <div class="pulse-step-title">${s3Title}</div>
              <div class="pulse-step-meta num">${s3Txt}</div>
            </div>
          </div>
        </div>
      `;
    }).join("");
    wrap.classList.add("open");
  }

  window.abortLiveReq = async function (reqId) {
    try {
      const r = await apiFetch("/api/live/" + encodeURIComponent(reqId) + "/abort", { method: "POST" });
      if (r.ok) showToast("已下发中止信号，正在释放上游连接");
      else showToast("该请求已结束或无需中止");
      fetchLive();
    } catch (e) {
      showToast("发送中止信号失败");
    }
  };

  /* =====================================================================
     3. CHANNEL, EXPRESS KEY POOL & COOKIE
     ===================================================================== */
  let KEYS = [];
  let activeMode = "api_key";
  let cookieConfigured = false;
  let cookieProjectId = "";

  const SUPPORTS_TEXT_SECURITY = Boolean(
    typeof CSS !== "undefined" &&
    CSS.supports &&
    (CSS.supports("-webkit-text-security", "disc") || CSS.supports("text-security", "disc"))
  );

  function maskKeyVisual(key) {
    const s = String(key || "");
    if (!s) return "";
    if (s.length <= 8) return "•".repeat(s.length);
    return s.slice(0, 4) + "••••••••••••" + s.slice(-4);
  }

  let keyRowUidSeq = 0;
  let keysMutationSeq = 0;
  let keysSaveInFlight = 0;
  let keysSaveChain = Promise.resolve();
  let keysProbeRefreshTimer = null;

  function ensureKeyUid(rec) {
    if (!rec._uid) rec._uid = ++keyRowUidSeq;
    return rec._uid;
  }

  function findKeyIndexByUid(uid) {
    return KEYS.findIndex((k) => k._uid === uid);
  }

  function findKeyByUid(uid) {
    const idx = findKeyIndexByUid(uid);
    return idx >= 0 ? KEYS[idx] : null;
  }

  window.setChannelSub = function (sub) {
    const isTest = sub === "test";
    $("subChanConfigBtn").classList.toggle("active", !isTest);
    $("subChanTestBtn").classList.toggle("active", isTest);
    $("channelSubConfig").style.display = isTest ? "none" : "block";
    $("channelSubTest").style.display = isTest ? "block" : "none";
    if (isTest) {
      syncLabTargetOptions();
      scheduleLabInspect();
    }
  };

  function applyChannelHighlight(mode) {
    activeMode = mode;
    const isWeb = mode === "web_proxy";
    $("chExpress").classList.toggle("active", !isWeb);
    $("chCookie").classList.toggle("active", isWeb);
    $("badgeExpressActive").textContent = isWeb ? "热备通道" : "当前主用";
    $("badgeExpressActive").className = isWeb ? "pill" : "pill primary";
    $("badgeCookieActive").textContent = isWeb ? "当前主用" : "热备通道";
    $("badgeCookieActive").className = isWeb ? "pill primary" : "pill";
    $("sidebarChannelStatus").textContent = isWeb ? "在线 · Cookie 主用" : "在线 · Express 主用";
  }

  window.selectChannel = async function (mode) {
    applyChannelHighlight(mode);
    try {
      const r = await apiFetch("/api/settings/mode", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode })
      });
      if (r.ok) {
        showToast(mode === "web_proxy" ? "已切换主通道为 Studio Cookie 直连" : "已切换主通道为 Express API Key");
        scheduleLabInspect();
      } else {
        showToast("切换通道失败");
      }
    } catch (e) {
      showToast("网络请求失败");
    }
  };

  async function loadRuntime() {
    try {
      const r = await apiFetch("/api/settings/runtime");
      if (!r.ok) return;
      const s = await r.json();
      cookieConfigured = Boolean(s.google_cookie_configured);
      cookieProjectId = s.google_project_id || "";
      applyChannelHighlight(s.use_web_proxy ? "web_proxy" : "api_key");
      const ckHits = Number(s.cookie_hits || 0);
      $("cookieHitsPill").innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>命中 ${fmtNum(ckHits)} 次`;
      $("cookieCardHitsPill").textContent = `Cookie 累计命中 ${fmtNum(ckHits)} 次`;
      const ci = $("cookieInput");
      if (ci && !ci.value) {
        ci.placeholder = s.google_cookie_configured
          ? `已配置：${s.google_cookie}（留空则保持不变，需更新时粘贴新 Cookie）`
          : "粘贴 console.cloud.google.com 的完整 Cookie（支持 Header 字符串或 Cookie-Editor JSON）";
      }
      const sp = $("cookieStatePill");
      if (sp) {
        sp.textContent = s.google_cookie_configured ? "已配置" : "未配置";
        sp.className = s.google_cookie_configured ? "pill success" : "pill";
      }
      if (s.google_project_id && $("cookieProj") && !$("cookieProj").value) {
        $("cookieProj").value = s.google_project_id;
      }
      syncLabTargetOptions();
    } catch (e) {}
  }

  function parseCookies(str) {
    const s = (str || "").trim();
    if (s.startsWith("[") && s.endsWith("]")) {
      try {
        const arr = JSON.parse(s);
        if (Array.isArray(arr)) {
          return arr.map((c) => {
            const n = c.name || c.key, v = c.value;
            return (n && v) ? `${n}=${v}` : "";
          }).filter(Boolean).join("; ");
        }
      } catch (e) {}
    }
    return s;
  }

  function extractProjectIdFromInput(raw) {
    const pid = (raw || "").trim();
    const m = pid.match(/[?&]project=([^&]+)/) || pid.match(/\/projects\/([^/]+)/);
    return m ? m[1] : pid;
  }

  if ($("cookieProj")) {
    $("cookieProj").addEventListener("input", (e) => {
      const cleaned = extractProjectIdFromInput(e.target.value);
      if (cleaned !== e.target.value.trim()) e.target.value = cleaned;
    });
  }

  window.saveCookie = async function () {
    const ck = parseCookies($("cookieInput").value);
    const pid = extractProjectIdFromInput($("cookieProj").value);
    $("cookieInput").value = ck;
    $("cookieProj").value = pid;
    if (!pid) { showToast("请填写 Google Cloud Project ID"); return; }
    try {
      const r = await apiFetch("/api/cookie", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cookie: ck, project_id: pid })
      });
      const d = await r.json().catch(() => ({}));
      if (r.ok) {
        $("cookieInput").value = "";
        showToast(d.message || "Studio Cookie 凭证已保存");
        loadRuntime();
        scheduleLabInspect();
      } else {
        showToast(d.error || "保存 Cookie 失败");
      }
    } catch (e) {
      showToast("网络请求失败");
    }
  };

  function keySrcPill(rec) {
    if (rec.project_source === "auto") return `<span class="pill success">自动探测</span>`;
    if (rec.project_source === "manual") return `<span class="pill primary">手动指定</span>`;
    return `<span class="pill warning">待探测</span>`;
  }

  function renderKeys() {
    const validKeys = KEYS.filter((k) => k.key);
    const enabledCount = validKeys.filter((k) => k.enabled !== false).length;
    const totalHits = validKeys.reduce((s, k) => s + Number(k.hits || 0), 0);
    $("navKeyBadge").textContent = String(enabledCount);
    $("keyPoolSub").textContent = `已启用 ${enabledCount} / ${validKeys.length} 把 · 命中次数按指纹长久留存，删改调序不串号`;
    $("expressTotalHitsPill").textContent = `池内累计命中 ${fmtNum(totalHits)} 次`;

    const box = $("keyPoolList");
    if (!KEYS.length) {
      box.innerHTML = `<div class="empty-state">当前没有任何 Express API Key。点击右上角「添加 Key」粘贴即可自动保存并探测归属 Project ID。</div>`;
      syncLabTargetOptions();
      return;
    }

    box.innerHTML = KEYS.map((rec, i) => {
      const uid = ensureKeyUid(rec);
      const enabled = rec.enabled !== false;
      const detStr = rec.detected_at ? `探测于 ${new Date(rec.detected_at * 1000).toLocaleString()}` : "";
      const hits = Number(rec.hits || 0);
      const fpHint = rec.key_fp ? `SHA-256 指纹 ${rec.key_fp}…（SQLite 持久化计数）` : "尚未保存";
      const msgHtml = rec._msg
        ? `<div class="key-probe-msg" style="color:${rec._msg_ok === false ? "var(--danger-text)" : "var(--success-text)"};">${escHtml(rec._msg)}</div>`
        : "";

      const isHidden = !rec._show && Boolean(rec.key);
      const useCssMask = isHidden && SUPPORTS_TEXT_SECURITY;
      const useFallbackMask = isHidden && !SUPPORTS_TEXT_SECURITY;
      const keyDisplayVal = useFallbackMask ? maskKeyVisual(rec.key) : (rec.key || "");
      const antiFillAttrs = `autocomplete="off" autocorrect="off" autocapitalize="off" spellcheck="false" data-lpignore="true" data-1p-ignore="true" data-bwignore="true" data-form-type="other"`;

      return `
        <div class="key-row ${enabled ? "" : "disabled"}" data-key-uid="${uid}">
          <div class="key-row-top">
            <div class="key-main-group">
              <span class="pill primary num" style="min-width:34px;justify-content:center;">#${i + 1}</span>
              <div class="key-input-wrap">
                <input class="field-input k-key${useCssMask ? " k-key-masked" : ""}" type="text" name="ap2a_ek_${uid}" ${antiFillAttrs} value="${escHtml(keyDisplayVal)}"${useFallbackMask ? ` readonly data-masked="1" onclick="revealAndEditKey(${uid})" title="点击明文显示并编辑此 Key"` : ""} placeholder="粘贴 Express API Key（移开焦点或回车自动保存）" onchange="onKeyFieldChange(${uid})" onkeydown="if(event.key==='Enter'&&!event.isComposing&&event.keyCode!==229){event.preventDefault();this.blur();}" />
                <button class="key-eye" type="button" onclick="toggleKeyVis(${uid})" aria-label="显示或隐藏 Key">
                  <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
                </button>
              </div>
            </div>
            <div class="key-actions-group">
              <label class="switch" title="启用或停用此 Key">
                <input type="checkbox" class="k-enabled" ${enabled ? "checked" : ""} onchange="onKeyFieldChange(${uid})" />
                <span class="switch-track"></span>
              </label>
              <button class="btn btn-ghost btn-sm" type="button" onclick="probeOneKey(${uid})">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
                探测
              </button>
              <button class="btn btn-ghost btn-sm" type="button" onclick="testKeyDirect(${uid})" title="跳转到连通实测并锁定此 Key 单发验活">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>
                实测
              </button>
              <button class="btn-icon" type="button" onclick="removeKeyRow(${uid})" title="删除此凭证" style="color:var(--danger-text);">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6m4-6v6"/><path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/></svg>
              </button>
            </div>
          </div>
          <div class="key-row-bottom">
            <div class="key-main-group">
              <span style="font-size:12px;color:var(--text-muted);font-weight:600;white-space:nowrap;">Project ID</span>
              <input class="field-input mono k-project" type="text" name="ap2a_ep_${uid}" ${antiFillAttrs} style="padding:6px 10px;font-size:12.5px;" value="${escHtml(rec.project_id || "")}" placeholder="留空 = 自动探测获取" onchange="onKeyFieldChange(${uid})" onkeydown="if(event.key==='Enter'&&!event.isComposing&&event.keyCode!==229){event.preventDefault();this.blur();}" />
            </div>
            <div class="key-actions-group">
              ${keySrcPill(rec)}
              ${detStr ? `<span style="font-size:11px;color:var(--text-muted);">${escHtml(detStr)}</span>` : ""}
              <span class="hit-pill num" title="${escHtml(fpHint)}">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
                命中 ${fmtNum(hits)} 次
              </span>
            </div>
          </div>
          ${msgHtml}
        </div>
      `;
    }).join("");
    syncLabTargetOptions();
  }

  function syncKeysFromDom() {
    document.querySelectorAll("#keyPoolList [data-key-uid]").forEach((row) => {
      const uid = Number(row.dataset.keyUid);
      const rec = findKeyByUid(uid);
      if (!rec) return;
      const keyInp = row.querySelector(".k-key");
      if (keyInp && keyInp.dataset.masked !== "1") {
        rec.key = keyInp.value.trim();
      }
      const projInp = row.querySelector(".k-project");
      if (projInp) {
        const pid = projInp.value.trim();
        if (pid !== (rec.project_id || "")) {
          rec.project_id = pid;
          rec.project_source = pid ? "manual" : "";
        }
      }
      const enInp = row.querySelector(".k-enabled");
      if (enInp) {
        rec.enabled = enInp.checked;
      }
    });
  }

  function cleanKeyPayload(list) {
    return list.filter((r) => r.key).map((r) => ({
      key: r.key,
      project_id: r.project_id || "",
      project_source: r.project_source || "",
      detected_at: r.detected_at || 0,
      note: r.note || "",
      enabled: r.enabled !== false
    }));
  }

  function mergeServerKeys(serverKeys, emptyRows) {
    const uiMap = {};
    KEYS.forEach((k) => {
      if (k.key) {
        uiMap[k.key] = {
          _uid: k._uid,
          _show: Boolean(k._show),
          _msg: k._msg,
          _msg_ok: k._msg_ok
        };
      }
    });
    const merged = (serverKeys || []).map((x) => {
      const item = Object.assign({}, x);
      const prev = uiMap[item.key];
      if (prev) {
        if (prev._uid) item._uid = prev._uid;
        item._show = prev._show;
        if (prev._msg) {
          item._msg = prev._msg;
          item._msg_ok = prev._msg_ok;
        }
      } else {
        item._show = false;
      }
      ensureKeyUid(item);
      return item;
    });
    (emptyRows || []).forEach((r) => ensureKeyUid(r));
    KEYS = merged.concat(emptyRows || []);
    renderKeys();
  }

  async function autoSaveExpressKeys(customMsg, opts) {
    if (!(opts && opts.skipSync)) syncKeysFromDom();
    const payload = cleanKeyPayload(KEYS);
    const mySeq = ++keysMutationSeq;
    if (keysProbeRefreshTimer) {
      clearTimeout(keysProbeRefreshTimer);
      keysProbeRefreshTimer = null;
    }
    const run = async () => {
      keysSaveInFlight++;
      try {
        const r = await apiFetch("/api/express-keys", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ keys: payload })
        });
        const d = await r.json().catch(() => ({}));
        if (r.ok) {
          if (mySeq === keysMutationSeq) {
            const emptyRows = KEYS.filter((item) => !item.key);
            mergeServerKeys(d.keys || [], emptyRows);
            showToast(customMsg || `已自动保存（共 ${payload.length} 把有效 Key）`);
            scheduleLabInspect();
            if (payload.some((k) => !k.project_id)) {
              keysProbeRefreshTimer = setTimeout(() => {
                keysProbeRefreshTimer = null;
                loadExpressKeysQuiet();
              }, 2400);
            }
          }
        } else if (mySeq === keysMutationSeq) {
          showToast(d.error || "自动保存失败");
        }
      } catch (e) {
        if (mySeq === keysMutationSeq) showToast("网络请求失败");
      } finally {
        keysSaveInFlight--;
      }
    };
    const chained = keysSaveChain.then(run, run);
    keysSaveChain = chained.catch(() => {});
    return chained;
  }

  async function loadExpressKeys() {
    try {
      const seqBefore = keysMutationSeq;
      const r = await apiFetch("/api/express-keys");
      if (!r.ok) return;
      const d = await r.json();
      if (seqBefore !== keysMutationSeq || keysSaveInFlight > 0) return;
      mergeServerKeys(d.keys || [], []);
    } catch (e) {}
  }

  async function loadExpressKeysQuiet() {
    try {
      // 若正在保存或用户正在编辑输入框，则暂不覆盖 DOM
      if (keysSaveInFlight > 0) return;
      const ae = document.activeElement;
      if (ae && $("keyPoolList") && $("keyPoolList").contains(ae)) return;
      const seqBefore = keysMutationSeq;
      syncKeysFromDom();
      const emptyRows = KEYS.filter((r) => !r.key);
      const r = await apiFetch("/api/express-keys");
      if (!r.ok) return;
      const d = await r.json();
      if (seqBefore !== keysMutationSeq || keysSaveInFlight > 0) return;
      const aeAfter = document.activeElement;
      if (aeAfter && $("keyPoolList") && $("keyPoolList").contains(aeAfter)) return;
      mergeServerKeys(d.keys || [], emptyRows);
    } catch (e) {}
  }

  window.revealAndEditKey = function (uid) {
    syncKeysFromDom();
    const rec = findKeyByUid(uid);
    if (!rec) return;
    rec._show = true;
    renderKeys();
    const inp = document.querySelector(`#keyPoolList [data-key-uid="${uid}"] .k-key`);
    if (inp) {
      inp.focus();
      const len = inp.value.length;
      try { inp.setSelectionRange(len, len); } catch (e) {}
    }
  };

  window.toggleKeyVis = function (uid) {
    syncKeysFromDom();
    const rec = findKeyByUid(uid);
    if (rec) { rec._show = !rec._show; renderKeys(); }
  };

  window.onKeyFieldChange = async function (uid) {
    syncKeysFromDom();
    const rec = findKeyByUid(uid);
    if (!rec || !rec.key) return;
    await autoSaveExpressKeys(undefined, { skipSync: true });
  };

  window.addKeyRow = function () {
    syncKeysFromDom();
    const newUid = ++keyRowUidSeq;
    KEYS.push({ _uid: newUid, key: "", project_id: "", project_source: "", enabled: true, hits: 0, _show: true });
    renderKeys();
    const inp = document.querySelector(`#keyPoolList [data-key-uid="${newUid}"] .k-key`);
    if (inp) inp.focus();
  };

  window.removeKeyRow = async function (uid) {
    syncKeysFromDom();
    const idx = findKeyIndexByUid(uid);
    if (idx < 0) return;
    const [removed] = KEYS.splice(idx, 1);
    renderKeys();
    if (!removed || (!removed.key && !removed.key_fp)) {
      showToast("已移除空白 Key 槽位");
      return;
    }
    await autoSaveExpressKeys("已删除 Key 并自动保存（历史命中保留于指纹库）", { skipSync: true });
  };

  window.probeOneKey = async function (uid) {
    syncKeysFromDom();
    const rec = findKeyByUid(uid);
    if (!rec || !rec.key) { showToast("请先在该行填入 Express API Key"); return; }
    const targetKey = rec.key;
    const payload = cleanKeyPayload(KEYS);
    const mySeq = ++keysMutationSeq;
    if (keysProbeRefreshTimer) {
      clearTimeout(keysProbeRefreshTimer);
      keysProbeRefreshTimer = null;
    }
    showToast("正在自动保存并探测归属项目…");
    const run = async () => {
      keysSaveInFlight++;
      try {
        const r = await apiFetch("/api/express-keys/probe", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ key: targetKey, force: true, keys: payload })
        });
        const d = await r.json().catch(() => ({}));
        if (mySeq !== keysMutationSeq) return;
        if (!r.ok) { showToast(d.error || "探测失败"); return; }
        const res = (d.results || [])[0] || {};
        const emptyRows = KEYS.filter((item) => !item.key);
        mergeServerKeys(d.keys || [], emptyRows);
        const hit = KEYS.find((x) => x.key === targetKey);
        if (hit) { hit._msg = res.message || ""; hit._msg_ok = Boolean(res.ok); }
        renderKeys();
        showToast(res.ok ? `探测成功：${res.project_id || "已识别"}` : `Key 已保存，探测未果：${res.message || "请手动填写项目"}`);
      } catch (e) {
        if (mySeq === keysMutationSeq) showToast("网络请求失败");
      } finally {
        keysSaveInFlight--;
      }
    };
    const chained = keysSaveChain.then(run, run);
    keysSaveChain = chained.catch(() => {});
    return chained;
  };

  window.probeAllKeys = async function () {
    syncKeysFromDom();
    const payload = cleanKeyPayload(KEYS);
    if (!payload.length) { showToast("请先至少添加一个 Express API Key"); return; }
    const mySeq = ++keysMutationSeq;
    if (keysProbeRefreshTimer) {
      clearTimeout(keysProbeRefreshTimer);
      keysProbeRefreshTimer = null;
    }
    showToast("正在自动保存并探测全部 Key…");
    const run = async () => {
      keysSaveInFlight++;
      try {
        const r = await apiFetch("/api/express-keys/probe", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ all: true, force: true, keys: payload })
        });
        const d = await r.json().catch(() => ({}));
        if (mySeq !== keysMutationSeq) return;
        if (!r.ok) { showToast(d.error || "探测失败"); return; }
        const results = d.results || [];
        const emptyRows = KEYS.filter((item) => !item.key);
        mergeServerKeys(d.keys || [], emptyRows);
        results.forEach((res) => {
          const hit = KEYS.find((x) => x.key === res.key);
          if (hit) { hit._msg = res.message || ""; hit._msg_ok = Boolean(res.ok); }
        });
        renderKeys();
        showToast(`已完成探测：成功 ${results.filter((x) => x.ok).length} / ${results.length}`);
      } catch (e) {
        if (mySeq === keysMutationSeq) showToast("网络请求失败");
      } finally {
        keysSaveInFlight--;
      }
    };
    const chained = keysSaveChain.then(run, run);
    keysSaveChain = chained.catch(() => {});
    return chained;
  };

  /* =====================================================================
     4. MODEL PARAMS & GLOBAL INFRASTRUCTURE (REAL API + DIRTY GUARD)
     ===================================================================== */
  let GLOBAL_SETTINGS = {};
  let OVERRIDES = {};
  let CAPS = {};
  let ALL_MODELS = [];
  let MODEL_CATEGORIES = [];
  let currentScope = "__global__";
  let curAR = "";
  let pendingAction = null;

  function setElVal(id, v) {
    const el = $(id);
    if (!el) return;
    if (el.type === "checkbox") el.checked = Boolean(v);
    else el.value = (v === null || v === undefined) ? "" : String(v);
  }

  function getElVal(id) {
    const el = $(id);
    if (!el) return "";
    return el.type === "checkbox" ? Boolean(el.checked) : el.value;
  }

  function normVal(key, val) {
    if (typeof val === "boolean") return val;
    if (GLOBAL_NUM_KEYS.has(key)) {
      const s = String(val == null ? "" : val).trim();
      return s === "" ? "" : String(Number(s));
    }
    return String(val == null ? "" : val);
  }

  function fillModelSelects() {
    let scopeHtml = `<option value="__global__">＊ 全局默认（对所有模型）</option>`;
    let labHtml = "";
    const grouped = {};

    if (MODEL_CATEGORIES.length) {
      MODEL_CATEGORIES.forEach((cat) => {
        const ms = (cat && Array.isArray(cat.models)) ? cat.models : [];
        if (!ms.length) return;
        scopeHtml += `<optgroup label="${escHtml(cat.label || cat.id || "模型分组")}">`;
        labHtml += `<optgroup label="${escHtml(cat.label || cat.id || "模型分组")}">`;
        ms.forEach((m) => {
          grouped[m] = true;
          const star = OVERRIDES[m] ? " ★" : "";
          const isImg = Boolean(CAPS[m] && CAPS[m].is_image);
          scopeHtml += `<option value="${escHtml(m)}">${escHtml(m)}${star}</option>`;
          labHtml += `<option value="${escHtml(m)}">${escHtml(m)}${isImg ? " [生图]" : ""}</option>`;
        });
        scopeHtml += `</optgroup>`;
        labHtml += `</optgroup>`;
      });
      const extras = ALL_MODELS.filter((m) => !grouped[m]);
      if (extras.length) {
        scopeHtml += `<optgroup label="其他模型">`;
        labHtml += `<optgroup label="其他模型">`;
        extras.forEach((m) => {
          const isImg = Boolean(CAPS[m] && CAPS[m].is_image);
          scopeHtml += `<option value="${escHtml(m)}">${escHtml(m)}${OVERRIDES[m] ? " ★" : ""}</option>`;
          labHtml += `<option value="${escHtml(m)}">${escHtml(m)}${isImg ? " [生图]" : ""}</option>`;
        });
        scopeHtml += `</optgroup>`;
        labHtml += `</optgroup>`;
      }
    } else {
      ALL_MODELS.forEach((m) => {
        const isImg = Boolean(CAPS[m] && CAPS[m].is_image);
        scopeHtml += `<option value="${escHtml(m)}">${escHtml(m)}${OVERRIDES[m] ? " ★" : ""}</option>`;
        labHtml += `<option value="${escHtml(m)}">${escHtml(m)}${isImg ? " [生图]" : ""}</option>`;
      });
    }

    $("scopeSelect").innerHTML = scopeHtml;
    $("scopeSelect").value = currentScope;

    const labSel = $("labModel");
    if (labSel) {
      const prev = labSel.value;
      labSel.innerHTML = labHtml || `<option value="gemini-4-carbon">gemini-4-carbon</option>`;
      if (prev && Array.from(labSel.options).some((o) => o.value === prev)) {
        labSel.value = prev;
      }
    }
  }

  function fillAspectRatioFor(cap, desired) {
    const sel = $("image_aspect_ratio");
    if (!sel) return;
    const list = (cap && cap.is_image && Array.isArray(cap.image_aspect_ratios) && cap.image_aspect_ratios.length)
      ? cap.image_aspect_ratios
      : COMMON_ARS;
    sel.innerHTML = `<option value="">自动（不强制指定）</option>` + list.map((a) => `<option value="${escHtml(a)}">${escHtml(a)}</option>`).join("");
    sel.value = (desired && list.includes(desired)) ? desired : "";
  }

  function getSavedScopeModel(scope) {
    const isGlobal = scope === "__global__";
    const ov = isGlobal ? {} : (OVERRIDES[scope] || {});
    const out = {};
    PER_MODEL_KEYS.forEach((k) => {
      out[k] = (k in ov) ? ov[k] : GLOBAL_SETTINGS[k];
    });
    return out;
  }

  function applyScopeToForm(scope) {
    currentScope = scope;
    const isGlobal = scope === "__global__";
    const cap = isGlobal ? null : (CAPS[scope] || null);
    const saved = getSavedScopeModel(scope);

    curAR = saved.image_aspect_ratio || "";
    fillAspectRatioFor(cap, curAR);
    PER_MODEL_KEYS.forEach((k) => {
      if (k !== "image_aspect_ratio") setElVal(k, saved[k]);
    });
    setElVal("prefill_cot_guard", true);

    const hasOv = !isGlobal && Boolean(OVERRIDES[scope] && Object.keys(OVERRIDES[scope]).length > 0);
    $("scopeSelect").value = scope;
    $("ovBadge").style.display = hasOv ? "inline-flex" : "none";
    $("btnScopeClear").style.display = isGlobal ? "none" : "inline-flex";
    $("btnScopeSaveText").textContent = isGlobal ? "保存为全局默认" : "保存为该模型专属";
    $("scopeHintText").textContent = isGlobal
      ? "当前编辑：所有未单独定制（无 ★）模型的全局默认参数"
      : (hasOv ? `当前编辑：${scope} 的专属覆盖参数（★ 已激活）` : `当前编辑：${scope}（当前继承全局默认，点保存即可创建专属覆盖）`);

    const capsBox = $("capsSummary");
    if (isGlobal || !cap) {
      capsBox.innerHTML = `<span class="pill primary">作用域：全局默认</span><span class="pill">切换具体模型可查看能力白名单</span>`;
      $("thinking_g3_level").disabled = false;
      $("wrapG3Level").style.opacity = "1";
      $("image_size").disabled = false;
      $("image_aspect_ratio").disabled = false;
      $("thinkNote").textContent = "全局默认：选「自动」时采用模型官方原生默认档位；压制思考或选 minimal 时，3.7+ 及 4.x 全系最低自动发送 low；仅四档思考的 4.x 机型（4-neon / 4-hydrogen）生效 xhigh，其余自动降级 high。";
      $("imageNote").textContent = "全局默认：分辨率与宽高比仅在调用生图模型（-image / nano-banana）时生效。";
    } else {
      const th = cap.thinking || {};
      const chips = [`<span class="pill primary">家族 ${escHtml(cap.family || "—")}</span>`];
      if (th.kind === "level") chips.push(`<span class="pill">思考档位 ${escHtml((th.levels || []).join(" / "))}</span>`);
      else chips.push(`<span class="pill">无思考调节</span>`);
      if (cap.is_image) {
        chips.push(`<span class="pill warning">生图专用</span>`);
        chips.push(`<span class="pill">分辨率 ${escHtml((cap.image_sizes || []).join("/"))}</span>`);
        chips.push(`<span class="pill">比例 ${(cap.image_aspect_ratios || []).length} 种</span>`);
      } else {
        if (cap.max_nonstream_output_tokens) {
          chips.push(`<span class="pill success">一元上限 ${Math.round(cap.max_nonstream_output_tokens / 1024)}K</span>`);
        }
        chips.push(`<span class="pill">支持函数调用</span>`);
      }
      if (cap.supports_search) chips.push(`<span class="pill">支持联网搜索</span>`);
      capsBox.innerHTML = chips.join("");

      const isLvl = th.kind === "level";
      $("thinking_g3_level").disabled = !isLvl;
      $("wrapG3Level").style.opacity = isLvl ? "1" : "0.48";
      if (cap.is_image) {
        $("thinkNote").textContent = "生图模型思考档位：选「自动」即采用官方原生默认，亦可在上方手动指定档位。";
      } else if (isLvl) {
        const lvs = th.levels || [];
        const b1 = lvs.includes("minimal") ? "最低合法档 minimal" : "最低合法档 low（不支持 minimal）";
        const b2 = lvs.includes("xhigh") ? "原生支持 xhigh" : "不支持 xhigh（自动降级 high）";
        $("thinkNote").textContent = `支持思考档位：${lvs.join(" / ")}（${b1}；${b2}）。`;
      } else {
        $("thinkNote").textContent = "该模型不支持思考档位调节。";
      }

      const imgOn = Boolean(cap.is_image);
      $("image_size").disabled = !imgOn;
      $("image_aspect_ratio").disabled = !imgOn;
      $("imageNote").textContent = imgOn
        ? `支持宽高比：${(cap.image_aspect_ratios || []).join("、")}；支持分辨率：${(cap.image_sizes || []).join(" / ")}。`
        : "当前所选为文本模型，无需配置生图分辨率与宽高比（切换到生图模型自动解锁）。";
    }

    updateLivePipeline();
    refreshDirtyState();
  }

  function applyGlobalToForm() {
    GLOBAL_KEYS.forEach((k) => setElVal(k, GLOBAL_SETTINGS[k]));
  }

  function countModelDirty() {
    if (!Object.keys(GLOBAL_SETTINGS).length) return 0;
    const baseM = getSavedScopeModel(currentScope);
    const cap = currentScope === "__global__" ? null : (CAPS[currentScope] || null);
    let diff = 0;
    PER_MODEL_KEYS.forEach((k) => {
      if ((k === "image_size" || k === "image_aspect_ratio") && cap && !cap.is_image) return;
      if (k === "thinking_g3_level" && cap && (cap.thinking || {}).kind !== "level") return;
      const expected = k === "image_aspect_ratio"
        ? ((baseM[k] && (!cap || !cap.is_image || (cap.image_aspect_ratios || []).includes(baseM[k]))) ? baseM[k] : "")
        : baseM[k];
      if (normVal(k, expected) !== normVal(k, getElVal(k))) diff++;
    });
    return diff;
  }

  function countGlobalDirty() {
    if (!Object.keys(GLOBAL_SETTINGS).length) return 0;
    let diff = 0;
    GLOBAL_KEYS.forEach((k) => {
      if (normVal(k, GLOBAL_SETTINGS[k]) !== normVal(k, getElVal(k))) diff++;
    });
    return diff;
  }

  function countDirtyDiffs() {
    return countModelDirty() + countGlobalDirty();
  }

  function updateLivePipeline() {
    const pre = getElVal("prefill_mode") || "smart";
    const ntm = getElVal("native_thinking_mode") || "request";
    const lv = getElVal("thinking_g3_level") || "";
    const sbox = Boolean(getElVal("synthetic_sandbox_enabled"));
    const fake = Boolean(getElVal("fake_streaming"));

    const preMap = { smart: "预填充：智能并入", keep_turn: "预填充：保留轮次", minimal: "预填充：最小占位", off: "预填充：关闭" };
    const thinkLabel = ntm === "off"
      ? "思考：压制关闭"
      : (ntm === "console" ? `思考：强制 ${lv || "默认"}` : `思考：跟随请求 (${lv || "自动"})`);
    const antiLabel = sbox ? "抗截断：compose_reply" : "抗截断：原生透传";
    const streamLabel = fake ? "输出：假流式保活" : "输出：真流式透传";
    const arrow = `<span class="pipe-arrow"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"/></svg></span>`;

    $("livePipelineTrack").innerHTML = `
      <span class="pipe-node on">客户端请求</span>
      ${arrow}
      <span class="pipe-node ${pre === "off" ? "muted" : "on"}">${preMap[pre] || pre}</span>
      ${arrow}
      <span class="pipe-node ${ntm === "off" ? "muted" : "think"}">${escHtml(thinkLabel)}</span>
      ${arrow}
      <span class="pipe-node ${sbox ? "mint" : "muted"}">${antiLabel}</span>
      ${arrow}
      <span class="pipe-node on">${streamLabel}</span>
    `;
  }

  function refreshDirtyState() {
    const diff = countDirtyDiffs();
    const isParamsActive = $("panel-params").classList.contains("active");
    const wrap = $("dirtyBarWrap");
    const show = diff > 0 && isParamsActive;
    $("dirtyBarText").textContent = `检测到 ${diff} 项未保存的配置改动`;
    wrap.classList.toggle("show", show);
    document.body.classList.toggle("has-dirty-bar", show);
  }

  window.onParamChange = function () {
    updateLivePipeline();
    refreshDirtyState();
  };

  function numOr(id, fallback) {
    const raw = String(getElVal(id) || "").trim();
    if (raw === "") return fallback;
    const v = Number(raw);
    return Number.isFinite(v) ? v : fallback;
  }

  function collectModelPatch() {
    return {
      native_thinking_mode: getElVal("native_thinking_mode"),
      thinking_g3_level: getElVal("thinking_g3_level"),
      image_size: getElVal("image_size"),
      image_aspect_ratio: getElVal("image_aspect_ratio"),
      default_temperature: null,
      default_top_p: null,
      default_max_tokens: null,
      sampling_policy: "auto",
      inject_system_instruction: getElVal("inject_system_instruction"),
      inject_prefill: getElVal("inject_prefill"),
      prefill_instruction: getElVal("prefill_instruction"),
      image_system_instruction: Boolean(getElVal("image_system_instruction")),
      inject_prefill_for_image: Boolean(getElVal("inject_prefill_for_image")),
      prefill_cot_guard: true,
      prefill_mode: getElVal("prefill_mode"),
      prefill_suppress_thinking: Boolean(getElVal("prefill_suppress_thinking")),
      synthetic_sandbox_enabled: Boolean(getElVal("synthetic_sandbox_enabled"))
    };
  }

  function collectGlobalInfraPatch() {
    return {
      express_location: getElVal("express_location"),
      img_compress_enabled: Boolean(getElVal("img_compress_enabled")),
      img_compress_max_dim: numOr("img_compress_max_dim", 1536),
      img_compress_max_mb: numOr("img_compress_max_mb", 1.5),
      img_compress_quality: numOr("img_compress_quality", 85),
      retry_max: numOr("retry_max", 10),
      retry_backoff_seconds: numOr("retry_backoff_seconds", 5),
      cross_channel_fallback: Boolean(getElVal("cross_channel_fallback")),
      fake_streaming: Boolean(getElVal("fake_streaming")),
      fake_streaming_interval: numOr("fake_streaming_interval", 1),
      resumable_enabled: Boolean(getElVal("resumable_enabled")),
      resumable_ttl_seconds: numOr("resumable_ttl_seconds", 1800),
      resumable_max_streams: numOr("resumable_max_streams", 200),
      roundrobin: Boolean(getElVal("roundrobin")),
      safety_score: Boolean(getElVal("safety_score")),
      debug_outbound: Boolean(getElVal("debug_outbound"))
    };
  }

  window.saveParams = async function (targetMode) {
    const mode = targetMode || "all";
    const modelPatch = collectModelPatch();
    const infraPatch = collectGlobalInfraPatch();
    const mDirty = countModelDirty();
    const gDirty = countGlobalDirty();
    try {
      if (mode === "global") {
        const r = await apiFetch("/api/settings", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(infraPatch)
        });
        if (!r.ok) { showToast("保存全局基础设施设置失败"); return false; }
        GLOBAL_SETTINGS = await r.json();
        applyGlobalToForm();
        updateLivePipeline();
        refreshDirtyState();
        showToast("全局基础设施设置已保存");
        scheduleLabInspect();
        return true;
      }

      if (mode === "model") {
        if (currentScope === "__global__") {
          const r = await apiFetch("/api/settings", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(modelPatch)
          });
          if (!r.ok) { showToast("保存全局默认参数失败"); return false; }
          GLOBAL_SETTINGS = await r.json();
          applyScopeToForm("__global__");
          showToast("全局默认模型参数已保存");
        } else {
          const rm = await apiFetch("/api/model-overrides/" + encodeURIComponent(currentScope), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(modelPatch)
          });
          const dm = await rm.json().catch(() => ({}));
          if (!rm.ok) { showToast("保存模型专属参数失败"); return false; }
          OVERRIDES[currentScope] = dm.override || modelPatch;
          fillModelSelects();
          applyScopeToForm(currentScope);
          showToast(`已保存 ${currentScope} 的专属参数配置`);
        }
        scheduleLabInspect();
        return true;
      }

      // mode === "all"（来自浮动未保存条或离开保护弹窗）：仅提交实际有改动的范围，避免误建模型专属覆盖
      if (currentScope === "__global__") {
        const merged = Object.assign({}, mDirty > 0 ? modelPatch : {}, gDirty > 0 ? infraPatch : {});
        const payload = Object.keys(merged).length ? merged : Object.assign({}, modelPatch, infraPatch);
        const r = await apiFetch("/api/settings", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        if (!r.ok) { showToast("保存全局配置失败"); return false; }
        GLOBAL_SETTINGS = await r.json();
        applyGlobalToForm();
        applyScopeToForm("__global__");
        showToast("全局配置已保存");
      } else {
        const savedParts = [];
        if (gDirty > 0) {
          const rg = await apiFetch("/api/settings", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(infraPatch)
          });
          if (!rg.ok) { showToast("保存全局基础设施设置失败"); return false; }
          GLOBAL_SETTINGS = await rg.json();
          applyGlobalToForm();
          savedParts.push("全局基础设施设置");
        }
        if (mDirty > 0) {
          const rm = await apiFetch("/api/model-overrides/" + encodeURIComponent(currentScope), {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(modelPatch)
          });
          const dm = await rm.json().catch(() => ({}));
          if (!rm.ok) { showToast("保存模型专属参数失败"); return false; }
          OVERRIDES[currentScope] = dm.override || modelPatch;
          fillModelSelects();
          applyScopeToForm(currentScope);
          savedParts.push(`${currentScope} 专属参数`);
        } else {
          updateLivePipeline();
          refreshDirtyState();
        }
        showToast(savedParts.length ? `已保存：${savedParts.join(" 与 ")}` : "当前没有需要保存的改动");
      }
      scheduleLabInspect();
      return true;
    } catch (e) {
      showToast("保存失败，请检查网络连接");
      return false;
    }
  };

  window.clearModelOverride = async function () {
    if (currentScope === "__global__") return;
    if (!OVERRIDES[currentScope]) { showToast("该模型当前未设置专属参数"); return; }
    try {
      const r = await apiFetch("/api/model-overrides/" + encodeURIComponent(currentScope), { method: "DELETE" });
      if (r.ok) {
        delete OVERRIDES[currentScope];
        fillModelSelects();
        applyScopeToForm(currentScope);
        scheduleLabInspect();
        showToast(`已清除 ${currentScope} 专属参数，恢复继承全局默认`);
      } else {
        showToast("清除失败");
      }
    } catch (e) {
      showToast("网络请求失败");
    }
  };

  window.discardParams = function () {
    applyGlobalToForm();
    applyScopeToForm(currentScope);
    showToast("已撤销未保存的修改");
  };

  window.setParamsSub = function (sub) {
    $("subParamModelBtn").classList.toggle("active", sub === "model");
    $("subParamGlobalBtn").classList.toggle("active", sub === "global");
    $("paramsSubModel").style.display = sub === "model" ? "block" : "none";
    $("paramsSubGlobal").style.display = sub === "global" ? "block" : "none";
  };

  function openDirtyModal() {
    const dlg = $("dirtyModal");
    if (!dlg) return;
    if (typeof dlg.showModal === "function") {
      if (!dlg.open) dlg.showModal();
    } else {
      dlg.classList.add("open");
    }
  }

  window.handleScopeChange = function (newScope) {
    if (countModelDirty() > 0) {
      $("scopeSelect").value = currentScope;
      pendingAction = () => applyScopeToForm(newScope);
      openDirtyModal();
    } else {
      applyScopeToForm(newScope);
    }
  };

  window.closeDirtyModal = function () {
    pendingAction = null;
    const dlg = $("dirtyModal");
    if (!dlg) return;
    if (typeof dlg.close === "function") {
      if (dlg.open) dlg.close();
    }
    dlg.classList.remove("open");
  };

  if ($("dirtyModal")) {
    $("dirtyModal").addEventListener("close", () => {
      pendingAction = null;
    });
  }

  window.confirmDirtyAction = async function (shouldSave) {
    if (shouldSave) {
      const ok = await window.saveParams("all");
      if (!ok) return;
    } else {
      applyGlobalToForm();
      applyScopeToForm(currentScope);
    }
    const fn = pendingAction;
    window.closeDirtyModal();
    if (fn) fn();
  };

  window.toggleHelpById = function (btn, id) {
    const drawer = $(id);
    if (!drawer) return;
    const open = drawer.classList.toggle("open");
    btn.classList.toggle("open", open);
    btn.setAttribute("aria-expanded", open ? "true" : "false");
  };

  async function loadParamsAndCaps() {
    try {
      const [rs, rc] = await Promise.all([
        apiFetch("/api/settings"),
        apiFetch("/api/capabilities")
      ]);
      if (rs.ok) GLOBAL_SETTINGS = await rs.json();
      if (rc.ok) {
        const c = await rc.json();
        CAPS = c.capabilities || {};
        OVERRIDES = c.overrides || {};
        ALL_MODELS = Array.isArray(c.models) ? c.models : [];
        MODEL_CATEGORIES = Array.isArray(c.categories) ? c.categories : [];
      }
      fillModelSelects();
      applyGlobalToForm();
      applyScopeToForm(currentScope);
      refreshLabInspector();
    } catch (e) {}
  }

  /* =====================================================================
     5. LIVE CONNECTIVITY TEST & REQUEST ANATOMY (MERGED IN CHANNEL TAB)
     ===================================================================== */
  let labStream = true;
  let labAbortCtrl = null;
  let labInspectTimer = null;

  function syncLabTargetOptions() {
    const sel = $("labTarget");
    if (!sel) return;
    const prev = sel.value || "auto";
    const opts = [
      `<option value="auto">跟随当前全局路由（含轮询与容灾）</option>`
    ];
    const savedKeys = KEYS.filter((k) => k && k.key && k.key_fp);
    if (savedKeys.length) {
      opts.push(`<optgroup label="定向单测 Express API Key（关闭兜底）">`);
      savedKeys.forEach((rec) => {
        const idx = KEYS.indexOf(rec) + 1;
        const pid = rec.project_id ? `项目 ${rec.project_id}` : "项目未探测";
        const st = rec.enabled === false ? " · 已停用" : "";
        opts.push(`<option value="key:${escHtml(rec.key_fp)}">仅测 Express Key #${idx}（${escHtml(pid)}${st}）</option>`);
      });
      opts.push(`</optgroup>`);
    }
    const ckLabel = cookieConfigured
      ? `仅测 Studio Cookie 直连（项目 ${cookieProjectId || "已配置"} · 单发直测）`
      : "仅测 Studio Cookie 直连（当前未配置）";
    opts.push(`<optgroup label="定向单测 Studio Cookie（不切全局主通道）"><option value="cookie">${escHtml(ckLabel)}</option></optgroup>`);

    sel.innerHTML = opts.join("");
    if (Array.from(sel.options).some((o) => o.value === prev)) {
      sel.value = prev;
    } else {
      sel.value = "auto";
    }
    updateLabTargetBadge();
  }

  function updateLabTargetBadge() {
    const sel = $("labTarget");
    const pill = $("labTargetModePill");
    if (!sel || !pill) return;
    const v = sel.value || "auto";
    if (v === "auto") {
      pill.className = "pill";
      pill.textContent = "跟随默认路由";
    } else if (v === "cookie") {
      pill.className = "pill success";
      pill.textContent = "单发直测 · 不切主通道";
    } else {
      pill.className = "pill primary";
      pill.textContent = "单发直测 · 关闭兜底";
    }
  }

  function updateLabModelHint() {
    const m = $("labModel") ? $("labModel").value : "";
    const isImg = Boolean(CAPS[m] && CAPS[m].is_image);
    const ta = $("labPrompt");
    const hint = $("labReadyHint");
    if (ta) {
      ta.placeholder = isImg
        ? "输入生图提示词…（留空直接点发送，将自动生成一张极简晴空小云朵测试图）"
        : "输入任意测试提示词…（留空直接点发送，将自动发送一句极简连通性探针）";
    }
    if (hint) {
      hint.textContent = isImg
        ? "生图模型实测：自动预览返回图片并展示分辨率与宽高比"
        : "将按当前生效规则发往真实上游";
    }
  }

  window.onLabTargetChange = function () {
    updateLabTargetBadge();
    scheduleLabInspect();
  };

  window.onLabModelChange = function () {
    updateLabModelHint();
    scheduleLabInspect();
  };

  window.testKeyDirect = async function (uid) {
    syncKeysFromDom();
    const idx = findKeyIndexByUid(uid);
    const rec = idx >= 0 ? KEYS[idx] : null;
    if (!rec || !rec.key) {
      showToast("请先在该行填入 Express API Key");
      return;
    }
    let fp = rec.key_fp;
    if (!fp) {
      await autoSaveExpressKeys("已自动保存 Key，正在跳转实测", { skipSync: true });
      const saved = KEYS.find((x) => x.key === rec.key);
      fp = saved && saved.key_fp;
    }
    if (!fp) {
      showToast("Key 尚未保存成功，无法定位指纹");
      return;
    }
    window.setChannelSub("test");
    syncLabTargetOptions();
    if ($("labTarget")) {
      $("labTarget").value = "key:" + fp;
      updateLabTargetBadge();
    }
    scheduleLabInspect();
    window.scrollTo({ top: 0, behavior: "smooth" });
    const finalIdx = KEYS.findIndex((x) => x.key === rec.key);
    showToast(`已锁定 Express Key #${(finalIdx >= 0 ? finalIdx : idx) + 1}（单发直测模式），点击「发送实测」即可验证`);
  };

  window.testCookieDirect = function () {
    window.setChannelSub("test");
    syncLabTargetOptions();
    if ($("labTarget")) {
      $("labTarget").value = "cookie";
      updateLabTargetBadge();
    }
    scheduleLabInspect();
    window.scrollTo({ top: 0, behavior: "smooth" });
    showToast("已锁定仅测 Studio Cookie（不影响当前全局主通道）");
  };

  window.jumpToModelTest = function () {
    const targetModel = currentScope !== "__global__" ? currentScope : "";
    const go = () => {
      applyTab("channel");
      window.setChannelSub("test");
      if (targetModel && $("labModel") && Array.from($("labModel").options).some((o) => o.value === targetModel)) {
        $("labModel").value = targetModel;
      }
      updateLabModelHint();
      scheduleLabInspect();
      showToast(targetModel ? `已带入模型 ${targetModel}，可直接实测或查看出站透视` : "已跳转至连通实测面板");
    };
    if (currentTab === "params" && countDirtyDiffs() > 0) {
      pendingAction = go;
      openDirtyModal();
      return;
    }
    go();
  };

  window.setLabStream = function (isStream) {
    labStream = Boolean(isStream);
    $("labStreamYes").classList.toggle("active", labStream);
    $("labStreamNo").classList.toggle("active", !labStream);
    scheduleLabInspect();
  };

  window.toggleLabOptional = function () {
    $("labOptionalArea").classList.toggle("open");
  };

  window.setLabView = function (view) {
    document.querySelectorAll("[data-labview]").forEach((b) => b.classList.toggle("active", b.dataset.labview === view));
    document.querySelectorAll(".lab-subpanel").forEach((p) => p.classList.remove("active"));
    $("labView-" + view).classList.add("active");
  };

  function getLabBody() {
    return {
      model: $("labModel") ? $("labModel").value : "gemini-4-carbon",
      stream: labStream,
      prompt: $("labPrompt") ? $("labPrompt").value : "",
      system: $("labSystem") ? $("labSystem").value : "",
      prefill: $("labPrefill") ? $("labPrefill").value : "",
      target: $("labTarget") ? $("labTarget").value : "auto"
    };
  }

  function renderLabInspectData(data) {
    if (!data) return;
    const cards = Array.isArray(data.cards) ? data.cards : [];
    $("labInspectGrid").innerHTML = cards.map((c) => `
      <div class="inspect-item">
        <div class="inspect-k">${escHtml(c.k)}</div>
        <div class="inspect-v">${escHtml(c.v)}</div>
      </div>
    `).join("");
    $("labInspectJson").textContent = JSON.stringify(data.payload || {}, null, 2);
  }

  function scheduleLabInspect() {
    clearTimeout(labInspectTimer);
    labInspectTimer = setTimeout(refreshLabInspector, 150);
  }

  async function refreshLabInspector() {
    try {
      const r = await apiFetch("/api/lab/inspect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(getLabBody())
      });
      if (r.ok) renderLabInspectData(await r.json());
    } catch (e) {}
  }
  window.refreshLabInspector = scheduleLabInspect;

  function renderLabMetaBar(meta, statusTxt, statusCls) {
    const bar = $("labMetaBar");
    const credPill = $("labCredPill");
    const tagsEl = $("labMetaTags");
    if (!bar || !credPill || !tagsEl) return;
    bar.style.display = "flex";
    const m = meta || {};
    const loc = m.location ? ` · ${m.location}` : "";
    const cred = m.cred_short
      ? `命中 ${m.cred_short}${loc}`
      : (m.target_label || "网关调度中");
    credPill.className = "pill " + (statusCls || "primary");
    credPill.textContent = statusTxt ? `${cred} · ${statusTxt}` : cred;

    const tags = [];
    if (m.targeted_single_shot) tags.push("单发直测");
    if (m.is_image) {
      const ic = m.image_config || {};
      tags.push(`生图 ${ic.image_size || "1K"}${ic.aspect_ratio ? " " + ic.aspect_ratio : ""}`);
    } else if (m.sandbox_active) {
      tags.push("compose_reply 抗截断");
    }
    if (m.prefill_active) tags.push("预填充");
    tagsEl.innerHTML = tags.map((t) => `<span class="mini-tag">${escHtml(t)}</span>`).join("");
  }

  function renderLabErrorCard(code, message, diag) {
    const card = $("labErrorCard");
    if (!card) return;
    const d = diag || {};
    const c = Number(d.code || code || 500);
    $("labErrorCode").textContent = `HTTP ${c}`;
    $("labErrorTitle").textContent = d.title || `上游返回异常 (HTTP ${c})`;
    $("labErrorHint").textContent = d.hint || "请核对凭证权限、区域设置或查看下方详细报错。";
    $("labErrorDetail").textContent = String(d.message || message || "未知错误");
    card.style.display = "block";
  }

  function renderLabContentWithImages(rawText, isLive, meta) {
    const streamOut = $("labStreamText");
    const gallery = $("labImageGallery");
    const images = [];
    // 提取 Markdown data:image/... 或裸 data:image/...，避免超长 Base64 刷爆正文框
    let cleaned = String(rawText || "").replace(/!\[([^\]]*)\]\((data:image\/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=\s]+)\)/g, (_, alt, uri) => {
      images.push({ alt: alt || "生成图像", src: uri.replace(/\s+/g, "") });
      return "";
    });
    if (!images.length) {
      cleaned = cleaned.replace(/(data:image\/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=]{80,})/g, (uri) => {
        images.push({ alt: "生成图像", src: uri });
        return "";
      });
    }
    cleaned = cleaned.trim();

    if (images.length) {
      const ic = (meta && meta.image_config) || {};
      const badge = ic.image_size ? `${ic.image_size}${ic.aspect_ratio ? " · " + ic.aspect_ratio : ""}` : "预览图";
      gallery.innerHTML = images.map((im, idx) => {
        const ext = (im.src.match(/^data:image\/([a-zA-Z0-9+.-]+);/) || [])[1] || "png";
        const fname = `vertex-test-${idx + 1}.${ext.replace("jpeg", "jpg")}`;
        return `
          <div class="lab-image-card">
            <img src="${im.src}" alt="${escHtml(im.alt)}" onclick="openDataImageNewTab(this.src)" title="点击在新标签页查看原图" />
            <div class="lab-image-foot">
              <span class="pill primary num">${escHtml(badge)} · 图 #${idx + 1}</span>
              <a class="btn btn-ghost btn-sm" href="${im.src}" download="${escHtml(fname)}">下载原图</a>
            </div>
          </div>
        `;
      }).join("");
      gallery.style.display = "grid";
      streamOut.style.color = cleaned ? "var(--text)" : "var(--text-muted)";
      streamOut.innerHTML = (cleaned ? escHtml(cleaned) : `已成功生成 ${images.length} 张图像（见下方预览）`) + (isLive ? `<span class="typing-cursor"></span>` : "");
    } else {
      gallery.style.display = "none";
      gallery.innerHTML = "";
      streamOut.style.color = "var(--text)";
      if (isLive) {
        streamOut.innerHTML = escHtml(rawText || "") + `<span class="typing-cursor"></span>`;
      } else {
        streamOut.textContent = rawText || "（无可见正文）";
      }
    }
  }

  window.openDataImageNewTab = function (src) {
    try {
      const w = window.open("");
      if (w) {
        w.document.write(`<!doctype html><html><body style="margin:0;background:#0f172a;display:flex;align-items:center;justify-content:center;min-height:100vh;"><img src="${src}" style="max-width:100%;max-height:100vh;object-fit:contain;" /></body></html>`);
        w.document.close();
      }
    } catch (e) {}
  };

  function renderLabTokens(u, finishReason) {
    const p = Number((u && u.prompt_tokens) || 0);
    const c = Number((u && u.cached_tokens) || 0);
    const th = Number((u && u.thoughts_tokens) || 0);
    const out = Number((u && u.output_tokens) || 0);
    const total = p + th + out;
    const cachePct = p > 0 ? ((c / p) * 100).toFixed(1) : "0.0";

    $("labTokTotal").textContent = total > 0 ? `合计 ${fmtNum(total)} Tokens` : "上游未回传用量（如部分 Cookie 接口）";
    if (total > 0) {
      $("labTokBar").innerHTML = `
        <div class="seg-in" style="width:${((p / total) * 100).toFixed(1)}%">
          <div class="seg-cache-overlay" style="width:${cachePct}%"></div>
        </div>
        <div class="seg-think" style="width:${((th / total) * 100).toFixed(1)}%"></div>
        <div class="seg-out" style="width:${((out / total) * 100).toFixed(1)}%"></div>
      `;
    } else {
      $("labTokBar").innerHTML = "";
    }

    const items = [
      { k: "输入 Prompt", v: `${fmtNum(p)} tk` },
      { k: "隐式前缀缓存命中", v: `${fmtNum(c)} tk (${cachePct}%)` },
      { k: "思考消耗", v: `${fmtNum(th)} tk` },
      { k: "可见正文输出", v: `${fmtNum(out)} tk` },
      { k: "结束原因", v: String(finishReason || "stop").toUpperCase() },
      { k: "传输模式", v: labStream ? "流式" : "非流式" }
    ];
    $("labTokenGrid").innerHTML = items.map((i) => `
      <div class="inspect-item">
        <div class="inspect-k">${escHtml(i.k)}</div>
        <div class="inspect-v num">${escHtml(i.v)}</div>
      </div>
    `).join("");
  }

  function updateLabTiming(conn, think, gen, total, tps) {
    const sum = Math.max(0.01, Number(total || (conn + think + gen)));
    const cPct = Math.max(2, (conn / sum) * 100);
    const thPct = think > 0 ? Math.max(2, (think / sum) * 100) : 0;
    const gPct = gen > 0 ? Math.max(2, (gen / sum) * 100) : 0;
    const norm = cPct + thPct + gPct;

    $("labSpeedPill").className = "pill success num";
    $("labSpeedPill").textContent = tps
      ? `${Number(tps).toFixed(1)} tok/s · 总耗时 ${sum.toFixed(2)}s`
      : `总耗时 ${sum.toFixed(2)}s`;
    $("labTimingBar").innerHTML = `
      <div class="t-conn" style="width:${((cPct / norm) * 100).toFixed(1)}%"></div>
      ${thPct > 0 ? `<div class="t-think" style="width:${((thPct / norm) * 100).toFixed(1)}%"></div>` : ""}
      ${gPct > 0 ? `<div class="t-gen" style="width:${((gPct / norm) * 100).toFixed(1)}%"></div>` : ""}
    `;
    $("labTimingLegend").innerHTML = `
      <span><i class="dot" style="background:var(--warning)"></i> 连接 ${Number(conn).toFixed(2)}s</span>
      <span><i class="dot" style="background:var(--think)"></i> 思考 ${Number(think).toFixed(2)}s</span>
      <span><i class="dot" style="background:var(--accent)"></i> 吐字 ${Number(gen).toFixed(2)}s</span>
    `;
  }

  function setLabBtnRunning(running) {
    const btn = $("labSendBtn");
    if (!btn) return;
    if (running) {
      btn.className = "btn btn-danger-soft";
      btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>中止`;
    } else {
      btn.className = "btn btn-primary";
      btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg>发送实测`;
    }
  }

  window.runLabTest = async function () {
    if (labAbortCtrl) {
      labAbortCtrl.abort();
      labAbortCtrl = null;
      setLabBtnRunning(false);
      showToast("已中止当前实测请求");
      return;
    }

    window.setLabView("stream");
    labAbortCtrl = new AbortController();
    setLabBtnRunning(true);

    const thoughtBox = $("labThoughtBox");
    const thoughtText = $("labThoughtText");
    const streamOut = $("labStreamText");
    const errCard = $("labErrorCard");
    const gallery = $("labImageGallery");
    thoughtBox.style.display = "none";
    thoughtBox.classList.remove("collapsed");
    thoughtText.textContent = "";
    errCard.style.display = "none";
    gallery.style.display = "none";
    gallery.innerHTML = "";
    streamOut.style.display = "block";
    streamOut.style.color = "var(--text)";
    streamOut.innerHTML = `<span class="typing-cursor"></span>`;
    $("labSpeedPill").className = "pill primary num";
    $("labSpeedPill").textContent = "连接上游中…";
    $("labTimingBar").innerHTML = `<div class="t-conn" style="width:100%;opacity:0.45"></div>`;

    const body = getLabBody();
    renderLabMetaBar({ target_label: $("labTarget") ? $("labTarget").options[$("labTarget").selectedIndex].text : "默认路由", targeted_single_shot: body.target !== "auto" }, "请求中…", "primary");

    let accThought = "";
    let accBody = "";
    let lastMeta = null;
    const t0 = performance.now();

    try {
      const r = await apiFetch("/api/lab/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: labAbortCtrl.signal
      });
      if (!r.ok || !r.body) {
        const errTxt = await r.text().catch(() => `HTTP ${r.status}`);
        streamOut.style.display = "none";
        renderLabErrorCard(r.status, errTxt, null);
        renderLabMetaBar(lastMeta, `HTTP ${r.status}`, "danger");
        $("labSpeedPill").className = "pill danger";
        $("labSpeedPill").textContent = `HTTP ${r.status}`;
        return;
      }

      const reader = r.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buf = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        while (buf.includes("\n\n")) {
          const idx = buf.indexOf("\n\n");
          const frame = buf.slice(0, idx);
          buf = buf.slice(idx + 2);
          for (const rawLine of frame.split("\n")) {
            const line = rawLine.trim();
            if (!line.startsWith("data:")) continue;
            const jsonStr = line.slice(5).trim();
            if (!jsonStr) continue;
            let ev;
            try { ev = JSON.parse(jsonStr); } catch (e) { continue; }
            if (ev.type === "inspect" && ev.inspect) {
              renderLabInspectData(ev.inspect);
            } else if (ev.type === "thought") {
              accThought += ev.delta || "";
              thoughtBox.style.display = "block";
              thoughtText.textContent = accThought;
              $("labThoughtMeta").textContent = `思考过程 · ${accThought.length} 字（点击折叠 / 展开）`;
              const sec = ((performance.now() - t0) / 1000).toFixed(1);
              $("labSpeedPill").textContent = `思考中 · ${sec}s`;
            } else if (ev.type === "content") {
              accBody += ev.delta || "";
              if (!accBody.includes("data:image/")) {
                streamOut.innerHTML = escHtml(accBody) + `<span class="typing-cursor"></span>`;
              } else {
                streamOut.innerHTML = `正在接收图像数据（${fmtNum(accBody.length)} 字符）…<span class="typing-cursor"></span>`;
              }
              const sec = ((performance.now() - t0) / 1000).toFixed(1);
              $("labSpeedPill").textContent = `接收中 · ${accBody.length} 字 · ${sec}s`;
            } else if (ev.type === "error") {
              lastMeta = ev.meta || lastMeta;
              streamOut.style.display = "none";
              renderLabErrorCard(ev.code || 500, ev.message || "未知错误", ev.diagnosis);
              renderLabMetaBar(lastMeta, `失败 HTTP ${ev.code || 500}`, "danger");
              $("labSpeedPill").className = "pill danger";
              $("labSpeedPill").textContent = `失败 (HTTP ${ev.code || 500})`;
            } else if (ev.type === "done") {
              lastMeta = ev.meta || lastMeta;
              renderLabMetaBar(lastMeta, "200 OK", "success");
              renderLabContentWithImages(accBody, false, lastMeta);
              if (ev.timing) {
                updateLabTiming(ev.timing.conn, ev.timing.think, ev.timing.gen, ev.timing.total, ev.timing.tps);
              }
              renderLabTokens(ev.usage, ev.finish_reason);
              showToast("连通实测完成");
            }
          }
        }
      }
      if (accBody && streamOut.querySelector(".typing-cursor")) {
        renderLabContentWithImages(accBody, false, lastMeta);
      }
    } catch (e) {
      if (e && e.name === "AbortError") {
        renderLabContentWithImages(accBody ? (accBody + "\n\n[已手动中止]") : "[已手动中止]", false, lastMeta);
        renderLabMetaBar(lastMeta, "已中止", "warning");
        $("labSpeedPill").className = "pill warning";
        $("labSpeedPill").textContent = "已中止";
      } else {
        streamOut.style.display = "none";
        renderLabErrorCard(0, e && e.message ? e.message : String(e), {
          code: 0,
          title: "网络连接异常",
          hint: "浏览器与本服务之间的连接被中断，请确认服务仍在运行。",
          message: e && e.message ? e.message : String(e)
        });
        renderLabMetaBar(lastMeta, "连接异常", "danger");
        $("labSpeedPill").className = "pill danger";
        $("labSpeedPill").textContent = "连接异常";
      }
    } finally {
      labAbortCtrl = null;
      setLabBtnRunning(false);
      fetchStats();
      fetchRecords();
      loadExpressKeysQuiet();
      loadRuntime();
    }
  };

  /* =====================================================================
     6. PERSISTENT CALL RECORDS & LIVE SSE TERMINAL
     ===================================================================== */
  let CALL_RECORDS = [];
  let recFilter = "all";
  let openRecIds = {};

  let logsSubMode = "records";

  function syncMobileTermLock() {
    const lock = currentTab === "logs" && logsSubMode === "terminal";
    document.body.classList.toggle("mobile-term-open", lock);
  }

  window.setLogsSub = function (sub) {
    logsSubMode = sub === "terminal" ? "terminal" : "records";
    const isRec = logsSubMode === "records";
    $("logsSubRecBtn").classList.toggle("active", isRec);
    $("logsSubTermBtn").classList.toggle("active", !isRec);
    $("logsSubRecords").style.display = isRec ? "block" : "none";
    $("logsSubTerminal").style.display = isRec ? "none" : "flex";
    $("logsSubTerminal").style.flexDirection = "column";
    syncMobileTermLock();
    if (!isRec) {
      window.renderTerminal();
      window.scrollTo({ top: 0, behavior: "instant" });
      if (termAutoscroll) scrollTerminalBottom();
    }
  };

  window.setRecFilter = function (f) {
    recFilter = f;
    document.querySelectorAll("[data-recfilter]").forEach((b) => b.classList.toggle("active", b.dataset.recfilter === f));
    renderCallRecords();
  };

  window.toggleRecordOpen = function (id) {
    openRecIds[id] = !openRecIds[id];
    renderCallRecords();
  };

  function renderRecordFinishPill(r) {
    const b = r.badge_type;
    const rt = Number(r.retries || 0);
    const rawFr = String(r.raw_finish_reason || "").toUpperCase();
    if (b === "ok") return `<span class="pill success">STOP</span>`;
    if (b === "retry_ok") return `<span class="pill warning">成功 · 重试×${rt}</span>`;
    if (b === "trunc") {
      if (rawFr === "MALFORMED_FUNCTION_CALL" || rawFr === "MALFORMED_RESPONSE" || rawFr === "UNEXPECTED_TOOL_CALL") {
        return `<span class="pill danger">${escHtml(rawFr)}</span>`;
      }
      if (rawFr === "MAX_TOKENS") {
        return `<span class="pill warning">MAX_TOKENS 达上限</span>`;
      }
      return `<span class="pill warning">截断 / 达上限</span>`;
    }
    if (b === "cancel") return `<span class="pill">已取消 (499)</span>`;
    return `<span class="pill danger">失败 (${Number(r.code || 500)})${rt > 0 ? ` · 重试×${rt}` : ""}</span>`;
  }

  function renderCallRecords() {
    const box = $("recordList");
    const list = CALL_RECORDS.filter((r) => recFilter === "all" || r.group === recFilter);
    if (!list.length) {
      box.innerHTML = `<div class="empty-state">${CALL_RECORDS.length === 0
        ? "暂无调用记录。发起一次请求后，此处将持久化展示模型、缓存命中率、耗时分段与 Token 明细（重启不丢）。"
        : "当前筛选条件下没有匹配的调用记录。"}</div>`;
      return;
    }

    box.innerHTML = list.map((r) => {
      const rid = Number(r.id || 0);
      const isOpen = Boolean(openRecIds[rid]);
      const pTk = Number(r.p_tk || 0);
      const cachedTk = Number(r.cached_tk || 0);
      const thTk = Number(r.th_tk || 0);
      const outTk = Number(r.out_tk || 0);
      const totTk = pTk + thTk + outTk;
      const cachePct = Number(r.cache_pct || (pTk > 0 && cachedTk > 0 ? ((cachedTk / pTk) * 100) : 0));
      const dotColor = r.group === "ok" ? "var(--success)" : (r.group === "cancel" ? "var(--text-muted)" : "var(--danger)");

      const cachePill = cachePct > 0
        ? `<span class="pill primary num"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>缓存 ${cachePct.toFixed(0)}%</span>`
        : "";

      const tags = [];
      if (r.sandbox) tags.push("抗截断");
      if (r.prefill) tags.push("预填充");
      if (r.search) tags.push("联网");
      if (r.tool_calls) tags.push("工具");
      const tagsHtml = tags.length ? `<span class="record-tags">${tags.map((t) => `<span class="mini-tag">${t}</span>`).join("")}</span>` : "";

      const totalT = Math.max(0.01, Number(r.total_time || r.gen_time || 0.01));
      const connT = Number(r.conn_time != null ? r.conn_time : (r.ttft != null ? r.ttft : totalT * 0.15));
      const thinkT = Number(r.think_time || 0);
      const bodyT = Number(r.body_time != null ? r.body_time : Math.max(0.01, totalT - connT - thinkT));
      const tSum = Math.max(0.01, connT + thinkT + bodyT);

      return `
        <details class="record-item" ${isOpen ? "open" : ""} ontoggle="onRecordToggle(${rid}, this.open)">
          <summary class="record-head">
            <div class="record-left">
              <span class="dot" style="background:${dotColor}"></span>
              <span class="mono" style="font-size:12px;color:var(--text-muted);" title="${escHtml(r.full_time || "")}">${escHtml(r.time || "")}</span>
              <strong class="mono" style="font-size:13px;">${escHtml(r.model || "")}</strong>
              <span class="pill">${escHtml(r.cred || "—")}</span>
              ${cachePill}
              ${renderRecordFinishPill(r)}
              ${tagsHtml}
            </div>
            <div class="record-right num">
              <span>${totalT.toFixed(2)}s</span>
              <span>${totTk > 0 ? fmtTok(totTk) + " tk" : (r.out_chars > 0 ? fmtNum(r.out_chars) + " 字" : "—")}</span>
              <svg class="record-chevron" viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>
            </div>
          </summary>
          <div class="record-detail">
            <div class="timing-box" style="margin-top:10px;">
              <div class="timing-head">
                <span>${escHtml(r.full_time || r.time || "")} · ${escHtml(r.mode || "")} · 思考 ${escHtml(r.thinking || "—")}</span>
                <span class="num">${r.tps ? `${r.tps_approx ? "≈" : ""}${Number(r.tps).toFixed(1)} tok/s` : `总耗时 ${totalT.toFixed(2)}s`}</span>
              </div>
              <div class="timing-bar">
                <div class="t-conn" style="width:${((connT / tSum) * 100).toFixed(1)}%"></div>
                ${thinkT > 0 ? `<div class="t-think" style="width:${((thinkT / tSum) * 100).toFixed(1)}%"></div>` : ""}
                ${bodyT > 0 ? `<div class="t-gen" style="width:${((bodyT / tSum) * 100).toFixed(1)}%"></div>` : ""}
              </div>
              <div class="timing-legend num">
                <span><i class="dot" style="background:var(--warning)"></i> 连接/首字 ${connT.toFixed(2)}s</span>
                <span><i class="dot" style="background:var(--think)"></i> 思考 ${thinkT.toFixed(2)}s</span>
                <span><i class="dot" style="background:var(--accent)"></i> 吐字 ${bodyT.toFixed(2)}s</span>
                ${r.has_usage ? `<span style="margin-left:auto;">入 ${fmtNum(pTk)}${cachedTk > 0 ? ` (缓 ${fmtNum(cachedTk)})` : ""} · 思 ${fmtNum(thTk)} · 正 ${fmtNum(outTk)}</span>` : ""}
              </div>
            </div>
            ${r.detail ? `<div class="record-detail-text">${r.detail}</div>` : ""}
          </div>
        </details>
      `;
    }).join("");
  }

  window.onRecordToggle = function (rid, isOpen) {
    if (isOpen) openRecIds[rid] = true;
    else delete openRecIds[rid];
  };

  async function fetchRecords() {
    try {
      const r = await apiFetch("/api/records");
      if (!r.ok) return;
      const d = await r.json();
      lastRecordsVersion = typeof d.version === "number" ? d.version : lastRecordsVersion;
      CALL_RECORDS = Array.isArray(d.records) ? d.records : [];
      const tot = Number(d.total || 0);
      const okc = Number(d.ok_count || 0);
      const errc = Number(d.err_count || 0);
      $("navRecBadge").textContent = String(tot);
      $("fltRecAll").textContent = `全部 (${tot})`;
      $("fltRecOk").textContent = `成功 (${okc})`;
      $("fltRecErr").textContent = `失败 (${errc})`;
      $("recAvgTtft").textContent = d.avg_ttft != null ? `平均首字 ${Number(d.avg_ttft).toFixed(2)}s` : "平均首字 —";
      $("recAvgTps").textContent = d.avg_tps != null ? `平均速度 ${Number(d.avg_tps).toFixed(1)} tok/s` : "平均速度 —";
      $("recAvgCache").textContent = d.avg_cache_pct != null ? `平均缓存 ${Number(d.avg_cache_pct).toFixed(1)}%` : "平均缓存 —";
      renderCallRecords();
    } catch (e) {}
  }

  window.clearCallRecords = async function () {
    try {
      const r = await apiFetch("/api/records/clear", { method: "POST" });
      if (r.ok) {
        openRecIds = {};
        await fetchRecords();
        showToast("已清空调用记录明细（历史聚合统计保持留存）");
      }
    } catch (e) {
      showToast("清空失败");
    }
  };

  /* ── Live SSE Terminal ── */
  let termAutoscroll = true;
  let termProgScroll = false;
  let LOG_RAW_LINES = [];

  function setTermAutoscroll(on) {
    termAutoscroll = Boolean(on);
    if ($("termPauseBtn")) {
      $("termPauseBtn").textContent = termAutoscroll ? "自动置底：开" : "自动置底：关";
    }
  }

  function scrollTerminalBottom() {
    const box = $("terminalBox");
    if (!box) return;
    requestAnimationFrame(() => {
      termProgScroll = true;
      box.scrollTop = box.scrollHeight;
      requestAnimationFrame(() => { termProgScroll = false; });
    });
  }

  if ($("terminalBox")) {
    $("terminalBox").addEventListener("scroll", function () {
      if (termProgScroll) return;
      const box = this;
      const distFromBottom = box.scrollHeight - box.scrollTop - box.clientHeight;
      if (distFromBottom > 40 && termAutoscroll) {
        setTermAutoscroll(false);
      } else if (distFromBottom <= 10 && !termAutoscroll) {
        setTermAutoscroll(true);
      }
    }, { passive: true });
  }

  function classifyLogLevel(line) {
    if (line.includes("❌") || line.includes("🔴") || line.includes("ERROR")) return "err";
    if (line.includes("⚠️") || line.includes("🔄") || line.includes("🛑") || line.includes("WARN") || line.includes("重试")) return "warn";
    if (line.includes("✅") || line.includes("🎉") || line.includes("💰")) return "ok";
    return "info";
  }

  function highlightTerminalMatches(box, query) {
    if (typeof CSS === "undefined" || !CSS.highlights || typeof Highlight !== "function") return;
    CSS.highlights.delete("term-match");
    if (!query || !box) return;
    const ranges = [];
    const walker = document.createTreeWalker(box, NodeFilter.SHOW_TEXT);
    const qLen = query.length;
    let node = walker.nextNode();
    while (node && ranges.length < 400) {
      const text = (node.nodeValue || "").toLowerCase();
      let pos = 0;
      while (pos < text.length && ranges.length < 400) {
        const idx = text.indexOf(query, pos);
        if (idx === -1) break;
        const r = new Range();
        r.setStart(node, idx);
        r.setEnd(node, idx + qLen);
        ranges.push(r);
        pos = idx + qLen;
      }
      node = walker.nextNode();
    }
    if (ranges.length) {
      CSS.highlights.set("term-match", new Highlight(...ranges));
    }
  }

  window.renderTerminal = function () {
    const box = $("terminalBox");
    if (!box) return;
    const q = ($("termSearch") ? $("termSearch").value : "").trim().toLowerCase();
    const filtered = [];
    LOG_RAW_LINES.forEach((raw, idx) => {
      if (!q || raw.toLowerCase().includes(q)) filtered.push({ raw, idx });
    });
    if (!filtered.length) {
      if (typeof CSS !== "undefined" && CSS.highlights) CSS.highlights.delete("term-match");
      box.innerHTML = `<div style="color:var(--terminal-muted);padding:12px;">${LOG_RAW_LINES.length ? "没有匹配的日志行。" : "等待实时日志流接入…"}</div>`;
      return;
    }
    box.innerHTML = filtered.map(({ raw, idx }) => {
      const lv = classifyLogLevel(raw);
      const m = raw.match(/^(\[\d{2}:\d{2}:\d{2}\])\s*([\s\S]*)$/);
      const ts = m ? m[1] : "";
      const body = (m ? m[2] : raw).replace(/\r?\n+$/, "");
      return `<div class="log-line" onclick="copySingleLog(${idx})" title="点击复制本条日志">`
        + (ts ? `<span class="log-ts">${escHtml(ts)}</span>` : "")
        + `<span class="log-lv-${lv}">${lv.toUpperCase()}</span>`
        + `<span>${escHtml(body)}</span></div>`;
    }).join("");
    highlightTerminalMatches(box, q);
    if (termAutoscroll && $("logsSubTerminal").style.display !== "none") scrollTerminalBottom();
  };

  window.copySingleLog = async function (idx) {
    const sel = window.getSelection ? window.getSelection().toString() : "";
    if (sel && sel.length > 0) return;
    const raw = LOG_RAW_LINES[idx] || "";
    if (!raw) return;
    try {
      await navigator.clipboard.writeText(raw);
      showToast("已复制该条日志");
    } catch (e) { showToast("复制失败，请手动选中复制"); }
  };

  window.copyAllLogs = async function () {
    if (!LOG_RAW_LINES.length) { showToast("当前没有可复制的日志"); return; }
    try {
      await navigator.clipboard.writeText(LOG_RAW_LINES.join("\n"));
      showToast(`已复制全部 ${LOG_RAW_LINES.length} 条日志`);
    } catch (e) { showToast("复制失败"); }
  };

  window.toggleTermPause = function () {
    setTermAutoscroll(!termAutoscroll);
    if (termAutoscroll) scrollTerminalBottom();
  };

  window.clearTerminal = function () {
    LOG_RAW_LINES = [];
    window.renderTerminal();
    showToast("已清空当前终端面板");
  };

  let termRafPending = false;

  function scheduleTerminalRender() {
    if (termRafPending) return;
    termRafPending = true;
    requestAnimationFrame(() => {
      termRafPending = false;
      if (currentTab === "logs" && logsSubMode === "terminal") {
        window.renderTerminal();
      }
    });
  }

  function connectLogSSE() {
    try {
      const es = new EventSource("/stream-logs");
      es.onmessage = function (e) {
        if (!e.data || e.data.includes("keep-alive")) return;
        if (LOG_RAW_LINES.length >= 250) LOG_RAW_LINES.shift();
        LOG_RAW_LINES.push(e.data);
        scheduleTerminalRender();
      };
    } catch (e) {}
  }

  /* =====================================================================
     7. NAVIGATION, COPY /V1, LOGOUT & POLLING LOOP
     ===================================================================== */
  const TAB_ORDER = { overview: 0, channel: 1, params: 2, logs: 3 };

  function applyTab(tab) {
    const prevTab = currentTab;
    const dirCls = (TAB_ORDER[tab] || 0) >= (TAB_ORDER[prevTab] || 0) ? "tab-enter-forward" : "tab-enter-backward";
    currentTab = tab;
    document.querySelectorAll(".tab-panel").forEach((p) => {
      p.classList.remove("active", "tab-enter-forward", "tab-enter-backward");
    });
    const nextPanel = $("panel-" + tab);
    if (nextPanel) {
      nextPanel.classList.add("active");
      if (prevTab !== tab) nextPanel.classList.add(dirCls);
    }
    document.querySelectorAll("[data-tab]").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
    refreshDirtyState();
    syncMobileTermLock();
    if (tab === "overview" && overviewDirty && lastStatsPayload) {
      renderOverview(lastStatsPayload);
      overviewDirty = false;
    } else if (tab === "logs") {
      fetchRecords();
      if (logsSubMode === "terminal") {
        window.renderTerminal();
        if (termAutoscroll) scrollTerminalBottom();
      }
    } else if (tab === "channel") {
      loadExpressKeysQuiet();
      loadRuntime();
      if ($("channelSubTest") && $("channelSubTest").style.display !== "none") {
        refreshLabInspector();
      }
    }
    window.scrollTo({ top: 0, behavior: "instant" });
  }

  /* ── Desktop Dark-Theme Pointer Spotlight Sheen (rAF throttled, touch-free) ── */
  if (window.matchMedia && window.matchMedia("(hover: hover) and (pointer: fine)").matches) {
    let spotRaf = 0;
    let spotCard = null;
    let spotX = 0;
    let spotY = 0;
    document.addEventListener("pointermove", (e) => {
      if (currentTheme !== "vinyl-night" && currentTheme !== "lacquer-night" && currentTheme !== "star-night") return;
      const card = e.target && e.target.closest ? e.target.closest(".glass-card") : null;
      if (!card) return;
      const rect = card.getBoundingClientRect();
      spotCard = card;
      spotX = e.clientX - rect.left;
      spotY = e.clientY - rect.top;
      if (!spotRaf) {
        spotRaf = requestAnimationFrame(() => {
          spotRaf = 0;
          if (spotCard) {
            spotCard.style.setProperty("--mx", spotX.toFixed(1) + "px");
            spotCard.style.setProperty("--my", spotY.toFixed(1) + "px");
          }
        });
      }
    }, { passive: true });
  }

  function switchTab(tab) {
    if (tab === currentTab) return;
    if (currentTab === "params" && countDirtyDiffs() > 0) {
      pendingAction = () => applyTab(tab);
      openDirtyModal();
      return;
    }
    applyTab(tab);
  }

  document.querySelectorAll("[data-tab]").forEach((btn) => {
    btn.addEventListener("click", () => switchTab(btn.dataset.tab));
  });

  async function copyEndpoint() {
    const url = window.location.origin + "/v1";
    try {
      await navigator.clipboard.writeText(url);
      showToast("已复制接口地址：" + url);
    } catch (e) {
      showToast("接口地址：" + url);
    }
  }
  if ($("sideCopyBtn")) $("sideCopyBtn").addEventListener("click", copyEndpoint);
  if ($("mobCopyBtn")) $("mobCopyBtn").addEventListener("click", copyEndpoint);

  window.logout = async function () {
    try { await fetch("/api/logout", { method: "POST" }); } catch (e) {}
    location.href = "/";
  };

  /* ── Bootstrap ── */
  syncSfxUi();
  renderThemePickers();
  fetchStats();
  fetchLive();
  fetchRecords();
  loadRuntime();
  loadExpressKeys();
  loadParamsAndCaps();
  connectLogSSE();

  setInterval(() => {
    if (!document.hidden) fetchStats();
  }, 4000);

  setInterval(() => {
    if (!document.hidden) fetchLive();
  }, 1200);

  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) {
      fetchLive();
      fetchStats();
    }
  });
})();
