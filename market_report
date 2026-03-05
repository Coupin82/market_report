import os
import math
from dataclasses import dataclass
from datetime import datetime, date
from zoneinfo import ZoneInfo
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import pandas as pd
import yfinance as yf

TZ = ZoneInfo("Europe/Madrid")

# ----------------------------
# Universe
# ----------------------------
INDICES = {
    "S&P 500": "^GSPC",
    "Nasdaq": "^IXIC",
    "Dow": "^DJI",
    "Russell 2000": "^RUT",
}

# Sector ETFs (SPDR Select Sector)
SECTORS = {
    "XLK (Tech)": "XLK",
    "XLF (Financials)": "XLF",
    "XLE (Energy)": "XLE",
    "XLV (Health Care)": "XLV",
    "XLI (Industrials)": "XLI",
    "XLY (Cons Disc)": "XLY",
    "XLP (Cons Staples)": "XLP",
    "XLU (Utilities)": "XLU",
    "XLB (Materials)": "XLB",
    "XLC (Comm)": "XLC",
    # (XLRE Real Estate) opcional
}

# Factors / risk proxies
FACTORS = {
    "SPY": "SPY",
    "QQQ": "QQQ",
    "IWM": "IWM",
    "HYG": "HYG",
    "LQD": "LQD",
    "MTUM": "MTUM",
    "SPLV": "SPLV",
}

VIX_TICKER = "^VIX"

# Upcoming events (manual “core calendar”; tú lo afinas cuando quieras)
# Formato: ("Nombre", "YYYY-MM-DD")
UPCOMING_EVENTS = [
    # Ejemplos (pon aquí tus fechas reales cuando quieras)
    # ("CPI (US)", "2026-03-12"),
    # ("FOMC", "2026-03-18"),
    # ("NFP", "2026-03-06"),
    # ("OPEX (Monthly)", "2026-03-20"),
]

# ----------------------------
# Helpers
# ----------------------------
def _to_float(x):
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None

def pct(a, b):
    if a is None or b in (None, 0):
        return None
    return (a / b - 1.0) * 100.0

def fmt_pct(x, nd=1, signed=True):
    if x is None:
        return "n/a"
    return f"{x:+.{nd}f}%" if signed else f"{x:.{nd}f}%"

def clamp(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, x))

def yfdl(ticker, period="400d"):
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
    if df is None or df.empty or "Close" not in df:
        return None
    return df

def compute_ma(series: pd.Series, n: int):
    if series is None or len(series) < n:
        return None
    return _to_float(series.rolling(n).mean().iloc[-1])

def compute_ret(series: pd.Series, n: int):
    if series is None or len(series) < n + 1:
        return None
    last = _to_float(series.iloc[-1])
    prev = _to_float(series.iloc[-(n+1)])
    if last is None or prev in (None, 0):
        return None
    return (last / prev - 1.0) * 100.0

def vix_regime(vix: float | None):
    if vix is None:
        return ("n/a", "n/a")
    if vix < 18:
        return ("Bajo", "Complacencia")
    if vix < 25:
        return ("Normal", "Rango habitual")
    if vix < 35:
        return ("Alto", "Estrés")
    return ("Extremo", "Crisis")

def semaforo(score: float):
    if score >= 70:
        return "🟢"
    if score >= 40:
        return "🟡"
    return "🔴"

def safe_rank(items, key, reverse=True):
    return sorted(items, key=lambda x: (x.get(key) is not None, x.get(key)), reverse=reverse)

# ----------------------------
# Data models
# ----------------------------
@dataclass
class InstrumentStats:
    name: str
    ticker: str
    close: float | None
    ret_1d: float | None
    ret_1w: float | None
    ret_1m: float | None
    dist_ma50: float | None
    dist_ma200: float | None
    above_ma200: bool | None
    above_ma50: bool | None

def analyze_instrument(name: str, ticker: str) -> InstrumentStats | None:
    df = yfdl(ticker, period="650d")
    if df is None:
        return None
    c = _to_float(df["Close"].iloc[-1])
    if c is None:
        return None
    prev = _to_float(df["Close"].iloc[-2]) if len(df) >= 2 else None

    ma50 = compute_ma(df["Close"], 50)
    ma200 = compute_ma(df["Close"], 200)

    dist50 = pct(c, ma50)
    dist200 = pct(c, ma200)

    r1d = (c / prev - 1.0) * 100.0 if prev not in (None, 0) else None
    r1w = compute_ret(df["Close"], 5)
    r1m = compute_ret(df["Close"], 21)

    above200 = (ma200 is not None and c > ma200)
    above50 = (ma50 is not None and c > ma50)

    return InstrumentStats(
        name=name, ticker=ticker,
        close=c, ret_1d=r1d, ret_1w=r1w, ret_1m=r1m,
        dist_ma50=dist50, dist_ma200=dist200,
        above_ma200=above200 if ma200 is not None else None,
        above_ma50=above50 if ma50 is not None else None,
    )

# ----------------------------
# Breadth proxy (sin pagar)
# ----------------------------
def breadth_proxy(indices_stats: dict, sector_stats: dict):
    """
    No podemos calcular % de SP500 componentes > MA200 sin dataset.
    Proxy profesional "good enough":
    - breadth_index: % de índices principales por encima de MA200 (4 índices)
    - breadth_sector: % de sectores por encima de MA200 (10 sectores)
    - confirm: Russell vs Nasdaq (amplitud vs concentración)
    """
    idx_list = [s for s in indices_stats.values() if s and s.above_ma200 is not None]
    sec_list = [s for s in sector_stats.values() if s and s.above_ma200 is not None]

    idx_above = sum(1 for s in idx_list if s.above_ma200)
    sec_above = sum(1 for s in sec_list if s.above_ma200)

    breadth_idx = 100.0 * idx_above / max(1, len(idx_list))
    breadth_sec = 100.0 * sec_above / max(1, len(sec_list))

    # confirmation: small caps participation
    nas = indices_stats.get("Nasdaq")
    rut = indices_stats.get("Russell 2000")
    confirmation = "n/a"
    if nas and rut and nas.ret_1m is not None and rut.ret_1m is not None:
        if rut.ret_1m >= nas.ret_1m - 0.5:
            confirmation = "Confirmación (amplitud OK)"
        else:
            confirmation = "Divergencia (rally estrecho)"

    return breadth_idx, breadth_sec, confirmation

# ----------------------------
# Factor flows
# ----------------------------
def ratio_strength(a: InstrumentStats | None, b: InstrumentStats | None):
    """
    Fuerza relativa simple: diferencia de retornos 1M como proxy.
    """
    if not a or not b:
        return None
    if a.ret_1m is None or b.ret_1m is None:
        return None
    return a.ret_1m - b.ret_1m

# ----------------------------
# Market Regime Score (0–100) + Overrides
# ----------------------------
def compute_market_score(indices_stats: dict, sector_stats: dict, vix_level: float | None, factors_stats: dict):
    # Components (0..100 each)
    # 1) Indices trend breadth (25)
    idx_list = [s for s in indices_stats.values() if s and s.above_ma200 is not None]
    pct_idx_above200 = 100.0 * sum(1 for s in idx_list if s.above_ma200) / max(1, len(idx_list))

    # 2) Sector trend breadth (15)
    sec_list = [s for s in sector_stats.values() if s and s.above_ma200 is not None]
    pct_sec_above200 = 100.0 * sum(1 for s in sec_list if s.above_ma200) / max(1, len(sec_list))

    # 3) Breadth proxy (20): blend of idx+sector
    breadth = 0.5 * pct_idx_above200 + 0.5 * pct_sec_above200

    # 4) VIX regime score (15): lower VIX => higher score
    vix_score = None
    if vix_level is not None:
        # piecewise mapping
        if vix_level < 14: vix_score = 95
        elif vix_level < 18: vix_score = 85
        elif vix_level < 25: vix_score = 65
        elif vix_level < 35: vix_score = 35
        else: vix_score = 15

    # 5) Momentum 1M indices (10): average 1M return mapped
    idx_r1m = [s.ret_1m for s in idx_list if s.ret_1m is not None]
    avg_r1m = sum(idx_r1m)/len(idx_r1m) if idx_r1m else None
    mom_score = None
    if avg_r1m is not None:
        # -6% -> 0, 0% -> 50, +6% -> 100 (clamped)
        mom_score = clamp(50 + (avg_r1m/6.0)*50, 0, 100)

    # 6) Small vs Large (10): IWM vs SPY 1M
    small = ratio_strength(factors_stats.get("IWM"), factors_stats.get("SPY"))
    small_score = None
    if small is not None:
        # -4pp -> 0, 0 -> 50, +4pp -> 100
        small_score = clamp(50 + (small/4.0)*50, 0, 100)

    # 7) Credit risk (5): HYG vs LQD 1M
    credit = ratio_strength(factors_stats.get("HYG"), factors_stats.get("LQD"))
    credit_score = None
    if credit is not None:
        credit_score = clamp(50 + (credit/3.0)*50, 0, 100)

    # Weighted sum
    def _nv(x, default=50):
        return default if x is None else x

    base = (
        0.25 * pct_idx_above200 +
        0.15 * pct_sec_above200 +
        0.20 * breadth +
        0.15 * _nv(vix_score) +
        0.10 * _nv(mom_score) +
        0.10 * _nv(small_score) +
        0.05 * _nv(credit_score)
    )

    # ---------------- Overrides (B) ----------------
    # Hard risk-off conditions
    penalty = 0.0

    # VIX stress penalty
    if vix_level is not None:
        if vix_level >= 35:
            penalty += 25
        elif vix_level >= 25:
            penalty += 12

    # Breadth collapse: if sectors above 200 < 30% => penalty
    if pct_sec_above200 < 30:
        penalty += 10

    # Index trend collapse: if indices above 200 <= 25% => penalty
    if pct_idx_above200 <= 25:
        penalty += 12

    # Credit warning: if HYG underperforms LQD strongly
    if credit is not None and credit <= -2.0:
        penalty += 8

    # Small caps warning: if IWM underperforms SPY strongly
    if small is not None and small <= -3.0:
        penalty += 6

    score = clamp(base - penalty, 0, 100)

    # Regime label
    label = "Risk-on" if score >= 70 else ("Neutral" if score >= 40 else "Risk-off")
    return score, label, {
        "pct_idx_above200": pct_idx_above200,
        "pct_sec_above200": pct_sec_above200,
        "breadth_blend": breadth,
        "vix": vix_level,
        "avg_r1m_idx": avg_r1m,
        "small_vs_large_pp": small,
        "credit_pp": credit,
        "penalty": penalty,
    }

# ----------------------------
# Upcoming events
# ----------------------------
def upcoming_events(today: date, max_items=6):
    events = []
    for name, d in UPCOMING_EVENTS:
        try:
            ed = datetime.fromisoformat(d).date()
        except Exception:
            continue
        delta = (ed - today).days
        if delta >= 0:
            events.append((delta, name, ed))
    events.sort(key=lambda x: x[0])
    return events[:max_items]

# ----------------------------
# Email build
# ----------------------------
def build_email(subject_date: str, score: float, label: str, score_meta: dict,
                indices_stats: dict, sector_stats: dict, factors_stats: dict,
                breadth_idx: float, breadth_sec: float, confirmation: str,
                vix_level: float | None, vix_change: float | None,
                events: list[tuple]):
    semoji = semaforo(score)

    # Executive snapshot key lines
    sp = indices_stats.get("S&P 500")
    nas = indices_stats.get("Nasdaq")
    rut = indices_stats.get("Russell 2000")

    vix_reg, vix_desc = vix_regime(vix_level)

    # Sector winners/losers (1W)
    sectors_list = [s for s in sector_stats.values() if s and s.ret_1w is not None]
    top_sec = sorted(sectors_list, key=lambda x: x.ret_1w, reverse=True)[:3]
    bot_sec = sorted(sectors_list, key=lambda x: x.ret_1w)[:3]

    # Factor summaries
    def line_factor(name, a, b):
        x = ratio_strength(factors_stats.get(a), factors_stats.get(b))
        if x is None:
            return f"- {name}: n/a"
        sign = "+" if x >= 0 else ""
        return f"- {name}: {sign}{x:.1f}pp (1M)"

    # Events line
    if events:
        ev_lines = "\n".join([f"- {name} — en {d} días ({ed.isoformat()})" for d, name, ed in events])
    else:
        ev_lines = "- (Configura UPCOMING_EVENTS en el script)"

    # Compose email (plain text, professional)
    body = []
    body.append(f"MARKET BRIEF — {subject_date}")
    body.append("")
    body.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    body.append("EXECUTIVE SNAPSHOT")
    body.append(f"Market Regime Score: {semoji} {score:.0f}/100 ({label})")
    if sp:
        body.append(f"S&P 500: {fmt_pct(sp.ret_1d)} (1D) | {fmt_pct(sp.ret_1m)} (1M) | {fmt_pct(sp.dist_ma200)} vs MA200")
    if nas and rut:
        body.append(f"Nasdaq vs Russell: {confirmation}")
    if vix_level is not None:
        ch = f"{vix_change:+.1f}" if vix_change is not None else "n/a"
        body.append(f"VIX: {vix_level:.1f} ({vix_reg}) | Δ1D {ch} | {vix_desc}")
    body.append(f"Breadth (proxy): Índices>{'MA200'} {breadth_idx:.0f}% | Sectores>{'MA200'} {breadth_sec:.0f}%")
    body.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    body.append("")
    body.append("MARKET STRUCTURE (Índices)")
    for name, st in indices_stats.items():
        if not st:
            body.append(f"- {name}: n/a")
            continue
        body.append(
            f"- {name}: {fmt_pct(st.ret_1d)} 1D | {fmt_pct(st.ret_1w)} 1W | {fmt_pct(st.ret_1m)} 1M | "
            f"MA200 {fmt_pct(st.dist_ma200)} | MA50 {fmt_pct(st.dist_ma50)}"
        )
    body.append("")
    body.append("VOLATILITY REGIME")
    if vix_level is None:
        body.append("- VIX: n/a")
    else:
        body.append(f"- VIX: {vix_level:.1f} ({vix_reg}) — {vix_desc}")
        if score_meta.get("penalty", 0) > 0:
            body.append(f"- Overrides/penalizaciones aplicadas: -{score_meta['penalty']:.0f} puntos (estrés/breadth/crédito)")
    body.append("")
    body.append("BREADTH & INTERNALS (proxy profesional)")
    body.append(f"- % Índices > MA200: {score_meta['pct_idx_above200']:.0f}%")
    body.append(f"- % Sectores > MA200: {score_meta['pct_sec_above200']:.0f}%")
    body.append(f"- Lectura: {confirmation}")
    body.append("")
    body.append("SECTOR ROTATION (1W)")
    if top_sec:
        body.append("- Top sectores: " + ", ".join([f"{s.name} {fmt_pct(s.ret_1w)}" for s in top_sec]))
    else:
        body.append("- Top sectores: n/a")
    if bot_sec:
        body.append("- Bottom sectores: " + ", ".join([f"{s.name} {fmt_pct(s.ret_1w)}" for s in bot_sec]))
    else:
        body.append("- Bottom sectores: n/a")
    body.append("")
    body.append("FACTOR FLOWS (1M, pp = puntos porcentuales)")
    body.append(line_factor("Small vs Large (IWM - SPY)", "IWM", "SPY"))
    body.append(line_factor("Growth bias (QQQ - SPY)", "QQQ", "SPY"))
    body.append(line_factor("Credit risk (HYG - LQD)", "HYG", "LQD"))
    body.append(line_factor("Momentum vs LowVol (MTUM - SPLV)", "MTUM", "SPLV"))
    body.append("")
    body.append("UPCOMING EVENTS")
    body.append(ev_lines)
    body.append("")
    body.append("OPERATIONAL CONCLUSION (auto)")
    # Simple rule-based conclusion (tight, executive)
    concl = []
    if label == "Risk-on":
        concl.append("- Sesgo: mantener/elevar exposición, priorizar sectores líderes.")
    elif label == "Neutral":
        concl.append("- Sesgo: selectivo; mantener exposición moderada, evitar añadir en debilidad.")
    else:
        concl.append("- Sesgo: defensivo; reducir riesgo y proteger capital.")
    # add nuance
    if vix_level is not None and vix_level >= 25:
        concl.append("- Nota: volatilidad elevada → reducir tamaño / esperar confirmaciones.")
    if score_meta["pct_sec_above200"] < 40:
        concl.append("- Nota: amplitud sectorial débil → rally frágil, exigir setups de alta calidad.")
    if score_meta.get("credit_pp") is not None and score_meta["credit_pp"] <= -2:
        concl.append("- Nota: crédito empeora → cuidado con high beta.")
    body.extend(concl)

    return "\n".join(body), f"Market Brief — {subject_date} — {semoji} {score:.0f}/100 ({label})"

# ----------------------------
# Email send (SMTP)
# ----------------------------
def send_email_smtp(subject: str, body: str):
    smtp_host = os.environ.get("SMTP_HOST")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER")
    smtp_pass = os.environ.get("SMTP_PASS")
    mail_from = os.environ.get("MAIL_FROM", smtp_user)
    mail_to = os.environ.get("MAIL_TO")  # comma separated
    if not all([smtp_host, smtp_user, smtp_pass, mail_to]):
        raise RuntimeError("Faltan variables SMTP_HOST/SMTP_USER/SMTP_PASS/MAIL_TO (y opcional MAIL_FROM).")

    recipients = [x.strip() for x in mail_to.split(",") if x.strip()]
    msg = MIMEMultipart()
    msg["From"] = mail_from
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain", "utf-8"))

    with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.sendmail(mail_from, recipients, msg.as_string())

# ----------------------------
# Main
# ----------------------------
def main():
    today = datetime.now(TZ).date().isoformat()

    # Fetch indices
    indices_stats = {}
    for name, t in INDICES.items():
        indices_stats[name] = analyze_instrument(name, t)

    # VIX
    vix_df = yfdl(VIX_TICKER, period="90d")
    vix_level = None
    vix_change = None
    if vix_df is not None and len(vix_df) >= 2:
        vix_level = _to_float(vix_df["Close"].iloc[-1])
        vix_prev = _to_float(vix_df["Close"].iloc[-2])
        if vix_level is not None and vix_prev is not None:
            vix_change = vix_level - vix_prev

    # Sectors
    sector_stats = {}
    for name, t in SECTORS.items():
        sector_stats[name] = analyze_instrument(name, t)

    # Factors
    factors_stats = {}
    for name, t in FACTORS.items():
        factors_stats[name] = analyze_instrument(name, t)

    # Breadth proxy
    b_idx, b_sec, confirmation = breadth_proxy(indices_stats, sector_stats)

    # Score + overrides
    score, label, meta = compute_market_score(indices_stats, sector_stats, vix_level, factors_stats)

    # Events
    ev = upcoming_events(datetime.now(TZ).date(), max_items=6)

    # Build email
    body, subject = build_email(
        subject_date=today,
        score=score, label=label, score_meta=meta,
        indices_stats=indices_stats,
        sector_stats=sector_stats,
        factors_stats=factors_stats,
        breadth_idx=b_idx, breadth_sec=b_sec, confirmation=confirmation,
        vix_level=vix_level, vix_change=vix_change,
        events=ev
    )

    # Send
    send_email_smtp(subject, body)

if __name__ == "__main__":
    main()
