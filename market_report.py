import os
import math
import smtplib
from datetime import datetime
from zoneinfo import ZoneInfo
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

import pandas as pd
import yfinance as yf

TZ = ZoneInfo("Europe/Madrid")

# ==============================
# UNIVERSO (ETFs PROXY robustos)
# ==============================
INDICES = {
    "S&P 500 (SPY)": "SPY",
    "Nasdaq (QQQ)": "QQQ",
    "Dow (DIA)": "DIA",
    "Russell 2000 (IWM)": "IWM",
}

SECTORS = {
    "Technology (XLK)": "XLK",
    "Financials (XLF)": "XLF",
    "Energy (XLE)": "XLE",
    "Health Care (XLV)": "XLV",
    "Industrials (XLI)": "XLI",
    "Cons Disc (XLY)": "XLY",
    "Cons Staples (XLP)": "XLP",
    "Utilities (XLU)": "XLU",
    "Materials (XLB)": "XLB",
    "Comm (XLC)": "XLC",
}

FACTORS = {
    "SPY": "SPY",
    "QQQ": "QQQ",
    "IWM": "IWM",
    "HYG": "HYG",
    "LQD": "LQD",
    "MTUM": "MTUM",
    "SPLV": "SPLV",
}

VIX_TICKER = "VIXY"  # proxy VIX (más fiable que ^VIX en CI)

# ==============================
# UTILS
# ==============================
def _scalar(x):
    """Convierte Series/numpy scalars a float python."""
    try:
        if hasattr(x, "iloc"):
            x = x.iloc[-1]
    except Exception:
        pass
    try:
        return None if x is None else float(x)
    except Exception:
        return None


def pct(a, b):
    a = _scalar(a)
    b = _scalar(b)
    if a is None or b is None or b == 0:
        return None
    return (a / b - 1.0) * 100.0


def fmt_pct(x, nd=1):
    if x is None:
        return "n/a"
    return f"{x:+.{nd}f}%"


def semaforo(score):
    if score is None:
        return "⚪"
    if score >= 70:
        return "🟢"
    if score >= 40:
        return "🟡"
    return "🔴"


# ==============================
# DESCARGA DATOS (robusta)
# ==============================
def download_series(ticker: str) -> pd.DataFrame | None:
    """
    Descarga con yfinance. En GitHub Actions a veces Yahoo rate-limit;
    este método es simple pero estable con ETFs.
    """
    try:
        df = yf.download(
            ticker,
            period="650d",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False,
            group_by="column",
        )
        if df is None or df.empty or "Close" not in df.columns:
            return None
        return df
    except Exception:
        return None


# ==============================
# ANALISIS
# ==============================
def analyze(ticker: str) -> dict | None:
    df = download_series(ticker)
    if df is None or df.empty:
        return None

    closes = df["Close"].dropna()
    if closes.empty or len(closes) < 2:
        return None

    close = _scalar(closes.iloc[-1])
    prev = _scalar(closes.iloc[-2])

    ma50 = _scalar(closes.rolling(50).mean().iloc[-1]) if len(closes) >= 50 else None
    ma200 = _scalar(closes.rolling(200).mean().iloc[-1]) if len(closes) >= 200 else None

    ret1d = pct(close, prev)

    ret1w = None
    if len(closes) >= 6:
        ret1w = pct(close, _scalar(closes.iloc[-6]))

    ret1m = None
    if len(closes) >= 22:
        ret1m = pct(close, _scalar(closes.iloc[-22]))

    dist200 = pct(close, ma200)
    dist50 = pct(close, ma50)

    above200 = (ma200 is not None and close > ma200)

    return {
        "close": close,
        "ret1d": ret1d,
        "ret1w": ret1w,
        "ret1m": ret1m,
        "dist200": dist200,
        "dist50": dist50,
        "above200": above200 if ma200 is not None else None,
    }


# ==============================
# BREADTH
# ==============================
def breadth(indices_stats: dict, sector_stats: dict):
    idx = [v for v in indices_stats.values() if v and v.get("above200") is not None]
    sec = [v for v in sector_stats.values() if v and v.get("above200") is not None]

    pct_idx = 100.0 * sum(1 for x in idx if x["above200"]) / max(1, len(idx))
    pct_sec = 100.0 * sum(1 for x in sec if x["above200"]) / max(1, len(sec))

    return pct_idx, pct_sec


# ==============================
# MARKET SCORE (0–100) + overrides simples
# ==============================
def market_score(indices_stats: dict, sector_stats: dict, vix_level: float | None):
    pct_idx, pct_sec = breadth(indices_stats, sector_stats)

    # base
    score = 0.50 * pct_idx + 0.50 * pct_sec

    # overrides (B): penalizar estrés de volatilidad
    if vix_level is not None:
        if vix_level >= 30:
            score -= 25
        elif vix_level >= 25:
            score -= 12

    # clamp
    score = max(0.0, min(100.0, score))

    label = "Risk-on" if score >= 70 else ("Neutral" if score >= 40 else "Risk-off")
    return score, label, {"pct_idx": pct_idx, "pct_sec": pct_sec}


# ==============================
# EMAIL
# ==============================
def send_email(subject: str, body: str):
    host = os.environ.get("SMTP_HOST")
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASS")
    mail_from = os.environ.get("MAIL_FROM", user)
    mail_to = os.environ.get("MAIL_TO")

    if not all([host, user, password, mail_to]):
        raise RuntimeError("Faltan secrets SMTP_HOST/SMTP_USER/SMTP_PASS/MAIL_TO (y opcional MAIL_FROM).")

    recipients = [x.strip() for x in mail_to.split(",") if x.strip()]

    msg = MIMEMultipart()
    msg["From"] = mail_from
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))

    with smtplib.SMTP(host, port, timeout=30) as server:
        server.starttls()
        server.login(user, password)
        server.sendmail(mail_from, recipients, msg.as_string())


# ==============================
# MAIN
# ==============================
def main():
    today = datetime.now(TZ).date().isoformat()

    indices_stats = {}
    sector_stats = {}
    factor_stats = {}
    failed = []

    # Índices (ETFs)
    for name, ticker in INDICES.items():
        r = analyze(ticker)
        indices_stats[name] = r
        if r is None:
            failed.append(ticker)

    # Sectores
    for name, ticker in SECTORS.items():
        r = analyze(ticker)
        sector_stats[name] = r
        if r is None:
            failed.append(ticker)

    # Factores (para futuras mejoras; ahora los descargamos por health)
    for name, ticker in FACTORS.items():
        r = analyze(ticker)
        factor_stats[name] = r
        if r is None:
            failed.append(ticker)

    # VIX proxy
    vix_data = analyze(VIX_TICKER)
    if vix_data is None:
        failed.append(VIX_TICKER)
    vix_level = vix_data["close"] if vix_data else None

    score, label, meta = market_score(indices_stats, sector_stats, vix_level)
    icon = semaforo(score)

    # Ranking sectores por 1W
    sector_list = []
    for name, stats in sector_stats.items():
        if stats and stats.get("ret1w") is not None:
            sector_list.append((name, stats["ret1w"]))

    sector_list.sort(key=lambda x: x[1])
    bottom3 = sector_list[:3]
    top3 = sector_list[-3:][::-1]

    # Email
    lines = []
    lines.append(f"MARKET BRIEF — {today}")
    lines.append("")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append("EXECUTIVE SNAPSHOT")
    lines.append(f"Market Regime Score: {icon} {score:.0f}/100 ({label})")
    lines.append(f"Breadth: Índices>MA200 {meta['pct_idx']:.0f}% | Sectores>MA200 {meta['pct_sec']:.0f}%")
    lines.append(f"VIX (proxy VIXY): {('n/a' if vix_level is None else f'{vix_level:.2f}')}")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append("")

    lines.append("MARKET STRUCTURE (ETFs proxy)")
    for name, stats in indices_stats.items():
        if not stats:
            lines.append(f"- {name}: n/a")
            continue
        lines.append(
            f"- {name}: {fmt_pct(stats['ret1d'])} 1D | {fmt_pct(stats['ret1w'])} 1W | {fmt_pct(stats['ret1m'])} 1M | "
            f"MA200 {fmt_pct(stats['dist200'])} | MA50 {fmt_pct(stats['dist50'])}"
        )

    lines.append("")
    lines.append("SECTOR ROTATION (1W)")
    if top3:
        lines.append("- Top 3: " + ", ".join([f"{n} {fmt_pct(r)}" for n, r in top3]))
    else:
        lines.append("- Top 3: n/a")
    if bottom3:
        lines.append("- Bottom 3: " + ", ".join([f"{n} {fmt_pct(r)}" for n, r in bottom3]))
    else:
        lines.append("- Bottom 3: n/a")

    lines.append("")
    lines.append("FACTOR FLOWS (download check)")
    # De momento solo confirmamos si hay datos, para ir subiendo nivel luego
    ok_factors = sum(1 for v in factor_stats.values() if v is not None)
    lines.append(f"- Factors OK: {ok_factors}/{len(FACTORS)}")

    lines.append("")
    lines.append("OPERATIONAL CONCLUSION")
    if label == "Risk-on":
        lines.append("- Sesgo constructivo. Mantener/elevar exposición y priorizar líderes.")
    elif label == "Neutral":
        lines.append("- Mercado mixto. Ser selectivo: setups fuertes y evitar debilidad.")
    else:
        lines.append("- Sesgo defensivo. Priorizar protección de capital y reducir riesgo.")

    lines.append("")
    lines.append("DATA HEALTH")
    ok_total = (len(INDICES) + len(SECTORS) + len(FACTORS) + 1) - len(set(failed))
    total = (len(INDICES) + len(SECTORS) + len(FACTORS) + 1)
    lines.append(f"- OK: {ok_total}/{total} | FAIL: {len(set(failed))}")
    if failed:
        lines.append("- Fail tickers: " + ", ".join(sorted(set(failed))[:20]))

    body = "\n".join(lines)
    subject = f"Market Brief — {today} — {icon} {score:.0f}/100 ({label})"

    send_email(subject, body)


if __name__ == "__main__":
    main()
