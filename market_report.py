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
# UNIVERSO
# ==============================

INDICES = {
    "S&P 500": "SPY",
    "Nasdaq": "QQQ",
    "Dow": "DIA",
    "Russell 2000": "IWM",
}

SECTORS = {
    "Technology": "XLK",
    "Financials": "XLF",
    "Energy": "XLE",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Utilities": "XLU",
    "Materials": "XLB",
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

VIX_TICKER = "VIXY"


# ==============================
# UTILS
# ==============================

def pct(a, b):
    if a is None or b in (None, 0):
        return None
    return (a / b - 1) * 100


def fmt(x):
    if x is None:
        return "n/a"
    return f"{x:+.1f}%"


def semaforo(score):

    if score is None:
        return "⚪"

    if score >= 70:
        return "🟢"

    if score >= 40:
        return "🟡"

    return "🔴"


# ==============================
# DESCARGA DATOS
# ==============================

def download_series(ticker):

    try:

        df = yf.download(
            ticker,
            period="650d",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )

        if df.empty:
            return None

        return df

    except Exception:
        return None


# ==============================
# ANALISIS
# ==============================

def analyze(ticker):

    df = download_series(ticker)

    if df is None:
        return None

    close = df["Close"].iloc[-1]

    ma50 = df["Close"].rolling(50).mean().iloc[-1]
    ma200 = df["Close"].rolling(200).mean().iloc[-1]

    prev = df["Close"].iloc[-2]

    ret1d = pct(close, prev)
    ret1w = pct(close, df["Close"].iloc[-6])
    ret1m = pct(close, df["Close"].iloc[-21])

    return {
        "close": close,
        "ret1d": ret1d,
        "ret1w": ret1w,
        "ret1m": ret1m,
        "dist200": pct(close, ma200),
        "dist50": pct(close, ma50),
        "above200": close > ma200,
    }


# ==============================
# BREADTH
# ==============================

def breadth(indices, sectors):

    idx = [v for v in indices.values() if v]

    sec = [v for v in sectors.values() if v]

    pct_idx = sum(1 for x in idx if x["above200"]) / max(len(idx),1) * 100
    pct_sec = sum(1 for x in sec if x["above200"]) / max(len(sec),1) * 100

    return pct_idx, pct_sec


# ==============================
# SCORE
# ==============================

def market_score(indices, sectors, vix):

    pct_idx, pct_sec = breadth(indices, sectors)

    score = pct_idx * 0.4 + pct_sec * 0.4

    if vix:

        if vix > 30:
            score -= 20

        elif vix > 25:
            score -= 10

    score = max(0, min(100, score))

    label = "Risk-on" if score >= 70 else "Neutral" if score >= 40 else "Risk-off"

    return score, label


# ==============================
# EMAIL
# ==============================

def send_email(subject, body):

    host = os.environ["SMTP_HOST"]
    port = int(os.environ["SMTP_PORT"])
    user = os.environ["SMTP_USER"]
    password = os.environ["SMTP_PASS"]
    mail_from = os.environ["MAIL_FROM"]
    mail_to = os.environ["MAIL_TO"]

    msg = MIMEMultipart()
    msg["From"] = mail_from
    msg["To"] = mail_to
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP(host, port) as server:

        server.starttls()

        server.login(user, password)

        server.sendmail(mail_from, mail_to, msg.as_string())


# ==============================
# MAIN
# ==============================

def main():

    today = datetime.now(TZ).date().isoformat()

    indices_stats = {}
    sector_stats = {}
    factor_stats = {}

    # indices
    for name, ticker in INDICES.items():

        indices_stats[name] = analyze(ticker)

    # sectors
    for name, ticker in SECTORS.items():

        sector_stats[name] = analyze(ticker)

    # factors
    for name, ticker in FACTORS.items():

        factor_stats[name] = analyze(ticker)

    # vix
    vix_data = analyze(VIX_TICKER)

    vix_level = vix_data["close"] if vix_data else None

    score, label = market_score(indices_stats, sector_stats, vix_level)

    pct_idx, pct_sec = breadth(indices_stats, sector_stats)

    # ======================
    # BUILD EMAIL
    # ======================

    lines = []

    lines.append(f"MARKET BRIEF — {today}")
    lines.append("")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    icon = semaforo(score)

    lines.append(f"EXECUTIVE SNAPSHOT")
    lines.append(f"Market Regime Score: {icon} {score:.0f}/100 ({label})")
    lines.append(f"Breadth: Índices>MA200 {pct_idx:.0f}% | Sectores>MA200 {pct_sec:.0f}%")

    if vix_level:
        lines.append(f"VIX proxy: {vix_level:.1f}")

    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append("")

    lines.append("MARKET STRUCTURE")

    for name, stats in indices_stats.items():

        if not stats:
            lines.append(f"{name}: n/a")
            continue

        lines.append(
            f"{name}: {fmt(stats['ret1d'])} 1D | {fmt(stats['ret1w'])} 1W | {fmt(stats['ret1m'])} 1M"
        )

    lines.append("")
    lines.append("SECTOR ROTATION (1W)")

    sec_list = [s for s in sector_stats.values() if s]

    sec_sorted = sorted(sec_list, key=lambda x: x["ret1w"] if x else -999)

    top = sec_sorted[-3:]
    bottom = sec_sorted[:3]

    lines.append("Top sectors: " + ", ".join(fmt(x["ret1w"]) for x in top))
    lines.append("Bottom sectors: " + ", ".join(fmt(x["ret1w"]) for x in bottom))

    lines.append("")
    lines.append("OPERATIONAL CONCLUSION")

    if label == "Risk-on":

        lines.append("Sesgo constructivo. Mantener exposición y buscar líderes.")

    elif label == "Neutral":

        lines.append("Mercado mixto. Ser selectivo.")

    else:

        lines.append("Sesgo defensivo. Priorizar protección de capital.")

    body = "\n".join(lines)

    subject = f"Market Brief — {icon} {score:.0f}/100 ({label})"

    send_email(subject, body)


if __name__ == "__main__":
    main()
