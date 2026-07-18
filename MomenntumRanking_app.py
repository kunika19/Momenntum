import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats
import os
import io
import re
from datetime import datetime
import streamlit as st

# ==============================================================================
# 1. 運用ルール・定数設定
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
# 環境に応じてCSVのパスを調整してください
CSV_FILE = os.path.join(current_dir, "tickers_topix500.csv")

RISK_FACTOR = 0.003       # 1銘柄あたりリスク0.3%（バックテスト検証済み）

# --- 追証耐性から建代金上限を逆算するための既定値 ---
DEFAULT_HAIRCUT = 80          # 代用掛目(%)。投資信託は通常80%
MAINTENANCE_LINE = 20         # SBI証券の最低委託保証金率(維持率)=20%。これを割ると追証
DEFAULT_STRESS_DROP = 20      # 想定担保下落率(%)
DEFAULT_ASSUMED_DD = 36       # 戦略の想定最大DD(%)。RISK 0.003 のバックテスト実測 -35.9%
DEFAULT_TARGET_RATIO = 30     # ストレス時に確保したい維持率(%)
DEFAULT_MARGIN_RATE = 2.8     # 信用金利(年率%)。SBIは手数料無料で金利のみ発生
DEFAULT_HOLDING_MONTHS = 12   # 想定保有期間(ヶ月)。この間の金利が実質保証金を削る
MIN_DAILY_TURNOVER = 1 * 10**8  
MAX_GAP_THRESHOLD = 15.0  
MOMENTUM_WINDOW = 90      
ATR_WINDOW = 14           # 真のATR(14)

# ==============================================================================
# 2. ロジック関数
# ==============================================================================
def calculate_momentum(prices):
    try:
        returns = prices.pct_change()
        adjusted_returns = returns.copy()
        adjusted_returns[returns.abs() >= 0.30] = 0
        clean_prices = (1 + adjusted_returns.fillna(0)).cumprod() * prices.iloc[0]
        log_prices = np.log(clean_prices)
        x = np.arange(len(log_prices))
        
        slope, _, r_value, _, _ = stats.linregress(x, log_prices)
        annualized_slope = (np.exp(slope) ** 252) - 1
        return annualized_slope * (r_value ** 2) * 100
    except:
        return -999

def calc_trading_limit(collateral_value, cash_value, haircut_pct,
                       stress_drop_pct, assumed_dd_pct, target_ratio_pct,
                       margin_rate_pct=DEFAULT_MARGIN_RATE,
                       holding_months=DEFAULT_HOLDING_MONTHS,
                       accrued_cost=0.0):
    """
    追証に耐えられる建代金上限を逆算する。

    ストレス時（担保がstress_drop%下落し、戦略が想定最大DDを引き、
    さらに保有期間分の金利が累積した状態）でも目標維持率を確保できる建代金Bを求める。

        実質保証金A = 代用評価額(=担保時価×掛目) + 現金 − 既発生の諸経費
        将来の諸経費 = B × 金利(年率) × 保有月数/12
        ストレス時A' = 代用評価額×(1−担保下落率) + 現金 − 既発生諸経費
                       − B×想定DD − B×金利負担
        A' / B ≧ 目標維持率

        ⇒ B ≦ (代用評価額×(1−担保下落率) + 現金 − 既発生諸経費)
               ÷ (目標維持率 + 想定DD + 金利負担)

    金利はBに比例して増えるため、分母に加算する形で厳密に解ける。
    （例: 年2.8%を12ヶ月保有 = 目標維持率を2.8%pt引き上げるのと等価）

    ※現金は掛目がかからず100%評価。下落もしないためストレス時もそのまま残る。
    ※現物購入分には金利がかからないため、この見積もりは安全側。
    """
    haircut = haircut_pct / 100.0
    stress_drop = stress_drop_pct / 100.0
    assumed_dd = assumed_dd_pct / 100.0
    target_ratio = target_ratio_pct / 100.0
    interest_burden = (margin_rate_pct / 100.0) * (holding_months / 12.0)

    collateral_evaluated = collateral_value * haircut   # 代用有価証券評価額（掛目後）
    # 実質保証金A: SBIの定義に合わせ、既に発生している諸経費を差し引く
    real_margin = collateral_evaluated + cash_value - accrued_cost

    stressed_margin = collateral_evaluated * (1 - stress_drop) + cash_value - accrued_cost
    denominator = target_ratio + assumed_dd + interest_burden
    limit = stressed_margin / denominator if denominator > 0 else 0
    limit = max(limit, 0)

    return {
        "collateral_evaluated": collateral_evaluated,
        "real_margin": real_margin,
        "stressed_margin": stressed_margin,
        "accrued_cost": accrued_cost,
        "limit": limit,
        "interest_burden": interest_burden,          # 建代金に対する金利負担率
        "interest_cost": limit * interest_burden,    # 保有期間中に発生する金利額
        # 参考: 限度額いっぱいに建てた場合の平常時レバレッジ・維持率
        "leverage": (limit / real_margin) if real_margin > 0 else 0,
        "current_ratio": (real_margin / limit * 100) if limit > 0 else 0,
    }


def run_ranking_process(ticker_df, trading_limit):
    """trading_limit: 建代金上限（UI側で追証耐性から逆算済み）"""
    tickers = ticker_df['Ticker'].tolist()
    name_map = dict(zip(ticker_df['Ticker'], ticker_df['Name']))

    raw_data = yf.download(tickers + ["^N225"], period="1y", progress=False)
    market_data = raw_data['Close']
    volume_data = raw_data['Volume']
    high_data = raw_data['High']
    low_data = raw_data['Low']

    virtual_capital = trading_limit
    risk_budget_per_stock = virtual_capital * RISK_FACTOR
    
    n225_series = market_data["^N225"].dropna()
    n225_ma200 = n225_series.rolling(200).mean().iloc[-1]
    market_ok = n225_series.iloc[-1] > n225_ma200

    results = []
    for ticker in tickers:
        try:
            if ticker not in market_data: continue
            prices = market_data[ticker].dropna()
            volumes = volume_data[ticker].dropna()
            if len(prices) < MOMENTUM_WINDOW: continue
            
            avg_turnover = (prices.tail(5) * volumes.tail(5)).mean()
            score = calculate_momentum(prices.tail(MOMENTUM_WINDOW))
            ma100 = prices.rolling(100).mean().iloc[-1]
            max_gap = prices.tail(MOMENTUM_WINDOW).pct_change().abs().max() * 100
            
            # --- 真のATR(14): TR = max(高値-安値, |高値-前日終値|, |安値-前日終値|) ---
            h = high_data[ticker]
            l = low_data[ticker]
            c = market_data[ticker]
            pc = c.shift(1)
            tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
            atr = tr.dropna().rolling(ATR_WINDOW).mean().iloc[-1]
            if np.isnan(atr) or atr <= 0: atr = 0.01
            
            current_price = float(prices.iloc[-1])
            
            # --- 株数計算（6捨7入ロジック） ---
            shares_raw = int(risk_budget_per_stock / atr)
            rem = shares_raw % 100
            shares_rounded = (shares_raw // 100) * 100 + (100 if rem >= 70 else 0)
            
            results.append({
                "Rank": 0,
                "Ticker": ticker,
                "Name": name_map.get(ticker, "N/A"),
                "Score": round(score, 2),
                "AvgTurnover_Oku": round(avg_turnover / 10**8, 2),
                "MaxGap%": round(max_gap, 2),
                "Price": round(current_price, 1),
                "Shares": shares_raw,
                "Shares_Rounded": shares_rounded,
                "Position_Size": int(current_price * shares_rounded),
                "MA100_OK": current_price > ma100 if not np.isnan(ma100) else True,
                "Liquidity_OK": avg_turnover >= MIN_DAILY_TURNOVER
            })
        except:
            continue

    df = pd.DataFrame(results).sort_values(by="Score", ascending=False).reset_index(drop=True)
    df['Rank'] = df.index + 1
    total = len(df)

    def judge_action(row):
        if not row['MA100_OK']: return "SELL(MA100)"
        if row['Rank'] > (total * 0.20): return "SELL(Rank)"
        if row['MaxGap%'] > MAX_GAP_THRESHOLD: return "SELL(Gap)"
        if not row['Liquidity_OK']: return "SKIP(Liquidity)"
        if row['Rank'] <= (total * 0.10):
            return "★BUY" if market_ok else "WAIT(Market)"
        return "HOLD"

    df['Action'] = df.apply(judge_action, axis=1)
    return df, market_ok


def normalize_ticker(t):
    """銘柄コードの表記揺れ（数字のみ／.T付き等）を吸収して比較できるようにする"""
    t = str(t).strip().upper()
    if t and "." not in t and t.replace(" ", "").isdigit():
        t = t + ".T"
    return t


TICKER_ALIASES = ["ticker", "code", "symbol", "証券コード", "銘柄コード", "コード", "銘柄"]
SHARES_ALIASES = ["shares", "quantity", "qty", "株数", "保有株数", "数量", "保有数量", "残数量", "保有数", "保有株式数"]


def _find_col(df, aliases_lower):
    cols = {c: str(c).strip().lower() for c in df.columns}
    for c, lc in cols.items():
        if lc in aliases_lower:
            return c
    for c, lc in cols.items():
        for a in aliases_lower:
            if a in lc:
                return c
    return None


def extract_ticker_shares(df):
    """様々な見出し（日本語/英語）や列順のテーブルから Ticker, Shares 列だけを抜き出す。
    見出しが認識できない場合は1列目をTicker、2列目をSharesとみなす。"""
    if df is None or df.empty:
        return None
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    ticker_col = _find_col(df, [a.lower() for a in TICKER_ALIASES])
    shares_col = _find_col(df, [a.lower() for a in SHARES_ALIASES])

    if ticker_col is None or shares_col is None:
        if df.shape[1] < 2:
            return None
        if ticker_col is None:
            ticker_col = df.columns[0]
        if shares_col is None:
            candidates = [c for c in df.columns if c != ticker_col]
            shares_col = candidates[0] if candidates else None
        if shares_col is None:
            return None

    out = df[[ticker_col, shares_col]].rename(columns={ticker_col: "Ticker", shares_col: "Shares"})
    out["Ticker"] = out["Ticker"].astype(str).str.strip()
    out["Shares"] = (
        out["Shares"].astype(str)
        .str.replace(",", "", regex=False)
        .str.extract(r"(-?\d+)")[0]
    )
    out["Shares"] = pd.to_numeric(out["Shares"], errors="coerce")
    out = out.dropna(subset=["Ticker", "Shares"])
    out = out[out["Ticker"] != ""]
    return out.reset_index(drop=True)


def parse_pasted_portfolio(text):
    """Excelや証券会社サイトからコピーした表（タブ・カンマ・スペース区切り）を解析する"""
    if not text or not text.strip():
        return None
    text = text.strip()
    if "\t" in text:
        sep = "\t"
    elif "," in text:
        sep = ","
    else:
        text = re.sub(r"[ \u3000]+", "\t", text)
        sep = "\t"
    try:
        parsed = pd.read_csv(io.StringIO(text), sep=sep, engine="python", header=None)
    except Exception:
        return None

    first_row = parsed.iloc[0].astype(str).tolist()
    looks_like_header = not any(re.search(r"\d", v) for v in first_row[:2])
    if looks_like_header and len(parsed) > 1:
        parsed.columns = first_row
        parsed = parsed.iloc[1:].reset_index(drop=True)
    else:
        parsed.columns = [f"col{i}" for i in range(parsed.shape[1])]

    return extract_ticker_shares(parsed)


def extract_margin_cost_stats(text):
    """SBI「信用建玉一覧」CSVから建代金・諸経費等・保有日数を集計し、
    実際に発生しているコストから実効金利（年率）を逆算する。
    諸経費等には金利のほか信用管理費・名義書換料が含まれるため、
    表面金利（2.8%）より高くなるのが通常。対応できない形式ならNoneを返す。"""
    if not text:
        return None
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if "銘柄コード" in line:
            header_idx = i
            break
    if header_idx is None:
        return None

    data_lines = []
    for line in lines[header_idx + 1:]:
        if line.strip() == "":
            break
        data_lines.append(line)
    if not data_lines:
        return None

    try:
        df = pd.read_csv(io.StringIO("\n".join([lines[header_idx]] + data_lines)))
    except Exception:
        return None
    df.columns = [str(c).strip() for c in df.columns]

    if "建代金" not in df.columns or "諸経費等" not in df.columns:
        return None

    if "売/買建" in df.columns:
        df = df[~df["売/買建"].astype(str).str.contains("売", na=False)]

    def to_num(col):
        return pd.to_numeric(
            df[col].astype(str).str.replace(",", "", regex=False).str.replace("+", "", regex=False),
            errors="coerce",
        )

    df["_daikin"] = to_num("建代金").fillna(0)
    df["_keihi"] = to_num("諸経費等").fillna(0)
    df = df[df["_daikin"] > 0]
    if df.empty:
        return None

    total_daikin = float(df["_daikin"].sum())
    total_keihi = float(df["_keihi"].sum())

    # 建日から保有日数を求め、建代金加重の平均日数と実効金利を逆算する。
    # 基準日はCSVの断面日。取得日が不明なため「最も新しい建日」を採用する
    # （週次リバランス運用では直近に必ず建玉があるため妥当な近似）。
    # 数日のズレでも保有日数が短い建玉では逆算金利が大きく振れるため、この判定は重要。
    avg_days = None
    implied_rate = None
    ref_date = None
    if "建日" in df.columns:
        tatehi = pd.to_datetime(df["建日"].astype(str).str.strip(),
                                format="%Y/%m/%d", errors="coerce")
        if tatehi.notna().any():
            today = pd.Timestamp(datetime.now().date())
            ref_date = min(tatehi.max(), today)   # 未来日付は今日で打ち切り
            days = (ref_date - tatehi).dt.days
            valid = days.notna() & (days >= 0)
            if valid.any():
                w = df.loc[valid, "_daikin"]
                avg_days = float((days[valid] * w).sum() / w.sum())
                # 実効金利 = 諸経費 ÷ (建代金 × 保有日数/365)
                denom = float((df.loc[valid, "_daikin"] * days[valid] / 365).sum())
                if denom > 0:
                    implied_rate = float(df.loc[valid, "_keihi"].sum() / denom * 100)

    return {
        "total_daikin": total_daikin,
        "total_keihi": total_keihi,
        "cost_pct": total_keihi / total_daikin * 100 if total_daikin > 0 else 0,
        "avg_days": avg_days,
        "implied_rate": implied_rate,
        "ref_date": ref_date,
        "positions": int(len(df)),
    }


def try_parse_sbi_margin_csv(text):
    """SBI証券の「信用建玉一覧」CSV（先頭に見出しテキストや空行、銘柄コード省略の継続行、
    末尾に注記がある特殊な形式）を解析する。対応できない形式ならNoneを返す。"""
    if not text:
        return None
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if "銘柄コード" in line:
            header_idx = i
            break
    if header_idx is None:
        return None

    data_lines = []
    for line in lines[header_idx + 1:]:
        if line.strip() == "":
            break
        data_lines.append(line)
    if not data_lines:
        return None

    csv_text = "\n".join([lines[header_idx]] + data_lines)
    try:
        raw_df = pd.read_csv(io.StringIO(csv_text))
    except Exception:
        return None
    raw_df.columns = [str(c).strip() for c in raw_df.columns]

    if "銘柄コード" not in raw_df.columns or "建株数" not in raw_df.columns:
        return None

    # 同一銘柄で複数回に分けて建てた行は銘柄コードが空欄になるため、直前の行から補完する
    raw_df["銘柄コード"] = pd.to_numeric(raw_df["銘柄コード"], errors="coerce").ffill()
    raw_df = raw_df.dropna(subset=["銘柄コード"])
    raw_df["銘柄コード"] = raw_df["銘柄コード"].astype(int).astype(str)
    if "銘柄名称" in raw_df.columns:
        raw_df["銘柄名称"] = raw_df["銘柄名称"].ffill()

    # このアプリはロング（買建）のモメンタム戦略を前提とするため、売建（ショート）は対象外とする
    if "売/買建" in raw_df.columns:
        raw_df = raw_df[~raw_df["売/買建"].astype(str).str.contains("売", na=False)]

    raw_df["Shares"] = pd.to_numeric(
        raw_df["建株数"].astype(str).str.replace(",", "", regex=False), errors="coerce"
    ).fillna(0)

    grouped = raw_df.groupby("銘柄コード", as_index=False)["Shares"].sum()
    grouped = grouped.rename(columns={"銘柄コード": "Ticker"})
    grouped = grouped[grouped["Shares"] > 0].reset_index(drop=True)
    return grouped if not grouped.empty else None


# アクション種別ごとの色（カード表示用）
ACTION_COLORS = {
    "🔴 全株売却": "#e74c3c",
    "⚠️ 流動性低下・売却検討": "#e67e22",
    "🆕 新規購入": "#2ecc71",
    "🔵 買い増し": "#3498db",
    "🟡 一部売却(調整)": "#f1c40f",
    "❓ 確認要": "#9b59b6",
    "✅ 保有継続": "#7f8c8d",
    "⏸ 保有継続(地合い待ち)": "#7f8c8d",
}


def render_action_cards(df):
    """アクション一覧を種別ごとにグループ化し、box-shadow付きのカードHTMLとして描画する（スマホ向け）。
    引数dfは呼び出し側で既にaction_order順にソート済みであることを前提とする。"""
    for action in df["PortfolioAction"].unique():
        group = df[df["PortfolioAction"] == action]
        color = ACTION_COLORS.get(action, "#7f8c8d")
        st.markdown(f"##### {action}（{len(group)}銘柄）")

        for _, row in group.iterrows():
            trade_shares = int(row["TradeShares"])
            trade_amount = int(row["TradeAmount"])

            if trade_shares > 0:
                trade_line = f"<b>+{trade_shares:,}株</b> 購入（約 {trade_amount:,}円）"
            elif trade_shares < 0:
                trade_line = f"<b>{trade_shares:,}株</b> 売却（約 {abs(trade_amount):,}円）"
            else:
                trade_line = "売買なし（株数の変更なし）"

            current = int(row["CurrentShares"])
            target = int(row["Shares_Rounded"])
            price = row["Price"]

            card_html = f"""
            <div style="border-left:6px solid {color};background:rgba(127,127,127,0.10);
                        border-radius:10px;padding:12px 14px;margin-bottom:10px;
                        box-shadow:0 1px 3px rgba(0,0,0,0.15);">
              <div style="font-size:1.15rem;font-weight:700;color:{color};margin-bottom:4px;">
                {action}
              </div>
              <div style="font-size:1.05rem;font-weight:600;margin-bottom:6px;">
                {row['Ticker']}　{row['Name']}
                <span style="font-size:0.8rem;color:#888;font-weight:400;">（{int(row['Rank'])}位）</span>
              </div>
              <div style="font-size:1.0rem;margin-bottom:6px;">{trade_line}</div>
              <div style="font-size:0.85rem;color:#888;">
                現在 {current:,}株 → 目標 {target:,}株 ／ 株価 {price:,.1f}円
              </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)


def decide_portfolio_action(row):
    """ランキングのAction判定と現在の保有株数を組み合わせ、具体的な売買アクションを決める"""
    held = row['CurrentShares'] > 0
    action = row['Action']
    target = row['Shares_Rounded']
    price = row['Price']
    current = row['CurrentShares']

    # ランキング側がSELL判定
    if action.startswith("SELL"):
        if held:
            return pd.Series(["🔴 全株売却", -current, -current * price])
        return pd.Series(["—", 0, 0])

    # 出来高不足でスキップ判定
    if action == "SKIP(Liquidity)":
        if held:
            return pd.Series(["⚠️ 流動性低下・売却検討", -current, -current * price])
        return pd.Series(["—", 0, 0])

    # 新規買い候補
    if action == "★BUY":
        if not held:
            if target <= 0:
                return pd.Series(["—", 0, 0])
            return pd.Series(["🆕 新規購入", target, target * price])
        diff = target - current
        if target <= 0:
            return pd.Series(["🟡 一部売却(調整)", -current, -current * price])
        if current < target * 0.9:
            return pd.Series(["🔵 買い増し", diff, diff * price])
        if current > target * 1.1:
            return pd.Series(["🟡 一部売却(調整)", diff, diff * price])
        return pd.Series(["✅ 保有継続", 0, 0])

    # 地合い悪化で新規買い見送り
    if action == "WAIT(Market)":
        if held:
            return pd.Series(["⏸ 保有継続(地合い待ち)", 0, 0])
        return pd.Series(["—", 0, 0])

    # 通常のHOLD
    if action == "HOLD":
        if held:
            diff = target - current
            if target <= 0:
                return pd.Series(["🟡 一部売却(調整)", -current, -current * price])
            if current < target * 0.9:
                return pd.Series(["🔵 買い増し", diff, diff * price])
            if current > target * 1.1:
                return pd.Series(["🟡 一部売却(調整)", diff, diff * price])
            return pd.Series(["✅ 保有継続", 0, 0])
        return pd.Series(["—", 0, 0])

    # 想定外のケース
    if held:
        return pd.Series(["❓ 確認要", 0, 0])
    return pd.Series(["—", 0, 0])


def compare_with_portfolio(rank_df, portfolio_df):
    """ランキング結果と現在のポートフォリオ（Ticker, Shares）を比較し、アクション表を作る"""
    portfolio_df = portfolio_df.copy()
    portfolio_df["Ticker_norm"] = portfolio_df["Ticker"].apply(normalize_ticker)
    portfolio_df["Shares"] = pd.to_numeric(portfolio_df["Shares"], errors="coerce").fillna(0).astype(int)
    portfolio_df = portfolio_df[portfolio_df["Shares"] > 0]
    portfolio_map = dict(zip(portfolio_df["Ticker_norm"], portfolio_df["Shares"]))

    merged = rank_df.copy()
    merged["Ticker_norm"] = merged["Ticker"].apply(normalize_ticker)
    merged["CurrentShares"] = merged["Ticker_norm"].map(portfolio_map).fillna(0).astype(int)

    merged[["PortfolioAction", "TradeShares", "TradeAmount"]] = merged.apply(decide_portfolio_action, axis=1)

    rank_tickers_norm = set(merged["Ticker_norm"])
    missing = portfolio_df[~portfolio_df["Ticker_norm"].isin(rank_tickers_norm)][["Ticker", "Shares"]]

    return merged, missing

# ==============================================================================
# 3. Streamlit WEB UI インターフェース
# ==============================================================================
st.set_page_config(
    page_title="モメンタム運用判断",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- グローバルCSS（余白調整・st.metricのカード化）を1箇所にまとめて注入 ---
# 色はライト/ダーク両対応のため rgba(127,127,127,...) 系のみを使用し、白/黒の直接指定は避ける。
st.markdown(
    """
    <style>
    /* 上部の余白を少し圧縮してコンテンツを詰める */
    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 2rem;
    }
    /* st.metric をカード風に（枠線＋角丸＋薄い背景） */
    div[data-testid="stMetric"] {
        background: rgba(127, 127, 127, 0.08);
        border: 1px solid rgba(127, 127, 127, 0.18);
        border-radius: 10px;
        padding: 12px 14px 8px 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 モメンタム運用判断")
st.caption("Topix500などの銘柄群から、クレノー流モメンタムスコアを計算し売買アクションを判定します。")


def read_csv_fallback_encoding(src):
    """utf-8-sig で読めなければ shift-jis で読み直す（証券会社CSV対策）"""
    try:
        return pd.read_csv(src, encoding="utf-8-sig")
    except Exception:
        if hasattr(src, "seek"):
            src.seek(0)
        return pd.read_csv(src, encoding="shift-jis")


def render_capital_summary(info, stress_drop, assumed_dd, target_ratio,
                           margin_rate, holding_months, stats=None):
    """建代金上限とリスク予算のサマリーを表示"""
    st.subheader("💰 建代金上限（追証耐性からの逆算）")

    c1, c2, c3 = st.columns(3)
    c1.metric("実質保証金", f"{info['real_margin']:,.0f} 円",
              help="代用評価額（担保時価×掛目）＋現金")
    c2.metric("建代金上限", f"{info['limit']:,.0f} 円",
              help=f"ストレス時（担保-{stress_drop}%＋戦略DD-{assumed_dd}%＋金利{holding_months}ヶ月分）でも維持率{target_ratio}%を確保できる上限。")
    c3.metric("1銘柄リスク予算", f"{info['limit'] * RISK_FACTOR:,.0f} 円",
              help=f"建代金上限 × RISK_FACTOR({RISK_FACTOR})。これをATRで割って株数を決めます。")

    if info["real_margin"] > 0:
        st.caption(
            f"実質保証金に対するレバレッジ **{info['leverage']:.2f}倍** ／ "
            f"平常時の維持率 **{info['current_ratio']:.0f}%**"
        )

    st.caption(
        f"金利 年{margin_rate}% × {holding_months}ヶ月 ＝ 建代金の **{info['interest_burden'] * 100:.1f}%**"
        f"（約 {info['interest_cost']:,.0f} 円）を織り込み済み。"
        f"実効的に目標維持率を {target_ratio}% → {target_ratio + info['interest_burden'] * 100:.1f}% に引き上げるのと等価です。"
    )

    if target_ratio <= MAINTENANCE_LINE:
        st.warning(
            f"目標維持率が追証ライン（{MAINTENANCE_LINE}%）ちょうどです。"
            "この設定は「追証は出ないが実質破綻」の水準なので、30%以上を推奨します。"
        )

    # --- 建玉CSVがあれば、現在の建代金と上限を突き合わせて過不足を表示 ---
    if stats:
        cur = stats["total_daikin"]
        diff = info["limit"] - cur
        st.markdown("**現在の建玉との比較**")
        d1, d2, d3 = st.columns(3)
        d1.metric("現在の建代金", f"{cur:,.0f} 円",
                  help=f"買建 {stats['positions']}建玉の合計")
        d2.metric("累積諸経費", f"{stats['total_keihi']:,.0f} 円",
                  help=f"建代金比 {stats['cost_pct']:.3f}%。実質保証金から差し引き済み。")
        if diff >= 0:
            d3.metric("上限までの余裕", f"{diff:,.0f} 円", delta="範囲内")
        else:
            d3.metric("超過額", f"{abs(diff):,.0f} 円", delta="要減玉", delta_color="inverse")
            st.error(
                f"現在の建代金が上限を **{abs(diff):,.0f}円** 超過しています。"
                f"想定ストレス（担保-{stress_drop}%＋戦略DD-{assumed_dd}%）が起きると"
                f"維持率{target_ratio}%を割り込みます。"
            )
        if stats.get("avg_days") is not None:
            _ref = stats.get("ref_date")
            _ref_txt = f"（基準日 {_ref.strftime('%Y/%m/%d')}）" if _ref is not None else ""
            _rate_txt = (f" ／ 諸経費から逆算した実効金利 {stats['implied_rate']:.2f}%"
                         if stats.get("implied_rate") else "")
            st.caption(
                f"建代金加重の平均保有日数 {stats['avg_days']:.1f}日{_ref_txt}{_rate_txt}"
            )

    st.caption(
        "※現物で買った分は追証の対象外かつ金利もかからないため、この上限は安全側の見積もりです。"
        "暴落時は掛目自体が引き下げられる場合があります。"
    )


# ------------------------------------------------------------------------
# ① 銘柄ユニバース
# ------------------------------------------------------------------------
st.sidebar.subheader("① 銘柄ユニバース")
uploaded_file = st.sidebar.file_uploader("CSVファイルをアップロード", type=["csv"])

ticker_df = None
if uploaded_file is not None:
    ticker_df = read_csv_fallback_encoding(uploaded_file)
    st.sidebar.success("CSVを読み込みました。")
elif os.path.exists(CSV_FILE):
    ticker_df = read_csv_fallback_encoding(CSV_FILE)
    st.sidebar.info(f"ローカルファイルを使用中: {os.path.basename(CSV_FILE)}")

# ------------------------------------------------------------------------
# ② 現在のポートフォリオ（任意）
# ------------------------------------------------------------------------
st.sidebar.subheader("② 現在のポートフォリオ（任意）")
portfolio_file = st.sidebar.file_uploader(
    "保有銘柄CSV（Ticker, Shares列）", type=["csv"], key="portfolio_csv_uploader"
)
st.sidebar.caption("CSVが無くても、本文の入力欄に貼り付け／手入力できます。")

# アップロードされた建玉CSVを先にデコードし、実際の諸経費・実効金利を抽出する
portfolio_raw_text = None
if portfolio_file is not None:
    _raw_bytes = portfolio_file.getvalue()
    for _enc in ["utf-8-sig", "cp932", "shift_jis"]:
        try:
            portfolio_raw_text = _raw_bytes.decode(_enc)
            break
        except Exception:
            continue
margin_stats = extract_margin_cost_stats(portfolio_raw_text) if portfolio_raw_text else None

# ------------------------------------------------------------------------
# ③ 資金入力
# ------------------------------------------------------------------------
st.sidebar.subheader("③ 資金入力")

collateral_input = st.sidebar.number_input(
    "担保評価額（投信等の時価・円）", min_value=0, value=0, step=100000,
    help="代用有価証券に入れている投資信託などの時価。掛目は「追証耐性の前提」で調整します。",
)

with st.sidebar.expander("⚠️ 追証耐性の前提", expanded=False):
    haircut_input = st.slider(
        "代用掛目 (%)", min_value=50, max_value=100, value=DEFAULT_HAIRCUT, step=5,
        help="投資信託は通常80%。増担保規制やレバレッジ型ETF等の適用で引き下げられることがあります。",
    )
    stress_drop_input = st.slider(
        "想定担保下落率 (%)", 0, 60, DEFAULT_STRESS_DROP, step=5,
        help="暴落時に担保がどこまで下がるか。NASDAQ100ゴールドプラス等の2倍レバ商品なら30〜35%も想定範囲。",
    )
    assumed_dd_input = st.slider(
        "戦略の想定最大DD (%)", 10, 60, DEFAULT_ASSUMED_DD, step=1,
        help=f"RISK {RISK_FACTOR} のバックテスト実測は約-36%（2018-19）。",
    )
    target_ratio_input = st.slider(
        "ストレス時の目標維持率 (%)", MAINTENANCE_LINE, 50, DEFAULT_TARGET_RATIO, step=5,
        help=f"{MAINTENANCE_LINE}%が追証ライン。ちょうどに設定すると『追証は出ないが実質破綻』の水準です。",
    )
    _rate_default = DEFAULT_MARGIN_RATE
    if margin_stats and margin_stats.get("implied_rate"):
        _rate_default = round(margin_stats["implied_rate"], 2)
        st.caption(
            f"建玉CSVの諸経費から実効金利 **{_rate_default}%** を逆算し、既定値に設定しました。"
        )
    margin_rate_input = st.number_input(
        "信用金利 (年率%)", min_value=0.0, max_value=10.0,
        value=float(_rate_default), step=0.1,
        help="諸経費等には金利のほか信用管理費(110円/銘柄/月)や名義書換料が含まれるため、表面金利より高くなります。",
    )
    holding_months_input = st.slider(
        "想定保有期間 (ヶ月)", 1, 24, DEFAULT_HOLDING_MONTHS, step=1,
        help="この期間に累積する金利を織り込みます。年2.8%×12ヶ月＝目標維持率を2.8%pt上げるのと等価。",
    )

cash_input = st.sidebar.number_input(
    "現金 (円) ※100%評価", min_value=0, value=0, step=100000,
    help="現金は掛目がかからず100%評価。現物購入にも充当できます。",
)

limit_info = calc_trading_limit(
    float(collateral_input), float(cash_input), haircut_input,
    stress_drop_input, assumed_dd_input, target_ratio_input,
    margin_rate_input, holding_months_input,
    accrued_cost=(margin_stats["total_keihi"] if margin_stats else 0.0),
)
trading_limit = limit_info["limit"]   # 建代金上限（安全余裕は目標維持率で明示的に確保）。ポジションサイズ計算のベース資金

# ------------------------------------------------------------------------
# 表示モード（スマホ/PC）
# ------------------------------------------------------------------------
st.sidebar.subheader("📱 表示モード")
view_mode = st.sidebar.radio(
    "画面の見やすさを選択",
    ["スマホ向け（カード表示）", "PC向け（表表示）"],
    index=0,
    help="iPhoneなど小さい画面ではカード表示が見やすく、PCの大画面では表表示が一覧しやすいです。",
)
is_mobile = view_mode.startswith("スマホ")

# ------------------------------------------------------------------------
# ④ 実行
# ------------------------------------------------------------------------
st.sidebar.subheader("④ 実行")
run_disabled = ticker_df is None
run_clicked = st.sidebar.button(
    "🚀 ランキングを生成する",
    type="primary",
    width="stretch",
    disabled=run_disabled,
)
if run_disabled:
    st.sidebar.caption("銘柄CSVを読み込むと実行できます。")

if run_clicked:
    with st.status("データ取得・計算中...", expanded=True) as status:
        st.write("株価データの取得とモメンタムスコアの計算を実行中...（銘柄数により1〜2分かかります）")
        rank_df, market_ok = run_ranking_process(ticker_df, trading_limit)
        status.update(label="完了", state="complete", expanded=False)
    # 結果をsession_stateへ永続化。生成後に他のウィジェットを操作しても結果が消えないようにする。
    st.session_state["rank_result"] = {
        "df": rank_df,
        "market_ok": market_ok,
        "generated_at": datetime.now(),
        "trading_limit": trading_limit,
    }
    st.toast("ランキングを生成しました")

# session_stateを読み直す（本ラン内でボタンが押されていれば直前の結果を即座に反映する）
rank_result = st.session_state.get("rank_result")
has_result = rank_result is not None and not rank_result["df"].empty

# ------------------------------------------------------------------------
# メイン画面: 結果が無い場合はガイダンス＋資金サマリー、ある場合は下部でタブ表示
# ------------------------------------------------------------------------
if not has_result:
    if ticker_df is None:
        st.info(
            "**使い方**\n\n"
            "① サイドバーから銘柄CSV（Ticker, Name列）をアップロード\n\n"
            "② 資金を入力\n\n"
            "③ ランキングを生成"
        )
    # 結果生成前はメイン画面に資金サマリーを表示する（生成後は資金サマリータブ内に表示）
    render_capital_summary(limit_info, stress_drop_input,
                           assumed_dd_input, target_ratio_input,
                           margin_rate_input, holding_months_input,
                           stats=margin_stats)

with st.expander("💼 現在のポートフォリオ（保有銘柄・株数）", expanded=not has_result):
    st.caption("ここに現在保有している銘柄と株数を入力してください。CSVアップロード／コピー&ペースト／直接入力のいずれでも構いません。ランキング生成後、ここに入力した内容と比較して取るべきアクションを表示します。")

    # --- ① CSVアップロード（SBI証券「信用建玉一覧」形式、または列名自動認識の通常形式に対応） ---
    if portfolio_file is not None:
        file_sig = f"{portfolio_file.name}_{portfolio_file.size}"
        if st.session_state.get("portfolio_csv_sig") != file_sig:
            decoded_text = portfolio_raw_text  # サイドバーでデコード済みのものを再利用

            extracted_csv = None
            parse_note = None
            if decoded_text is not None:
                extracted_csv = try_parse_sbi_margin_csv(decoded_text)
                if extracted_csv is not None and not extracted_csv.empty:
                    parse_note = "SBI証券「信用建玉一覧」形式として読み込みました（同一銘柄の複数建玉は合算、売建は対象外）。"
                else:
                    try:
                        raw_df = pd.read_csv(io.StringIO(decoded_text))
                    except Exception:
                        raw_df = None
                    if raw_df is not None:
                        extracted_csv = extract_ticker_shares(raw_df)

            st.session_state["portfolio_csv_sig"] = file_sig
            if extracted_csv is None or extracted_csv.empty:
                st.warning("CSVから「コード」「株数」に該当する列を認識できませんでした。列名や内容をご確認ください。")
            else:
                if parse_note:
                    st.info(parse_note)
                st.session_state["portfolio_data"] = extracted_csv[["Ticker", "Shares"]].reset_index(drop=True)
                st.session_state["portfolio_editor_nonce"] = st.session_state.get("portfolio_editor_nonce", 0) + 1
                st.rerun()

    # --- ② コピー&ペースト（Excelや証券会社サイトの保有銘柄一覧をそのまま貼り付け） ---
    st.markdown("**📋 Excel／証券会社サイトの保有銘柄一覧をそのまま貼り付け**")
    st.caption("見出し行（コード・株数等）があれば自動認識します。見出しがない場合は1列目をコード、2列目を株数として読み込みます。反映すると下の表が上書きされます。")
    pasted_text = st.text_area(
        "ここに貼り付け",
        height=100,
        key="portfolio_paste_area",
        placeholder="例）\n7203\tトヨタ自動車\t100\n6758\tソニーグループ\t50",
    )
    if st.button("📋 貼り付け内容を反映（表を上書き）", key="paste_overwrite_btn"):
        extracted_paste = parse_pasted_portfolio(pasted_text)
        if extracted_paste is None or extracted_paste.empty:
            st.error("コードと株数を認識できませんでした。貼り付けた内容の形式をご確認ください。")
        else:
            st.session_state["portfolio_data"] = extracted_paste[["Ticker", "Shares"]].reset_index(drop=True)
            st.session_state["portfolio_editor_nonce"] = st.session_state.get("portfolio_editor_nonce", 0) + 1
            st.rerun()

    # --- ③ 直接編集できる表（①②を使わない場合はここに手入力） ---
    # portfolio_data をこのアプリ唯一の「保有データの正本」として扱い、再実行をまたいで保持する。
    # CSV取り込み・貼り付け時は nonce を更新して data_editor を作り直し、新しい内容を確実に反映させる。
    base_cols = ["Ticker", "Shares"]
    stored = st.session_state.get("portfolio_data")
    if stored is not None and not stored.empty:
        base_df = stored.copy()
    else:
        base_df = pd.DataFrame([{"Ticker": "", "Shares": 0} for _ in range(5)])

    for c in base_cols:
        if c not in base_df.columns:
            base_df[c] = None
    base_df = base_df[base_cols].reset_index(drop=True)

    editor_key = f"portfolio_editor_{st.session_state.get('portfolio_editor_nonce', 0)}"
    edited_df = st.data_editor(
        base_df,
        num_rows="dynamic",
        width="stretch",
        key=editor_key,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker"),
            "Shares": st.column_config.NumberColumn("Shares", min_value=0, step=100),
        },
    )

    # 手入力での変更も含め、編集結果を毎回正本へ書き戻す（次の再実行で復元できるように）
    st.session_state["portfolio_data"] = edited_df.reset_index(drop=True)
    portfolio_df = edited_df

# ------------------------------------------------------------------------
# 結果表示（session_stateにあれば常に描画する。他ウィジェット操作で結果が消えない）
# ------------------------------------------------------------------------
if rank_result is not None:
    if abs(rank_result["trading_limit"] - trading_limit) > 1e-6:
        st.warning("資金設定が変更されています。最新の設定で再生成してください。")
    st.caption(f"生成日時: {rank_result['generated_at'].strftime('%Y-%m-%d %H:%M:%S')}")

    rank_df = rank_result["df"]
    market_ok = rank_result["market_ok"]

    if rank_df.empty:
        st.warning("条件に合致する銘柄がありませんでした。")
    else:
        tab_rank, tab_action, tab_capital = st.tabs(
            ["📊 ランキング", "🔄 売買アクション", "💰 資金サマリー"]
        )

        # ====================================================================
        # タブ1: ランキング
        # ====================================================================
        with tab_rank:
            total_n = len(rank_df)
            n_buy_total = int((rank_df["Action"] == "★BUY").sum())

            m1, m2, m3 = st.columns(3)
            m1.metric("対象銘柄数", f"{total_n} 銘柄")
            m2.metric("★BUY銘柄数", f"{n_buy_total} 銘柄")
            m3.metric("地合い", "良好" if market_ok else "慎重")

            if not market_ok:
                st.warning("地合い判定: 慎重 — 新規買いは見送り推奨（WAIT判定）")

            # --- フィルタUI（表示のみに作用。CSVダウンロードは全件のまま） ---
            f1, f2, f3 = st.columns([2, 2, 1])
            with f1:
                action_group_filter = st.pills(
                    "Actionで絞り込み",
                    options=["★BUY", "HOLD", "SELL系", "その他"],
                    selection_mode="multi",
                    default=["★BUY", "HOLD", "SELL系", "その他"],
                    key="rank_action_filter",
                )
            with f2:
                search_text = st.text_input(
                    "Ticker / Name で検索", key="rank_search_text", placeholder="例）7203 やトヨタ"
                )
            with f3:
                show_n_option = st.selectbox(
                    "表示件数",
                    [20, 50, 100, "全件"],
                    index=1,
                    key="rank_show_n",
                    format_func=lambda x: f"{x}件" if isinstance(x, int) else x,
                )

            # Action文字列を表示フィルタ用の4分類にまとめる
            def _action_group(a):
                if a == "★BUY":
                    return "★BUY"
                if a == "HOLD":
                    return "HOLD"
                if a.startswith("SELL"):
                    return "SELL系"
                return "その他"

            display_cols = [
                "Rank", "Ticker", "Name", "Action", "Shares_Rounded", "Score",
                "Price", "MaxGap%", "AvgTurnover_Oku", "Shares", "Position_Size"
            ]
            output_df = rank_df[display_cols]

            filtered_df = output_df.copy()
            if action_group_filter:
                filtered_df = filtered_df[filtered_df["Action"].apply(_action_group).isin(action_group_filter)]
            if search_text and search_text.strip():
                _q = search_text.strip().lower()
                _mask = (
                    filtered_df["Ticker"].astype(str).str.lower().str.contains(_q, na=False)
                    | filtered_df["Name"].astype(str).str.lower().str.contains(_q, na=False)
                )
                filtered_df = filtered_df[_mask]
            if show_n_option != "全件":
                filtered_df = filtered_df.head(int(show_n_option))

            # Action列に応じて行に薄い背景色を付ける（rgbaでライト/ダーク両対応）
            def _row_bg(row):
                a = row["Action"]
                if a == "★BUY":
                    c = "rgba(46, 204, 113, 0.18)"
                elif a.startswith("SELL"):
                    c = "rgba(231, 76, 60, 0.18)"
                elif a.startswith("WAIT") or a.startswith("SKIP"):
                    c = "rgba(241, 196, 15, 0.18)"
                else:
                    c = ""
                return [f"background-color: {c}" if c else "" for _ in row]

            format_dict_full = {
                'Score': '{:.2f}',
                'MaxGap%': '{:.2f}%',
                'AvgTurnover_Oku': '{:.2f} 億円',
                'Price': '{:,.1f} 円',
                'Shares': '{:,}',
                'Shares_Rounded': '{:,}',
                'Position_Size': '{:,} 円'
            }

            if is_mobile:
                # スマホでは横スクロールを避けるため要点列のみ表示
                mobile_cols = ["Rank", "Ticker", "Name", "Action", "Score", "Price"]
                st.dataframe(
                    filtered_df[mobile_cols].style.apply(_row_bg, axis=1).format({
                        'Score': '{:.2f}',
                        'Price': '{:,.1f} 円',
                    }),
                    width="stretch",
                    hide_index=True,
                )
                with st.expander("詳細列も表示する"):
                    st.dataframe(
                        filtered_df.style.apply(_row_bg, axis=1).format(format_dict_full),
                        width="stretch",
                        hide_index=True,
                    )
            else:
                st.dataframe(
                    filtered_df.style.apply(_row_bg, axis=1).format(format_dict_full),
                    width="stretch",
                    hide_index=True,
                )

            st.caption(f"フィルタ後 {len(filtered_df)} 件 / 全 {len(output_df)} 件を表示中（CSVダウンロードは全件です）")

            csv_data = output_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                "📥 結果をCSVとしてダウンロード",
                data=csv_data,
                file_name=f"ranking_result_{rank_result['generated_at'].strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )

        # ====================================================================
        # タブ2: 売買アクション
        # ====================================================================
        with tab_action:
            valid_portfolio = portfolio_df.dropna(subset=["Ticker"]).copy()
            valid_portfolio = valid_portfolio[valid_portfolio["Ticker"].astype(str).str.strip() != ""]

            if valid_portfolio.empty:
                st.info("上の「現在のポートフォリオ」に保有銘柄と株数を入力すると、ランキングと比較して具体的な売買アクション（新規購入・売却・買い増し等）が表示されます。")
            else:
                compared_df, missing_df = compare_with_portfolio(rank_df, valid_portfolio)

                action_order = [
                    "🔴 全株売却", "⚠️ 流動性低下・売却検討", "🆕 新規購入",
                    "🔵 買い増し", "🟡 一部売却(調整)", "❓ 確認要",
                    "✅ 保有継続", "⏸ 保有継続(地合い待ち)",
                ]
                actionable_df = compared_df[compared_df["PortfolioAction"] != "—"].copy()
                actionable_df["__order"] = actionable_df["PortfolioAction"].apply(
                    lambda x: action_order.index(x) if x in action_order else len(action_order)
                )
                actionable_df = actionable_df.sort_values(["__order", "Rank"]).drop(columns="__order")

                n_buy = (actionable_df["PortfolioAction"] == "🆕 新規購入").sum()
                n_sell = actionable_df["PortfolioAction"].isin(["🔴 全株売却", "⚠️ 流動性低下・売却検討"]).sum()
                n_add = (actionable_df["PortfolioAction"] == "🔵 買い増し").sum()
                n_trim = (actionable_df["PortfolioAction"] == "🟡 一部売却(調整)").sum()

                if is_mobile:
                    r1c1, r1c2 = st.columns(2)
                    r1c1.metric("🆕 新規購入", f"{n_buy} 銘柄")
                    r1c2.metric("🔴 売却", f"{n_sell} 銘柄")
                    r2c1, r2c2 = st.columns(2)
                    r2c1.metric("🔵 買い増し", f"{n_add} 銘柄")
                    r2c2.metric("🟡 調整売却", f"{n_trim} 銘柄")
                else:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("🆕 新規購入", f"{n_buy} 銘柄")
                    c2.metric("🔴 売却", f"{n_sell} 銘柄")
                    c3.metric("🔵 買い増し", f"{n_add} 銘柄")
                    c4.metric("🟡 調整売却", f"{n_trim} 銘柄")

                if actionable_df.empty:
                    st.success("現在のポートフォリオに対して、新たに取るべきアクションはありません（全銘柄が保有継続）。")
                else:
                    compare_cols = [
                        "Ticker", "Name", "Rank", "Action", "PortfolioAction",
                        "CurrentShares", "Shares_Rounded", "TradeShares", "TradeAmount", "Price",
                    ]
                    compare_display = actionable_df[compare_cols].rename(
                        columns={"Shares_Rounded": "TargetShares"}
                    )

                    if is_mobile:
                        render_action_cards(actionable_df)
                    else:
                        st.dataframe(
                            compare_display.style.format({
                                "Price": "{:,.1f} 円",
                                "CurrentShares": "{:,}",
                                "TargetShares": "{:,}",
                                "TradeShares": "{:+,}",
                                "TradeAmount": "{:+,.0f} 円",
                            }),
                            width="stretch"
                        )

                    compare_csv = compare_display.to_csv(index=False, encoding="utf-8-sig")
                    st.download_button(
                        "📥 アクションリストをCSVでダウンロード",
                        data=compare_csv,
                        file_name=f"portfolio_actions_{rank_result['generated_at'].strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        key="portfolio_action_download",
                    )

                if not missing_df.empty:
                    st.warning("以下の保有銘柄はランキング対象外（データ取得不可・対象ユニバース外）のため個別にご確認ください。")
                    st.dataframe(missing_df, width="stretch")

        # ====================================================================
        # タブ3: 資金サマリー
        # ====================================================================
        with tab_capital:
            render_capital_summary(limit_info, stress_drop_input,
                                   assumed_dd_input, target_ratio_input,
                                   margin_rate_input, holding_months_input,
                                   stats=margin_stats)
