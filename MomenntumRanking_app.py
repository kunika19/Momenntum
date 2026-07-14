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

RISK_FACTOR = 0.002  
CASH_ADJUST = 0.96  
MIN_DAILY_TURNOVER = 1 * 10**8  
MAX_GAP_THRESHOLD = 15.0  
MOMENTUM_WINDOW = 90      
ATR_WINDOW = 14           # 真のATR(14)（旧: VOLATILITY_WINDOW=20 の20日レンジ/2）

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

def run_ranking_process(ticker_df, current_golnas_value):
    tickers = ticker_df['Ticker'].tolist()
    name_map = dict(zip(ticker_df['Ticker'], ticker_df['Name']))

    raw_data = yf.download(tickers + ["^N225"], period="1y", progress=False)
    market_data = raw_data['Close']
    volume_data = raw_data['Volume']
    high_data = raw_data['High']
    low_data = raw_data['Low']
    
    virtual_capital = current_golnas_value * CASH_ADJUST
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


def _rerun_app():
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()


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
    """アクション一覧をスマホで見やすい縦並びのカードHTMLとして描画する"""
    for _, row in df.iterrows():
        action = row["PortfolioAction"]
        color = ACTION_COLORS.get(action, "#7f8c8d")
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
                    border-radius:8px;padding:12px 14px;margin-bottom:10px;">
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
st.set_page_config(page_title="モメンタム運用判断システム", layout="wide")

st.title("📈 モメンタム運用判断")
st.write("Topix500などの銘柄群から、クレノー流モメンタムスコアを計算し売買アクションを判定します。")

st.sidebar.header("🔧 設定・資金入力")
capital_input = st.sidebar.number_input("投資可能額 (円)", min_value=0, value=10000000, step=100000)

st.sidebar.subheader("📱 表示モード")
view_mode = st.sidebar.radio(
    "画面の見やすさを選択",
    ["スマホ向け（カード表示）", "PC向け（表表示）"],
    index=0,
    help="iPhoneなど小さい画面ではカード表示が見やすく、PCの大画面では表表示が一覧しやすいです。",
)
is_mobile = view_mode.startswith("スマホ")

st.sidebar.subheader("銘柄データの読み込み")
uploaded_file = st.sidebar.file_uploader("CSVファイルをアップロード", type=["csv"])

ticker_df = None
if uploaded_file is not None:
    try:
        ticker_df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
    except:
        ticker_df = pd.read_csv(uploaded_file, encoding='shift-jis')
    st.sidebar.success("CSVを読み込みました。")
elif os.path.exists(CSV_FILE):
    try:
        ticker_df = pd.read_csv(CSV_FILE, encoding='utf-8-sig')
    except:
        ticker_df = pd.read_csv(CSV_FILE, encoding='shift-jis')
    st.sidebar.info(f"ローカルファイルを使用中: {os.path.basename(CSV_FILE)}")

st.sidebar.subheader("💼 現在のポートフォリオ")
portfolio_file = st.sidebar.file_uploader(
    "保有銘柄CSV（Ticker, Shares列）", type=["csv"], key="portfolio_csv_uploader"
)
st.sidebar.caption("CSVが無くても、本文の入力欄に貼り付け／手入力できます。")

with st.expander("💼 現在のポートフォリオ（保有銘柄・株数）", expanded=True):
    st.caption("ここに現在保有している銘柄と株数を入力してください。CSVアップロード／コピー&ペースト／直接入力のいずれでも構いません。ランキング生成後、ここに入力した内容と比較して取るべきアクションを表示します。")

    # --- ① CSVアップロード（SBI証券「信用建玉一覧」形式、または列名自動認識の通常形式に対応） ---
    if portfolio_file is not None:
        file_sig = f"{portfolio_file.name}_{portfolio_file.size}"
        if st.session_state.get("portfolio_csv_sig") != file_sig:
            raw_bytes = portfolio_file.getvalue()
            decoded_text = None
            for enc in ["utf-8-sig", "cp932", "shift_jis"]:
                try:
                    decoded_text = raw_bytes.decode(enc)
                    break
                except Exception:
                    continue

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
                _rerun_app()

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
            _rerun_app()

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
        use_container_width=True,
        key=editor_key,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker"),
            "Shares": st.column_config.NumberColumn("Shares", min_value=0, step=100),
        },
    )

    # 手入力での変更も含め、編集結果を毎回正本へ書き戻す（次の再実行で復元できるように）
    st.session_state["portfolio_data"] = edited_df.reset_index(drop=True)
    portfolio_df = edited_df

if st.sidebar.button("🚀 ランキングを生成する") and ticker_df is not None:
    with st.spinner("データ取得・計算中..."):
        rank_df, market_ok = run_ranking_process(ticker_df, float(capital_input))
        
    if not rank_df.empty:
        st.subheader(f"📊 運用判断レポート ({datetime.now().strftime('%Y-%m-%d')})")
        if market_ok: st.success("地合い判定: 良好")
        else: st.warning("地合い判定: 慎重")
            
        display_cols = [
            "Rank", "Ticker", "Name", "Action", "Shares_Rounded", "Score",
             "Price","MaxGap%", "AvgTurnover_Oku",  "Shares", "Position_Size"
        ]
        output_df = rank_df[display_cols]
        
        if is_mobile:
            # スマホでは横スクロールを避けるため要点列のみ表示
            mobile_cols = ["Rank", "Ticker", "Name", "Action", "Score", "Price"]
            st.dataframe(
                output_df[mobile_cols].style.format({
                    'Score': '{:.2f}',
                    'Price': '{:,.1f} 円',
                }),
                use_container_width=True,
                hide_index=True,
            )
            with st.expander("詳細列も表示する"):
                st.dataframe(
                    output_df.style.format({
                        'Score': '{:.2f}',
                        'MaxGap%': '{:.2f}%',
                        'AvgTurnover_Oku': '{:.2f} 億円',
                        'Price': '{:,.1f} 円',
                        'Shares': '{:,}',
                        'Shares_Rounded': '{:,}',
                        'Position_Size': '{:,} 円'
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.dataframe(
                output_df.style.format({
                    'Score': '{:.2f}',
                    'MaxGap%': '{:.2f}%',
                    'AvgTurnover_Oku': '{:.2f} 億円',
                    'Price': '{:,.1f} 円',
                    'Shares': '{:,}',
                    'Shares_Rounded': '{:,}',
                    'Position_Size': '{:,} 円'
                }),
                use_container_width=True
            )
        
        csv_data = output_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button("📥 結果をCSVとしてダウンロード", data=csv_data, file_name="ranking_result.csv", mime="text/csv")

        # --------------------------------------------------------------------
        # ポートフォリオ比較 - 取るべきアクション
        # --------------------------------------------------------------------
        st.markdown("---")
        st.subheader("🔄 現在のポートフォリオと比較 - 取るべきアクション")

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
                        use_container_width=True
                    )

                compare_csv = compare_display.to_csv(index=False, encoding="utf-8-sig")
                st.download_button(
                    "📥 アクションリストをCSVでダウンロード",
                    data=compare_csv,
                    file_name="portfolio_actions.csv",
                    mime="text/csv",
                    key="portfolio_action_download",
                )

            if not missing_df.empty:
                st.warning("以下の保有銘柄はランキング対象外（データ取得不可・対象ユニバース外）のため個別にご確認ください。")
                st.dataframe(missing_df, use_container_width=True)
