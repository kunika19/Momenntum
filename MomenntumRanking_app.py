import pandas as pd
import numpy as np
import yfinance as yf
from scipy import stats
import os
from datetime import datetime
import streamlit as str  # stとしてインポートするのが一般的ですが、競合を避けるためフルスペルか別名にします
import streamlit as st

# ==============================================================================
# 1. 運用ルール・定数設定
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
# 環境に応じてCSVのパスを調整してください
CSV_FILE = os.path.join(current_dir, "tickers_filtered.csv")
#CSV_FILE = os.path.join(parent_dir, "Topix500", "tickers_filtered.csv") 

RISK_FACTOR = 0.005  
CASH_ADJUST = 0.96  
MIN_DAILY_TURNOVER = 1 * 10**8  
MAX_GAP_THRESHOLD = 15.0  
MOMENTUM_WINDOW = 60      
VOLATILITY_WINDOW = 20    

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

    # yfinanceでの一括ダウンロード
    raw_data = yf.download(tickers + ["^N225"], period="1y", progress=False)
    market_data = raw_data['Close']
    volume_data = raw_data['Volume']
    
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
            
            vol_range = prices.tail(VOLATILITY_WINDOW)
            atr = (vol_range.max() - vol_range.min()) / 2
            if atr <= 0: atr = 0.01
            
            current_price = float(prices.iloc[-1])
            
            shares = int(risk_budget_per_stock / atr)
            sharesFloorTo100 = (int(risk_budget_per_stock / atr) // 100) * 100
            
            results.append({
                "Rank": 0,
                "Ticker": ticker,
                "Name": name_map.get(ticker, "N/A"),
                "Score": round(score, 2),
                "AvgTurnover_Oku": round(avg_turnover / 10**8, 2),
                "MaxGap%": round(max_gap, 2),
                "Price": round(current_price, 1),
                "Order_Shares": shares,
                "Order_Shares_floorTo100": sharesFloorTo100,
                "Position_Size": int(current_price * shares),
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

# ==============================================================================
# 3. Streamlit WEB UI インターフェース
# ==============================================================================
st.set_page_config(page_title="モメンタム運用判断システム", layout="wide")

st.title("📈 モメンタム運用判断")
st.write("Topix500などの銘柄群から、クレノー流モメンタムスコアを計算し売買アクションを判定します。")

# サイドバー: 設定と入力
st.sidebar.header("🔧 設定・資金入力")

# 資金入力 (カンマ区切り対応のテキスト入力、または数値入力)
capital_input = st.sidebar.number_input(
    "資金 (円)", 
    min_value=0, 
    value=10,000,000, 
    step=100,000
)

# 銘柄CSVファイルのアップローダー（ローカルパスに見つからない場合のバックアップ）
st.sidebar.subheader("銘柄データの読み込み")
uploaded_file = st.sidebar.file_uploader("CSVファイルをアップロード (未選択時はローカルファイルを探索)", type=["csv"])

# CSVデータの確定
ticker_df = None
if uploaded_file is not None:
    try:
        ticker_df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
    except:
        ticker_df = pd.read_csv(uploaded_file, encoding='shift-jis')
    st.sidebar.success("アップロードされたCSVを使用します。")
else:
    if os.path.exists(CSV_FILE):
        try:
            ticker_df = pd.read_csv(CSV_FILE, encoding='utf-8-sig')
        except:
            ticker_df = pd.read_csv(CSV_FILE, encoding='shift-jis')
        st.sidebar.info(f"ローカルファイルを使用します:\n{os.path.basename(CSV_FILE)}")
    else:
        st.sidebar.error("CSVファイルが見つかりません。アップロードしてください。")

# 実行ボタン
if st.sidebar.button("🚀 ランキングを生成する") and ticker_df is not None:
    with st.spinner("Yahoo Finance から最新データを取得し、スコアを計算中..."):
        rank_df, market_ok = run_ranking_process(ticker_df, float(capital_input))
        
    if not rank_df.empty:
        # メイン画面の情報表示
        st.subheader(f"📊 運用判断レポート ({datetime.now().strftime('%Y-%m-%d')})")
        
        # 地合い判定のステータス表示
        if market_ok:
            st.success("地合い判定(日経225 > 200日線): 良好 (新規買い有効)")
        else:
            st.warning("地合い判定(日経225 > 200日線): 慎重 (キャッシュ維持・新規買い抑制)")
            
        # 表示列の整形
        display_cols = [
            "Rank", "Ticker", "Name", "Score", "Action", 
            "MaxGap%", "AvgTurnover_Oku", "Price", "Order_Shares", "Order_Shares_floorTo100", "Position_Size"
        ]
        output_df = rank_df[display_cols]
        
        # 画面上のデータテーブル表示 (スクロールやソートが可能)
        st.dataframe(
            output_df.style.format({
                'Score': '{:.2f}',
                'MaxGap%': '{:.2f}%',
                'AvgTurnover_Oku': '{:.2f} 億円',
                'Price': '{:,.1f} 円',
                'Order_Shares': '{:,}',
                'Order_Shares_floorTo100': '{:,}',
                'Position_Size': '{:,} 円'
            }),
            use_container_width=True
        )
        
        # CSVダウンロードボタン (Tkinterの保存ダイアログの代わり)
        csv_data = output_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 結果をCSVとしてダウンロード",
            data=csv_data,
            file_name=f"ranking_result_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.error("有効なデータを出力できませんでした。銘柄コードや市場データを確認してください。")
