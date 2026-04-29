import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import streamlit as st
from streamlit_folium import folium_static
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(
    page_title="Olist E-Commerce Dashboard",
    page_icon="📦",
    layout="wide"
)

# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_csv('dashboard/main_data.csv')
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    df['year']       = df['order_purchase_timestamp'].dt.year
    df['month']      = df['order_purchase_timestamp'].dt.month
    df['year_month'] = df['order_purchase_timestamp'].dt.to_period('M').astype(str)
    return df

df = load_data()

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/5/5c/Olist_logo.png", width=200)
st.sidebar.title("📦 Olist Dashboard")
st.sidebar.markdown("---")

year_filter = st.sidebar.multiselect(
    "Pilih Tahun:",
    options=[2017, 2018],
    default=[2017, 2018]
)

df_filtered = df[df['year'].isin(year_filter)]

st.sidebar.markdown("---")
st.sidebar.markdown("**Dibuat oleh:** Aldini Dziaul Haq")
st.sidebar.markdown("**Dataset:** Olist E-Commerce")

# ============================================================
# HEADER
# ============================================================
st.title("📦 Olist E-Commerce Dashboard")
st.markdown("Analisis data transaksi e-commerce Olist Brazil periode 2017–2018")
st.markdown("---")

# ============================================================
# METRIC CARDS
# ============================================================
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Order",    f"{df_filtered['order_id'].nunique():,}")
col2.metric("Total Revenue",  f"R$ {df_filtered['payment_value'].sum():,.0f}")
col3.metric("Total Customer", f"{df_filtered['customer_unique_id'].nunique():,}")
col4.metric("Rata-rata Score",f"{df_filtered['review_score'].mean():.2f}" if 'review_score' in df_filtered.columns else "N/A")

st.markdown("---")

# ============================================================
# TAB
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Tren Penjualan",
    "⭐ Kepuasan Pelanggan",
    "👥 RFM Analysis",
    "🗺️ Geospatial"
])

# ── TAB 1: TREN PENJUALAN ────────────────────────────────
with tab1:
    st.subheader("📈 Tren Penjualan Bulanan 2017–2018")

    monthly = (df_filtered.groupby('year_month', as_index=False)
        .agg(
            total_order   = ('order_id',      'nunique'),
            total_revenue = ('payment_value', 'sum')
        )
        .sort_values('year_month')
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    axes[0].plot(monthly['year_month'], monthly['total_order'],
                 marker='o', color='#2196F3', linewidth=2.5)
    axes[0].fill_between(monthly['year_month'], monthly['total_order'],
                         alpha=0.15, color='#2196F3')
    axes[0].set_title('Tren Jumlah Order Bulanan', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Jumlah Order')
    axes[0].tick_params(axis='x', rotation=45)

    axes[1].plot(monthly['year_month'], monthly['total_revenue'],
                 marker='o', color='#4CAF50', linewidth=2.5)
    axes[1].fill_between(monthly['year_month'], monthly['total_revenue'],
                         alpha=0.15, color='#4CAF50')
    axes[1].set_title('Tren Total Revenue Bulanan', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Total Revenue (R$)')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    st.pyplot(fig)

# ── TAB 2: KEPUASAN PELANGGAN ────────────────────────────
with tab2:
    st.subheader("⭐ Review Score per Kategori Produk")

    if 'review_score' in df_filtered.columns:
        cat_review = (df_filtered
            .groupby('product_category_name_english', as_index=False)
            .agg(
                avg_score     = ('review_score', 'mean'),
                total_order   = ('order_id',     'nunique'),
                low_score_pct = ('review_score', lambda x: (x <= 2).sum() / len(x) * 100)
            )
        )
        cat_review = cat_review.dropna(subset=['product_category_name_english'])
        cat_review = cat_review[cat_review['total_order'] >= 30].sort_values('avg_score')
        bottom10   = cat_review.head(10)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        colors = ['#ef5350' if s < 3.5 else '#66BB6A' for s in bottom10['avg_score']]
        axes[0].barh(bottom10['product_category_name_english'], bottom10['avg_score'], color=colors)
        axes[0].axvline(x=3.5, color='gray', linestyle='--', linewidth=1.5)
        axes[0].set_title('10 Kategori Review Score Terendah', fontsize=13, fontweight='bold')
        axes[0].set_xlabel('Rata-Rata Review Score')

        axes[1].barh(bottom10['product_category_name_english'], bottom10['low_score_pct'], color='#FF7043')
        axes[1].set_title('Persentase Order Score ≤ 2', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('Persentase (%)')

        plt.tight_layout()
        st.pyplot(fig)

# ── TAB 3: RFM ANALYSIS ──────────────────────────────────
with tab3:
    st.subheader("👥 RFM Analysis — Segmentasi Pelanggan")

    snapshot_date = df_filtered['order_purchase_timestamp'].max() + pd.Timedelta(days=1)

    rfm = (df_filtered.groupby('customer_unique_id', as_index=False)
        .agg(
            recency   = ('order_purchase_timestamp', lambda x: (snapshot_date - x.max()).days),
            frequency = ('order_id',                 'nunique'),
            monetary  = ('payment_value',            'sum')
        )
    )

    rfm['r_score'] = pd.qcut(rfm['recency'],   5, labels=[5,4,3,2,1])
    rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
    rfm['m_score'] = pd.qcut(rfm['monetary'],  5, labels=[1,2,3,4,5])

    def segment_rfm(row):
        r, f, m = int(row['r_score']), int(row['f_score']), int(row['m_score'])
        if r >= 4 and f >= 4 and m >= 4:
            return 'Champions'
        elif r >= 3 and f >= 3:
            return 'Loyal Customers'
        elif r >= 4 and f <= 2:
            return 'New Customers'
        elif r <= 2 and f >= 3:
            return 'At Risk'
        else:
            return 'Lost Customers'

    rfm['segment'] = rfm.apply(segment_rfm, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    segment_count = rfm['segment'].value_counts()
    colors = ['#4CAF50','#2196F3','#FF9800','#F44336','#9C27B0']

    axes[0].pie(segment_count, labels=segment_count.index,
                autopct='%1.1f%%', colors=colors,
                startangle=140, wedgeprops={'edgecolor':'white'})
    axes[0].set_title('Distribusi Segmen Pelanggan', fontsize=13, fontweight='bold')

    seg_monetary = rfm.groupby('segment')['monetary'].mean().sort_values(ascending=False)
    axes[1].bar(seg_monetary.index, seg_monetary.values, color=colors)
    axes[1].set_title('Rata-rata Monetary per Segmen', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Segmen')
    axes[1].set_ylabel('Rata-rata Total Belanja (R$)')
    axes[1].tick_params(axis='x', rotation=15)

    plt.tight_layout()
    st.pyplot(fig)

    st.dataframe(rfm[['customer_unique_id','recency','frequency','monetary','segment']]
                 .sort_values('monetary', ascending=False).head(20))

# ── TAB 4: GEOSPATIAL ────────────────────────────────────
with tab4:
    st.subheader("🗺️ Distribusi Order per State di Brazil")

    state_stats = (df_filtered.groupby('customer_state', as_index=False)
        .agg(
            total_order   = ('order_id',      'nunique'),
            total_revenue = ('payment_value', 'sum')
        )
        .sort_values('total_order', ascending=False)
    )

    state_coords = {
        'SP': (-23.55,-46.63), 'RJ': (-22.91,-43.17), 'MG': (-19.92,-43.93),
        'RS': (-30.03,-51.22), 'PR': (-25.43,-49.27), 'SC': (-27.60,-48.55),
        'BA': (-12.97,-38.50), 'GO': (-16.69,-49.26), 'ES': (-20.32,-40.31),
        'PE': (-8.05, -34.88), 'CE': (-3.72, -38.54), 'MT': (-15.60,-56.10),
        'MS': (-20.45,-54.63), 'DF': (-15.78,-47.93), 'PB': (-7.12, -34.86),
        'MA': (-2.53, -44.30), 'AM': (-3.10, -60.03), 'RN': (-5.79, -35.21),
        'AL': (-9.67, -35.74), 'PI': (-5.09, -42.80), 'PA': (-1.46, -48.50),
        'SE': (-10.95,-37.07), 'RO': (-8.76, -63.90), 'TO': (-10.25,-48.32),
        'AC': (-9.02, -70.81), 'AP': (0.03,  -51.07), 'RR': (2.82,  -60.68)
    }

    state_stats['lat'] = state_stats['customer_state'].map(lambda x: state_coords.get(x,(0,0))[0])
    state_stats['lng'] = state_stats['customer_state'].map(lambda x: state_coords.get(x,(0,0))[1])

    m = folium.Map(location=[-15.0, -50.0], zoom_start=4, tiles='CartoDB positron')

    for _, row in state_stats.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lng']],
            radius=row['total_order'] / state_stats['total_order'].max() * 40,
            color='#2196F3',
            fill=True,
            fill_color='#2196F3',
            fill_opacity=0.6,
            tooltip=f"{row['customer_state']}: {row['total_order']:,} orders | R$ {row['total_revenue']:,.0f}"
        ).add_to(m)

    folium_static(m)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top 10 State — Jumlah Order**")
        st.dataframe(state_stats[['customer_state','total_order']].head(10))
    with col2:
        st.markdown("**Top 10 State — Total Revenue**")
        st.dataframe(state_stats[['customer_state','total_revenue']]
                     .sort_values('total_revenue', ascending=False).head(10))