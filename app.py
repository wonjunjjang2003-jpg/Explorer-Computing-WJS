import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from io import BytesIO
from urllib.parse import quote
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import platform
import os

# ─── OS별 한글 폰트 설정 (Ch.12/13 폰트 설정 응용) ─────────────────
# 수업에서는 로컬(Windows/Mac) 기준 폰트 설정을 배웠다.
# Streamlit Cloud(Linux)에는 한글 폰트가 기본 설치되어 있지 않으므로,
# 레포지토리 루트의 packages.txt에 'fonts-nanum'을 추가해 설치한 뒤
# font_manager에 직접 등록해야 차트의 한글이 깨지지 않는다.
os_name = platform.system()
wc_font_path = None

if os_name == 'Darwin':                       # Mac OS
    plt.rc('font', family='AppleGothic')
    wc_font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
elif os_name == 'Windows':                    # Windows
    plt.rc('font', family='Malgun Gothic')
    wc_font_path = "C:/Windows/Fonts/malgun.ttf"
else:                                         # Linux (Streamlit Cloud)
    _font_candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "NanumGothic.ttf"),
    ]
    for _fp in _font_candidates:
        if os.path.exists(_fp):
            fm.fontManager.addfont(_fp)
            _font_name = fm.FontProperties(fname=_fp).get_name()
            plt.rc('font', family=_font_name)
            wc_font_path = _fp
            break
    else:
        plt.rc('font', family='DejaVu Sans')  # 폰트 미발견 시 앱은 계속 동작

plt.rcParams['axes.unicode_minus'] = False    # 음수 표시 깨짐 방지

try:
    from wordcloud import WordCloud
except ImportError:
    WordCloud = None

# 공통 색상 팔레트
COLORS = {'단백질': '#FF6B6B', '탄수화물': '#4ECDC4', '지방': '#FFE66D'}


# ════════════════════════════════════════════════════════════════
# 데이터 로드 (Ch.11 load_data() 패턴 + @st.cache_data)
# ════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    df_shop  = pd.read_csv("protein_products.csv")
    df_words = pd.read_csv("community_keywords.csv")
    df_ml    = pd.read_csv("ml_data.csv")

    # 구버전 CSV에 없는 영양소 컬럼 기본값 추가 (하위 호환)
    macro_defaults = {
        "단백질(g/100g)":  15.0,
        "탄수화물(g/100g)": 15.0,
        "지방(g/100g)":     5.0,
        "수집시각":         "알 수 없음",
    }
    for col, val in macro_defaults.items():
        if col not in df_shop.columns:
            df_shop[col] = val

    if "수집시각" not in df_words.columns:
        df_words["수집시각"] = "알 수 없음"
    if "사이트" not in df_words.columns:
        df_words["사이트"] = "알 수 없음"

    if "수집방식" not in df_shop.columns:
        df_shop["수집방식"] = "알 수 없음"
    if "검색어" not in df_shop.columns:
        df_shop["검색어"] = ""

    # ── 커뮤니티 데이터 정제 ──────────────────────────────────
    # 크롤링 시 페이지네이션 문제로 동일 게시글이 중복 수집될 수 있어
    # 분석 왜곡을 막기 위해 제목 기준으로 중복을 제거한다.
    n_raw = len(df_words)
    df_words = df_words.drop_duplicates(subset="title").reset_index(drop=True)

    # 사이트 공통 내비게이션·광고성 텍스트 제거
    JUNK_PATTERNS = ["SELECT ITEMS", "임상시험", "생동성시험", "참여자 모집"]
    df_words = df_words[~df_words["title"].astype(str)
                        .str.contains("|".join(JUNK_PATTERNS), na=False)]
    df_words = df_words.reset_index(drop=True)

    # ── 쇼핑 데이터 숫자 타입 보정 및 가격 지표 재계산 ────────────
    # 구버전 CSV의 단위 오류(100g당 가격이 g당 가격으로 저장된 경우 등)를 방어한다.
    for col in ['가격(원)', '총중량(g)', '핵심영양소(g)',
                '단백질(g/100g)', '탄수화물(g/100g)', '지방(g/100g)']:
        if col in df_shop.columns:
            df_shop[col] = pd.to_numeric(df_shop[col], errors='coerce')

    # 구버전 CSV에 핵심영양소(g) 컬럼이 없거나 모두 0/NaN이면 매크로 프로필로 재계산
    if ('핵심영양소(g)' not in df_shop.columns
            or df_shop['핵심영양소(g)'].replace(0, np.nan).isna().all()):
        def _calc_core_g(row):
            w = row.get('총중량(g)', 0) or 0
            cat = str(row.get('카테고리', ''))
            if cat == '단백질':
                return round(w * (row.get('단백질(g/100g)', 0) or 0) / 100)
            elif cat == '탄수화물':
                return round(w * (row.get('탄수화물(g/100g)', 0) or 0) / 100)
            else:
                return round(w * (row.get('지방(g/100g)', 0) or 0) / 100)
        df_shop['핵심영양소(g)'] = df_shop.apply(_calc_core_g, axis=1)

    df_shop['100g당 가격(원)'] = (
        df_shop['가격(원)'] / df_shop['총중량(g)'] * 100
    ).round(1)

    df_shop['핵심영양소 1g당 가격(원/g)'] = (
        df_shop['가격(원)'] / df_shop['핵심영양소(g)'].replace(0, np.nan)
    ).round(1)

    # 총중량 0 등 비정상값으로 생기는 inf를 NaN으로 교체 후 제거
    df_shop['100g당 가격(원)'] = df_shop['100g당 가격(원)'].replace([np.inf, -np.inf], np.nan)
    df_shop['핵심영양소 1g당 가격(원/g)'] = df_shop['핵심영양소 1g당 가격(원/g)'].replace([np.inf, -np.inf], np.nan)

    df_shop = df_shop.dropna(
        subset=['100g당 가격(원)', '가격(원)', '총중량(g)']
    ).reset_index(drop=True)

    return df_shop, df_words, df_ml, n_raw


# ════════════════════════════════════════════════════════════════
# ML 모델 학습 (Ch.11 cars_predict()의 LinearRegression을 확장)
# 수업에서 배운 선형 회귀를 기반으로, 운동 강도와 단백질 필요량의
# 비선형 관계를 반영하기 위해 2차 다항 특성 + Ridge 정규화를 적용
# ════════════════════════════════════════════════════════════════
@st.cache_resource
def train_model(n_samples: int, data_mtime: float = 0.0):
    _ = (n_samples, data_mtime)  # @st.cache_resource 캐시 키 — 행 수 + 파일 수정 시각
    df = pd.read_csv("ml_data.csv")
    X = df[['체중', '운동일수', '운동시간']]
    y = df['필요단백질']
    # 학습 데이터와 테스트 데이터 분리 (Ch.11 train_test_split)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe = Pipeline([
        ('poly',  PolynomialFeatures(degree=2, include_bias=False)),
        ('ridge', Ridge(alpha=1.0)),
    ])
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    metrics = {
        "r2":   r2_score(y_te, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_te, y_pred)),
        "mae":  mean_absolute_error(y_te, y_pred),
        "n_train": len(X_tr),
        "n_test":  len(X_te),
        "y_test":  y_te,
        "y_pred":  y_pred,
    }
    return pipe, metrics, X


# ════════════════════════════════════════════════════════════════
# ① 프로젝트 개요 & 데이터 수집
# ════════════════════════════════════════════════════════════════
def page_overview(df_shop, df_words, df_ml, n_words_raw,
                  shop_collected_at, words_collected_at):
    st.header("① 프로젝트 개요 & 데이터 수집")

    st.subheader("💡 기획 의도")
    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.write("""
        고강도 운동을 병행하는 자취생은 두 가지 딜레마에 직면합니다.

        1. **비용 문제** – 단백질 식품은 일반 식품보다 비싸 식비 부담이 큽니다.
        2. **정보 부족** – 어떤 식품이 가장 가성비가 좋은지 비교하기 어렵습니다.

        이 프로젝트는 **크롤링 → 데이터 정제 → 머신러닝 → 시각화**의 전 과정을 자동화하여,
        사용자가 자신의 신체 조건과 예산에 맞는 비용 효율적인 단백질 구매 조합을 즉시 확인할 수 있도록 합니다.
        """)
    with col_r:
        st.metric("수집된 상품 수", f"{len(df_shop)}개")
        st.metric("고유 게시글 수", f"{len(df_words)}건",
                  help=f"원본 수집 {n_words_raw}건에서 중복·광고성 게시글을 제거한 수치입니다.")
        st.metric("ML 학습 데이터", f"{len(df_ml)}개 샘플")

    st.markdown("---")

    st.subheader("🛠️ 데이터 파이프라인")
    st.markdown("""
    <div style="display:flex; justify-content:center; align-items:center;
                padding:18px; background:#f8f9fa; border-radius:12px;
                font-size:15px; gap:10px; flex-wrap:wrap; margin-bottom:8px;">
        <span style="background:#FF6B6B; color:white; padding:8px 16px; border-radius:8px; font-weight:bold;">📰 커뮤니티 크롤링<br><small style='font-weight:normal'>BeautifulSoup</small></span>
        <span style="font-size:20px; color:#888;">→</span>
        <span style="background:#4ECDC4; color:white; padding:8px 16px; border-radius:8px; font-weight:bold;">🛒 쇼핑 크롤링<br><small style='font-weight:normal'>Selenium</small></span>
        <span style="font-size:20px; color:#888;">→</span>
        <span style="background:#45B7D1; color:white; padding:8px 16px; border-radius:8px; font-weight:bold;">🔧 데이터 정제<br><small style='font-weight:normal'>Pandas</small></span>
        <span style="font-size:20px; color:#888;">→</span>
        <span style="background:#96CEB4; color:white; padding:8px 16px; border-radius:8px; font-weight:bold;">🤖 ML 학습<br><small style='font-weight:normal'>scikit-learn</small></span>
        <span style="font-size:20px; color:#888;">→</span>
        <span style="background:#FFE66D; color:#333; padding:8px 16px; border-radius:8px; font-weight:bold;">📊 시각화·배포<br><small style='font-weight:normal'>Streamlit</small></span>
    </div>
    <p style="text-align:center; color:#888; font-size:12px; margin-top:4px;">데이터 수집 → 정제 → 분석 → 시각화 파이프라인</p>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📰 커뮤니티 크롤링 (BS4)", "🛒 쇼핑 크롤링 (Selenium)", "🧠 ML 데이터 생성"])

    with tab1:
        st.markdown("**대상:** 디시인사이드 영양갤 · 몬스터짐 커뮤니티 · 블라인드")
        st.caption(f"🕐 커뮤니티 데이터 수집 시각: **{words_collected_at}**")
        # 수업(Ch.12)에서 배운 requests + BeautifulSoup + find_all(class_=...) 패턴
        st.code("""
url = f"https://gall.dcinside.com/mgallery/board/lists/?id=nutrient&s_keyword=닭가슴살&page={page}"
res = requests.get(url, headers=headers)           # User-Agent 헤더 필수 (Ch.12)
soup = BeautifulSoup(res.text, "html.parser")      # html.parser로 파싱
titles = soup.find_all('td', class_='gall_tit')    # 개발자 도구로 태그·속성 확인
for t in titles:
    a_tag = t.find('a')
    if a_tag:
        keywords.append(a_tag.get_text().strip())
        """, language="python")
        st.info("💡 **데이터 정제**: 원본 커뮤니티 제목을 수집한 뒤 중복·광고성 문구를 제거하고, "
                "단백질 관련 타겟 키워드가 포함된 제목을 중심으로 빈도 분석합니다. "
                "페이지네이션 중복 수집 방지(seen 집합), '마프→마이프로틴' 등 "
                "동의어 처리, **부정적 맥락('최악', '환불' 등)** 게시글은 텍스트 마이닝에서 별도 처리합니다.")

    with tab2:
        st.markdown("**대상:** 네이버 쇼핑 — 단백질·탄수화물·지방 카테고리별 최저가 수집")
        st.caption(f"🕐 쇼핑 데이터 수집/작성 시각: **{shop_collected_at}** *(백업 데이터의 경우 데이터 작성 시각이며, 실시간 최저가와 다를 수 있습니다)*")
        # 수업(Ch.13)에서 배운 Selenium + 스크롤 + find_elements + 다단계 fallback 패턴
        st.code("""
driver.get(url)
driver.implicitly_wait(10)                         # 로딩 대기 (Ch.13)
driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
time.sleep(2)

# [1단계] CSS selector 다단계 시도 (UI 업데이트 대비)
for sel in ['[class^="product_item__"]', '[class*="product_card"]', ...]:
    items = driver.find_elements(By.CSS_SELECTOR, sel)
    if items: break

# [2단계] 카드 못 찾으면 data-shp-contents-dtl 속성 a태그 fallback
if not items:
    items = driver.find_elements(By.CSS_SELECTOR, 'a[data-shp-contents-dtl]')

# [3단계] 여전히 없으면 쇼핑 홈 검색창 입력 방식으로 재시도

for item in items[:20]:
    title = _try_find_text(item, TITLE_SELECTORS)   # selector 다단계 시도
    if not title or not price:                       # item.text 최종 fallback
        title, price = _extract_from_raw_text(item.text)
    if is_ad_product(title): continue               # 광고 상품 제외
    weight_g = extract_total_weight_g(title)         # 사은품/증정 표기 무시

# 크롤링 실패 시 사전 정제된 백업 데이터로 자동 전환 (Fail-safe)
        """, language="python")
        if "수집방식" in df_shop.columns:
            modes = set(df_shop["수집방식"].dropna().unique())
            if modes == {"크롤링"}:
                _mode_msg = "현재 표시 중인 쇼핑 데이터는 **Selenium으로 수집된 크롤링 결과**입니다."
            elif modes == {"백업"}:
                _mode_msg = ("현재 표시 중인 쇼핑 데이터는 **크롤링 차단으로 백업 데이터가 사용된 결과**이며, "
                             "가격은 백업 데이터 작성 시점 기준입니다.")
            else:
                _mode_msg = f"현재 표시 중인 쇼핑 데이터는 **크롤링/백업 데이터 혼합** 결과입니다. (수집 방식: {', '.join(sorted(modes))})"
        else:
            _mode_msg = "데이터 수집 방식을 확인할 수 없습니다."
        st.warning(f"⚠️ **Fail-safe 시스템:** 안티봇 차단 시 사전 정제된 백업 데이터로 자동 전환됩니다. {_mode_msg}")

    with tab3:
        st.markdown("스포츠 영양학 **권장 범위를 참고한** 시뮬레이션 데이터 (체중 1kg당 단백질 1.2~2.0g 섭취 범위 가정)")
        st.warning("""
        **⚠️ 시뮬레이션 데이터 한계 명시**
        - 이 데이터는 실측 임상 데이터가 아닌 **영양학 공식으로 역산된 가상 데이터**입니다.
        - ML 모델은 실제 패턴을 발견하는 것이 아니라, **공식을 근사 학습**하므로 R² 값이 높게 나타납니다.
        - 개인의 근육량·기초대사량·건강 상태는 반영되지 않으며, **전문가 상담이 필요합니다.**
        """)
        st.code("""
# 운동 강도에 따라 단백질 필요량 1.2~2.0 g/kg 구간에서 증가
intensity = (days / 7) * (hours / 3)           # 0~1 운동강도 지수
protein_per_kg = 1.2 + intensity * 0.8         # 1.2~2.0 g/kg
protein_per_kg = np.minimum(protein_per_kg, 2.2)  # 영양학 권장 상한(2.2g/kg) 안전장치
protein = (weight * protein_per_kg) + noise
        """, language="python")

    st.markdown("---")

    st.subheader("📋 수집된 데이터 미리보기")
    preview_tab1, preview_tab2, preview_tab3 = st.tabs(["🛒 쇼핑 데이터", "💬 커뮤니티 데이터", "🧠 ML 데이터"])
    with preview_tab1:
        st.dataframe(df_shop, use_container_width=True)
    with preview_tab2:
        st.dataframe(df_words.head(20), use_container_width=True)
    with preview_tab3:
        st.dataframe(df_ml.head(10), use_container_width=True)
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("평균 체중", f"{df_ml['체중'].mean():.1f} kg")
        col_b.metric("평균 운동일수", f"{df_ml['운동일수'].mean():.1f} 일/주")
        col_c.metric("평균 권장 단백질", f"{df_ml['필요단백질'].mean():.1f} g/일")


# ════════════════════════════════════════════════════════════════
# ② 식재료 분석 & 가성비 랭킹
# ════════════════════════════════════════════════════════════════
def page_ranking(df_shop, shop_collected_at):
    st.header("② 식재료 분석 & 가성비 랭킹")
    st.caption(f"🕐 데이터 수집 시각: **{shop_collected_at}** *(가격은 수집 시점 기준이며 실시간 최저가와 다를 수 있습니다)*")
    st.write("수집된 쇼핑 데이터를 바탕으로 카테고리별 가성비(100g당 가격)를 분석합니다.")

    df_protein = df_shop[df_shop["카테고리"] == "단백질"]
    df_carb    = df_shop[df_shop["카테고리"] == "탄수화물"]
    df_fat     = df_shop[df_shop["카테고리"] == "지방"]

    st.info("""💡 **두 가지 가성비 기준** — 이 분석에서는 두 기준을 모두 사용합니다.
- **식품 100g당 가격**: 같은 무게의 식품을 구매할 때 드는 비용 (용량 기준 가성비)
- **핵심영양소 1g당 가격**: 실제 목표 영양소 1g을 얻기 위한 비용 (영양 효율 기준 가성비)

예시: WPC는 식품 100g당 가격이 비싸지만 단백질 함량(80g/100g)이 높아 **단백질 1g당 비용은 낮을 수 있습니다**.
반면 두부는 식품 100g당 가격이 저렴해도 단백질 밀도(8g/100g)가 낮아 **단백질 1g당 비용은 높을 수 있습니다**.""")

    c1, c2, c3 = st.columns(3)
    if not df_protein.empty:
        c1.metric("🐔 단백질 식품 100g 최저가",
                  f"{df_protein['100g당 가격(원)'].min():,.0f}원/100g",
                  help="식품 무게 기준 가성비 — 용량 대비 가격")
    if not df_carb.empty:
        c2.metric("🍚 탄수화물 식품 100g 최저가",
                  f"{df_carb['100g당 가격(원)'].min():,.0f}원/100g",
                  help="식품 무게 기준 가성비 — 용량 대비 가격")
    if not df_fat.empty:
        c3.metric("🥜 지방 식품 100g 최저가",
                  f"{df_fat['100g당 가격(원)'].min():,.0f}원/100g",
                  help="식품 무게 기준 가성비 — 용량 대비 가격")

    c4, c5, c6 = st.columns(3)
    if not df_protein.empty:
        c4.metric("🐔 단백질 1g 최저 비용",
                  f"{df_protein['핵심영양소 1g당 가격(원/g)'].min():,.1f}원/g",
                  help="영양 효율 기준 가성비 — 단백질 1g을 얻기 위한 최저 비용")
    if not df_carb.empty:
        c5.metric("🍚 탄수화물 1g 최저 비용",
                  f"{df_carb['핵심영양소 1g당 가격(원/g)'].min():,.1f}원/g",
                  help="영양 효율 기준 가성비 — 탄수화물 1g을 얻기 위한 최저 비용")
    if not df_fat.empty:
        c6.metric("🥜 지방 1g 최저 비용",
                  f"{df_fat['핵심영양소 1g당 가격(원/g)'].min():,.1f}원/g",
                  help="영양 효율 기준 가성비 — 지방 1g을 얻기 위한 최저 비용")

    st.markdown("---")

    st.subheader("📊 카테고리별 가성비 비교 시각화")
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("**① 식품 100g당 가격 비교 (용량 기준)**")
        n_items = len(df_shop)
        fig_h = max(5, n_items * 0.45)
        fig1, ax1 = plt.subplots(figsize=(6, fig_h))
        for cat, grp in df_shop.groupby("카테고리"):
            ax1.barh(grp["상품명"].str[:15], grp["100g당 가격(원)"],
                     color=COLORS.get(cat, '#ccc'), label=cat, height=0.6)
        ax1.set_xlabel("식품 100g당 가격 (원)")
        ax1.tick_params(axis='y', labelsize=8)
        ax1.legend()
        ax1.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close()
        st.caption("낮을수록 같은 무게의 식품을 더 저렴하게 구매 가능")

    with col_chart2:
        st.markdown("**② 핵심영양소 1g당 가격 비교 (영양 효율 기준)**")
        fig2, ax2 = plt.subplots(figsize=(6, fig_h))
        for cat, grp in df_shop.groupby("카테고리"):
            ax2.barh(grp["상품명"].str[:15], grp["핵심영양소 1g당 가격(원/g)"],
                     color=COLORS.get(cat, '#ccc'), label=cat, height=0.6)
        ax2.set_xlabel("핵심영양소 1g당 가격 (원/g)")
        ax2.tick_params(axis='y', labelsize=8)
        ax2.legend()
        ax2.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()
        st.caption("낮을수록 원하는 영양소 1g을 더 저렴하게 섭취 가능")

    st.markdown("**③ 카테고리별 평균 비교**")
    col_avg1, col_avg2 = st.columns(2)
    with col_avg1:
        fig3, ax3 = plt.subplots(figsize=(5, 3))
        cat_avg = df_shop.groupby("카테고리")["100g당 가격(원)"].mean()
        bar_colors = [COLORS.get(c, '#ccc') for c in cat_avg.index]
        ax3.bar(cat_avg.index, cat_avg.values, color=bar_colors, width=0.5)
        for i, v in enumerate(cat_avg.values):
            ax3.text(i, v + 5, f"{v:,.0f}원", ha='center', fontsize=9, fontweight='bold')
        ax3.set_ylabel("평균 100g당 가격 (원)")
        ax3.set_title("카테고리별 평균 (용량 기준)")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()
    with col_avg2:
        fig4, ax4 = plt.subplots(figsize=(5, 3))
        cat_eff = df_shop.groupby("카테고리")["핵심영양소 1g당 가격(원/g)"].mean()
        bar_colors2 = [COLORS.get(c, '#ccc') for c in cat_eff.index]
        ax4.bar(cat_eff.index, cat_eff.values, color=bar_colors2, width=0.5)
        for i, v in enumerate(cat_eff.values):
            ax4.text(i, v + 0.5, f"{v:.1f}원", ha='center', fontsize=9, fontweight='bold')
        ax4.set_ylabel("평균 핵심영양소 1g당 가격 (원/g)")
        ax4.set_title("카테고리별 평균 (영양 효율 기준)")
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()

    st.markdown("---")

    st.subheader("🏆 전체 랭킹")
    tab_rank1, tab_rank2 = st.tabs(["📦 식품 100g당 가격 기준", "💊 핵심영양소 1g당 가격 기준"])

    with tab_rank1:
        st.caption("* 쇼핑 링크 클릭 시 네이버 쇼핑에서 실제 검색 결과를 확인할 수 있습니다.")
        df_display = df_shop.copy().sort_values("100g당 가격(원)")
        df_display["쇼핑 링크"] = df_display["상품명"].apply(
            lambda x: f"https://search.shopping.naver.com/search/all?query={quote(x[:10])}"
        )
        st.dataframe(
            df_display[["카테고리", "상품명", "가격(원)", "총중량(g)", "100g당 가격(원)",
                         "핵심영양소(g)", "핵심영양소 1g당 가격(원/g)", "쇼핑 링크"]],
            column_config={"쇼핑 링크": st.column_config.LinkColumn("🛒 검색하기")},
            use_container_width=True
        )

    with tab_rank2:
        st.caption("영양소 1g을 얻기 위한 비용이 낮을수록 영양 효율 기준 가성비가 높습니다.")
        df_display2 = df_shop.copy().sort_values("핵심영양소 1g당 가격(원/g)")
        df_display2["쇼핑 링크"] = df_display2["상품명"].apply(
            lambda x: f"https://search.shopping.naver.com/search/all?query={quote(x[:10])}"
        )
        st.dataframe(
            df_display2[["카테고리", "상품명", "가격(원)", "총중량(g)", "100g당 가격(원)",
                          "핵심영양소(g)", "핵심영양소 1g당 가격(원/g)", "쇼핑 링크"]],
            column_config={"쇼핑 링크": st.column_config.LinkColumn("🛒 검색하기")},
            use_container_width=True
        )

    st.markdown("---")
    st.subheader("📝 분석 결과 해석")

    if df_protein.empty or df_carb.empty or df_fat.empty:
        st.warning("⚠️ 일부 카테고리 데이터가 없어 분석 결과를 표시할 수 없습니다.")
    else:
        best_p = df_protein.loc[df_protein["100g당 가격(원)"].idxmin()]
        best_c = df_carb.loc[df_carb["100g당 가격(원)"].idxmin()]
        best_f = df_fat.loc[df_fat["100g당 가격(원)"].idxmin()]

        best_p_eff = df_protein.loc[df_protein["핵심영양소 1g당 가격(원/g)"].idxmin()]
        best_c_eff = df_carb.loc[df_carb["핵심영양소 1g당 가격(원/g)"].idxmin()]
        best_f_eff = df_fat.loc[df_fat["핵심영양소 1g당 가격(원/g)"].idxmin()]

        st.success(f"""
**식품 100g당 가격 기준 TOP 1 (용량 대비 가성비)**
- 🐔 단백질: **{best_p['상품명']}** — 100g당 {best_p['100g당 가격(원)']:,.0f}원
- 🍚 탄수화물: **{best_c['상품명']}** — 100g당 {best_c['100g당 가격(원)']:,.0f}원
- 🥜 지방: **{best_f['상품명']}** — 100g당 {best_f['100g당 가격(원)']:,.0f}원
        """)

        st.success(f"""
**핵심영양소 1g당 가격 기준 TOP 1 (영양 효율 가성비)**
- 🐔 단백질 1g당: **{best_p_eff['상품명']}** — {best_p_eff['핵심영양소 1g당 가격(원/g)']:,.1f}원/g
- 🍚 탄수화물 1g당: **{best_c_eff['상품명']}** — {best_c_eff['핵심영양소 1g당 가격(원/g)']:,.1f}원/g
- 🥜 지방 1g당: **{best_f_eff['상품명']}** — {best_f_eff['핵심영양소 1g당 가격(원/g)']:,.1f}원/g
        """)

        st.info("""💡 **기준별 해석**
- **식품 100g당 가격**이 낮다고 영양 효율까지 높은 것은 아닙니다.
- **WPC 보충제**는 단백질 함량(80g/100g)이 높아 단백질 1g당 비용 기준으로 효율적일 수 있습니다.
- **닭가슴살**은 포만감·조리 다양성 측면에서 실사용에 더 적합합니다.
- **두부**는 식품 100g당 가격이 저렴해도 단백질 밀도(8g/100g)가 낮아 단백질 1g당 비용은 상대적으로 높습니다.""")

    st.markdown("---")
    # 엑셀 다운로드 (Ch.14에서 배운 BytesIO + download_button 패턴)
    buffer = BytesIO()
    df_shop.to_excel(buffer, index=False, engine='openpyxl')
    st.download_button(
        "📥 전체 데이터 엑셀 다운로드",
        data=buffer.getvalue(),
        file_name="protein_product_ranking.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# ════════════════════════════════════════════════════════════════
# ③ ML 예측 & 식단 최적화 (Ch.11 cars_predict() 구조 확장)
# ════════════════════════════════════════════════════════════════
def page_predict(df_shop, model, ml_metrics, X_full, budget, shop_collected_at=""):
    st.header("③ ML 예측 & 구매 조합 탐색")

    # ── 모델 설명 ──────────────────────────────────────────────
    st.subheader("🤖 모델: 2차 다항 Ridge 회귀")
    st.markdown("""
    수업에서 배운 **선형 회귀(Linear Regression)**는 변수 간 비선형 관계를
    반영하지 못한다는 한계가 있습니다. 이를 보완하기 위해
    **2차 다항식 특성(제곱항·교호작용항)**을 추가한 뒤,
    과적합 방지를 위해 **Ridge 정규화**를 적용했습니다.

    > ⚠️ **설계 한계:** 학습 데이터는 실측이 아닌 영양학 공식 기반 시뮬레이션이므로
    > R² 값이 높게 나타나는 것은 공식을 역산(reverse-engineering)하는 것에 가깝습니다.
    > 개인별 임상 적용 시 전문가 상담을 권장합니다.
    """)

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("R² (설명력)", f"{ml_metrics['r2']:.3f}", help="1.0에 가까울수록 높은 예측력 (시뮬레이션 기반이므로 참고용)")
    col_m2.metric("RMSE (평균 오차)", f"±{ml_metrics['rmse']:.1f}g", help="예측값이 실제값과 평균적으로 ±이 만큼 차이남")
    col_m3.metric("MAE (절대 평균 오차)", f"±{ml_metrics['mae']:.1f}g", help="이상값 영향이 제거된 평균 오차")
    col_m4.metric("학습 샘플", f"{ml_metrics['n_train']}개")

    col_chart_a, col_chart_b = st.columns(2)

    with col_chart_a:
        with st.expander("📈 실제값 vs 예측값 산점도 보기"):
            fig_val, ax_val = plt.subplots(figsize=(5, 4))
            ax_val.scatter(ml_metrics["y_test"], ml_metrics["y_pred"],
                           alpha=0.5, color='steelblue', label='예측 결과')
            mn = min(ml_metrics["y_test"].min(), ml_metrics["y_pred"].min())
            mx = max(ml_metrics["y_test"].max(), ml_metrics["y_pred"].max())
            ax_val.plot([mn, mx], [mn, mx], 'r--', lw=2, label='완벽한 예측선')
            ax_val.set_xlabel("실제 단백질 필요량 (g)")
            ax_val.set_ylabel("예측 단백질 필요량 (g)")
            ax_val.set_title(f"실제값 vs 예측값 (R²={ml_metrics['r2']:.3f})")
            ax_val.legend()
            plt.tight_layout()
            st.pyplot(fig_val)
            plt.close()
            st.caption(f"RMSE={ml_metrics['rmse']:.1f}g: 예측값이 실제값과 평균 ±{ml_metrics['rmse']:.1f}g 차이가 납니다.")

    with col_chart_b:
        with st.expander("🔗 독립변수 상관관계 히트맵 (다중공선성 확인)"):
            corr = X_full.corr()
            fig_corr, ax_corr = plt.subplots(figsize=(4, 3))
            cmap = plt.get_cmap('coolwarm')
            im = ax_corr.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
            plt.colorbar(im, ax=ax_corr, shrink=0.8)
            ax_corr.set_xticks(range(len(corr.columns)))
            ax_corr.set_yticks(range(len(corr.columns)))
            ax_corr.set_xticklabels(corr.columns, rotation=30, ha='right', fontsize=9)
            ax_corr.set_yticklabels(corr.columns, fontsize=9)
            for i in range(len(corr)):
                for j in range(len(corr.columns)):
                    val = corr.values[i, j]
                    color = 'white' if abs(val) > 0.5 else 'black'
                    ax_corr.text(j, i, f"{val:.2f}", ha='center', va='center',
                                 fontsize=9, color=color)
            ax_corr.set_title("독립변수 상관행렬", fontsize=10)
            plt.tight_layout()
            st.pyplot(fig_corr)
            plt.close()
            max_corr = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().abs().max()
            if max_corr > 0.7:
                st.warning(f"⚠️ 최대 상관계수 {max_corr:.2f}: 일부 변수 간 다중공선성 의심. Ridge 정규화로 부분 완화됩니다.")
            else:
                st.success(f"✅ 최대 상관계수 {max_corr:.2f}: 심각한 다중공선성 없음.")

    st.markdown("---")

    # ── 빈 카테고리 방어 ────────────────────────────────────────
    _missing_cats = [c for c in ["단백질", "탄수화물", "지방"]
                     if df_shop[df_shop["카테고리"] == c].empty]
    if _missing_cats:
        st.error(f"⚠️ {', '.join(_missing_cats)} 카테고리 상품이 없어 구매 조합 탐색이 불가합니다. "
                 "`python collect_data.py`를 재실행해 주세요.")
        return

    # ── 사용자 입력 (Ch.10/11에서 배운 number_input, slider, radio) ──
    st.subheader("👇 나의 정보 입력 → 단백질 필요량 예측")
    col1, col2, col3 = st.columns(3)
    with col1:
        my_weight = st.number_input("체중 (kg)", min_value=40.0, max_value=150.0, value=75.0, step=1.0)
    with col2:
        my_days = st.slider("이번 주 운동 (일)", 1, 7, 5)
    with col3:
        my_hours = st.slider("하루 운동 시간 (시간)", 0.5, 3.0, 1.5, step=0.5)

    goal = st.radio(
        "목표 선택",
        ["💪 벌크업 (고단백/고탄수)", "🔥 다이어트 (고단백/저탄수)"],
        horizontal=True
    )

    # ── 입력값 엣지케이스 검증 ──────────────────────────────────
    if my_weight < 50 and "벌크업" in goal:
        st.warning("⚠️ 체중 50kg 미만의 벌크업 목표는 전문가 상담을 권장합니다. 예측값은 참고용으로만 활용하세요.")
    if budget < 35000 and "벌크업" in goal:
        st.warning(f"⚠️ 예산 {budget:,}원으로 벌크업 영양소를 충족하는 조합을 찾기 어려울 수 있습니다. 예산을 35,000원 이상으로 설정해보세요.")
    if my_days <= 1 and my_hours <= 0.5:
        st.info("💡 운동량이 매우 낮습니다. 기초 단백질(체중 × 1.2g/kg)만으로도 충분할 수 있습니다.")

    # 입력값으로 예측 수행 (Ch.11 cars_predict() 패턴)
    input_df = pd.DataFrame([[my_weight, my_days, my_hours]], columns=['체중', '운동일수', '운동시간'])
    base_protein = model.predict(input_df)[0]
    base_protein = max(base_protein, my_weight * 0.8)  # 하한선: 체중 × 0.8 g/kg

    # 목표별 배율은 프로젝트 가정값임 (실제 임상 기준과 다를 수 있음)
    if "벌크업" in goal:
        p_target = round(base_protein * 1.2, 1)   # 가정: 벌크업 시 ML 예측 × 1.2
        c_target = round(p_target * 2.5, 1)        # 가정: 단백질 대비 탄수화물 2.5배
        f_target = round(p_target * 0.5, 1)        # 가정: 단백질 대비 지방 0.5배
    else:
        p_target = round(base_protein * 1.1, 1)   # 가정: 다이어트 시 ML 예측 × 1.1
        c_target = round(p_target * 1.0, 1)
        f_target = round(p_target * 0.3, 1)

    st.success(f"🎯 **예측 결과:** 단백질 **{p_target:.0f}g** / 탄수화물 **{c_target:.0f}g** / 지방 **{f_target:.0f}g** (하루 목표, 참고용)")
    st.caption("⚠️ 탄수화물·지방 목표는 단백질 예측값에 비례한 **프로젝트 가정값**입니다. 개인 상황에 따라 달라질 수 있습니다.")

    with st.expander("🥧 목표 영양소 비율 파이차트"):
        fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
        ax_pie.pie(
            [p_target, c_target, f_target],
            labels=["단백질", "탄수화물", "지방"],
            colors=["#FF6B6B", "#4ECDC4", "#FFE66D"],
            autopct="%1.1f%%", startangle=90
        )
        ax_pie.set_title(f"목표 영양소 비율 ({'벌크업' if '벌크업' in goal else '다이어트'})")
        st.pyplot(fig_pie)
        plt.close()

    st.markdown("---")

    # ── 식단 최적화 ────────────────────────────────────────────
    SHIPPING_COST  = 3 * 3000   # 판매자 3곳 × 배송비 3,000원 추정
    MAX_PKG_WEIGHT = 2000       # 자취방 냉동고 기준 1회 구매 최대 2kg

    st.subheader(f"🛒 예산 {budget:,}원 내 구매 조합 탐색")
    st.caption(f"배송비 약 {SHIPPING_COST:,}원 포함 (단순 추정값) / 2kg 초과 대용량 단백질·탄수화물 상품 자동 제외 (냉동 보관 제약)")
    st.warning("⚠️ **단위 주의:** 예산은 **1회 구매 기준**, 단백질·탄수화물·지방 목표는 **하루 섭취 기준**입니다. "
               "구매한 상품 패키지는 여러 날에 걸쳐 나눠 먹게 되며, 아래 영양소 합계는 구매 패키지 전체 기준입니다.")
    st.caption("ℹ️ **참고:** 본 추천 조합은 건강성 평가가 아닌 가격·영양소 기준의 비용 효율 비교입니다. "
               "나트륨, 당류, 포화지방, 식품 가공도는 반영하지 않았습니다.")

    # ── session_state 캐시 키 (데이터 버전 포함하여 CSV 교체 시 자동 무효화) ──
    opt_key = (budget, goal, round(my_weight), my_days, my_hours, shop_collected_at)
    if "opt_cache" not in st.session_state:
        st.session_state.opt_cache = {}

    show_cached = opt_key in st.session_state.opt_cache

    if show_cached:
        st.info("이전 계산 결과를 표시합니다. 다른 조건으로 바꾼 후 버튼을 다시 누르면 재계산합니다.")

    run_opt = st.button("🚀 구매 조합 탐색 시작" + (" (재계산)" if show_cached else ""))

    if run_opt or show_cached:
        if run_opt:
            with st.spinner("영양소 교차 기여량까지 반영한 추천 조합 탐색 중..."):

                # 보관 제약: 단백질·탄수화물 상품은 2kg 이하만 사용
                df_p_items = df_shop[df_shop['카테고리'] == '단백질']
                df_c_items = df_shop[df_shop['카테고리'] == '탄수화물']
                df_f_items = df_shop[df_shop['카테고리'] == '지방']

                df_p = df_p_items[df_p_items['총중량(g)'] <= MAX_PKG_WEIGHT].reset_index(drop=True)
                df_c = df_c_items[df_c_items['총중량(g)'] <= MAX_PKG_WEIGHT].reset_index(drop=True)
                df_f = df_f_items.reset_index(drop=True)  # 지방 식품은 상온 보관

                # 필터 후 후보가 없으면 제약 해제
                if df_p.empty:
                    df_p = df_p_items.reset_index(drop=True)
                    st.warning("⚠️ 2kg 이하 단백질 상품이 없어 대용량 포함 탐색합니다.")
                if df_c.empty:
                    df_c = df_c_items.reset_index(drop=True)

                # ── 상품별 패키지당 영양소(g) ──────────────────────
                def pkg_macro(df, key):
                    return (df['총중량(g)'].values * df[f'{key}(g/100g)'].values / 100)

                pp_prot = pkg_macro(df_p, '단백질')
                pp_carb = pkg_macro(df_p, '탄수화물')
                pp_fat  = pkg_macro(df_p, '지방')
                pp_cost = df_p['가격(원)'].values

                cc_prot = pkg_macro(df_c, '단백질')
                cc_carb = pkg_macro(df_c, '탄수화물')
                cc_fat  = pkg_macro(df_c, '지방')
                cc_cost = df_c['가격(원)'].values

                ff_prot = pkg_macro(df_f, '단백질')
                ff_carb = pkg_macro(df_f, '탄수화물')
                ff_fat  = pkg_macro(df_f, '지방')
                ff_cost = df_f['가격(원)'].values

                # ── 수량 계산: 교차 영양소 기여량 반영 ────────────
                # Step 1: 단백질 목표를 충족하는 최소 단백질 식품 수량
                qty_p = np.maximum(1, np.ceil(p_target / np.maximum(pp_prot, 0.01)))  # (n_p,)

                # Step 2: 단백질 식품이 이미 공급하는 탄수화물/지방 차감
                carb_from_p = qty_p * pp_carb   # (n_p,)
                fat_from_p  = qty_p * pp_fat    # (n_p,)

                resid_carb = np.maximum(0, c_target - carb_from_p)  # (n_p,)

                # Step 3: 탄수화물 목표를 충족하는 최소 탄수화물 식품 수량
                qty_c = np.maximum(1, np.ceil(
                    resid_carb[:, None] / np.maximum(cc_carb[None, :], 0.01)
                ))  # (n_p, n_c)

                # Step 4: p+c 합산 후 잔여 지방 계산
                fat_from_pc = fat_from_p[:, None] + qty_c * cc_fat[None, :]   # (n_p, n_c)
                resid_fat   = np.maximum(0, f_target - fat_from_pc)           # (n_p, n_c)

                # Step 5: 지방 목표를 충족하는 최소 지방 식품 수량
                qty_f = np.maximum(1, np.ceil(
                    resid_fat[:, :, None] / np.maximum(ff_fat[None, None, :], 0.01)
                ))  # (n_p, n_c, n_f)

                # ── 총 비용 (배송비 포함) ───────────────────────────
                cost_p = qty_p * pp_cost                            # (n_p,)
                cost_c = qty_c * cc_cost[None, :]                   # (n_p, n_c)
                cost_f = qty_f * ff_cost[None, None, :]             # (n_p, n_c, n_f)
                total_cost = (cost_p[:, None, None]
                              + cost_c[:, :, None]
                              + cost_f
                              + SHIPPING_COST)                      # (n_p, n_c, n_f)

                valid_mask = total_cost <= budget
                valid_idx  = np.argwhere(valid_mask)

                results = {"combinations": [], "df_p": df_p, "df_c": df_c, "df_f": df_f}

                if len(valid_idx) > 0:
                    costs_at_valid = total_cost[valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]]
                    order = np.argsort(costs_at_valid)
                    for k in order:
                        pi, ci, fi = valid_idx[k]
                        qp = int(qty_p[pi])
                        qc = int(qty_c[pi, ci])
                        qf = int(qty_f[pi, ci, fi])

                        t_prot = qp*pp_prot[pi] + qc*cc_prot[ci] + qf*ff_prot[fi]
                        t_carb = qp*pp_carb[pi] + qc*cc_carb[ci] + qf*ff_carb[fi]
                        t_fat  = qp*pp_fat[pi]  + qc*cc_fat[ci]  + qf*ff_fat[fi]

                        tc = int(total_cost[pi, ci, fi])
                        results["combinations"].append({
                            "p": df_p.iloc[pi]['상품명'][:22],
                            "c": df_c.iloc[ci]['상품명'][:22],
                            "f": df_f.iloc[fi]['상품명'][:22],
                            "qp": qp, "qc": qc, "qf": qf,
                            "t_prot": round(t_prot, 1),
                            "t_carb": round(t_carb, 1),
                            "t_fat":  round(t_fat,  1),
                            "product_cost": tc - SHIPPING_COST,
                            "total_cost":   tc,
                        })

                st.session_state.opt_cache[opt_key] = results

        results = st.session_state.opt_cache[opt_key]
        combos  = results["combinations"]

        if not combos:
            st.error(
                f"❌ 예산 {budget:,}원(배송비 {SHIPPING_COST:,}원 포함)으로는 목표 영양소를 충족하는 조합이 없습니다. "
                f"예산을 늘리거나, 목표를 다이어트로 변경하거나, 운동량을 줄여보세요."
            )
        else:
            st.success(f"✅ {len(combos)}개의 예산 내 조합 발견!")
            best  = combos[0]
            worst = combos[-1]

            col_a, col_b = st.columns(2)

            def show_combo(col, title, combo, is_info=True):
                func = col.info if is_info else col.warning
                func(f"**{title}**")
                col.write(f"🐔 단백질: {combo['p']} ×{combo['qp']}")
                col.write(f"🍚 탄수화물: {combo['c']} ×{combo['qc']}")
                col.write(f"🥜 지방: {combo['f']} ×{combo['qf']}")
                col.markdown(f"""
                | 실제 영양소 | 단백질 | 탄수화물 | 지방 |
                |---|---|---|---|
                | **달성량** | {combo['t_prot']:.0f}g | {combo['t_carb']:.0f}g | {combo['t_fat']:.0f}g |
                | **목표** | {p_target:.0f}g | {c_target:.0f}g | {f_target:.0f}g |
                """)
                col.metric("상품 금액", f"{combo['product_cost']:,}원")
                col.metric("배송비 추정", f"+{SHIPPING_COST:,}원 (3개 판매자)")
                col.metric("총 지출", f"{combo['total_cost']:,}원",
                           delta=f"-{budget - combo['total_cost']:,}원 절약")
                if combo['qp'] > 5 or combo['qc'] > 5 or combo['qf'] > 5:
                    col.warning("⚠️ 한 상품을 5개 이상 구매해야 하는 조합입니다. "
                                "식단 단조로움·보관 공간·유통기한을 고려해 주세요.")

            with col_a:
                show_combo(col_a, "🪙 메뉴 A: 예산 절약 세트 (최저 비용)", best, is_info=True)
            with col_b:
                if best["total_cost"] != worst["total_cost"]:
                    show_combo(col_b, "🔖 메뉴 B: 비용 상위 조합 (참고용)", worst, is_info=False)
                else:
                    col_b.info("현재 예산 내에서는 메뉴 A가 유일한 예산 내 조합입니다.")


# ════════════════════════════════════════════════════════════════
# ④ 텍스트 마이닝 & 결론 (Ch.12 단어 빈도 분석 + WordCloud 응용)
# ════════════════════════════════════════════════════════════════
def page_textmining(df_words, words_collected_at, df_shop, ml_metrics):
    st.header("④ 텍스트 마이닝 & 결론")

    st.subheader("🔍 커뮤니티 텍스트 마이닝 분석")
    st.caption(f"🕐 커뮤니티 데이터 수집 시각: **{words_collected_at}**")
    st.write("""
    크롤링한 커뮤니티 게시글 제목에서 단백질 관련 키워드를 추출합니다.
    **부정적 맥락**(최악, 환불, 맛없 등)이 포함된 게시글은 언급 횟수에서 제외합니다.
    """)

    # ── 부정적 맥락 감지 ──────────────────────────────────────
    # 주의: 한 글자 단어는 '노력', '~했노'(긍정 어미) 등에서
    # 오탐을 일으키므로 2글자 이상 표현만 사용한다.
    NEGATIVE_WORDS = {"싫어", "싫음", "최악", "별로", "맛없", "불만", "반품", "환불", "쓰레기",
                      "실망", "구려", "후회", "비추", "노맛", "안좋", "안 좋", "돈아깝"}

    def has_negative_context(text: str, keyword: str) -> bool:
        """키워드 전후 30자 내에 부정어가 있으면 True."""
        pos = text.find(keyword)
        if pos == -1:
            return False
        window = text[max(0, pos - 30): pos + len(keyword) + 30]
        return any(neg in window for neg in NEGATIVE_WORDS)

    category_dict = {
        "🐔 닭가슴살 브랜드": ["허닭", "랭킹닭컴", "맛있닭", "아임닭", "바르닭", "잇메이트", "하림",
                             "한끼통살", "마이닭", "미트리", "푸드나무"],
        "🧃 프로틴 음료":    ["더단백", "셀렉스", "하이뮨", "테이크핏", "랩노쉬", "마이밀",
                              "칼로바이", "이지프로틴", "얼티브"],
        "🍫 보충제/기타":    ["퀘스트바", "베어벨스", "마이프로틴", "신타6", "옵티멈"],
    }
    synonyms = {"마프": "마이프로틴", "헬앤뷰": "헬스앤뷰티", "퀘바": "퀘스트바", "신타": "신타6"}

    all_targets = (
        sum(category_dict.values(), [])
        + list(synonyms.keys())
        + ["스팀", "소시지", "생닭", "스테이크", "햇반", "식빵", "쉐이크", "닭가슴살", "단백질",
           "프로틴", "보충제", "WPC", "WPI", "단백질쉐이크", "프로틴바",
           "닭가슴살소시지", "냉동닭가슴살"]
    )

    words, neg_words = [], []
    for title in df_words["title"]:
        text = str(title)
        seen_pos = set()   # 제목 하나에서 같은 canonical 키워드 중복 카운트 방지
        seen_neg = set()
        for target in all_targets:
            if target in text:
                canonical = synonyms.get(target, target)
                if has_negative_context(text, target):
                    if canonical not in seen_neg:
                        seen_neg.add(canonical)
                        neg_words.append(canonical)
                else:
                    if canonical not in seen_pos:
                        seen_pos.add(canonical)
                        words.append(canonical)       # 긍정/중립 언급만 카운트

    word_counts     = pd.Series(words).value_counts().head(10)
    neg_word_counts = pd.Series(neg_words).value_counts().head(5)

    if word_counts.empty:
        st.warning("현재 수집된 게시글에서 타겟 키워드가 검색되지 않았습니다.")
        st.info("`python collect_data.py`를 재실행해 보세요.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**긍정/중립 언급 TOP 10**")
            fig_bar, ax_bar = plt.subplots(figsize=(5, 4))
            ax_bar.barh(word_counts.index[::-1], word_counts.values[::-1], color='#FF7F50')
            ax_bar.set_xlabel("언급 횟수 (부정 맥락 제외)")
            ax_bar.set_title("커뮤니티 인기 키워드 TOP 10")
            plt.tight_layout()
            st.pyplot(fig_bar)
            plt.close()

            if not neg_word_counts.empty:
                st.markdown("**부정 맥락 언급 TOP 5** (별도 집계)")
                st.dataframe(
                    pd.DataFrame({"키워드": neg_word_counts.index, "부정 언급 수": neg_word_counts.values}),
                    hide_index=True, use_container_width=True
                )

        with col2:
            # 워드클라우드 (Ch.12에서 배운 WordCloud + font_path 패턴)
            if WordCloud is not None and wc_font_path is not None:
                st.markdown("**키워드 워드클라우드** (긍정/중립 언급)")
                wc = WordCloud(
                    font_path=wc_font_path,
                    background_color="white",
                    width=400, height=300,
                    colormap="Oranges"
                ).generate_from_frequencies(word_counts.to_dict())
                fig_wc, ax_wc = plt.subplots(figsize=(5, 4))
                ax_wc.imshow(wc)
                ax_wc.axis("off")
                plt.tight_layout()
                st.pyplot(fig_wc)
                plt.close()
            elif WordCloud is None:
                st.info("💡 `pip install wordcloud`를 설치하면 워드클라우드를 볼 수 있습니다.")
            else:
                st.info("💡 한글 폰트를 찾지 못해 워드클라우드를 생략했습니다. (packages.txt에 fonts-nanum 추가 필요)")

        top_keyword = word_counts.index[0]
        st.success(f"📢 커뮤니티 긍정/중립 언급 1위: **'{top_keyword}'** — 부정 맥락 필터링 후 기준")
        st.write(f"상위 5개 키워드: **{', '.join(word_counts.index[:5].tolist())}**")

    st.markdown("---")

    st.subheader("📌 종합 분석 결론")

    top_kw = word_counts.index[0] if not word_counts.empty else "데이터 없음"

    df_p_all = df_shop[df_shop["카테고리"] == "단백질"]
    df_c_all = df_shop[df_shop["카테고리"] == "탄수화물"]
    df_f_all = df_shop[df_shop["카테고리"] == "지방"]

    if not df_p_all.empty and not df_c_all.empty and not df_f_all.empty:
        best_p   = df_p_all.loc[df_p_all["100g당 가격(원)"].idxmin()]
        best_c   = df_c_all.loc[df_c_all["100g당 가격(원)"].idxmin()]
        best_p_e = df_p_all.loc[df_p_all["핵심영양소 1g당 가격(원/g)"].idxmin()]
        best_c_e = df_c_all.loc[df_c_all["핵심영양소 1g당 가격(원/g)"].idxmin()]
        best_f_e = df_f_all.loc[df_f_all["핵심영양소 1g당 가격(원/g)"].idxmin()]

        st.markdown(f"""
| 분석 항목 | 핵심 결과 |
|---|---|
| 식품 100g당 최저가 단백질 | {best_p['상품명'][:20]} ({best_p['100g당 가격(원)']:,.0f}원/100g) |
| 식품 100g당 최저가 탄수화물 | {best_c['상품명'][:20]} ({best_c['100g당 가격(원)']:,.0f}원/100g) |
| 단백질 1g당 최저 비용 | {best_p_e['상품명'][:20]} ({best_p_e['핵심영양소 1g당 가격(원/g)']:,.1f}원/g) |
| 탄수화물 1g당 최저 비용 | {best_c_e['상품명'][:20]} ({best_c_e['핵심영양소 1g당 가격(원/g)']:,.1f}원/g) |
| 지방 1g당 최저 비용 | {best_f_e['상품명'][:20]} ({best_f_e['핵심영양소 1g당 가격(원/g)']:,.1f}원/g) |
| 커뮤니티 부정 맥락 제외 언급 1위 | {top_kw} |
| ML 예측 오차 (RMSE) | ±{ml_metrics['rmse']:.1f}g (시뮬레이션 기반, 참고용) |
        """)

        st.info(f"""💡 **결론 (데이터 기반)**
- **식품 100g당 가격** 기준 단백질 가성비: **{best_p['상품명'][:15]}** ({best_p['100g당 가격(원)']:,.0f}원/100g)
- **단백질 1g당 비용** 기준 영양 효율: **{best_p_e['상품명'][:15]}** ({best_p_e['핵심영양소 1g당 가격(원/g)']:,.1f}원/g)
- 실제 식단에서는 포만감·맛·조리 편의성을 함께 고려해야 하므로, 위 수치는 참고용으로 활용하세요.
- 개인 건강 상태에 따른 식단은 **전문가 상담**을 권장합니다.
        """)
    else:
        st.warning("⚠️ 카테고리 데이터가 부족해 종합 결론을 생성할 수 없습니다.")

    st.markdown("---")
    st.subheader("⚠️ 프로젝트 한계점 및 개선 방향")

    if "수집방식" in df_shop.columns:
        _modes = set(df_shop["수집방식"].dropna().unique())
        _shop_mode_label = ("크롤링" if _modes == {"크롤링"}
                            else "백업" if _modes == {"백업"}
                            else "혼합/미상")
    else:
        _shop_mode_label = "미상"

    with st.expander("한계점 상세 보기"):
        st.markdown(f"""
        **1. ML 모델 한계**
        - 학습 데이터가 실측이 아닌 영양학 공식 기반 시뮬레이션이므로, ML이 패턴을 발견하는 게 아니라 공식을 역산합니다.
        - 2차 다항 Ridge 회귀로 비선형성을 부분 포착했으나, 실제 개인 임상 데이터에는 더 복잡한 모델이 필요합니다.
        - RMSE/MAE로 실제 예측 오차 범위를 명시했습니다.

        **2. 구매 조합 탐색 한계**
        - 각 식품의 단백질·탄수화물·지방 교차 기여량을 모두 반영했습니다.
        - 배송비(약 9,000원)를 예산에 포함했습니다.
        - 2kg 초과 대용량 상품은 보관 제약으로 기본 제외됩니다.
        - 식단 단조로움 방지를 위해 단일 상품 5개 이상 구매 시 경고를 표시합니다.
        - 나트륨·당류·포화지방·식품 가공도는 반영하지 않으며, 건강성 평가가 아닌 비용 효율 비교입니다.
        - 실제 탐색 문제는 선형 계획법(LP)으로 더 정교하게 풀 수 있습니다.

        **3. 크롤링 한계**
        - CSS 선택자는 플랫폼 UI 업데이트 시 깨질 수 있습니다.
        - 디시인사이드 검색 페이지네이션이 page 파라미터로 동작하지 않아 동일 페이지가 반복 수집되는 문제를 발견했고,
          수집 단계(seen 집합)와 분석 단계(drop_duplicates) 양쪽에서 중복을 제거하도록 개선했습니다.
        - 광고 상품을 키워드 기반으로 필터링하나, 마크업이 없는 광고는 걸러낼 수 없습니다.
        - 안티봇 차단 시 백업 데이터로 자동 전환됩니다. (현재 데이터: {_shop_mode_label} 기준)

        **4. 텍스트 마이닝 한계**
        - 형태소 분석기(KoNLPy) 없이 단순 키워드 매칭으로, 복합 부정 표현("이게 맛없는 건 아님")을 완전히 처리하지 못합니다.
        - 부정 맥락 창(window ±30자) 기반 감지로 완전한 감성 분석을 대체합니다.

        **5. 개선 방향**
        - 식약처 공공 API 연동으로 정확한 영양 성분 데이터 확보 (Ch.11에서 배운 공공데이터포털 인증키 활용)
        - KoNLPy 형태소 분석 기반 감성 분석 적용
        - 실제 임상 데이터를 활용한 랜덤 포레스트·XGBoost 모델 적용
        - 선형 계획법(scipy.optimize.linprog)으로 식단 최적화 고도화
        """)


# ════════════════════════════════════════════════════════════════
# 메인 화면: main() (Ch.11에서 배운 사이드바 radio 메뉴 구조)
# ════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(
        page_title="자취생 단백질 물가지수 대시보드",
        page_icon="🏋️",
        layout="wide"
    )

    # 한글 폰트 미설치 환경 안내 (Linux에서 나눔폰트를 못 찾은 경우)
    if os_name not in ('Darwin', 'Windows') and wc_font_path is None:
        st.warning("⚠️ 한글 폰트(NanumGothic)를 찾지 못했습니다. 차트의 한글이 깨질 수 있습니다. "
                   "배포 환경이라면 레포지토리 루트에 `packages.txt` 파일을 만들고 `fonts-nanum`을 추가해주세요.")

    # ── 데이터 로드 ────────────────────────────────────────────
    try:
        df_shop, df_words, df_ml, n_words_raw = load_data()
    except FileNotFoundError:
        st.error("⚠️ 데이터 파일이 없습니다! 터미널에서 `python collect_data.py`를 먼저 실행해주세요.")
        st.stop()

    # ── 수집 시각 추출 ─────────────────────────────────────────
    shop_collected_at  = df_shop["수집시각"].iloc[0] if "수집시각" in df_shop.columns else "알 수 없음"
    words_collected_at = df_words["수집시각"].iloc[0] if "수집시각" in df_words.columns else "알 수 없음"

    # ── ML 모델 학습 (캐시: 행 수 + 파일 수정 시각으로 무효화) ───
    ml_mtime = os.path.getmtime("ml_data.csv") if os.path.exists("ml_data.csv") else 0.0
    model, ml_metrics, X_full = train_model(len(df_ml), ml_mtime)

    # ── 사이드바 메뉴 (Ch.11 main() 패턴) ──────────────────────
    st.sidebar.title("🗂️ 메뉴")
    menu = st.sidebar.radio(
        "페이지 선택",
        [
            "① 프로젝트 개요 & 데이터 수집",
            "② 식재료 분석 & 가성비 랭킹",
            "③ ML 예측 & 구매 조합 탐색",
            "④ 텍스트 마이닝 & 결론",
        ]
    )
    st.sidebar.markdown("---")
    budget = st.sidebar.slider(
        "💰 이번 주 식비 예산 (원)",
        min_value=20000, max_value=150000, value=70000, step=5000
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("**👨‍💻 개발자:** 경제학부 손원준")
    st.sidebar.caption("F37.206 컴퓨팅 탐색: 실생활에서 활용하기")

    # ── 타이틀 ─────────────────────────────────────────────────
    st.title("🏋️‍♂️ 자취생 맞춤 단백질 물가지수 대시보드")
    st.markdown("*BeautifulSoup · Selenium · 다항 Ridge 회귀 기반 구매 조합 탐색 시스템*")
    st.markdown("---")

    # ── 메뉴별 화면 실행 (Ch.11 패턴: 메뉴 클릭 시 해당 함수 동작) ──
    if menu == "① 프로젝트 개요 & 데이터 수집":
        page_overview(df_shop, df_words, df_ml, n_words_raw,
                      shop_collected_at, words_collected_at)
    elif menu == "② 식재료 분석 & 가성비 랭킹":
        page_ranking(df_shop, shop_collected_at)
    elif menu == "③ ML 예측 & 구매 조합 탐색":
        page_predict(df_shop, model, ml_metrics, X_full, budget, shop_collected_at)
    elif menu == "④ 텍스트 마이닝 & 결론":
        page_textmining(df_words, words_collected_at, df_shop, ml_metrics)


# 파일이 직접 실행될 때만 main()이 동작하도록 하는 안전장치 (Ch.11)
if __name__ == "__main__":
    main()
