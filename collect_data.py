
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import numpy as np
import re
from datetime import datetime

# 식품별 100g당 단백질/탄수화물/지방(g) — 식품의약품안전처 DB 기준
MACRO_PROFILES = {
    "닭가슴살":         {"단백질": 23.0, "탄수화물":  0.0, "지방":  1.2},
    "단백질 보충제 WPC": {"단백질": 80.0, "탄수화물":  5.0, "지방":  5.0},
    "두부":             {"단백질":  8.0, "탄수화물":  1.0, "지방":  4.0},
    "고구마":           {"단백질":  2.0, "탄수화물": 20.0, "지방":  0.1},
    "현미 즉석밥":      {"단백질":  2.6, "탄수화물": 40.0, "지방":  0.3},
    "귀리 오트밀":      {"단백질": 12.0, "탄수화물": 65.0, "지방":  6.0},
    "구운 아몬드":      {"단백질": 21.0, "탄수화물": 22.0, "지방": 50.0},
    "피넛버터":         {"단백질": 25.0, "탄수화물": 20.0, "지방": 55.0},
    "올리브오일":       {"단백질":  0.0, "탄수화물":  0.0, "지방": 99.0},
}

# 검색어 → MACRO_PROFILES 키 매핑
QUERY_TO_PROFILE = {
    "닭가슴살":         "닭가슴살",
    "단백질 보충제 WPC": "단백질 보충제 WPC",
    "두부":             "두부",
    "고구마":           "고구마",
    "현미 즉석밥":      "현미 즉석밥",
    "귀리 오트밀":      "귀리 오트밀",
    "구운 아몬드":      "구운 아몬드",
    "피넛버터":         "피넛버터",
    "올리브오일":       "올리브오일",
}

# 광고 상품 판별 키워드
AD_MARKERS = ["[ad]", "(ad)", "광고", "스폰서", "sponsored", "ad상품"]


def is_ad_product(title: str) -> bool:
    t = title.lower()
    return any(marker in t for marker in AD_MARKERS)


def extract_total_weight_g(title: str):
    """상품명에서 총 중량(g)을 추출. 사은품/증정 표기는 무시."""
    # 사은품·증정 부분 제거 ("+ 사은품 50g", "(증정품 포함)" 등)
    clean = re.sub(r'[+＋]\s*사은품.*', '', title, flags=re.IGNORECASE)
    clean = re.sub(r'증정.*', '', clean, flags=re.IGNORECASE)
    clean = re.sub(r'\(포함\)', '', clean)

    # "Xg × N팩/개/입/봉/포"
    m = re.search(r'(\d+(?:\.\d+)?)\s*g\s*[xX×]\s*(\d+)\s*(?:팩|개|입|봉|포)', clean)
    if m:
        return float(m.group(1)) * int(m.group(2))
    # "Xg N팩/개/입/봉/포"
    m = re.search(r'(\d+(?:\.\d+)?)\s*g\s*(\d+)\s*(?:팩|개|입|봉|포)', clean)
    if m:
        return float(m.group(1)) * int(m.group(2))
    # "N팩/개/입 × Xg"
    m = re.search(r'(\d+)\s*(?:팩|개|입|봉|포)\s*[xX×]?\s*(\d+(?:\.\d+)?)\s*g(?![가-힣a-zA-Z])', clean)
    if m:
        return int(m.group(1)) * float(m.group(2))
    # "Xkg × N팩/개/입"
    m = re.search(r'(\d+(?:\.\d+)?)\s*kg\s*[xX×]?\s*(\d+)\s*(?:팩|개|입|봉|포)', clean)
    if m:
        return float(m.group(1)) * 1000 * int(m.group(2))
    # "N팩/개/입 × Xkg"
    m = re.search(r'(\d+)\s*(?:팩|개|입|봉|포)\s*[xX×]?\s*(\d+(?:\.\d+)?)\s*kg', clean)
    if m:
        return int(m.group(1)) * float(m.group(2)) * 1000
    # "Xkg" 단독
    m = re.search(r'(\d+(?:\.\d+)?)\s*kg', clean)
    if m:
        return float(m.group(1)) * 1000
    # "Xml" (ml → g 근사)
    m = re.search(r'(\d+(?:\.\d+)?)\s*ml', clean, re.IGNORECASE)
    if m:
        return float(m.group(1))
    # "Xg" 단독 (mg/ug 제외)
    m = re.search(r'(?<![mu])(\d+(?:\.\d+)?)\s*g(?![가-힣a-zA-Z])', clean)
    if m:
        return float(m.group(1))
    return None


# ════════════════════════════════════════════════════════════════
# [1/3] 커뮤니티 키워드 수집 — BeautifulSoup 정적 크롤링
# ════════════════════════════════════════════════════════════════
def get_keywords():
    """
    [Ch.12 정적 크롤링] 디시인사이드 영양갤, 몬스터짐, 블라인드에서
    단백질 관련 게시글 제목을 수집해 community_keywords.csv 저장.
    - requests.get() + User-Agent 헤더 (봇 차단 방지, Ch.12)
    - BeautifulSoup(res.text, "html.parser")로 파싱 후 find_all()로 추출
    - 페이지네이션 미작동 시 동일 페이지 반복 수집을 막기 위해 seen 집합 사용
    - 크롤링 차단 시 사전 수집된 백업 데이터로 자동 전환(Fail-safe)
    """
    print("1. 커뮤니티 키워드 통합 수집 시작 (디시 + 몬짐 + 블라인드)...")
    keywords = []  # list of dicts {"title": ..., "사이트": ...}
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/114.0.0.0 Safari/537.36"
        )
    }

    seen = set()  # 페이지네이션 미작동 시 동일 페이지 반복 수집을 막기 위한 중복 추적

    print("\n -> [1/3] 디시인사이드 수집 중... (최대 50페이지)")
    for page in range(1, 51):
        url = (
            "https://gall.dcinside.com/mgallery/board/lists/"
            f"?id=nutrient&s_type=search_subject_memo"
            f"&s_keyword=%EB%8B%AD%EA%B0%80%EC%8A%B4%EC%82%B4&page={page}"
        )
        try:
            res = requests.get(url, headers=headers, timeout=5)
            res.raise_for_status()
            soup = BeautifulSoup(res.text, "html.parser")
            titles = soup.find_all('td', class_='gall_tit')
            new_count = 0
            for t in titles:
                a_tag = t.find('a')
                if a_tag:
                    text = a_tag.get_text().strip()
                    if text and "설문" not in text and text not in seen:
                        seen.add(text)
                        keywords.append({"title": text, "사이트": "디시인사이드"})
                        new_count += 1
            # 새 게시글이 하나도 없으면 페이지가 더 이상 넘어가지 않는 것이므로 중단
            if new_count == 0:
                print(f"    (page {page}: 새 게시글 없음 → 수집 종료)")
                break
            time.sleep(1.5)
        except Exception:
            break

    print("\n -> [2/3] 몬스터짐 수집 중... (최대 50페이지)")
    NAV_JUNK = ["로그인", "회원가입", "게시판", "SELECT ITEMS", "이벤트", "공지"]
    for page in range(1, 51):
        url = f"https://www.monsterzym.com/community/?category_id=8&page={page}"
        try:
            res = requests.get(url, headers=headers, timeout=5)
            res.raise_for_status()
            soup = BeautifulSoup(res.text, "html.parser")
            new_count = 0
            for a_tag in soup.find_all('a'):
                text = a_tag.get_text().strip()
                if (len(text) > 8
                        and not any(x in text for x in NAV_JUNK)
                        and text not in seen):
                    seen.add(text)
                    keywords.append({"title": text, "사이트": "몬스터짐"})
                    new_count += 1
            if new_count == 0:
                print(f"    (page {page}: 새 게시글 없음 → 수집 종료)")
                break
            time.sleep(1.5)
        except Exception:
            break

    print("\n -> [3/3] 블라인드 수집 중...")
    url = "https://www.teamblind.com/kr/search/%EC%8B%9D%EB%8B%A8"
    try:
        res = requests.get(url, headers=headers, timeout=5)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        for a_tag in soup.find_all('a'):
            text = a_tag.get_text().strip()
            if len(text) > 8 and text not in seen:
                seen.add(text)
                keywords.append({"title": text, "사이트": "블라인드"})
        time.sleep(1.5)
    except Exception:
        pass

    backup_data = [
        {"title": "허닭 스팀 닭가슴살 존맛인정",             "사이트": "백업"},
        {"title": "마이프로틴 할인 언제함?",                  "사이트": "백업"},
        {"title": "가성비 단백질 보충제 추천 부탁드려요",     "사이트": "백업"},
        {"title": "자취생 식단 공유합니다",                   "사이트": "백업"},
        {"title": "닭가슴살 맛있게 먹는 방법 알려주세요",     "사이트": "백업"},
        {"title": "WPC vs WPI 차이가 뭔가요",                 "사이트": "백업"},
        {"title": "프로틴 먹으면 살찌나요?",                  "사이트": "백업"},
        {"title": "운동 후 식단 어떻게 구성하세요",           "사이트": "백업"},
        {"title": "고구마 vs 현미밥 탄수화물 비교",           "사이트": "백업"},
        {"title": "아몬드 하루 권장량은?",                    "사이트": "백업"},
        {"title": "허닭 vs 랭킹닭컴 어디가 더 나음",         "사이트": "백업"},
        {"title": "맛있닭 후기 공유",                         "사이트": "백업"},
        {"title": "단백질 하루 얼마나 먹어야 하나요",         "사이트": "백업"},
        {"title": "벌크업 식단 공유해요",                     "사이트": "백업"},
        {"title": "다이어트 중 단백질 섭취량 질문",           "사이트": "백업"},
        {"title": "닭가슴살 매일 먹어도 괜찮나요",            "사이트": "백업"},
        {"title": "마이프로틴 초코맛 추천",                   "사이트": "백업"},
        {"title": "퀘스트바 칼로리 얼마예요",                 "사이트": "백업"},
        {"title": "운동 전후 뭐 먹으면 좋아요",               "사이트": "백업"},
        {"title": "자취생 단백질 식단 짜는 방법",             "사이트": "백업"},
        {"title": "닭가슴살 추천 좀",                         "사이트": "백업"},
        {"title": "가성비 프로틴 보충제 추천",                "사이트": "백업"},
        {"title": "WPI vs WPC 어떤 게 더 나음",               "사이트": "백업"},
        {"title": "단백질쉐이크 언제 먹어야 하나요",          "사이트": "백업"},
        {"title": "프로틴바 간식으로 괜찮나요",               "사이트": "백업"},
        # 부정적 맥락 예시 (NLP 테스트용)
        {"title": "닭가슴살 너무 맛없고 최악이에요",          "사이트": "백업"},
        {"title": "허닭 불만 환불 신청했어요",                "사이트": "백업"},
        {"title": "단백질 보충제 맛없어서 반품함",            "사이트": "백업"},
        {"title": "WPC 별로라 안 사먹음",                     "사이트": "백업"},
    ]

    collected_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    final_keywords = keywords if len(keywords) > 20 else backup_data
    df = pd.DataFrame(final_keywords)
    df["수집시각"] = collected_at
    if "사이트" not in df.columns:
        df["사이트"] = "알 수 없음"
    df.to_csv("community_keywords.csv", index=False, encoding="utf-8-sig")
    print(f"-> community_keywords.csv 저장 완료! (총 {len(final_keywords)}건, 수집시각: {collected_at})")


# ════════════════════════════════════════════════════════════════
# [2/3] 네이버 쇼핑 크롤링 — Selenium 동적 크롤링
# ════════════════════════════════════════════════════════════════
def _try_find_text(element, selectors, by=None):
    """여러 selector를 순서대로 시도해 첫 번째로 텍스트를 반환."""
    from selenium.webdriver.common.by import By as _By
    _by = by or _By.CSS_SELECTOR
    for sel in selectors:
        try:
            el = element.find_element(_by, sel)
            text = el.text.strip()
            if text:
                return text
        except Exception:
            continue
    return None


def _extract_from_raw_text(raw_text):
    """item.text 전체에서 상품명·가격을 fallback으로 추출."""
    lines = [l.strip() for l in raw_text.split('\n') if l.strip()]
    title = lines[0] if lines else None
    price = None
    price_m = re.search(r'(\d{1,3}(?:,\d{3})+|\d{4,})\s*원', raw_text)
    if price_m:
        price = int(re.sub(r'[^0-9]', '', price_m.group(1)))
    return title, price


def get_shopping_data():
    """
    [Ch.13 동적 크롤링] Selenium으로 네이버 쇼핑 카테고리별 최저가를 크롤링.
    - chromedriver_autoinstaller.install()로 드라이버 자동 설치 (Ch.13)
    - implicitly_wait() + time.sleep()으로 로딩 대기
    - execute_script("window.scrollTo(...)")로 스크롤하여 상품 목록 로딩
    - 상품 카드·제목·가격 selector 다단계 fallback (수업 코드 data-shp-contents-dtl 방식 포함)
    - item.text 전체 파싱을 최종 fallback으로 사용
    - 광고 상품 자동 필터링 / 2kg 초과 대용량 상품 경고
    - 단백질/탄수화물/지방 전체 영양소 기록 (식약처 식품영양DB 기준)
    """
    import json as _json
    print("\n2. 네이버 쇼핑 100g당 최저단가 크롤링 중 (Selenium)...")
    import chromedriver_autoinstaller
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.options import Options

    chromedriver_autoinstaller.install()

    options = Options()
    # headless 비활성화: 수업 예제처럼 실제 Chrome 창으로 수집해야 네이버 차단 우회에 유리함.
    # Streamlit 배포 환경에서는 Selenium을 직접 실행하지 않고 사전 수집된 CSV를 읽으므로
    # headless가 필요 없음. 서버 환경에서만 아래 주석 해제.
    # options.add_argument('--headless')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--no-sandbox')
    options.add_argument('--window-size=1920,1080')
    options.add_argument(
        "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )

    # 상품 카드 selector — 순서대로 시도
    CARD_SELECTORS = [
        '[class^="product_item__"]',
        '[class*="basicProductCard_basic_product_card"]',
        '[class*="product_card"]',
        '[class*="ProductCard"]',
        'li[class*="product"]',
    ]
    # 상품명 selector — 순서대로 시도
    TITLE_SELECTORS = [
        '[class^="product_title__"]',
        '[class*="product_title"]',
        '[class*="product_name"]',
        '[class*="item_name"]',
        'a[class*="title"]',
        'strong',
    ]
    # 가격 selector — 순서대로 시도
    PRICE_SELECTORS = [
        '[class^="price_num__"]',
        '[class*="price_num"]',
        '[class*="price"]',
        'strong[class*="price"]',
        'em[class*="price"]',
    ]

    SEARCH_TARGETS = [
        ("단백질",   "닭가슴살"),
        ("단백질",   "단백질 보충제 WPC"),
        ("단백질",   "두부"),
        ("탄수화물", "고구마"),
        ("탄수화물", "현미 즉석밥"),
        ("탄수화물", "귀리 오트밀"),
        ("지방",     "구운 아몬드"),
        ("지방",     "피넛버터"),
        ("지방",     "올리브오일"),
    ]

    collected_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    crawled = []
    dom_empty_queries = []  # 상품 DOM이 0개였던 검색어 목록

    driver = None
    try:
        driver = webdriver.Chrome(options=options)
        driver.implicitly_wait(10)

        for category, query in SEARCH_TARGETS:
            encoded = requests.utils.quote(query)
            url = (
                "https://search.shopping.naver.com/search/all"
                f"?query={encoded}&sort=price_asc"
            )
            try:
                driver.get(url)
                driver.implicitly_wait(10)

                # 단계적 스크롤로 동적 상품 목록 충분히 로딩
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight / 2);")
                time.sleep(1.5)
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight / 2);")
                time.sleep(1)

                # ── [1단계] 상품 카드 selector 다단계 시도 ──────────────────
                items = []
                used_card_sel = None
                for sel in CARD_SELECTORS:
                    items = driver.find_elements(By.CSS_SELECTOR, sel)
                    if items:
                        used_card_sel = sel
                        break

                # ── [2단계] 카드 못 찾으면 data-shp-contents-dtl 속성 a태그 시도 ─
                use_attr_mode = False
                if not items:
                    items = driver.find_elements(
                        By.CSS_SELECTOR, 'a[data-shp-contents-dtl]'
                    )
                    if items:
                        used_card_sel = 'a[data-shp-contents-dtl]'
                        use_attr_mode = True

                # ── [3단계] 여전히 0개면 — 디버그 로그 + 검색창 입력 fallback ──
                if not items:
                    dom_empty_queries.append(query)
                    # 현재 화면 상태 저장 (원인 진단용)
                    screenshot_path = f"debug_{query.replace(' ', '_')}.png"
                    driver.save_screenshot(screenshot_path)
                    print(f"  [디버그] 페이지 제목: {driver.title[:60]}")
                    print(f"  [디버그] URL: {driver.current_url[:80]}")
                    print(f"  [디버그] 스크린샷 저장: {screenshot_path}")
                    print(f"  [디버그] 소스 앞 200자: {driver.page_source[:200]!r}")

                    # 수업 방식처럼 쇼핑 홈 → 검색창 입력 방식으로 fallback
                    try:
                        from selenium.webdriver.common.keys import Keys
                        driver.get("https://shopping.naver.com/home")
                        time.sleep(2)
                        for input_sel in [
                            'input[class*="input_text"]',
                            'input[name="query"]',
                            'input[type="search"]',
                            'input[placeholder*="검색"]',
                        ]:
                            try:
                                search_input = driver.find_element(By.CSS_SELECTOR, input_sel)
                                search_input.clear()
                                search_input.send_keys(query)
                                search_input.send_keys(Keys.RETURN)
                                time.sleep(4)
                                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                                time.sleep(2)
                                for card_sel in CARD_SELECTORS:
                                    items = driver.find_elements(By.CSS_SELECTOR, card_sel)
                                    if items:
                                        used_card_sel = card_sel
                                        use_attr_mode = False
                                        break
                                if not items:
                                    items = driver.find_elements(
                                        By.CSS_SELECTOR, 'a[data-shp-contents-dtl]'
                                    )
                                    if items:
                                        used_card_sel = 'a[data-shp-contents-dtl]'
                                        use_attr_mode = True
                                if items:
                                    print(f"  [검색창 fallback 성공] {query}: 카드={len(items)}개")
                                    break
                            except Exception:
                                continue
                    except Exception as fb_e:
                        print(f"  [검색창 fallback 실패] {fb_e}")

                # 디버그: 최종 카드 수·가격요소 수 출력
                price_debug = driver.find_elements(
                    By.CSS_SELECTOR, '[class*="price_num"]'
                )
                print(f"  [디버그] {query}: 카드={len(items)}개"
                      f"(sel={used_card_sel}), 가격요소={len(price_debug)}개")

                candidates = []
                for item in items[:20]:
                    try:
                        title = None
                        price = None

                        # ── 제목 추출: selector 다단계 시도 ──────────────────
                        title = _try_find_text(item, TITLE_SELECTORS)

                        # ── 가격 추출: selector 다단계 시도 ──────────────────
                        price_raw = _try_find_text(item, PRICE_SELECTORS)
                        if price_raw:
                            nums = re.sub(r'[^0-9]', '', price_raw)
                            price = int(nums) if nums else None

                        # ── data-shp-contents-dtl 속성으로 제목 보완 ─────────
                        if not title or use_attr_mode:
                            data_attr = item.get_attribute('data-shp-contents-dtl')
                            if data_attr:
                                try:
                                    data = _json.loads(data_attr)
                                    title = (
                                        data.get('productName')
                                        or data.get('product_name')
                                        or data.get('title')
                                        or title
                                    )
                                except Exception:
                                    pass

                        # ── 최종 fallback: item.text 전체에서 추출 ────────────
                        if not title or not price:
                            raw_text = item.text
                            if raw_text:
                                fb_title, fb_price = _extract_from_raw_text(raw_text)
                                if not title:
                                    title = fb_title
                                if not price:
                                    price = fb_price

                        if not title or not price:
                            continue

                        # 광고 상품 필터링
                        if is_ad_product(title):
                            continue

                        if price < 500:
                            continue

                        weight_g = extract_total_weight_g(title)
                        if weight_g and weight_g >= 50:
                            price_per_100g = round(price / weight_g * 100, 1)
                            profile = MACRO_PROFILES.get(QUERY_TO_PROFILE.get(query, ""), {})
                            prot  = profile.get("단백질", 0.0)
                            carb  = profile.get("탄수화물", 0.0)
                            fat   = profile.get("지방", 0.0)
                            if category == "단백질":
                                core_g = round(weight_g * prot / 100)
                            elif category == "탄수화물":
                                core_g = round(weight_g * carb / 100)
                            else:
                                core_g = round(weight_g * fat / 100)

                            core_price = round(price / core_g, 1) if core_g > 0 else None
                            candidates.append({
                                "카테고리":                   category,
                                "검색어":                     query,
                                "상품명":                     title,
                                "가격(원)":                   price,
                                "총중량(g)":                  int(weight_g),
                                "100g당 가격(원)":            price_per_100g,
                                "핵심영양소(g)":              core_g,
                                "핵심영양소 1g당 가격(원/g)": core_price,
                                "단백질(g/100g)":             prot,
                                "탄수화물(g/100g)":           carb,
                                "지방(g/100g)":               fat,
                                "수집시각":                   collected_at,
                                "수집방식":                   "크롤링",
                            })
                    except Exception:
                        continue

                if candidates:
                    best = min(candidates, key=lambda x: x["100g당 가격(원)"])
                    if best["총중량(g)"] > 2000:
                        print(f"  ⚠ 대용량 주의({best['총중량(g)']}g): {best['상품명'][:20]}")
                    crawled.append(best)
                    print(f"  ✓ [{category}] {query}: {best['상품명'][:20]}... "
                          f"→ 100g당 {best['100g당 가격(원)']}원")
                else:
                    print(f"  ⚠ [{category}] {query}: 파싱 가능 상품 없음")

            except Exception as e:
                print(f"  ⚠ {query} 크롤링 실패: {e}")

    except Exception as e:
        print(f"  ⚠ Selenium 실패 → 백업 데이터 사용: {e}")
    finally:
        if driver:
            driver.quit()

    cats_crawled = {r["카테고리"] for r in crawled}
    crawl_ok = (
        len(crawled) >= 6
        and "단백질" in cats_crawled
        and "탄수화물" in cats_crawled
        and "지방" in cats_crawled
    )
    if crawl_ok:
        df_shop = pd.DataFrame(crawled)
        print(f"  → 크롤링 성공: 총 {len(crawled)}개 상품 수집 (카테고리: {', '.join(sorted(cats_crawled))})")
    else:
        if crawled:
            print(f"  → 크롤링 데이터 부족({len(crawled)}개, 카테고리: {cats_crawled}) → 백업 데이터 사용")
        else:
            print("  → 크롤링 데이터 없음 → 백업 데이터 사용")
        df_shop = pd.DataFrame([
            {"카테고리": "단백질",   "검색어": "닭가슴살",         "상품명": "스팀 닭가슴살 100g 10팩",      "가격(원)": 14900, "총중량(g)": 1000, "100g당 가격(원)": 1490.0, "핵심영양소(g)": 230, "핵심영양소 1g당 가격(원/g)":  64.8, "단백질(g/100g)": 23.0, "탄수화물(g/100g)":  0.0, "지방(g/100g)":  1.2, "수집시각": collected_at, "수집방식": "백업"},
            {"카테고리": "단백질",   "검색어": "단백질 보충제 WPC", "상품명": "WPC 단백질 보충제 1kg",         "가격(원)": 29800, "총중량(g)": 1000, "100g당 가격(원)": 2980.0, "핵심영양소(g)": 800, "핵심영양소 1g당 가격(원/g)":  37.3, "단백질(g/100g)": 80.0, "탄수화물(g/100g)":  5.0, "지방(g/100g)":  5.0, "수집시각": collected_at, "수집방식": "백업"},
            {"카테고리": "단백질",   "검색어": "두부",              "상품명": "국산 두부 300g",                "가격(원)":  2900, "총중량(g)":  300, "100g당 가격(원)":  966.7, "핵심영양소(g)":  24, "핵심영양소 1g당 가격(원/g)": 120.8, "단백질(g/100g)":  8.0, "탄수화물(g/100g)":  1.0, "지방(g/100g)":  4.0, "수집시각": collected_at, "수집방식": "백업"},
            {"카테고리": "탄수화물", "검색어": "고구마",            "상품명": "유기농 고구마 1kg",              "가격(원)":  8900, "총중량(g)": 1000, "100g당 가격(원)":  890.0, "핵심영양소(g)": 200, "핵심영양소 1g당 가격(원/g)":  44.5, "단백질(g/100g)":  2.0, "탄수화물(g/100g)": 20.0, "지방(g/100g)":  0.1, "수집시각": collected_at, "수집방식": "백업"},
            {"카테고리": "탄수화물", "검색어": "현미 즉석밥",       "상품명": "현미 즉석밥 210g 12개",         "가격(원)": 12500, "총중량(g)": 2520, "100g당 가격(원)":  496.0, "핵심영양소(g)":1008, "핵심영양소 1g당 가격(원/g)":  12.4, "단백질(g/100g)":  2.6, "탄수화물(g/100g)": 40.0, "지방(g/100g)":  0.3, "수집시각": collected_at, "수집방식": "백업"},
            {"카테고리": "탄수화물", "검색어": "귀리 오트밀",       "상품명": "유기농 귀리 오트밀 1kg",        "가격(원)":  8900, "총중량(g)": 1000, "100g당 가격(원)":  890.0, "핵심영양소(g)": 650, "핵심영양소 1g당 가격(원/g)":  13.7, "단백질(g/100g)": 12.0, "탄수화물(g/100g)": 65.0, "지방(g/100g)":  6.0, "수집시각": collected_at, "수집방식": "백업"},
            {"카테고리": "지방",     "검색어": "구운 아몬드",       "상품명": "구운 아몬드 500g",              "가격(원)": 12500, "총중량(g)":  500, "100g당 가격(원)": 2500.0, "핵심영양소(g)": 250, "핵심영양소 1g당 가격(원/g)":  50.0, "단백질(g/100g)": 21.0, "탄수화물(g/100g)": 22.0, "지방(g/100g)": 50.0, "수집시각": collected_at, "수집방식": "백업"},
            {"카테고리": "지방",     "검색어": "피넛버터",          "상품명": "무가당 피넛버터 400g",          "가격(원)":  9900, "총중량(g)":  400, "100g당 가격(원)": 2475.0, "핵심영양소(g)": 220, "핵심영양소 1g당 가격(원/g)":  45.0, "단백질(g/100g)": 25.0, "탄수화물(g/100g)": 20.0, "지방(g/100g)": 55.0, "수집시각": collected_at, "수집방식": "백업"},
            {"카테고리": "지방",     "검색어": "올리브오일",        "상품명": "엑스트라버진 올리브오일 500ml", "가격(원)": 11900, "총중량(g)":  500, "100g당 가격(원)": 2380.0, "핵심영양소(g)": 495, "핵심영양소 1g당 가격(원/g)":  24.0, "단백질(g/100g)":  0.0, "탄수화물(g/100g)":  0.0, "지방(g/100g)": 99.0, "수집시각": collected_at, "수집방식": "백업"},
        ])

    # ── 저장 직전 재계산 (백업 데이터의 수기 입력 오류 방어) ──────
    df_shop['총중량(g)'] = pd.to_numeric(df_shop['총중량(g)'], errors='coerce')
    df_shop['100g당 가격(원)'] = (
        df_shop['가격(원)'] / df_shop['총중량(g)'] * 100
    ).round(1)
    df_shop['핵심영양소 1g당 가격(원/g)'] = (
        df_shop['가격(원)'] / df_shop['핵심영양소(g)'].replace(0, np.nan)
    ).round(1)
    # inf는 총중량 0 등 비정상값에서 발생할 수 있으므로 NaN으로 교체
    df_shop['100g당 가격(원)'] = df_shop['100g당 가격(원)'].replace([np.inf, -np.inf], np.nan)
    df_shop['핵심영양소 1g당 가격(원/g)'] = df_shop['핵심영양소 1g당 가격(원/g)'].replace([np.inf, -np.inf], np.nan)
    df_shop = df_shop.sort_values(
        by=["카테고리", "핵심영양소 1g당 가격(원/g)", "100g당 가격(원)"],
        na_position="last"
    )
    df_shop.to_csv("protein_products.csv", index=False, encoding="utf-8-sig")
    print("-> protein_products.csv 저장 완료 (핵심영양소 1g당 가격 기준 정렬)!")


# ════════════════════════════════════════════════════════════════
# [3/3] ML 학습 데이터 생성 — 스포츠 영양학 공식 기반 시뮬레이션
# ════════════════════════════════════════════════════════════════
def make_ml_data():
    """
    스포츠 영양학 논문(체중 1kg당 단백질 1.2~2.0g 권장) 공식을 바탕으로
    비선형 항(체중², 운동강도 교호작용)을 포함한 500개 시뮬레이션 데이터 생성.

    [설계 한계 명시]
    - 이 데이터는 실측이 아닌 스포츠 영양학 공식으로 역산된 시뮬레이션입니다.
    - ML 모델은 공식을 근사 학습하므로 R² 값이 높게 나타나는 것은 자연스럽습니다.
    - 개인의 근육량·기초대사량·건강 상태는 반영되지 않습니다.
    """
    print("\n3. ML 학습용 다중 변수 데이터 생성 중 (비선형 항 포함)...")
    np.random.seed(42)
    n_samples = 500  # 샘플 수 증가

    weight = np.random.uniform(45, 110, size=n_samples)
    days   = np.random.randint(1, 8,   size=n_samples)
    hours  = np.random.uniform(0.5, 3.0, size=n_samples)

    # 비선형 효과: 운동 강도에 따라 1.2~2.0 g/kg 구간에서 증가
    intensity = (days / 7) * (hours / 3)                   # 0~1 운동강도 지수
    protein_per_kg = 1.2 + intensity * 0.8                 # 1.2~2.0 g/kg
    protein_per_kg = np.minimum(protein_per_kg, 2.2)       # 영양학 권장 상한(2.2g/kg) 안전장치
    noise   = np.random.normal(0, 4, size=n_samples)
    protein = (weight * protein_per_kg) + noise
    protein = np.maximum(protein, 50)                       # 최소값 보정

    df_ml = pd.DataFrame({
        "체중":       np.round(weight,  1),
        "운동일수":   days,
        "운동시간":   np.round(hours,   1),
        "필요단백질": np.round(protein, 1),
    })
    df_ml.to_csv("ml_data.csv", index=False, encoding="utf-8-sig")
    print(f"-> ml_data.csv 저장 완료! (샘플 수: {n_samples}개, 비선형 항 포함)")


if __name__ == "__main__":
    get_keywords()
    get_shopping_data()
    make_ml_data()
    print("\n=== 모든 데이터 수집 및 정제 완료! ===")
