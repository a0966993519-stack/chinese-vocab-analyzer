
import streamlit as st
import json
import re
from collections import Counter
from ckip_transformers.nlp import CkipWordSegmenter


# =========================================================
# 1. 網站設定
# =========================================================

st.set_page_config(
    page_title="華語文本詞彙分析工具",
    page_icon="📚",
    layout="wide"
)


# =========================================================
# 2. 載入整合詞庫
# =========================================================

@st.cache_data
def load_database():

    with open(
        "華語三來源_整合詞庫_正式命名版.json",
        "r",
        encoding="utf-8"
    ) as f:
        database = json.load(f)

    vocabulary = database["entries"]

    vocab_db = {
        item["詞語"]: item
        for item in vocabulary
    }

    return vocabulary, vocab_db


vocabulary, vocab_db = load_database()


# =========================================================
# 3. CKIP
# =========================================================

@st.cache_resource
def load_ckip():

    return CkipWordSegmenter(
        model="bert-base"
    )


ws_driver = load_ckip()


def segment_text(text):

    result = ws_driver([text])

    if not result:
        return []

    return result[0]


# =========================================================
# 4. 基本文本處理
# =========================================================

def clean_text(text):

    text = re.sub(
        r"\s+",
        "",
        text
    )

    text = re.sub(
        r"[^\u4e00-\u9fffA-Za-z0-9]",
        "",
        text
    )

    return text


def merge_known_words(
    words,
    vocab_db,
    max_merge=4
):

    result = []
    i = 0

    while i < len(words):

        best_word = None
        best_length = 0

        max_length = min(
            max_merge,
            len(words) - i
        )

        for length in range(
            max_length,
            1,
            -1
        ):

            candidate = "".join(
                words[i:i + length]
            )

            if candidate in vocab_db:

                best_word = candidate
                best_length = length
                break

        if best_word:

            result.append(best_word)
            i += best_length

        else:

            result.append(words[i])
            i += 1

    return result


# =========================================================
# 5. 《當代中文》課次
# =========================================================

chinese_numbers = {
    1: "一",
    2: "二",
    3: "三",
    4: "四",
    5: "五",
    6: "六",
    7: "七",
    8: "八",
    9: "九",
    10: "十",
    11: "十一",
    12: "十二"
}


lesson_order = []

for book in range(1, 4):

    for lesson in range(1, 13):

        lesson_order.append(
            f"第{book}冊第{chinese_numbers[lesson]}課"
        )


def calculate_textbook_coverage(tokens):

    unique_words = list(
        dict.fromkeys(tokens)
    )

    token_counts = Counter(tokens)

    results = []

    learned_words = set()

    for current_lesson in lesson_order:

        for word, data in vocab_db.items():

            locations = data.get(
                "當代中文所有課次",
                ""
            )

            if not locations:
                continue

            word_lessons = [
                x.strip()
                for x in locations.split("；")
            ]

            if current_lesson in word_lessons:
                learned_words.add(word)

        covered_types = [
            word
            for word in unique_words
            if word in learned_words
        ]

        type_coverage = (
            len(covered_types)
            / len(unique_words)
            if unique_words
            else 0
        )

        total_tokens = len(tokens)

        covered_token_count = sum(
            count
            for word, count
            in token_counts.items()
            if word in learned_words
        )

        token_coverage = (
            covered_token_count
            / total_tokens
            if total_tokens
            else 0
        )

        uncovered_words = [
            word
            for word in unique_words
            if word not in learned_words
        ]

        results.append({
            "課次": current_lesson,
            "詞型覆蓋率": type_coverage,
            "詞次覆蓋率": token_coverage,
            "未覆蓋詞": uncovered_words
        })

    return results


# =========================================================
# 6. 華語八千詞
# =========================================================

tocfl_levels = [
    "準備級一級",
    "準備級二級",
    "入門級",
    "基礎級",
    "進階級",
    "高階級",
    "流利級"
]


def get_tocfl_level(word):

    data = vocab_db.get(word)

    if not data:
        return None

    level = data.get(
        "華語八千詞最低等級",
        ""
    )

    if level in tocfl_levels:
        return level

    return None


# =========================================================
# 7. 國教院三等七級
# =========================================================

naer_levels = [
    "第1級",
    "第2級",
    "第3級",
    "第4級",
    "第5級",
    "第6級",
    "第7級"
]


def get_naer_level(word):

    data = vocab_db.get(word)

    if not data:
        return None

    level = data.get(
        "國教院三等七級詞語表｜級別",
        ""
    )

    if not level:
        return None

    levels = [
        x.strip()
        for x in level.split("；")
        if x.strip() in naer_levels
    ]

    if not levels:
        return None

    return min(
        levels,
        key=lambda x:
        naer_levels.index(x)
    )


# =========================================================
# 8. 程度詞表 Coverage
# =========================================================

def calculate_coverage(
    tokens,
    level_order,
    level_function
):

    unique_words = list(
        dict.fromkeys(tokens)
    )

    token_counts = Counter(tokens)

    results = []

    for index, target_level \
            in enumerate(level_order):

        allowed_levels = set(
            level_order[:index + 1]
        )

        graded_types = []
        covered_types = []

        for word in unique_words:

            level = level_function(word)

            if level is not None:

                graded_types.append(word)

                if level in allowed_levels:
                    covered_types.append(word)

        type_coverage = (
            len(covered_types)
            / len(graded_types)
            if graded_types
            else 0
        )

        graded_token_count = 0
        covered_token_count = 0

        for word, count \
                in token_counts.items():

            level = level_function(word)

            if level is not None:

                graded_token_count += count

                if level in allowed_levels:
                    covered_token_count += count

        token_coverage = (
            covered_token_count
            / graded_token_count
            if graded_token_count
            else 0
        )

        results.append({
            "級別": target_level,
            "詞型覆蓋率": type_coverage,
            "詞次覆蓋率": token_coverage
        })

    return results


# =========================================================
# 9. 正式文章分析
# =========================================================

def analyze_text(
    text,
    exclude_words=None
):

    if exclude_words is None:
        exclude_words = []

    cleaned = clean_text(text)

    words = segment_text(cleaned)

    words = merge_known_words(
        words,
        vocab_db
    )

    all_tokens = [
        word.strip()
        for word in words
        if word.strip()
        and re.search(
            r"[\u4e00-\u9fffA-Za-z0-9]",
            word
        )
    ]

    actual_excluded = list(
        dict.fromkeys(
            word
            for word in all_tokens
            if word in exclude_words
        )
    )

    tokens = [
        word
        for word in all_tokens
        if word not in exclude_words
    ]

    unique_words = list(
        dict.fromkeys(tokens)
    )

    textbook = \
        calculate_textbook_coverage(
            tokens
        )

    tocfl = calculate_coverage(
        tokens,
        tocfl_levels,
        get_tocfl_level
    )

    naer = calculate_coverage(
        tokens,
        naer_levels,
        get_naer_level
    )

    ungraded = [
        word
        for word in unique_words
        if get_tocfl_level(word) is None
        and get_naer_level(word) is None
    ]

    categories = {
        "數字／年份": [],
        "外語／英數詞": [],
        "其他待確認詞": []
    }

    for word in ungraded:

        if (
            re.fullmatch(r"\d+", word)
            or
            re.fullmatch(
                r"\d{2,4}年",
                word
            )
        ):

            categories[
                "數字／年份"
            ].append(word)

        elif re.search(
            r"[A-Za-z]",
            word
        ):

            categories[
                "外語／英數詞"
            ].append(word)

        else:

            categories[
                "其他待確認詞"
            ].append(word)

    return {
        "字數": len(cleaned),
        "原始總詞次": len(all_tokens),
        "總詞次": len(tokens),
        "詞型數": len(unique_words),
        "排除詞": actual_excluded,
        "當代中文": textbook,
        "華語八千詞": tocfl,
        "國教院": naer,
        "無分級詞": ungraded,
        "無分級分類": categories
    }


# =========================================================
# 10. 顯示 Coverage 表格
# =========================================================

def show_coverage_table(data):

    rows = []

    for item in data:

        rows.append({
            "程度": item["級別"],
            "詞型覆蓋率":
                f'{item["詞型覆蓋率"] * 100:.1f}%',
            "詞次覆蓋率":
                f'{item["詞次覆蓋率"] * 100:.1f}%'
        })

    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# 11. 網站畫面
# =========================================================

st.title(
    "📚 華語文本詞彙分析工具"
)

st.write(
    "分析文本與《當代中文課程》、"
    "華語八千詞及國教院三等七級詞語表"
    "之詞彙覆蓋情形。"
)

with st.expander(
    "ℹ️ 分析方式說明"
):

    st.markdown(
        """
        **詞型覆蓋率**：以不重複詞彙種類計算。

        **詞次覆蓋率**：依詞彙在文章中的實際出現次數計算。

        《當代中文課程》的覆蓋率以全文詞彙為分母；
        華語八千詞與國教院三等七級詞語表則以
        各自具有分級資料的詞彙為分母。

        無分級資料不代表詞彙一定較困難。
        """
    )


text = st.text_area(
    "請貼上要分析的中文文章",
    height=300,
    placeholder="在這裡貼上文章……"
)


exclude_text = st.text_input(
    "排除詞（選填）",
    placeholder=(
        "例如：東京、上海、Prada"
    ),
    help=(
        "適合排除不希望影響分析結果的"
        "人名、地名、品牌名稱等。"
    )
)


if st.button(
    "🔍 開始分析",
    type="primary",
    use_container_width=True
):

    if not text.strip():

        st.warning(
            "請先貼上要分析的文章。"
        )

    else:

        raw = exclude_text.strip()

        if raw:

            exclude_words = [
                word.strip()
                for word in raw
                .replace("，", ",")
                .replace("、", ",")
                .replace(" ", ",")
                .split(",")
                if word.strip()
            ]

        else:

            exclude_words = []

        with st.spinner(
            "正在進行 CKIP 斷詞與詞彙分析……"
        ):

            result = analyze_text(
                text,
                exclude_words
            )

        st.success("分析完成！")


        # =====================================
        # 文本資訊
        # =====================================

        st.header("📄 文本資訊")

        col1, col2, col3 = \
            st.columns(3)

        col1.metric(
            "字數",
            result["字數"]
        )

        col2.metric(
            "分析詞次",
            result["總詞次"]
        )

        col3.metric(
            "不重複詞型",
            result["詞型數"]
        )


        # =====================================
        # 當代中文
        # =====================================

        st.header(
            "📘《當代中文課程》教材詞彙覆蓋"
        )

        st.caption(
            "顯示學生學完各冊後，"
            "已學教材生詞可覆蓋本文多少詞彙。"
        )

        textbook = \
            result["當代中文"]

        book_rows = []

        for book in range(1, 4):

            target = (
                f"第{book}冊第十二課"
            )

            item = next(
                (
                    x
                    for x in textbook
                    if x["課次"] == target
                ),
                None
            )

            if item:

                book_rows.append({
                    "教材進度":
                        f"第{book}冊學完",

                    "詞型覆蓋率":
                        f'{item["詞型覆蓋率"] * 100:.1f}%',

                    "詞次覆蓋率":
                        f'{item["詞次覆蓋率"] * 100:.1f}%'
                })

        st.dataframe(
            book_rows,
            use_container_width=True,
            hide_index=True
        )


        # =====================================
        # 華語八千詞
        # =====================================

        st.header(
            "📗 華語八千詞"
        )

        st.caption(
            "以下為累積至各程度的詞彙覆蓋率。"
        )

        show_coverage_table(
            result["華語八千詞"]
        )


        # =====================================
        # 國教院
        # =====================================

        st.header(
            "📙 國教院三等七級詞語表"
        )

        st.caption(
            "以下為第1級至第7級的"
            "累積詞彙覆蓋率。"
        )

        show_coverage_table(
            result["國教院"]
        )


        # =====================================
        # 教材外詞
        # =====================================

        st.header(
            "🔎 詞彙檢視"
        )

        final_textbook = \
            textbook[-1]

        textbook_unknown = list(
            dict.fromkeys(
                final_textbook[
                    "未覆蓋詞"
                ]
            )
        )

        with st.expander(
            f"《當代中文》前三冊未覆蓋詞 "
            f"({len(textbook_unknown)} 詞)"
        ):

            if textbook_unknown:

                st.write(
                    "、".join(
                        textbook_unknown
                    )
                )

            else:

                st.write(
                    "本文詞彙皆已由前三冊覆蓋。"
                )


        # =====================================
        # 無分級資料
        # =====================================

        with st.expander(
            f"兩份程度詞表皆無分級資料 "
            f"({len(result['無分級詞'])} 詞)"
        ):

            for category, words in \
                    result[
                        "無分級分類"
                    ].items():

                if words:

                    st.markdown(
                        f"**{category}**"
                    )

                    st.write(
                        "、".join(words)
                    )


        # =====================================
        # 排除詞
        # =====================================

        if result["排除詞"]:

            with st.expander(
                f"🚫 本次排除詞 "
                f"({len(result['排除詞'])} 詞)"
            ):

                st.write(
                    "、".join(
                        result["排除詞"]
                    )
                )

                st.caption(
                    "以上詞彙存在於原文，"
                    "但未納入教材及程度覆蓋率計算。"
                )
