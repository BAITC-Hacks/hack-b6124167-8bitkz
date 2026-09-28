"""GovTech Camp, кейс 1 — мониторинг и прогнозирование нагрузки на стационары.

Запуск из корня проекта:  streamlit run app/main.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import anomalies, loaders, model_load, model_waiting
from src.regions import region_name

# ---------------------------------------------------------------- палитры
THEMES = {
    "Светлая": dict(
        app_bg="#F4F7F8", card="#FFFFFF", border="#E2E8F0", text="#0F172A",
        text_mut="#5B6B7B", shadow="0 1px 3px rgba(15,23,42,.06)",
        side_a="#0B4F4A", side_b="#083A36", side_tx="#D9F5EF",
        hero_b="#FFFFFF", hero_br="#C8F1E8", hero_h="#0E5A52",
        hero_light="1", hero_dark="0",
        badge_bg="#D5F4EC", badge_tx="#0F766E",
        template="plotly_white", font="#334155", grid="#E8EEF0", axis="#CBD5E1",
        accent="#0E9384", accent_soft="rgba(20,184,166,.15)", warn="#D97706",
        warn_soft="rgba(245,158,11,.12)", grey="#94A3B8", sel_bg="#FFFFFF",
        hover="#F1F5F9", scale_a="rgba(20,184,166,.30)", scale_w="rgba(245,158,11,.30)",
    ),
    "Тёмная": dict(
        app_bg="#0B1117", card="#141D26", border="#243039", text="#E6EDF3",
        text_mut="#8CA0B3", shadow="0 1px 6px rgba(0,0,0,.45)",
        side_a="#0B4F4A", side_b="#083A36", side_tx="#D9F5EF",
        hero_b="#101A22", hero_br="#1E4D48", hero_h="#8CE8DC",
        hero_light="0", hero_dark="1",
        badge_bg="#113B36", badge_tx="#5EEAD4",
        template="plotly_dark", font="#C7D2DA", grid="#1D2833", axis="#31414F",
        accent="#2DD4BF", accent_soft="rgba(45,212,191,.16)", warn="#FBBF24",
        warn_soft="rgba(251,191,36,.14)", grey="#64748B", sel_bg="#18222D",
        hover="#1B2732", scale_a="rgba(45,212,191,.32)", scale_w="rgba(251,191,36,.32)",
    ),
}
T = THEMES["Светлая"]  # переопределяется ниже по выбору темы

LOGO_SVG = (
    '<svg width="30" height="30" viewBox="0 0 24 24" style="flex:none">'
    '<rect rx="7" width="24" height="24" fill="#2DD4BF"/>'
    '<path d="M4 13h3.5l2-5.5 4 9.5 2-4H20" stroke="#06302C" stroke-width="2.1" '
    'fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>'
)

CSS_TEMPLATE = """
<style>
    .stApp {background: __APP_BG__; color: __TEXT__; --hero-light: __HERO_LIGHT__; --hero-dark: __HERO_DARK__;}
    [data-testid="stHeader"] {background: transparent;}
    h1, h2, h3 {color: __TEXT__;}

    /* плавная смена темы */
    .stApp, div[data-testid="stMetric"], .roadmap-card, .badge, hr,
    div[data-testid="stSelectbox"] .react-aria-ComboBox > div,
    section[data-testid="stSidebar"] div[role="radiogroup"] button[role="radio"] {
        transition: background-color .5s ease, color .5s ease,
                    border-color .5s ease, box-shadow .5s ease, fill .5s ease;
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp span, .stApp label,
    div[data-testid="stMetric"] *, div[data-testid="stMarkdownContainer"] * {
        transition: color .5s ease;
    }

    .block-container {padding-top: 1.2rem; max-width: 1240px;}
    #MainMenu, footer, [data-testid="stAppDeployButton"] {visibility: hidden;}
    [data-testid="stCaptionContainer"], [data-testid="stWidgetLabel"] p {color: __TEXT_MUT__ !important;}
    hr {border-color: __BORDER__ !important;}

    /* боковая панель */
    section[data-testid="stSidebar"] {background: linear-gradient(180deg, __SIDE_A__ 0%, __SIDE_B__ 100%);}
    section[data-testid="stSidebar"] * {color: __SIDE_TX__ !important;}
    section[data-testid="stSidebar"] hr {border-color: rgba(255,255,255,.18) !important;}
    .side-brand {display: flex; align-items: center; gap: 10px; margin: 4px 0 2px;}
    .side-brand b {font-size: 1.18rem; letter-spacing: .2px;}
    .side-brand div {line-height: 1.2;}
    .side-brand small {opacity: .75; font-size: .8rem;}

    /* навигация-пилюли */
    section[data-testid="stSidebar"] .stRadio label {
        background: transparent; border-radius: 10px; padding: 7px 12px;
        margin-bottom: 4px; transition: background .15s; font-weight: 500;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {background: rgba(255,255,255,.12);}
    section[data-testid="stSidebar"] .stRadio label:has(input:checked) {
        background: rgba(94,234,212,.16); outline: 1px solid rgba(94,234,212,.45);}
    section[data-testid="stSidebar"] .stRadio label div[data-testid="stRadioOptionMark"],
    section[data-testid="stSidebar"] .stRadio input {display: none;}

    /* переключатель темы (Streamlit рисует его как radiogroup из button[role=radio]) */
    section[data-testid="stSidebar"] div[role="radiogroup"]:has(button[role="radio"]) {
        background: rgba(255,255,255,.10); border-radius: 10px; padding: 3px;}
    section[data-testid="stSidebar"] div[role="radiogroup"] button[role="radio"] {
        color: __SIDE_TX__ !important; background: rgba(4,33,30,.55) !important;
        border-radius: 8px; font-weight: 500; opacity: 1;}
    section[data-testid="stSidebar"] div[role="radiogroup"] button[role="radio"][aria-checked="true"] {
        background: __ACCENT__ !important; color: #04211E !important;}

    /* карточки-метрики */
    div[data-testid="stMetric"] {
        background: __CARD__; border: 1px solid __BORDER__; border-radius: 14px;
        padding: 14px 18px 10px; box-shadow: __SHADOW__;
        border-left: 4px solid __ACCENT__;
    }
    div[data-testid="stMetric"] label {color: __TEXT_MUT__ !important;}
    div[data-testid="stMetric"] div {color: __TEXT__ !important;}

    /* фильтры (React Aria ComboBox в Streamlit 1.59) */
    div[data-testid="stSelectbox"] label {font-weight: 600; color: __TEXT__;}
    div[data-testid="stSelectbox"] .react-aria-ComboBox > div {
        background: __SEL_BG__ !important; border: 1px solid __BORDER__ !important;}
    div[data-testid="stSelectbox"] .react-aria-ComboBox input,
    div[data-testid="stSelectbox"] .react-aria-ComboBox button {color: __TEXT__ !important;}
    div[data-testid="stSelectbox"] .react-aria-ComboBox svg {fill: __TEXT__;}
    div[data-react-aria-popover], div[role="listbox"] {
        background: __SEL_BG__ !important; color: __TEXT__ !important; border: 1px solid __BORDER__;}
    div[data-react-aria-popover] *, div[role="listbox"] * {color: __TEXT__ !important;}
    div[role="listbox"] li:hover, div[role="option"]:hover {background: __HOVER__ !important;}

    /* заголовок страницы: перекрёстное растворение светлого и тёмного градиентов */
    .page-hero {
        position: relative; background-color: __HERO_B__;
        border: 1px solid __HERO_BR__; border-radius: 16px;
        padding: 18px 24px; margin-bottom: 18px;
        transition: background-color .5s ease, border-color .5s ease;
    }
    .page-hero::before, .page-hero::after {
        content: ""; position: absolute; inset: 0; border-radius: inherit;
        pointer-events: none; transition: opacity .5s ease;
    }
    .page-hero::before {
        background: linear-gradient(90deg, #ECFDF9 0%, rgba(255,255,255,0) 75%);
        opacity: var(--hero-light, 1);
    }
    .page-hero::after {
        background: linear-gradient(90deg, #0E1F1D 0%, rgba(16,26,34,0) 75%);
        opacity: var(--hero-dark, 0);
    }
    .page-hero > * {position: relative; z-index: 1;}
    .page-hero h1 {font-size: 1.55rem; margin: 0 0 4px; color: __HERO_H__;}
    .page-hero p {margin: 0; color: __TEXT_MUT__; font-size: .95rem;}
    .badge {
        display: inline-block; background: __BADGE_BG__; color: __BADGE_TX__ !important;
        border-radius: 999px; padding: 2px 12px; font-size: .78rem;
        font-weight: 600; margin-top: 8px; margin-right: 6px;
    }
    .roadmap-card {
        background: __CARD__; border: 1px solid __BORDER__; border-radius: 14px;
        padding: 16px 20px; margin-bottom: 12px; color: __TEXT__;
        border-left: 4px solid __WARN__; box-shadow: __SHADOW__;
    }
    .roadmap-card b {color: __HERO_H__;}
"""

OUTCOME_RU = {"hospitalized": "Госпитализирован", "refused": "Отказ", "pending": "Ожидает"}
MONTHS_RU = {1: "Январь", 2: "Февраль", 3: "Март", 4: "Апрель", 5: "Май", 6: "Июнь"}
ICD_CHAPTERS = {
    "A": "Инфекции", "B": "Инфекции", "C": "Онкология", "D": "Кровь/новообразования",
    "E": "Эндокринные", "F": "Психика", "G": "Нервная система", "H": "Глаз и ухо",
    "I": "Кровообращение", "J": "Дыхание", "K": "Пищеварение", "L": "Кожа",
    "M": "Костно-мышечная", "N": "Мочеполовая", "O": "Беременность и роды",
    "P": "Перинатальный период", "Q": "Врождённые", "R": "Симптомы",
    "S": "Травмы", "T": "Отравления/травмы", "Z": "Прочие факторы",
}
NA_LABEL = "(не определён)"
FEATURE_RU = {
    "hospital_mo": "Стационар", "bed_profile": "Профиль коек", "icd_group": "Группа МКБ-10",
    "region": "Регион", "territorial_type": "Тип населённого пункта",
    "referral_purpose": "Цель направления", "finance_source": "Источник финансирования",
    "month": "Месяц", "dow": "День недели регистрации",
}


def css_for(theme: dict) -> str:
    reps = {
        "__APP_BG__": theme["app_bg"], "__TEXT__": theme["text"], "__TEXT_MUT__": theme["text_mut"],
        "__CARD__": theme["card"], "__BORDER__": theme["border"], "__SHADOW__": theme["shadow"],
        "__SIDE_A__": theme["side_a"], "__SIDE_B__": theme["side_b"], "__SIDE_TX__": theme["side_tx"],
        "__HERO_B__": theme["hero_b"], "__HERO_BR__": theme["hero_br"],
        "__HERO_H__": theme["hero_h"], "__HERO_LIGHT__": theme["hero_light"],
        "__HERO_DARK__": theme["hero_dark"],
        "__BADGE_BG__": theme["badge_bg"], "__BADGE_TX__": theme["badge_tx"],
        "__ACCENT__": theme["accent"], "__WARN__": theme["warn"], "__SEL_BG__": theme["sel_bg"],
        "__HOVER__": theme["hover"],
    }
    css = CSS_TEMPLATE
    for k, v in reps.items():
        css = css.replace(k, v)
    return css


def plot_layout() -> dict:
    return dict(
        template=T["template"],
        font=dict(family="Segoe UI, Arial, sans-serif", size=13, color=T["font"]),
        margin=dict(l=10, r=10, t=48, b=10),
        title=dict(font=dict(size=15, color=T["text"])),
        hoverlabel=dict(font_size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )


def style_fig(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(**plot_layout(), height=height)
    fig.update_xaxes(showgrid=True, gridcolor=T["grid"], linecolor=T["axis"])
    fig.update_yaxes(showgrid=True, gridcolor=T["grid"], linecolor=T["axis"])
    fig.update_coloraxes(showscale=False)
    return fig


st.set_page_config(page_title="Стационары РК — GovTech Camp", layout="wide")


@st.cache_data(show_spinner="Загружаю направления…")
def referrals() -> pd.DataFrame:
    return loaders.load_referrals()


@st.cache_data(show_spinner="Загружаю очередь…")
def queue() -> pd.DataFrame:
    return loaders.load_queue()


@st.cache_data(show_spinner="Загружаю отказы…")
def refusals() -> pd.DataFrame:
    return loaders.load_refusals()


@st.cache_data
def treated() -> pd.DataFrame:
    return loaders.load_treated()


@st.cache_resource(show_spinner="Обучаю модель времени ожидания…")
def waiting_artifact() -> dict:
    return model_waiting.load_or_train()


@st.cache_resource(show_spinner="Обучаю модель нагрузки…")
def load_artifact() -> dict:
    return model_load.load_or_train()


@st.cache_resource
def _waiting_explainer():
    import shap
    return shap.TreeExplainer(waiting_artifact()["booster"])


@st.cache_data(show_spinner="Готовлю ряды нагрузки…")
def load_counts() -> pd.DataFrame:
    return model_load.build_counts(referrals())


@st.cache_data(show_spinner="Ищу всплески отказов…")
def anomaly_daily() -> pd.DataFrame:
    return anomalies.detect_spikes(anomalies.daily_by_region(refusals()))


def nfmt(n) -> str:
    return f"{n:,}".replace(",", " ")


def hero(title: str, subtitle: str, *badges: str) -> None:
    tags = "".join(f'<span class="badge">{b}</span>' for b in badges)
    st.markdown(
        f'<div class="page-hero"><h1>{title}</h1><p>{subtitle}</p>{tags}</div>',
        unsafe_allow_html=True,
    )


def card(title: str, body: str) -> None:
    st.markdown(f'<div class="roadmap-card"><b>{title}</b><br>{body}</div>',
                unsafe_allow_html=True)


def short_mo(name, maxlen: int = 46) -> str:
    """Короткое имя МО: из правовой формы оставляем суть в кавычках."""
    s = str(name)
    m = re.search(r'«([^»]+)»|"([^"]+)"', s)
    if m:
        return (m.group(1) or m.group(2)).strip()[:maxlen]
    return s[:maxlen]


def fmt_cat(v) -> str:
    s = str(v)
    return NA_LABEL if s in ("nan", "None", "") else s


# ---------------------------------------------------------------- страницы
def page_overview() -> None:
    hero("Обзор данных",
         "Единая картина плановой госпитализации: направления, очередь, отказы и пролеченные случаи",
         "2025 Q1", "Открытые данные МЗ РК", "ИС БГ · ЭРСБ")

    ref, q, refu, tr = referrals(), queue(), refusals(), treated()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Направления", nfmt(len(ref)), help="ИС БГ, события 2025 Q1")
    c2.metric("В очереди", nfmt(len(q)), help="Снимок очереди на конец периода (31.03.2025)")
    c3.metric("Отказы", nfmt(len(refu)), help="Приёмный покой, события 2025 Q1")
    c4.metric("Медорганизации", nfmt(len(tr)), help="ЭРСБ: число МО с пролеченными случаями")

    left, right = st.columns([1, 2], gap="medium")
    with left:
        outcome = ref["outcome"].value_counts().reset_index()
        outcome.columns = ["Исход", "Число направлений"]
        outcome["Исход"] = outcome["Исход"].map(OUTCOME_RU)
        outcome["Цвет"] = [T["accent"], T["warn"], T["grey"]][: len(outcome)]
        fig = px.pie(outcome, names="Исход", values="Число направлений", hole=0.62,
                     color="Исход", color_discrete_sequence=outcome["Цвет"],
                     title="Чем закончились направления")
        fig.update_traces(texttemplate="%{percent:.1%}", textfont_size=13)
        fig.update_layout(**plot_layout(), showlegend=True,
                          legend=dict(orientation="h", y=-0.12, title=None,
                                      font=dict(color=T["font"])))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    with right:
        by_day = (ref.assign(Дата=ref["registration_dt"].dt.to_period("D").astype(str))
                  .groupby("Дата").size().reset_index(name="Направлений"))
        fig = px.area(by_day, x="Дата", y="Направлений", title="Регистрация направлений по дням")
        fig.update_traces(line_color=T["accent"], fillcolor=T["accent_soft"])
        st.plotly_chart(style_fig(fig), width="stretch", config={"displayModeBar": False})

    w = ref.loc[ref["waiting_days"].between(0, 90), "waiting_days"]
    left2, right2 = st.columns([2, 1], gap="medium")
    with left2:
        fig = px.histogram(x=w, nbins=90, title="Сколько дней ждал пациент на госпитализацию (0–90 дней)",
                           labels={"x": "Дней ожидания", "count": "Пациентов"})
        fig.update_traces(marker_color=T["accent"])
        fig.add_vline(x=w.median(), line_dash="dash", line_color=T["warn"],
                      annotation_text=f"медиана {w.median():.0f} дн.",
                      annotation_font_color=T["font"])
        st.plotly_chart(style_fig(fig), width="stretch", config={"displayModeBar": False})
    with right2:
        card("Медиана ожидания",
             f"{w.median():.1f} дня от подачи направления до госпитализации. "
             f"{(ref['waiting_days'] >= 30).mean():.0%} пациентов ждали месяц и дольше.")
        card("Качество данных",
             f"{nfmt((ref['waiting_days'] < 0).sum())} направлений "
             f"({(ref['waiting_days'] < 0).mean():.1%}) имеют дату госпитализации раньше "
             "регистрации — артефакт выгрузки, исключены из гистограммы и модели.")


def page_queue() -> None:
    hero("Очереди на плановую госпитализацию",
         "Кто, куда и в каком порядке ждёт госпитализацию — по регионам, стационарам и профилям коек",
         "765 тыс. пациентов", "1 406 стационаров", "20 регионов")

    q = queue()
    q["region_label"] = q["region_origin_code"].map(region_name)

    f1, f2 = st.columns(2)
    with f1:
        region = st.selectbox("Регион направителя", ["Все"] + sorted(q["region_label"].unique().tolist()))
    with f2:
        profile = st.selectbox("Профиль коек", ["Все"] + sorted(q["profile_code"].cat.categories.tolist()))

    dfq = q
    if region != "Все":
        dfq = dfq[dfq["region_label"] == region]
    if profile != "Все":
        dfq = dfq[dfq["profile_code"] == profile]

    c1, c2, c3 = st.columns(3)
    c1.metric("Пациентов в очереди", nfmt(len(dfq)))
    c2.metric("Стационаров-получателей", nfmt(dfq["mo_destination_code"].nunique()))
    age = (pd.Timestamp("2025-03-31") - dfq["registration_dt"]).dt.days
    c3.metric("Медианный возраст заявки", f"{age.median():.0f} дн.",
              help="Сколько дней в среднем заявка ждёт в очереди по состоянию на 31.03.2025")

    left, right = st.columns([3, 2], gap="medium")
    with left:
        top = (dfq.groupby("mo_destination_code", observed=True)
               .agg(Ожидают=("patient_seq_no", "size"))
               .sort_values("Ожидают", ascending=False).head(15).reset_index())
        fig = px.bar(top.sort_values("Ожидают"), x="Ожидают", y="mo_destination_code",
                     orientation="h", title="Топ-15 стационаров по числу ожидающих",
                     color="Ожидают", color_continuous_scale=[T["scale_a"], T["accent"]],
                     labels={"mo_destination_code": "Стационар (код)"})
        st.plotly_chart(style_fig(fig, 460), width="stretch", config={"displayModeBar": False})
    with right:
        prof = dfq["profile_code"].value_counts().head(12).reset_index()
        prof.columns = ["Профиль коек", "Ожидают"]
        fig = px.bar(prof.sort_values("Ожидают"), x="Ожидают", y="Профиль коек",
                     orientation="h", title="Ожидающие по профилям коек (топ-12)",
                     color="Ожидают",
                     color_continuous_scale=[T["scale_w"], T["warn"]])
        st.plotly_chart(style_fig(fig, 460), width="stretch", config={"displayModeBar": False})

    st.divider()
    st.subheader("Сравнение стационаров по потоку направлений")
    st.caption("По данным направлений 2025 Q1: объём, доля отказов и медианное ожидание "
               "(сроки 0–90 дней, без артефактов выгрузки).")
    ref = referrals()
    comp = (ref.groupby("hospital_mo", observed=True)
            .agg(Направлений=("hospitalization_code", "size"),
                 Доля_отказов=("outcome", lambda s: (s == "refused").mean()),
                 Медиана_ожидания=("waiting_days", lambda s: s[s.between(0, 90)].median()))
            .reset_index())
    comp["Регион"] = comp["hospital_mo"].map(loaders.parse_region).fillna(NA_LABEL)
    comp = comp.sort_values("Направлений", ascending=False).head(15)
    comp["Стационар"] = comp["hospital_mo"].map(short_mo)
    table = pd.DataFrame({
        "Стационар": comp["Стационар"],
        "Регион": comp["Регион"],
        "Направлений": comp["Направлений"].map(nfmt),
        "Доля отказов": comp["Доля_отказов"].map(lambda v: f"{v:.0%}"),
        "Медиана ожидания, дн.": comp["Медиана_ожидания"].round(1),
    }).reset_index(drop=True)
    st.dataframe(table, use_container_width=True, height=430, hide_index=True)


def page_refusals() -> None:
    hero("Отказы в плановой госпитализации",
         "Отказы приёмного покоя: география, диагнозы, динамика и всплески",
         "1.5 млн событий", "20 регионов", "449 организаций")

    refu = refusals()

    c1, c2, c3 = st.columns(3)
    c1.metric("Всего отказов", nfmt(len(refu)))
    c2.metric("Застрахованных", f"{(refu['insured'] == 'Застрахован').mean():.0%}",
              help="Доля застрахованных ОСМС среди отказанных")
    c3.metric("Сумма по отказам", f"{refu['amount'].sum() / 1e9:.1f}".replace(".", ",") + " млрд ₸",
              help="Сумма поля amount по всем отказам периода")

    fig = px.area(refu.assign(Дата=refu["refuse_dt"].dt.to_period("D").astype(str))
                  .groupby("Дата").size().reset_index(name="Отказов"),
                  x="Дата", y="Отказов", title="Отказы по дням")
    fig.update_traces(line_color=T["warn"], fillcolor=T["warn_soft"])
    st.plotly_chart(style_fig(fig, 300), width="stretch", config={"displayModeBar": False})

    left, right = st.columns(2, gap="medium")
    with left:
        by_region = refu["region_in"].value_counts().head(20).reset_index()
        by_region.columns = ["Регион", "Отказов"]
        fig = px.bar(by_region.sort_values("Отказов"), x="Отказов", y="Регион",
                     orientation="h", title="Отказы по регионам (топ-20)",
                     color="Отказов",
                     color_continuous_scale=[T["scale_w"], T["warn"]])
        fig.update_yaxes(title=None)
        st.plotly_chart(style_fig(fig, 520), width="stretch", config={"displayModeBar": False})
    with right:
        by_icd = refu.groupby("icd_name", observed=True).size().sort_values(ascending=False).head(20).reset_index()
        by_icd.columns = ["Диагноз (МКБ-10)", "Отказов"]
        by_icd["Диагноз (МКБ-10)"] = by_icd["Диагноз (МКБ-10)"].astype(str).str.slice(0, 42).str.rstrip() + "…"
        fig = px.bar(by_icd.sort_values("Отказов"), x="Отказов", y="Диагноз (МКБ-10)",
                     orientation="h", title="Отказы по диагнозам (топ-20)",
                     color="Отказов",
                     color_continuous_scale=[T["scale_a"], T["accent"]])
        fig.update_yaxes(categoryorder="total ascending", title=None)
        st.plotly_chart(style_fig(fig, 520), width="stretch", config={"displayModeBar": False})

    st.divider()
    st.subheader("Аномальные всплески за последние 30 дней")
    daily = anomaly_daily()
    tbl = anomalies.spikes_table(daily)
    if tbl.empty:
        st.success("За последние 30 дней аномальных всплесков не обнаружено.")
    else:
        st.dataframe(tbl, use_container_width=True, hide_index=True, height=220)
        st.caption("Всплеск: день выше скользящей медианы предыдущих 14 дней при z ≥ 3,5 "
                   "(разброс — MAD, устойчивый к выбросам). Полный детектор — на странице «Прогнозы и модели».")


def page_treated() -> None:
    hero("Пролеченные случаи (ЭРСБ)",
         "Объёмы и эффективность коечного фонда в разрезе медорганизаций",
         f"{nfmt(len(treated()))} организаций", "снимок ЭРСБ", "без временной оси")

    tr = treated()
    regions = [r for r in tr["region"].dropna().astype(str).unique().tolist() if r != "nan"]
    region = st.selectbox("Регион", ["Все", NA_LABEL] + sorted(regions))
    dtr = tr if region == "Все" else tr[tr["region"].astype(str) == ("" if region == NA_LABEL else region)]
    if region == NA_LABEL:
        dtr = tr[tr["region"].isna() | (tr["region"].astype(str) == "nan")]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Пролечено случаев", nfmt(int(dtr["discharged_total"].sum())))
    c2.metric("Койко-дней", nfmt(int(dtr["bed_days"].sum())))
    c3.metric("Койко-дней на случай", f"{dtr['bed_days_per_case'].median():.1f}",
              help="Медиана по организациям региона/страны")
    amount_bln = dtr["amount_to_pay"].sum() / 1e9
    amount_txt = (f"{amount_bln / 1000:.2f}".replace(".", ",") + " трлн ₸"
                  if amount_bln >= 1000
                  else f"{amount_bln:.1f}".replace(".", ",") + " млрд ₸")
    c4.metric("Сумма к оплате", amount_txt, help="Кумулятивно по всем случаям снимка ЭРСБ")

    top = dtr.sort_values("discharged_total", ascending=False).head(15).copy()
    top["МО"] = top["medicine_organization"].map(short_mo)
    fig = px.bar(top.sort_values("discharged_total"), x="discharged_total", y="МО",
                 orientation="h", title="Топ-15 организаций по числу пролеченных случаев",
                 color="discharged_total",
                 color_continuous_scale=[T["scale_a"], T["accent"]],
                 labels={"discharged_total": "Пролечено случаев"})
    fig.update_yaxes(title=None)
    st.plotly_chart(style_fig(fig, 500), width="stretch", config={"displayModeBar": False})

    left, right = st.columns([1, 1], gap="medium")
    with left:
        bd = tr.loc[tr["bed_days_per_case"].between(0, 40), "bed_days_per_case"]
        fig = px.histogram(x=bd, nbins=60, title="Койко-дней на случай (0–40)",
                           labels={"x": "Койко-дней на случай", "count": "Число МО"})
        fig.update_traces(marker_color=T["accent"])
        st.plotly_chart(style_fig(fig), width="stretch", config={"displayModeBar": False})
    with right:
        big = dtr[dtr["discharged_total"] >= 100].copy()
        big["Летальность"] = (big["deaths_share"] * 100).round(2)
        worst = big.sort_values("Летальность", ascending=False).head(15)
        fig = px.bar(worst.sort_values("Летальность"), x="Летальность", y="МО" if "МО" in worst else worst.index,
                     orientation="h", title="Летальность, % (МО ≥ 100 случаев, топ-15)",
                     color="Летальность", color_continuous_scale=[T["scale_w"], T["warn"]])
        fig.update_yaxes(title=None)
        st.plotly_chart(style_fig(fig), width="stretch", config={"displayModeBar": False})

    st.divider()
    st.subheader("Сводная таблица (топ-15 по объёму)")
    tbl = pd.DataFrame({
        "Организация": top["МО"],
        "Регион": top["region"].map(fmt_cat),
        "Пролечено": top["discharged_total"].map(nfmt),
        "Койко-дней на случай": top["bed_days_per_case"].round(1),
        "Летальность, %": (top["deaths_share"] * 100).round(2),
        "Сумма, млрд ₸": (top["amount_to_pay"] / 1e9).round(2),
    }).reset_index(drop=True)
    st.dataframe(tbl, use_container_width=True, height=430, hide_index=True)
    st.caption("ЭРСБ — кумулятивный снимок без разбивки по периодам: годится для сравнения "
               "организаций, но не для динамики. Регион восстановлен по названию организации "
               "(покрытие ~44% — у части частных клиник регион в названии не указан).")


def page_forecast() -> None:
    hero("Прогнозы и модели",
         "Прогноз времени ожидания, прогноз нагрузки стационаров и детектор всплесков",
         "LightGBM", "Валидация по времени", "SHAP-объяснения")

    tab_wait, tab_load, tab_anom = st.tabs(
        ["Время ожидания", "Нагрузка стационара", "Всплески отказов"])

    with tab_wait:
        art = waiting_artifact()
        cats = art["categories"]
        m = art["metrics"]

        c1, c2 = st.columns(2)
        mo = c1.selectbox("Стационар", cats["hospital_mo"], index=cats["hospital_mo"].index(
            next((x for x in cats["hospital_mo"] if "Областная клиническая больница" in x and "Туркестанской" in x), 0)),
            help="Начните вводить название для поиска")
        profile = c2.selectbox("Профиль коек", cats["bed_profile"])
        c3, c4, c5 = st.columns(3)
        icd = c3.selectbox("Группа МКБ-10", cats["icd_group"],
                           format_func=lambda x: f"{x} — {ICD_CHAPTERS.get(x, 'прочее')}")
        region = c4.selectbox("Регион стационара", cats["region"], format_func=fmt_cat)
        terr = c5.selectbox("Тип населённого пункта", cats["territorial_type"], format_func=fmt_cat)
        c6, c7, c8 = st.columns(3)
        purpose = c6.selectbox("Цель направления", cats["referral_purpose"], format_func=fmt_cat)
        fin = c7.selectbox("Источник финансирования", cats["finance_source"], format_func=fmt_cat)
        month = c8.selectbox("Месяц плановой госпитализации", [1, 2, 3, 4, 5, 6],
                             format_func=lambda x: MONTHS_RU.get(x, x),
                             help="Обучена на Q1 2025; апрель–июнь — экстраполяция")

        X, disp = model_waiting.make_row(
            art, hospital_mo=mo, bed_profile=profile, icd_group=icd, region=region,
            territorial_type=terr, referral_purpose=purpose, finance_source=fin,
            month=int(month), dow=0)
        pred, contrib = model_waiting.predict_one(art, X, disp, explainer=_waiting_explainer())
        contrib["Признак"] = contrib["Признак"].map(FEATURE_RU)

        left, right = st.columns([1, 2], gap="medium")
        with left:
            st.metric("Прогноз ожидания", f"≈ {pred:.0f} дн.",
                      help=f"Медианная ошибка модели на мартовской проверке: {m['median_ae']:.1f} дня")
            st.caption("Прогноз — поддержка решения, окончательное решение за специалистом "
                       "(human-in-the-loop).")
            card("Качество модели (март 2025)",
                 f"Обучена на {nfmt(m['n_train'])} направлениях янв–фев, проверена на "
                 f"{nfmt(m['n_test'])} мартовских. MAE {m['mae']:.1f} дн., WAPE {m['wape']*100:.0f}%, "
                 f"±1 день: {m['within_1d']*100:.0f}% прогнозов.")
        with right:
            plot_df = contrib.head(7).iloc[::-1]
            plot_df = plot_df[plot_df["Вклад (дней)"].abs() > 0.05]
            fig = px.bar(plot_df, x="Вклад (дней)", y="Признак", orientation="h",
                         title="Что повлияло на этот прогноз (SHAP)",
                         color="Вклад (дней)",
                         color_continuous_scale=[T["accent"], T["grid"], T["warn"]],
                         color_continuous_midpoint=0,
                         hover_data={"Значение": True})
            fig.update_yaxes(title=None)
            st.plotly_chart(style_fig(fig, 340), width="stretch", config={"displayModeBar": False})
            with st.expander("Значения признаков этого направления"):
                st.dataframe(contrib[["Признак", "Значение", "Вклад (дней)"]].round(2),
                             use_container_width=True, hide_index=True)

        st.divider()
        imp = pd.DataFrame(art["importance"], columns=["Признак", "Важность (gain)"])
        imp["Признак"] = imp["Признак"].map(FEATURE_RU)
        fig = px.bar(imp.head(8).sort_values("Важность (gain)"), x="Важность (gain)", y="Признак",
                     orientation="h", title="Глобальная важность факторов модели",
                     color="Важность (gain)", color_continuous_scale=[T["scale_a"], T["accent"]])
        fig.update_yaxes(title=None)
        st.plotly_chart(style_fig(fig, 320), width="stretch", config={"displayModeBar": False})

    with tab_load:
        art2 = load_artifact()
        counts = load_counts()
        lm = art2["metrics"]

        vol = counts.groupby("mo", observed=True)["count"].sum().sort_values(ascending=False)
        c1, c2 = st.columns(2)
        mo = c1.selectbox("Стационар", vol.index.tolist(), format_func=lambda x: short_mo(x, 60),
                          help="Отсортированы по объёму госпитализаций")
        profiles = (counts[counts["mo"] == mo].groupby("profile", observed=True)["count"]
                    .sum().sort_values(ascending=False))
        profile = c2.selectbox("Профиль коек", profiles.index.tolist(),
                               format_func=lambda p: f"{p} ({nfmt(int(profiles[p]))} за Q1)")

        fc = model_load.forecast(art2, counts, mo, profile)
        hist = (counts[(counts["mo"] == mo) & (counts["profile"] == profile)]
                .sort_values("date").tail(45))

        if fc.empty:
            st.warning("По этой паре нет истории госпитализаций — выберите другую.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist["date"], y=hist["count"], mode="lines",
                                     name="Факт", line=dict(color=T["accent"], width=2)))
            fig.add_trace(go.Scatter(x=fc["date"], y=fc["hi"], mode="lines",
                                     line=dict(width=0), showlegend=False,
                                     hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=fc["date"], y=fc["lo"], mode="lines",
                                     line=dict(width=0), fill="tonexty",
                                     fillcolor=T["accent_soft"], showlegend=False,
                                     hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=fc["date"], y=fc["yhat"], mode="lines",
                                     name=f"Прогноз на {art2['horizon']} дн.",
                                     line=dict(color=T["warn"], width=3, dash="dash")))
            fig.update_layout(title="Госпитализации по дням: факт и прогноз")
            st.plotly_chart(style_fig(fig, 420), width="stretch", config={"displayModeBar": False})

            c1, c2, c3 = st.columns(3)
            c1.metric("Прогноз на 7 дней", nfmt(int(fc["yhat"].head(7).sum())),
                      help="Сумма прогнозных госпитализаций на первые 7 дней")
            c2.metric("Прогноз на 14 дней", nfmt(int(fc["yhat"].sum())),
                      help="Сумма прогнозных госпитализаций на весь горизонт")
            c3.metric("Ошибка модели (holdout)", f"WAPE {lm['wape']*100:.0f}%",
                      help=f"Проверка на {art2['cutoff']}–31.03.2025: {nfmt(lm['n_test'])} "
                           "паро-дней, MAE " + f"{lm['mae']:.1f} пациента/день")
            st.caption("Прогноз рекурсивный на 14 дней от последней даты выгрузки (31.03.2025); "
                       "лента — ±1,28 стандартного отклонения ошибок (~80% интервал). "
                       "Модель — поддержка планирования, решение за руководителем.")

    with tab_anom:
        daily = anomaly_daily()
        regions = sorted(daily["region_in"].astype(str).unique())
        region = st.selectbox("Регион", regions)
        d = daily[daily["region_in"].astype(str) == region]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d["date"], y=d["baseline"], mode="lines",
                                 name="Базовая линия (медиана 14 дн.)",
                                 line=dict(color=T["grey"], width=2, dash="dot")))
        fig.add_trace(go.Scatter(x=d["date"], y=d["count"], mode="lines",
                                 name="Отказы по дням",
                                 line=dict(color=T["warn"], width=2)))
        sp = d[d["spike"]]
        fig.add_trace(go.Scatter(x=sp["date"], y=sp["count"], mode="markers",
                                 name=f"Всплески (z≥3,5): {len(sp)}",
                                 marker=dict(color=T["accent"], size=12, symbol="diamond"),
                                 text=[f"z={z:.1f}" for z in sp["z"]]))
        fig.update_layout(title=f"{region}: отказы, базовая линия и всплески")
        st.plotly_chart(style_fig(fig, 400), width="stretch", config={"displayModeBar": False})
        tbl = anomalies.spikes_table(daily, last_days=90)
        st.subheader("Все всплески за последние 90 дней")
        st.dataframe(tbl, use_container_width=True, hide_index=True, height=300)
        st.caption("Детектор: скользящая медиана предыдущих 14 дней как базовая линия, "
                   "MAD как устойчивая мера разброса, порог z ≥ 3,5 только вверх. "
                   "Всплеск — сигнал аналитику проверить причину, а не автоматическое решение.")


PAGES = {
    "Обзор данных": page_overview,
    "Очереди на госпитализацию": page_queue,
    "Отказы": page_refusals,
    "Пролеченные случаи (ЭРСБ)": page_treated,
    "Прогнозы и модели": page_forecast,
}

with st.sidebar:
    st.markdown(
        f'<div class="side-brand">{LOGO_SVG}<div><b>Стационары РК</b><br>'
        f'<small>GovTech Camp · Кейс 1</small></div></div>',
        unsafe_allow_html=True,
    )
    st.divider()
    theme_pick = st.segmented_control("Тема", ["Светлая", "Тёмная"], default="Светлая", key="theme")
    st.divider()
    page = st.radio("Раздел", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.caption("Данные: открытые наборы МЗ РК\n\nИС БГ · ЭРСБ · 2025 Q1")

T = THEMES[theme_pick or "Светлая"]
st.markdown(css_for(T), unsafe_allow_html=True)

PAGES[page]()
