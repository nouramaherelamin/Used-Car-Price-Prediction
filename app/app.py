from pathlib import Path
import base64
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / 'data' / 'raw' / 'car data.csv'
MODEL_PATH = BASE_DIR / 'models' / 'best_car_price_model.joblib'

PAGE_TITLE = 'Used Car Price Intelligence'
BG = '#0A0F0D'
PANEL = '#111815'
PANEL2 = '#18231D'
LINE = '#26372F'
TEXT = '#F4F7F5'
MUTED = '#9AA8A0'
ACCENT = '#19B779'
ACCENT2 = '#42D99A'
BLUE = '#79C9B0'
GREEN = '#27C985'
RED = '#E56B78'
HERO_IMAGE = BASE_DIR / 'app' / 'assets' / 'used_cars_hero.png'
CAR_ICON = BASE_DIR / 'app' / 'assets' / 'car_favicon.png'

PAGES = ['Overview', 'Price Analysis', 'Market Insights', 'Price Predictor', 'Model Performance', 'Data Explorer', 'Data Quality']

st.set_page_config(page_title=PAGE_TITLE, page_icon=str(CAR_ICON), layout='wide', initial_sidebar_state='expanded')

st.markdown('''<style>
:root{--bg:#0A0F0D;--panel:#111815;--panel2:#18231D;--line:#26372F;--text:#F4F7F5;--muted:#9AA8A0;--accent:#19B779;--accent2:#42D99A;--soft:#F4F7F5}
*{scrollbar-width:thin;scrollbar-color:#28533F #08100C}
.stApp{background:radial-gradient(circle at 85% 8%,rgba(25,183,121,.055),transparent 28%),var(--bg)!important;color:var(--text)!important;animation:appReveal .65s ease both}.block-container{max-width:1500px!important;padding:1.15rem 2rem 3rem!important}
.stApp .stMarkdown,.stApp .stMarkdown p,.stApp label,.stApp [data-testid="stWidgetLabel"],.stApp [data-testid="stWidgetLabel"] p{color:var(--text)!important}.stApp [data-testid="stCaptionContainer"],.stApp [data-testid="stCaptionContainer"] p{color:var(--muted)!important}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#07100B 0%,#0A1510 52%,#07100B 100%)!important;border-right:1px solid #1E3028!important;box-shadow:18px 0 55px rgba(0,0,0,.18)}section[data-testid="stSidebar"] .block-container{padding:1rem .82rem 2rem!important}
section[data-testid="stSidebar"] [data-testid="stRadio"] label{border-radius:12px!important;padding:.58rem .72rem!important;color:#C9D4CE!important;transition:transform .25s cubic-bezier(.2,.8,.2,1),background .25s ease,box-shadow .25s ease!important;animation:navIn .5s ease both}section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover{background:#142019!important;transform:translateX(5px)!important;box-shadow:inset 3px 0 0 rgba(66,217,154,.35)}section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked){background:linear-gradient(90deg,var(--accent),#21A873)!important;color:#06110C!important;font-weight:900!important;box-shadow:0 8px 26px rgba(25,183,121,.2),inset 3px 0 0 #8FF3C5!important}section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p,section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) span{color:#06110C!important}
[data-baseweb="select"]>div{background:linear-gradient(145deg,#18231D,#111815)!important;border-color:#30483D!important;color:var(--soft)!important;transition:.25s ease}[data-baseweb="select"]>div:hover{border-color:var(--accent)!important;box-shadow:0 0 0 1px rgba(25,183,121,.15),0 8px 20px rgba(0,0,0,.18)}[data-baseweb="select"] input{color:var(--soft)!important}div[role="listbox"],div[role="option"]{background:#111815!important;color:var(--soft)!important}
.stButton button{position:relative;overflow:hidden;border-radius:11px!important;border:1px solid #30483D!important;background:#111815!important;color:var(--soft)!important;font-weight:800!important;transition:transform .25s ease,border-color .25s ease,box-shadow .25s ease!important}.stButton button:before{content:"";position:absolute;top:0;left:-120%;width:70%;height:100%;background:linear-gradient(100deg,transparent,rgba(255,255,255,.16),transparent);transform:skewX(-20deg);transition:left .65s ease}.stButton button:hover:before{left:140%}.stButton button:hover{border-color:var(--accent)!important;color:var(--accent2)!important;transform:translateY(-3px)!important;box-shadow:0 12px 28px rgba(0,0,0,.28),0 0 18px rgba(25,183,121,.08)}button[kind="primary"]{background:linear-gradient(135deg,#19B779,#2ED18B)!important;color:#06110C!important;border-color:var(--accent)!important;font-weight:950!important;box-shadow:0 10px 28px rgba(25,183,121,.2)!important}button[kind="primary"]:hover{color:#021008!important;box-shadow:0 16px 34px rgba(25,183,121,.3)!important}.stTextInput input,.stNumberInput input{background:linear-gradient(145deg,#18231D,#111815)!important;color:#fff!important;border-color:#30483D!important;transition:.25s ease}.stTextInput input:focus,.stNumberInput input:focus{border-color:var(--accent)!important;box-shadow:0 0 0 1px var(--accent),0 0 20px rgba(25,183,121,.09)!important}
.hero{position:relative!important;overflow:hidden!important;min-height:455px!important;margin:0 0 2.5rem!important;padding:0!important;border:1px solid rgba(25,183,121,.34)!important;border-radius:28px!important;background:#07100C!important;box-shadow:0 28px 80px rgba(0,0,0,.5),inset 0 1px 0 rgba(255,255,255,.045)!important;isolation:isolate}.hero:before{content:"";position:absolute;inset:0;z-index:1;background-image:linear-gradient(rgba(66,217,154,.045) 1px,transparent 1px),linear-gradient(90deg,rgba(66,217,154,.045) 1px,transparent 1px);background-size:44px 44px;mask-image:linear-gradient(90deg,#000,transparent 76%);animation:gridMove 18s linear infinite}.hero:after{content:"";position:absolute;z-index:2;top:-35%;right:18%;width:2px;height:170%;background:linear-gradient(transparent,rgba(66,217,154,.28),transparent);filter:blur(1px);transform:rotate(16deg);animation:scan 7s ease-in-out infinite}.hero-bg{position:absolute!important;inset:0!important;background-position:center!important;background-size:cover!important;background-repeat:no-repeat!important;transform:scale(1.015);animation:heroImage 13s ease-in-out infinite alternate!important;z-index:0}.hero-overlay{position:absolute!important;inset:0!important;z-index:1;background:linear-gradient(90deg,rgba(4,9,7,.99) 0%,rgba(6,13,10,.96) 25%,rgba(7,13,10,.74) 45%,rgba(7,13,10,.22) 69%,rgba(7,13,10,.06) 100%),linear-gradient(180deg,rgba(0,0,0,.05),rgba(0,0,0,.30))!important}.hero-glow{position:absolute!important;left:-120px!important;bottom:-190px!important;width:720px!important;height:470px!important;border-radius:50%!important;background:radial-gradient(circle,rgba(25,183,121,.30),transparent 66%)!important;filter:blur(12px)!important;z-index:1;animation:emeraldGlow 5s ease-in-out infinite!important}.hero-orb{position:absolute;z-index:2;right:5%;top:8%;width:170px;height:170px;border-radius:50%;border:1px solid rgba(66,217,154,.16);box-shadow:0 0 50px rgba(25,183,121,.07),inset 0 0 40px rgba(25,183,121,.04);animation:orbFloat 6s ease-in-out infinite}.hero-orb:after{content:"";position:absolute;inset:20px;border-radius:50%;border:1px dashed rgba(66,217,154,.18);animation:spin 16s linear infinite}.hero-content{position:relative!important;z-index:4!important;min-height:455px!important;display:flex!important;align-items:center!important;padding:3rem 3.15rem!important}.hero-copy{max-width:620px!important;animation:heroCopy .9s cubic-bezier(.2,.8,.2,1) both!important}.hero-accent{width:88px;height:5px;margin-bottom:1.25rem;border-radius:99px;background:linear-gradient(90deg,#19B779,#42D99A);box-shadow:0 0 25px rgba(25,183,121,.58);animation:accent 2.8s ease-in-out infinite}.hero h1{margin:0;color:#F4F7F5;font-size:clamp(2.45rem,4.35vw,4.25rem);line-height:1.01;font-weight:950;letter-spacing:-.058em;text-shadow:0 4px 28px rgba(0,0,0,.42)}.hero h1 span{color:#35D18F!important}.hero p{max-width:585px;margin:1.1rem 0 0;color:#D9E2DD;font-size:1rem;line-height:1.72;text-shadow:0 2px 14px rgba(0,0,0,.58)}.hero-badge{display:inline-flex;margin-top:1.2rem;padding:.58rem .9rem;border:1px solid rgba(66,217,154,.34);border-radius:999px;background:rgba(8,20,14,.62);backdrop-filter:blur(12px);color:#BDEED8;font-size:.73rem;font-weight:850;letter-spacing:.02em;animation:badgePulse 3.2s ease-in-out infinite}.hero-chip-row{display:flex;gap:.55rem;flex-wrap:wrap;margin-top:1rem}.hero-chip{padding:.42rem .68rem;border:1px solid rgba(244,247,245,.12);border-radius:999px;background:rgba(7,13,10,.56);color:#DCE6E1;font-size:.68rem;font-weight:750;backdrop-filter:blur(8px);transition:.3s ease;animation:chipFloat 4s ease-in-out infinite}.hero-chip:nth-child(2){animation-delay:.5s}.hero-chip:nth-child(3){animation-delay:1s}.hero-chip:hover{border-color:rgba(66,217,154,.55);transform:translateY(-4px);box-shadow:0 10px 25px rgba(0,0,0,.2),0 0 18px rgba(25,183,121,.1)}.hero-float{position:absolute;z-index:5;right:3.2%;top:13%;display:grid;gap:.7rem;width:180px;animation:floatPanel 5s ease-in-out infinite}.hero-float-card{padding:.78rem .88rem;border:1px solid rgba(66,217,154,.20);border-radius:14px;background:rgba(9,20,14,.54);backdrop-filter:blur(14px);box-shadow:0 16px 30px rgba(0,0,0,.25);animation:floatCard 4.8s ease-in-out infinite}.hero-float-card:nth-child(2){animation-delay:.6s}.hero-float-card:nth-child(3){animation-delay:1.2s}.hero-float-label{color:#8FA59B;font-size:.58rem;font-weight:800;text-transform:uppercase;letter-spacing:.1em}.hero-float-value{margin-top:.25rem;color:#F4F7F5;font-size:.82rem;font-weight:900}.hero-float-dot{display:inline-block;width:7px;height:7px;margin-right:.35rem;border-radius:50%;background:#42D99A;box-shadow:0 0 12px rgba(66,217,154,.8);animation:dotPulse 1.8s ease-in-out infinite}
.section-title{position:relative;margin:2.15rem 0 .55rem;color:#F4F7F5;font-size:1.42rem;font-weight:900;letter-spacing:-.025em;animation:rise .55s ease both}.section-title:after{content:"";display:block;width:42px;height:3px;margin-top:.55rem;border-radius:99px;background:linear-gradient(90deg,var(--accent),transparent);animation:sectionLine 1.8s ease-in-out infinite}.section-subtitle{margin:0 0 1.15rem;color:#9AA8A0;font-size:.88rem;line-height:1.55;animation:fadeIn .65s ease both}.kpi{box-sizing:border-box;position:relative;overflow:hidden;min-height:128px;padding:1.2rem 1.25rem;border:1px solid #294037;border-radius:17px;background:linear-gradient(145deg,#151E19 0%,#101713 72%,#0C120F 100%);box-shadow:0 10px 28px rgba(0,0,0,.25);transition:transform .32s cubic-bezier(.2,.8,.2,1),border-color .32s ease,box-shadow .32s ease;animation:card .65s ease both}.kpi:after{content:"";position:absolute;top:-80%;left:-30%;width:30%;height:260%;background:linear-gradient(90deg,transparent,rgba(255,255,255,.055),transparent);transform:rotate(20deg);animation:shine 5.5s ease-in-out infinite}.kpi:before{content:"";position:absolute;left:0;top:0;width:100%;height:3px;background:linear-gradient(90deg,#19B779,#42D99A,transparent);transform:scaleX(.35);transform-origin:left;transition:.35s ease}.kpi:hover{transform:translateY(-8px) scale(1.015);border-color:rgba(25,183,121,.52);box-shadow:0 20px 40px rgba(0,0,0,.38),0 0 28px rgba(25,183,121,.08)}.kpi:hover:before{transform:scaleX(1)}.kpi-label{color:#9FAFA7;font-size:.66rem;line-height:1.2;font-weight:850;letter-spacing:.095em;text-transform:uppercase}.kpi-value{margin-top:.55rem;color:#F4F7F5;font-size:1.55rem;line-height:1.12;font-weight:900;letter-spacing:-.035em}
.insight{position:relative;overflow:hidden;margin:.7rem 0 1.15rem;padding:1.05rem 1.2rem 1.05rem 1.25rem;border:1px solid #29483A;border-left:4px solid #19B779;border-radius:15px;background:linear-gradient(145deg,#101B15,#0D1511);color:#E3EBE6;box-shadow:0 8px 22px rgba(0,0,0,.18);line-height:1.5;animation:slideUp .65s ease both}.insight:after{content:"";position:absolute;inset:0;background:linear-gradient(110deg,transparent 20%,rgba(66,217,154,.07),transparent 70%);transform:translateX(-100%);animation:insightSweep 6s ease-in-out infinite}.insight strong{color:#42D99A}[data-testid="stMetric"]{min-height:118px!important;border:1px solid #294037!important;border-radius:17px!important;background:linear-gradient(145deg,#151E19,#101713)!important;box-shadow:0 10px 28px rgba(0,0,0,.25)!important;transition:.3s ease!important}[data-testid="stMetric"]:hover{transform:translateY(-6px) scale(1.01);border-color:rgba(25,183,121,.45)!important;box-shadow:0 18px 36px rgba(0,0,0,.28),0 0 20px rgba(25,183,121,.06)!important}[data-testid="stMetricLabel"],[data-testid="stMetricLabel"] p{color:#9FAFA7!important}[data-testid="stMetricValue"],[data-testid="stMetricValue"] div{color:#F4F7F5!important;font-weight:900!important}[data-testid="stPlotlyChart"],[data-testid="stDataFrame"]{background:#111815!important;border:1px solid #294037!important;border-radius:17px!important;box-shadow:0 10px 28px rgba(0,0,0,.25)!important;transition:transform .35s ease,box-shadow .35s ease,border-color .35s ease!important;animation:panelIn .7s ease both}[data-testid="stPlotlyChart"]:hover,[data-testid="stDataFrame"]:hover{transform:translateY(-4px);border-color:rgba(25,183,121,.3)!important;box-shadow:0 18px 40px rgba(0,0,0,.34),0 0 25px rgba(25,183,121,.045)!important}div[data-testid="stHorizontalBlock"]{gap:1.25rem!important;margin-bottom:1.15rem!important;align-items:stretch!important}.quality{position:relative;overflow:hidden;min-height:135px;padding:1.1rem 1.15rem;border:1px solid #294037;border-radius:16px;background:linear-gradient(145deg,#151E19,#101713);box-shadow:0 10px 28px rgba(0,0,0,.2);transition:.32s ease;animation:card .6s ease both}.quality:after{content:"";position:absolute;width:100px;height:100px;right:-50px;top:-50px;border-radius:50%;background:rgba(25,183,121,.08);filter:blur(2px);transition:.35s ease}.quality:hover{transform:translateY(-7px);border-color:rgba(25,183,121,.42);box-shadow:0 18px 34px rgba(0,0,0,.3)}.quality:hover:after{transform:scale(1.6);background:rgba(25,183,121,.12)}.quality b{display:block;color:#42D99A;font-size:.67rem;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.45rem}.quality span{color:#E6EEE9;font-size:.87rem;line-height:1.5}.predict-result{position:relative;overflow:hidden;padding:1.55rem;border:1px solid rgba(25,183,121,.38);border-radius:19px;background:radial-gradient(circle at 85% 10%,rgba(66,217,154,.12),transparent 34%),linear-gradient(145deg,#102018,#0D1611);box-shadow:0 15px 35px rgba(0,0,0,.3);animation:resultPop .65s cubic-bezier(.2,.8,.2,1) both}.predict-result:before{content:"";position:absolute;inset:0;background:linear-gradient(120deg,transparent 15%,rgba(66,217,154,.08),transparent 55%);transform:translateX(-100%);animation:resultSweep 4s ease-in-out infinite}.predict-price{font-size:2.65rem;font-weight:950;color:#42D99A;letter-spacing:-.04em;text-shadow:0 0 24px rgba(66,217,154,.18)}.predict-shell{padding:1.25rem;border:1px solid #294037;border-radius:19px;background:linear-gradient(145deg,#121B17,#0D1411);box-shadow:0 12px 30px rgba(0,0,0,.22);animation:panelIn .7s ease both}.mini-badge{display:inline-flex;padding:.35rem .58rem;border:1px solid rgba(66,217,154,.2);border-radius:999px;background:rgba(25,183,121,.06);color:#9FE8C4;font-size:.62rem;font-weight:850;letter-spacing:.05em;text-transform:uppercase}.footer{margin:2.2rem 0 .4rem;padding:.75rem 1rem;border:1px solid rgba(25,183,121,.18);border-radius:12px;background:#0C1410;text-align:center;color:#AEB9B2;font-size:.66rem;font-weight:600;animation:fadeIn 1s ease both}.footer a{color:#8FDDB9;text-decoration:none;font-weight:700;margin:0 .35rem;transition:.2s ease}.footer a:hover{color:#42D99A;text-shadow:0 0 12px rgba(66,217,154,.35)}
@keyframes appReveal{from{opacity:0}to{opacity:1}}@keyframes heroImage{from{transform:scale(1.015) translate3d(0,0,0)}to{transform:scale(1.055) translate3d(-.6%,.4%,0)}}@keyframes heroCopy{from{opacity:0;transform:translateX(-30px) translateY(8px)}to{opacity:1;transform:translateX(0) translateY(0)}}@keyframes emeraldGlow{0%,100%{opacity:.5;transform:scale(.94) translate(0,0)}50%{opacity:1;transform:scale(1.08) translate(30px,-15px)}}@keyframes accent{0%,100%{width:88px;box-shadow:0 0 16px rgba(25,183,121,.35)}50%{width:122px;box-shadow:0 0 30px rgba(25,183,121,.72)}}@keyframes rise{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}@keyframes fadeIn{from{opacity:0}to{opacity:1}}@keyframes slideUp{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:translateY(0)}}@keyframes panelIn{from{opacity:0;transform:translateY(18px) scale(.985)}to{opacity:1;transform:translateY(0) scale(1)}}@keyframes card{from{opacity:0;transform:translateY(18px) scale(.97)}to{opacity:1;transform:translateY(0) scale(1)}}@keyframes shine{0%,55%{left:-35%}75%,100%{left:145%}}@keyframes insightSweep{0%,55%{transform:translateX(-100%)}75%,100%{transform:translateX(100%)}}@keyframes sectionLine{0%,100%{width:42px;opacity:.7}50%{width:82px;opacity:1}}@keyframes badgePulse{0%,100%{box-shadow:0 0 0 rgba(25,183,121,0)}50%{box-shadow:0 0 25px rgba(25,183,121,.11)}}@keyframes chipFloat{0%,100%{transform:translateY(0)}50%{transform:translateY(-3px)}}@keyframes floatPanel{0%,100%{transform:translateY(0) rotate(0deg)}50%{transform:translateY(-12px) rotate(.6deg)}}@keyframes floatCard{0%,100%{transform:translateX(0)}50%{transform:translateX(-8px)}}@keyframes dotPulse{0%,100%{transform:scale(.8);opacity:.65}50%{transform:scale(1.25);opacity:1}}@keyframes orbFloat{0%,100%{transform:translate(0,0)}50%{transform:translate(-18px,14px)}}@keyframes spin{to{transform:rotate(360deg)}}@keyframes scan{0%,100%{opacity:0;transform:translateX(-160px) rotate(16deg)}45%,60%{opacity:1}70%{opacity:0;transform:translateX(260px) rotate(16deg)}}@keyframes gridMove{to{background-position:44px 44px}}@keyframes navIn{from{opacity:0;transform:translateX(-8px)}to{opacity:1;transform:translateX(0)}}@keyframes resultPop{from{opacity:0;transform:scale(.94) translateY(12px)}to{opacity:1;transform:scale(1) translateY(0)}}@keyframes resultSweep{0%,60%{transform:translateX(-100%)}80%,100%{transform:translateX(100%)}}
@media(max-width:1100px){.hero-float{right:2.5%;width:155px}.hero-copy{max-width:570px}.block-container{padding-left:1.15rem;padding-right:1.15rem}}@media(max-width:950px){.block-container{padding:1rem .9rem 2.5rem}.hero{min-height:500px}.hero-content{min-height:500px;padding:2rem 1.35rem;align-items:flex-end;padding-bottom:2.1rem}.hero-overlay{background:linear-gradient(180deg,rgba(5,10,8,.42) 0%,rgba(5,10,8,.70) 44%,rgba(5,10,8,.98) 100%)}.hero-copy{max-width:100%}.hero h1{font-size:2.35rem}.hero p{font-size:.92rem}.hero-float,.hero-orb{display:none}}@media(max-width:640px){.block-container{padding:.8rem .65rem 2rem}.hero{min-height:455px;border-radius:21px;margin-bottom:2rem}.hero-content{min-height:455px;padding:1.5rem 1rem 1.5rem}.hero h1{font-size:2.05rem}.hero p{font-size:.85rem;line-height:1.6}.hero-chip{font-size:.61rem}.section-title{font-size:1.2rem}.kpi{min-height:112px;padding:1rem}.kpi-value{font-size:1.25rem}}@media(prefers-reduced-motion:reduce){*,*::before,*::after{animation-duration:.01ms!important;animation-iteration-count:1!important;transition-duration:.01ms!important}}footer{visibility:hidden}
</style>''', unsafe_allow_html=True)

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. "
            "Make sure data/raw/car data.csv is committed to GitHub."
        )
    data = pd.read_csv(DATA_PATH)
    data = data.copy()
    data['Car_Age'] = int(data['Year'].max()) - data['Year']
    return data

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. "
            "Make sure models/best_car_price_model.joblib is committed to GitHub."
        )
    return joblib.load(MODEL_PATH)

df = load_data()
model = load_model()


def plot_theme(fig, height=430):
    fig.update_layout(template='plotly_dark', paper_bgcolor=PANEL, plot_bgcolor=PANEL, font=dict(color=TEXT), height=height, colorway=['#19B779','#42D99A','#8FDDB9','#79C9B0','#C9E8DA','#0D6B4A'],
                      margin=dict(l=24,r=24,t=64,b=34), legend=dict(bgcolor='rgba(0,0,0,0)'), title=dict(x=.02,xanchor='left'))
    fig.update_xaxes(gridcolor='rgba(255,255,255,.08)', zerolinecolor='rgba(255,255,255,.08)', automargin=True)
    fig.update_yaxes(gridcolor='rgba(255,255,255,.08)', zerolinecolor='rgba(255,255,255,.08)', automargin=True)
    return fig


def kpi(label, value):
    st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>', unsafe_allow_html=True)


def section(title, subtitle=''):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle: st.markdown(f'<div class="section-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def hero():
    years = f"{int(df.Year.min())}–{int(df.Year.max())}"
    encoded = base64.b64encode(HERO_IMAGE.read_bytes()).decode("utf-8") if HERO_IMAGE.exists() else ""
    st.markdown(f'''<div class="hero"><div class="hero-bg" style="background-image:url('data:image/png;base64,{encoded}');"></div><div class="hero-overlay"></div><div class="hero-glow"></div><div class="hero-orb"></div><div class="hero-float"><div class="hero-float-card"><div class="hero-float-label"><span class="hero-float-dot"></span>Live Dataset</div><div class="hero-float-value">{len(df):,} Vehicles</div></div><div class="hero-float-card"><div class="hero-float-label">Market Range</div><div class="hero-float-value">{years}</div></div><div class="hero-float-card"><div class="hero-float-label">AI Engine</div><div class="hero-float-value">Gradient Boosting</div></div></div><div class="hero-content"><div class="hero-copy"><div class="hero-accent"></div><h1>Used Car <span>Price Intelligence</span></h1><p>Explore resale-market patterns, understand the factors associated with vehicle value, and estimate a selling price with the trained machine-learning model.</p><span class="hero-badge">Interactive analytics · {len(df):,} vehicle records · {years}</span><div class="hero-chip-row"><span class="hero-chip">Market Analysis</span><span class="hero-chip">Price Prediction</span><span class="hero-chip">Machine Learning</span></div></div></div></div>''', unsafe_allow_html=True)


def sidebar():
    with st.sidebar:
        st.markdown('# 🚗 Car Price AI')
        st.caption('Interactive used-car analytics')
        st.markdown('### Navigation')
        st.radio('Go to', PAGES, key='nav', label_visibility='collapsed')
        st.divider()
        st.markdown('### Market Filters')
        st.selectbox('Fuel Type', ['All'] + sorted(df.Fuel_Type.unique().tolist()), key='fuel_filter')
        st.selectbox('Selling Type', ['All'] + sorted(df.Selling_type.unique().tolist()), key='selling_filter')
        st.selectbox('Transmission', ['All'] + sorted(df.Transmission.unique().tolist()), key='trans_filter')
        st.slider('Manufacturing Year', int(df.Year.min()), int(df.Year.max()), (int(df.Year.min()), int(df.Year.max())), key='year_filter')
        st.slider('Owner Count', int(df.Owner.min()), int(df.Owner.max()), (int(df.Owner.min()), int(df.Owner.max())), key='owner_filter')
        if st.button('↺ Reset Filters', use_container_width=True):
            for k,v in [('fuel_filter','All'),('selling_filter','All'),('trans_filter','All'),('year_filter',(int(df.Year.min()),int(df.Year.max()))),('owner_filter',(int(df.Owner.min()),int(df.Owner.max())))]: st.session_state[k]=v
            st.rerun()
        st.divider()
        st.metric('Vehicles in Dataset', f'{len(df):,}')
        st.markdown('<div style="text-align:center;margin-top:18px;color:#8EA098;font-size:11px;line-height:1.7"><strong style="color:#A8E8CA;font-size:12px">© 2026 Noura Maher</strong><br><span>Used Car Price Intelligence</span><br><a href="https://www.linkedin.com/in/nouramaherelamin/" target="_blank" style="color:#8FDDB9;text-decoration:none;font-weight:700">LinkedIn</a> &nbsp; <a href="https://github.com/nouramaherelamin" target="_blank" style="color:#8FDDB9;text-decoration:none;font-weight:700">GitHub</a></div>', unsafe_allow_html=True)


def filtered_data():
    x=df.copy()
    if st.session_state.fuel_filter!='All': x=x[x.Fuel_Type==st.session_state.fuel_filter]
    if st.session_state.selling_filter!='All': x=x[x.Selling_type==st.session_state.selling_filter]
    if st.session_state.trans_filter!='All': x=x[x.Transmission==st.session_state.trans_filter]
    x=x[x.Year.between(*st.session_state.year_filter)]
    x=x[x.Owner.between(*st.session_state.owner_filter)]
    return x


def overview(x):
    section('Overview','A live market summary based on the active filters.')
    vals=[('Vehicles',f'{len(x):,}'),('Average Selling Price',f'{x.Selling_Price.mean():.2f}'),('Median Selling Price',f'{x.Selling_Price.median():.2f}'),('Average Present Price',f'{x.Present_Price.mean():.2f}'),('Average KM Driven',f'{x.Driven_kms.mean():,.0f}'),('Average Car Age',f'{x.Car_Age.mean():.1f} yrs')]
    for row in (vals[:3], vals[3:]):
        cols=st.columns(3, gap='large')
        for c,(l,v) in zip(cols,row):
            with c:kpi(l,v)
    st.markdown('<div class="insight"><strong>Key Insight</strong> Selling price is strongly connected to the vehicle\'s present price, age, and usage. Use the interactive filters to see how market segments change the observed price distribution.</div>',unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1:
        fig=px.scatter(x,x='Present_Price',y='Selling_Price',color='Fuel_Type',hover_data=['Car_Name','Year','Driven_kms'],title='Present Price vs Selling Price')
        st.plotly_chart(plot_theme(fig,420),use_container_width=True)
    with c2:
        avg=x.groupby('Year',as_index=False)['Selling_Price'].mean()
        fig=px.line(avg,x='Year',y='Selling_Price',markers=True,title='Average Selling Price by Year')
        fig.update_traces(line_color=ACCENT,marker_color=ACCENT)
        st.plotly_chart(plot_theme(fig,420),use_container_width=True)


def price_analysis(x):
    section('Price Analysis','Explore resale prices across fuel, transmission, selling type, age, and usage.')
    c1,c2=st.columns(2)
    with c1:
        fig=px.histogram(x,x='Selling_Price',nbins=30,title='Selling Price Distribution')
        fig.update_traces(marker_color=ACCENT)
        st.plotly_chart(plot_theme(fig),use_container_width=True)
    with c2:
        fig=px.box(x,x='Fuel_Type',y='Selling_Price',color='Fuel_Type',title='Selling Price by Fuel Type')
        st.plotly_chart(plot_theme(fig),use_container_width=True)
    c1,c2=st.columns(2)
    with c1:
        fig=px.box(x,x='Transmission',y='Selling_Price',color='Transmission',title='Selling Price by Transmission')
        st.plotly_chart(plot_theme(fig),use_container_width=True)
    with c2:
        fig=px.scatter(x,x='Driven_kms',y='Selling_Price',size='Present_Price',color='Car_Age',hover_data=['Car_Name','Year'],title='Mileage vs Selling Price')
        st.plotly_chart(plot_theme(fig),use_container_width=True)


def insights(x):
    section('Market Insights','Descriptive comparisons that explain the observed used-car market.')
    g=x.groupby('Fuel_Type')['Selling_Price'].agg(['mean','median','count']).reset_index().sort_values('mean',ascending=False)
    c1,c2=st.columns(2)
    with c1:
        fig=px.bar(g,x='Fuel_Type',y='mean',text='mean',title='Average Selling Price by Fuel Type')
        fig.update_traces(marker_color=ACCENT,texttemplate='%{text:.2f}')
        st.plotly_chart(plot_theme(fig),use_container_width=True)
    with c2:
        owner=x.groupby('Owner')['Selling_Price'].mean().reset_index()
        fig=px.bar(owner,x='Owner',y='Selling_Price',text='Selling_Price',title='Average Selling Price by Previous Owners')
        fig.update_traces(marker_color=BLUE,texttemplate='%{text:.2f}')
        st.plotly_chart(plot_theme(fig),use_container_width=True)
    st.dataframe(g.rename(columns={'mean':'Mean Selling Price','median':'Median Selling Price','count':'Vehicles'}),use_container_width=True,hide_index=True)
    st.markdown('<div class="insight"><strong>Interpretation</strong> These are observed group differences in this dataset. They should not be interpreted as causal effects or as guaranteed market pricing rules.</div>',unsafe_allow_html=True)


def predictor():
    section('Price Predictor','Build a vehicle profile, then generate an estimated resale price from the trained model.')
    left,right=st.columns([1.38,.82],gap='large')
    with left:
        st.markdown('<div class="predict-shell"><span class="mini-badge">Vehicle Profile</span>',unsafe_allow_html=True)
        a,b=st.columns(2,gap='medium')
        with a:
            year=st.number_input('Manufacturing Year',int(df.Year.min()),int(df.Year.max()),int(df.Year.median()))
            present=st.number_input('Present Price',0.1,float(df.Present_Price.max()*1.5),float(df.Present_Price.median()),step=0.1)
            kms=st.number_input('Driven Kilometers',0,int(df.Driven_kms.max()*1.5),int(df.Driven_kms.median()),step=1000)
        with b:
            fuel=st.selectbox('Fuel Type',sorted(df.Fuel_Type.unique()))
            selling=st.selectbox('Selling Type',sorted(df.Selling_type.unique()))
            transmission=st.selectbox('Transmission',sorted(df.Transmission.unique()))
            owner=st.number_input('Previous Owners',int(df.Owner.min()),int(df.Owner.max()),0,step=1)
        st.caption('The prediction uses the same feature structure as the trained pipeline, including calculated car age.')
        estimate=st.button('✦  Estimate Selling Price',type='primary',use_container_width=True)
        st.markdown('</div>',unsafe_allow_html=True)
    with right:
        st.markdown('<div class="predict-shell"><span class="mini-badge">AI Estimate</span><div style="height:14px"></div>',unsafe_allow_html=True)
        if estimate:
            row=pd.DataFrame([{'Year':year,'Present_Price':present,'Driven_kms':kms,'Fuel_Type':fuel,'Selling_type':selling,'Transmission':transmission,'Owner':owner,'Car_Age':int(df.Year.max())-year}])
            price=max(0,float(model.predict(row)[0]))
            st.markdown(f'<div class="predict-result"><div style="color:#AEB6C2;font-size:.72rem;font-weight:800;text-transform:uppercase;letter-spacing:.09em">Estimated Selling Price</div><div class="predict-price">{price:.2f}</div><div style="color:#9CA3AF;font-size:.82rem;margin-top:.45rem">Model estimate · not a guaranteed market price</div></div>',unsafe_allow_html=True)
        else:
            st.markdown('<div class="predict-result" style="min-height:205px;display:flex;flex-direction:column;justify-content:center"><div style="font-size:2rem">◈</div><div style="color:#F4F7F5;font-size:1.05rem;font-weight:900;margin-top:.45rem">Ready to predict</div><div style="color:#9AA8A0;font-size:.82rem;line-height:1.55;margin-top:.35rem">Fill in the vehicle profile and launch the estimator to see the model output here.</div></div>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)
    st.markdown('<div class="insight"><strong>Prediction note</strong> The estimator is a machine-learning decision-support tool. Real transaction prices can differ because of condition, location, negotiation, accessories, and other factors not represented in the dataset.</div>',unsafe_allow_html=True)


def model_performance():
    section('Model Performance','Evaluation of the selected Gradient Boosting Regressor on the held-out test set.')
    vals=[('Model','Gradient Boosting'),('MAE','1.09'),('RMSE','2.26'),('R²','0.80'),('CV Folds','5')]
    cols=st.columns(5)
    for c,(l,v) in zip(cols,vals):
        with c:kpi(l,v)
    comp=pd.read_csv(BASE_DIR/'reports'/'model_comparison.csv')
    left,right=st.columns([.9,1.1],gap='large')
    with left:
        st.markdown('<div class="mini-badge">Model Comparison</div>',unsafe_allow_html=True)
        st.dataframe(comp,use_container_width=True,hide_index=True)
    with right:
        fig=px.bar(comp,x=comp.columns[0],y='RMSE',title='Cross-Model RMSE Comparison')
        fig.update_traces(marker_color=ACCENT)
        st.plotly_chart(plot_theme(fig),use_container_width=True)
    pred=pd.read_csv(BASE_DIR/'reports'/'test_predictions.csv')
    if {'Actual_Selling_Price','Predicted_Selling_Price'}.issubset(pred.columns):
        fig=px.scatter(pred,x='Actual_Selling_Price',y='Predicted_Selling_Price',title='Actual vs Predicted Selling Price')
        lo=min(pred.Actual_Selling_Price.min(),pred.Predicted_Selling_Price.min()); hi=max(pred.Actual_Selling_Price.max(),pred.Predicted_Selling_Price.max())
        fig.add_trace(go.Scatter(x=[lo,hi],y=[lo,hi],mode='lines',name='Perfect prediction',line=dict(color=ACCENT,dash='dash')))
        st.plotly_chart(plot_theme(fig),use_container_width=True)


def explorer(x):
    section('Data Explorer','Search, inspect, and download the currently filtered vehicle records.')
    top_left,top_right=st.columns([1.4,.6],gap='large')
    with top_left:
        q=st.text_input('Search car name',placeholder='Type a model or brand...')
    view=x.copy()
    if q.strip(): view=view[view.Car_Name.astype(str).str.contains(q,case=False,na=False,regex=False)]
    with top_right:
        st.markdown(f'<div class="mini-badge" style="margin-top:28px">Showing {len(view):,} / {len(df):,} records</div>',unsafe_allow_html=True)
        st.download_button('↓  Download filtered CSV',view.to_csv(index=False).encode('utf-8'),file_name='filtered_car_data.csv',mime='text/csv',use_container_width=True)
    st.dataframe(view,use_container_width=True,hide_index=True)


def quality():
    section('Data Quality','Checks performed on the supplied vehicle dataset.')
    missing=int(df.isna().sum().sum()); dup=int(df.duplicated().sum())
    cards=[('Rows',f'{len(df):,} records in source dataset'),('Columns',f'{df.shape[1]} variables available'),('Missing Values',f'{missing} missing cells'),('Duplicate Rows',f'{dup} exact duplicate rows'),('Target', 'Selling_Price used for regression'),('Categorical Fields','Fuel, selling type, transmission')]
    cols=st.columns(3)
    for i,(title,body) in enumerate(cards):
        with cols[i%3]: st.markdown(f'<div class="quality"><b>{title}</b><span>{body}</span></div>',unsafe_allow_html=True)
    st.markdown('<div class="insight"><strong>Cleaning decision</strong> Exact duplicate rows were identified during preparation and removed from the processed dataset used for modeling. Car_Name is retained for analysis and display but is not used as a predictive feature.</div>',unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({'Column':df.columns,'Dtype':[str(x) for x in df.dtypes],'Missing':[int(df[c].isna().sum()) for c in df.columns],'Unique':[int(df[c].nunique()) for c in df.columns]}),use_container_width=True,hide_index=True)


def footer():
    st.markdown('''<div class="footer">© 2026 <b style="color:#42D99A">Noura Maher Elamin</b> · Used Car Price Intelligence · <a href="https://www.linkedin.com/in/nouramaherelamin/" target="_blank">LinkedIn</a> · <a href="https://github.com/nouramaherelamin" target="_blank">GitHub</a></div>''',unsafe_allow_html=True)

sidebar()
x=filtered_data()
hero()
if x.empty:
    st.warning('No vehicles match the current filters. Broaden the filters from the sidebar.')
else:
    page=st.session_state.nav
    if page=='Overview': overview(x)
    elif page=='Price Analysis': price_analysis(x)
    elif page=='Market Insights': insights(x)
    elif page=='Price Predictor': predictor()
    elif page=='Model Performance': model_performance()
    elif page=='Data Explorer': explorer(x)
    elif page=='Data Quality': quality()
footer()
