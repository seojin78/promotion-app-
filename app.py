import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

# --- 설정 ---
IMAGE_FILE = "growth_image.jpg"
MODEL_NAME = 'jhgan/ko-sroberta-multitask'

# 모바일 친화적인 레이아웃 설정 (wide 모드 사용 고려, 여기선 centered 유지하며 CSS로 조절)
st.set_page_config(page_title="'26년 승진자 교육 안내", layout="centered")

# --- 스타일 설정 (모바일 최적화 포함) ---
st.markdown("""
<style>
    /* 웹폰트 Pretendard 적용 (가독성 좋은 한글 폰트) */
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');

    /* 기본 스타일 (PC 기준) */
    .stApp {
        background-color: #F3F4F8;
        font-family: 'Pretendard', 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif;
    }
    .chat-container {
        background-color: #AEC6CF;
        border-radius: 20px;
        padding: 20px;
    }
    /* 채팅 메시지 공통 스타일 */
    div[data-testid="stChatMessage"] {
        background-color: transparent !important;
    }
    div[data-testid="stChatMessage"] .stMarkdown {
        word-break: keep-all; /* 한글 단어 단위 줄바꿈 (중요!) */
        line-height: 1.6; /* 줄 간격 넓게 */
    }
    /* 봇 메시지 스타일 */
    div[data-testid="stChatMessage"] div[data-testid="chatAvatarIcon-assistant"] + div {
        background-color: #FFFFFF !important;
        border-radius: 15px;
        padding: 12px 18px; /* 패딩 약간 늘림 */
        color: #333333;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    /* 유저 메시지 스타일 */
    div[data-testid="stChatMessage"] div[data-testid="chatAvatarIcon-user"] + div {
        background-color: #FEE500 !important;
        border-radius: 15px;
        padding: 12px 18px; /* 패딩 약간 늘림 */
        color: #333333;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    /* 정보 카드 스타일 (PC) */
    .info-card {
        background-color: #FFFFFF;
        border-radius: 15px;
        padding: 20px;
        margin: 10px;
        text-align: center;
        border: 2px solid #E0E0E0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        transition: all 0.3s;
        /* 핵심: 긴 글씨 줄바꿈 처리 */
        word-break: keep-all;
        white-space: normal;
    }
    .info-card:hover {
        border-color: #AEC6CF;
        transform: translateY(-3px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .info-card h4 {
        margin-bottom: 12px;
        color: #444;
        font-size: 1.1rem;
        font-weight: bold;
    }
    .info-card p {
        color: #777;
        font-size: 0.95rem;
        margin: 0;
        line-height: 1.5;
    }

    /* =========================================
       모바일 전용 스타일 (화면 너비 768px 이하)
    ========================================= */
    @media (max-width: 768px) {
        /* 전체 컨테이너 여백 줄이기 (화면 넓게 쓰기) */
        .main .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            padding-top: 2rem !important;
        }

        /* 시작 화면 제목 및 내용 폰트 키우기 */
        h1 { font-size: 2.2rem !important; }
        p { font-size: 1.1rem !important; }

        /* 시작하기 버튼 크기 키우기 */
        .stButton button {
            font-size: 1.2rem !important;
            padding: 0.8rem 1rem !important;
        }

        /* 정보 카드 모바일 최적화 */
        .info-card {
            margin: 10px 0 !important; /* 좌우 마진 제거 */
            padding: 15px !important; /* 패딩 조절 */
            width: 100%; /* 너비 꽉 채우기 */
        }
        .info-card h4 {
            font-size: 1.2rem !important; /* 제목 폰트 키움 */
        }
        .info-card p {
            font-size: 1rem !important; /* 내용 폰트 키움 */
        }
        /* 카드 안의 상세보기 버튼 */
        .info-card .stButton button {
            width: 100%; /* 버튼 너비 꽉 채우기 */
            margin-top: 10px;
            font-size: 1rem !important;
        }

        /* 채팅 메시지 폰트 키우기 */
        .stChatMessage .stMarkdown {
            font-size: 1.05rem !important;
        }
        
        /* 사이드바 조정 */
        [data-testid="stSidebar"] {
             width: 85% !important; /* 사이드바 너비 넓게 */
        }
        [data-testid="stSidebar"] .stMarkdown {
             font-size: 1rem !important;
        }
        [data-testid="stSidebar"] img {
            margin-bottom: 15px;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- AI 및 데이터 로드 ---
@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

@st.cache_data
def load_data_from_secrets():
    # Secrets에서 데이터 가져오기 확인
    if "knowledge_base" not in st.secrets:
        return [], []
    
    text = st.secrets["knowledge_base"]
    sections = [s.strip() for s in text.split('###') if s.strip()]
    
    titles = []
    contents = []
    for section in sections:
        lines = section.split('\n', 1)
        if len(lines) >= 1:
            titles.append(lines[0].strip())
            contents.append(section)
    return titles, contents

@st.cache_resource
def create_vector_index(contents, _model):
    if not contents: return None
    embeddings = _model.encode(contents)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def get_answer(query, index, contents, model):
    if index is None or not contents:
        return "⚠️ 데이터가 연결되지 않았습니다. Secrets 설정을 확인해주세요."
    
    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, 1)
    
    # 무조건 가장 유사한 답변 반환 (Threshold 제거)
    return contents[I[0][0]]

# --- 초기화 ---
if 'page' not in st.session_state: st.session_state.page = 'start'
if 'messages' not in st.session_state: st.session_state.messages = []

# 데이터 로드 시도
with st.spinner("데이터 연결 중..."):
    embedder = load_model()
    titles, kb_contents = load_data_from_secrets()
    
    if kb_contents:
        vector_index = create_vector_index(kb_contents, embedder)
        st.session_state.data_ready = True
    else:
        st.session_state.data_ready = False
        vector_index = None

# --- 화면 로직 ---
def handle_user_input(user_query):
    st.session_state.messages.append({"role": "user", "content": user_query})
    answer = get_answer(user_query, vector_index, kb_contents, embedder)
    st.session_state.messages.append({"role": "assistant", "content": answer.replace("\n", "  \n")})

def show_chat_screen():
    with st.sidebar:
        st.markdown("### ✨ 지속적인 성장")
        if os.path.exists(IMAGE_FILE): st.image(IMAGE_FILE, caption="Keep Growing!", use_column_width=True)
        
        # [디버깅 기능] 데이터 연결 상태 표시
        st.markdown("---")
        st.markdown("### 🛠 시스템 상태")
        if st.session_state.data_ready:
            st.success(f"데이터 연결 성공! ({len(titles)}개 주제)")
        else:
            st.error("데이터 연결 실패")
            st.info("Secrets 설정에 'knowledge_base'가 있는지 확인하세요.")
        
        if st.button("처음으로", use_container_width=True):
            st.session_state.page = 'start'; st.session_state.messages = []; st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("type") == "welcome":
                # 모바일에서는 컬럼을 1개로 보여주는게 나을 수 있음.
                # 화면 너비에 따라 자동으로 반응하도록 설정
                cols = st.columns(2)
                cards = [
                    ("🏢 연수원 안내", "시설/위치 안내", "연수원 안내"),
                    ("📅 교육 시간표", "상세 일정 확인", "교육 시간표"),
                    ("🚌 이동방법", "셔틀/주차 안내", "이동방법"),
                    ("📜 Ground Rule", "생활 수칙", "Ground Rule")
                ]
                for i, (title, desc, query) in enumerate(cards):
                    # 모바일에서는 한 줄에 하나씩 보이게 하려면 아래 주석 해제하고 cols[i%2] 주석 처리
                    # with st.container(): 
                    with cols[i % 2]:
                        st.markdown(f"<div class='info-card'><h4>{title}</h4><p>{desc}</p></div>", unsafe_allow_html=True)
                        if st.button("상세보기", key=f"btn_{i}", use_container_width=True):
                            handle_user_input(query)
                            st.rerun()

    if prompt := st.chat_input("질문을 입력하세요"): handle_user_input(prompt); st.rerun()

def show_start_screen():
    st.markdown("<div style='text-align: center; padding: 50px 20px;'>", unsafe_allow_html=True)
    st.markdown("<h1>🎉 '26년 승진자 교육</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #888; margin: 30px 0;'>승진을 축하드립니다!<br>교육 안내를 도와드릴 챗봇입니다.</p>", unsafe_allow_html=True)
    if st.button("시작하기", use_container_width=True):
        st.session_state.page = 'chat'
        st.session_state.messages.append({"role": "assistant", "content": "안녕하세요! '26년 승진자 교육 안내 봇입니다. 🤖\n아래 메뉴를 선택하거나 궁금한 점을 입력해주세요.", "type": "welcome"})
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.page == 'start': show_start_screen()
else: show_chat_screen()
