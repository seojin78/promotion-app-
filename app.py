import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

# --- 설정 ---
IMAGE_FILE = "growth_image.jpg"
MODEL_NAME = 'jhgan/ko-sroberta-multitask'
SIMILARITY_THRESHOLD = 0.5  # 유사도 기준을 약간 완화

st.set_page_config(page_title="'26년 승진자 교육 안내", layout="centered")

# --- 스타일 설정 ---
st.markdown("""
<style>
    .stApp { background-color: #F3F4F8; font-family: 'Apple SD Gothic Neo', sans-serif; }
    .chat-container { background-color: #AEC6CF; border-radius: 20px; padding: 20px; }
    div[data-testid="stChatMessage"] { background-color: transparent !important; }
    div[data-testid="stChatMessage"] div[data-testid="chatAvatarIcon-assistant"] + div {
        background-color: #FFFFFF !important; border-radius: 15px; padding: 10px 15px; color: #333333; box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    div[data-testid="stChatMessage"] div[data-testid="chatAvatarIcon-user"] + div {
        background-color: #FEE500 !important; border-radius: 15px; padding: 10px 15px; color: #333333; box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .info-card {
        background-color: #FFFFFF; border-radius: 15px; padding: 15px; margin: 5px; text-align: center; border: 2px solid #E0E0E0; cursor: pointer;
    }
    .info-card:hover { border-color: #AEC6CF; transform: translateY(-3px); transition: all 0.3s; }
    .info-card h4 { margin-bottom: 8px; color: #555; font-size: 16px; font-weight: bold; }
    .info-card p { color: #888; font-size: 13px; margin: 0; }
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
    
    # 디버깅용: 유사도 점수 출력 (개발자 도구 등에서 확인 가능)
    # print(f"Query: {query}, Distance: {D[0][0]}")

    if D[0][0] > 60: # 임계값 (L2 거리 기준)
         return "죄송합니다. 해당 내용은 안내 자료에 없습니다. 운영진에게 문의해주세요. (운영진 : 홍길동)"
    
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
        if os.path.exists(IMAGE_FILE): st.image(IMAGE_FILE, caption="Keep Growing!")
        
        # [디버깅 기능] 데이터 연결 상태 표시
        st.markdown("---")
        st.markdown("### 🛠 시스템 상태")
        if st.session_state.data_ready:
            st.success(f"데이터 연결 성공! ({len(titles)}개 주제)")
            with st.expander("로드된 주제 확인"):
                for t in titles:
                    st.markdown(f"- {t}")
        else:
            st.error("데이터 연결 실패")
            st.info("Secrets 설정에 'knowledge_base'가 있는지 확인하세요.")
        
        if st.button("처음으로"):
            st.session_state.page = 'start'; st.session_state.messages = []; st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("type") == "welcome":
                cols = st.columns(2)
                # 카드 클릭 시 보낼 질문 매핑
                cards = [
                    ("🏢 연수원 안내", "시설/위치 안내", "연수원 안내"),
                    ("📅 교육 시간표", "상세 일정 확인", "교육 시간표"),
                    ("🚌 이동방법", "셔틀/주차 안내", "이동방법"),
                    ("📜 Ground Rule", "생활 수칙", "Ground Rule")
                ]
                for i, (title, desc, query) in enumerate(cards):
                    with cols[i % 2]:
                        st.markdown(f"<div class='info-card'><h4>{title}</h4><p>{desc}</p></div>", unsafe_allow_html=True)
                        if st.button("상세보기", key=f"btn_{i}"):
                            handle_user_input(query)
                            st.rerun()

    if prompt := st.chat_input("질문을 입력하세요"): handle_user_input(prompt); st.rerun()

def show_start_screen():
    st.markdown("<div style='text-align: center; padding: 50px;'>", unsafe_allow_html=True)
    st.markdown("<h1>🎉 '26년 승진자 교육</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #888;'>승진을 축하드립니다! 교육 안내 봇입니다.</p>", unsafe_allow_html=True)
    if st.button("시작하기", use_container_width=True):
        st.session_state.page = 'chat'
        st.session_state.messages.append({"role": "assistant", "content": "안녕하세요! 무엇을 도와드릴까요?", "type": "welcome"})
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

if st.session_state.page == 'start': show_start_screen()
else: show_chat_screen()
