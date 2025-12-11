import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os
import time

# --- 설정 ---
DATA_FILE = "data.txt"  # 지식 베이스 텍스트 파일 이름
IMAGE_FILE = "growth_image.jpg" # 성장 독려 이미지 파일 이름
MODEL_NAME = 'jhgan/ko-sroberta-multitask' # 한국어 문장 임베딩 모델
SIMILARITY_THRESHOLD = 0.4 # 답변 유사도 임계값

# --- 페이지 설정 및 CSS 스타일링 ---
st.set_page_config(page_title="'26년 승진자 교육 안내", layout="centered")

st.markdown("""
<style>
    /* 전체 배경 및 폰트 */
    .stApp {
        background-color: #F3F4F8;
        font-family: 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif;
    }
    
    /* 챗봇 컨테이너 */
    .chat-container {
        background-color: #AEC6CF;
        border-radius: 20px;
        padding: 20px;
    }

    /* 메시지 스타일 */
    .stChatMessage {
        background-color: transparent !important;
    }
    
    /* 봇 메시지 (흰색) */
    div[data-testid="stChatMessage"] div[data-testid="chatAvatarIcon-assistant"] + div {
        background-color: #FFFFFF !important; 
        border-radius: 15px;
        padding: 10px 15px;
        color: #333333;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }

    /* 유저 메시지 (노란색) */
    div[data-testid="stChatMessage"] div[data-testid="chatAvatarIcon-user"] + div {
        background-color: #FEE500 !important;
        border-radius: 15px;
        padding: 10px 15px;
        color: #333333;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    
    /* 카드 스타일 */
    .info-card {
        background-color: #FFFFFF;
        border-radius: 15px;
        padding: 15px;
        margin: 5px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 2px solid #E0E0E0;
    }
    .info-card h4 { margin-bottom: 8px; color: #555; }
    .info-card p { color: #888; font-size: 14px; }
</style>
""", unsafe_allow_html=True)

# --- AI 모델 및 데이터 로딩 함수 ---
@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

@st.cache_data
def load_and_process_data(file_path):
    """로컬 파일을 읽어서 처리"""
    if not os.path.exists(file_path):
        # 파일이 없으면 빈 리스트 반환 (에러 방지)
        return [], []

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

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
    if not contents:
        return None
    embeddings = _model.encode(contents)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def get_answer(query, index, contents, model):
    if index is None or not contents:
        return "데이터 파일(data.txt)이 업로드되지 않았거나 비어있습니다."

    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, 1)

    if D[0][0] > (1 - SIMILARITY_THRESHOLD) * 100:
         return "해당 문의 사항은 운영진에게 문의해주세요. (운영진 : 홍길동)"

    return contents[I[0][0]]

# --- 세션 상태 초기화 ---
if 'page' not in st.session_state:
    st.session_state.page = 'start'
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# --- 데이터 로드 실행 ---
if not st.session_state.data_loaded:
    with st.spinner("교육 안내 데이터를 준비 중입니다..."):
        embedder = load_model()
        titles, kb_contents = load_and_process_data(DATA_FILE)
        
        if kb_contents:
            vector_index = create_vector_index(kb_contents, embedder)
            st.session_state.embedder = embedder
            st.session_state.titles = titles
            st.session_state.kb_contents = kb_contents
            st.session_state.vector_index = vector_index
            st.session_state.data_loaded = True
        else:
            # 데이터 파일이 없을 경우 처리를 위해 임시 통과
            st.session_state.data_loaded = True
            st.session_state.kb_contents = []
            st.session_state.vector_index = None

# --- 화면 구성 ---
def show_start_screen():
    st.markdown("<div style='text-align: center; padding: 50px;'>", unsafe_allow_html=True)
    st.markdown("<h1 style='color: #555;'>🎉 '26년 승진자 교육</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #888; margin-bottom: 50px;'>승진을 진심으로 축하드립니다!<br>교육 안내를 도와드릴 챗봇입니다.</p>", unsafe_allow_html=True)
    
    if st.button("시작하기", use_container_width=True):
        st.session_state.page = 'chat'
        welcome_msg = "안녕하세요! '26년 승진자 교육 안내 봇입니다. 🤖\n아래 메뉴를 선택하거나 궁금한 점을 입력해주세요."
        st.session_state.messages.append({"role": "assistant", "content": welcome_msg, "type": "welcome"})
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

def show_chat_screen():
    with st.sidebar:
        st.markdown("### ✨ 지속적인 성장")
        if os.path.exists(IMAGE_FILE):
            st.image(IMAGE_FILE, caption="Keep Growing!")
        
        st.info("승진은 끝이 아닌 새로운 시작입니다.\n당신의 열정을 응원합니다!")
        
        if st.button("처음으로"):
            st.session_state.page = 'start'
            st.session_state.messages = []
            st.rerun()

    # 채팅 기록
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message.get("type") == "welcome":
                cols = st.columns(2)
                cards = [
                    ("🏢 연수원 안내", "시설/위치 안내", "연수원 안내 알려줘"),
                    ("📅 교육 시간표", "상세 일정 확인", "교육 시간표 알려줘"),
                    ("🚌 이동방법", "셔틀/주차 안내", "이동방법 알려줘"),
                    ("📜 Ground Rule", "생활 수칙", "Ground rule 알려줘"),
                ]
                for i, (title, desc, query) in enumerate(cards):
                    with cols[i % 2]:
                        st.markdown(f"<div class='info-card'><h4>{title}</h4><p>{desc}</p></div>", unsafe_allow_html=True)
                        if st.button(f"상세보기", key=f"btn_{i}"):
                            handle_user_input(query)
                            st.rerun()

    # 입력창
    if prompt := st.chat_input("질문을 입력하세요"):
        handle_user_input(prompt)
        st.rerun()

def handle_user_input(user_query):
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    with st.spinner("..."):
        answer = get_answer(
            user_query, 
            st.session_state.vector_index, 
            st.session_state.kb_contents,
            st.session_state.embedder if hasattr(st.session_state, 'embedder') else None
        )
    
    st.session_state.messages.append({"role": "assistant", "content": answer.replace("\n", "  \n")})

if st.session_state.page == 'start':
    show_start_screen()
else:
    show_chat_screen()
