import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import smtplib
from email.mime.text import MIMEText
from email.header import Header

# --- 설정 ---
IMAGE_FILE = "growth_image.jpg"
MODEL_NAME = 'jhgan/ko-sroberta-multitask'

# 모바일 친화적인 레이아웃 설정
st.set_page_config(page_title="'26년 승진자 교육 안내", layout="centered")

# --- 스타일 설정 ---
st.markdown("""
<style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    .stApp { background-color: #F3F4F8; font-family: 'Pretendard', sans-serif; }
    .chat-container { background-color: #AEC6CF; border-radius: 20px; padding: 20px; }
    div[data-testid="stChatMessage"] { background-color: transparent !important; }
    div[data-testid="stChatMessage"] .stMarkdown { word-break: keep-all; line-height: 1.6; }
    div[data-testid="stChatMessage"] div[data-testid="chatAvatarIcon-assistant"] + div {
        background-color: #FFFFFF !important; border-radius: 15px; padding: 12px 18px; color: #333333; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    div[data-testid="stChatMessage"] div[data-testid="chatAvatarIcon-user"] + div {
        background-color: #FEE500 !important; border-radius: 15px; padding: 12px 18px; color: #333333; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .info-card {
        background-color: #FFFFFF; border-radius: 15px; padding: 20px; margin: 10px; text-align: center; border: 2px solid #E0E0E0; box-shadow: 0 2px 5px rgba(0,0,0,0.05); transition: all 0.3s; word-break: keep-all;
    }
    .info-card:hover { border-color: #AEC6CF; transform: translateY(-3px); }
    .info-card h4 { margin-bottom: 12px; color: #444; font-size: 1.1rem; font-weight: bold; }
    .info-card p { color: #777; font-size: 0.95rem; margin: 0; }
    
    @media (max-width: 768px) {
        .main .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
        .info-card { margin: 10px 0 !important; width: 100%; }
        .info-card .stButton button { width: 100%; margin-top: 10px; }
    }
</style>
""", unsafe_allow_html=True)

# --- 이메일 발송 함수 (NEW!) ---
def send_email_alert(user_query):
    """사용자 질문을 이메일로 발송하는 함수"""
    # Secrets에 이메일 정보가 없으면 조용히 패스
    if "EMAIL_ID" not in st.secrets or "EMAIL_PW" not in st.secrets:
        return

    try:
        smtp_server = "smtp.naver.com"
        smtp_port = 587
        
        email_id = st.secrets["EMAIL_ID"]
        email_pw = st.secrets["EMAIL_PW"]
        email_to = st.secrets.get("EMAIL_TO", email_id) # 받는 사람 없으면 나에게 보내기

        # 메일 내용 작성
        subject = f"[챗봇 알림] 새로운 문의가 도착했습니다!"
        content = f"""
        🔔 챗봇에 새로운 질문이 등록되었습니다.
        
        ----------------------------------------
        📝 질문 내용:
        {user_query}
        ----------------------------------------
        
        (이 메일은 자동 발송되었습니다.)
        """
        
        msg = MIMEText(content, 'plain', 'utf-8')
        msg['Subject'] = Header(subject, 'utf-8')
        msg['From'] = email_id
        msg['To'] = email_to

        # 네이버 서버 접속 및 전송
        s = smtplib.SMTP(smtp_server, smtp_port)
        s.starttls() # 보안 연결
        s.login(email_id, email_pw)
        s.sendmail(email_id, email_to, msg.as_string())
        s.quit()
        
        # print("이메일 전송 성공") # 디버깅용

    except Exception as e:
        print(f"이메일 전송 실패: {e}") 
        # 사용자에겐 에러를 보여주지 않음 (앱은 계속 돌아가야 하니까)

# --- AI 및 데이터 로드 ---
@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

@st.cache_data
def load_data_from_secrets():
    if "knowledge_base" not in st.secrets: return [], []
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
    
    return contents[I[0][0]]

# --- 초기화 ---
if 'page' not in st.session_state: st.session_state.page = 'start'
if 'messages' not in st.session_state: st.session_state.messages = []

# 데이터 로드
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
    
    # [핵심] 질문 들어오면 이메일 발송! 🚀
    # (너무 자주 보내면 스팸 처리될 수 있으니 주의)
    # 버튼 클릭(상세보기)은 제외하고, 직접 입력한 경우만 보낼 수도 있지만
    # 일단은 모든 질문에 대해 알림을 보내도록 설정함
    send_email_alert(user_query)
    
    answer = get_answer(user_query, vector_index, kb_contents, embedder)
    st.session_state.messages.append({"role": "assistant", "content": answer.replace("\n", "  \n")})

def show_chat_screen():
    with st.sidebar:
        st.markdown("### ✨ 지속적인 성장")
        if os.path.exists(IMAGE_FILE): st.image(IMAGE_FILE, caption="Keep Growing!", use_column_width=True)
        
        st.markdown("---")
        st.markdown("### 🛠 시스템 상태")
        if st.session_state.data_ready:
            st.success(f"데이터 연결 성공!")
        else:
            st.error("데이터 연결 실패")
        
        if st.button("처음으로", use_container_width=True):
            st.session_state.page = 'start'; st.session_state.messages = []; st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("type") == "welcome":
                cols = st.columns(2)
                cards = [
                    ("🏢 연수원 안내", "시설/위치 안내", "연수원 안내"),
                    ("📅 교육 시간표", "상세 일정 확인", "교육 시간표"),
                    ("🚌 이동방법", "셔틀/주차 안내", "이동방법"),
                    ("📜 Ground Rule", "생활 수칙", "Ground Rule")
                ]
                for i, (title, desc, query) in enumerate(cards):
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
