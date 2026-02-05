import streamlit as st
from api_client import api_client

# 頁面全域設定 
st.set_page_config(
    page_title="AutoFill AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 
st.markdown("""
<style>
    /* 調整標題間距 */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    /* 隱藏預設 Footer */
    footer {visibility: hidden;}
    /* 調整按鈕樣式 */
    .stButton button {
        width: 100%;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# 初始化 Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "upload_status" not in st.session_state:
    st.session_state.upload_status = None

# --- Sidebar: 文件上傳區 ---
with st.sidebar:
    st.markdown("### Knowledge Base")
    st.caption("Upload reference documents to provide context.")
    
    uploaded_file = st.file_uploader("Select PDF Document", type=["pdf"], label_visibility="collapsed")
    
    if uploaded_file:
        if st.button("Process Document", type="primary"):
            # 顯示進度
            with st.status("Processing document...", expanded=True) as status:
                st.write("Uploading to server...")
                result = api_client.upload_reference(uploaded_file)
                
                if "error" in result:
                    status.update(label="Process Failed", state="error", expanded=True)
                    st.error(result['error'])
                else:
                    chunks = result.get('result', {}).get('chunks', 0)
                    st.write("Indexing content...")
                    st.write(f"Vectorizing {chunks} data chunks...")
                    status.update(label="System Ready", state="complete", expanded=False)
                    st.session_state.upload_status = "ready"
    
    st.markdown("---")
    st.markdown("### System Status")
    if st.session_state.get("upload_status") == "ready":
        st.success("Index Active")
    else:
        st.warning("Waiting for Data")

# --- Main Area: 聊天問答區 ---
st.markdown("## AutoFill AI Workspace")

# 使用容器來區隔聊天區
chat_container = st.container()

# 顯示歷史訊息
with chat_container:
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        
        with st.chat_message(role, avatar=None):
            st.markdown(content)

# 輸入區 (Input Area)
if prompt := st.chat_input("Ask a question about the document..."):
    # 顯示使用者輸入
    st.session_state.messages.append({"role": "user", "content": prompt})
    with chat_container:
        with st.chat_message("user", avatar=None):
            st.markdown(prompt)

    # 呼叫 AI (如果沒有上傳文件，給予提示)
    if st.session_state.get("upload_status") != "ready":
        response_text = "Please upload and process a document in the sidebar first."
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        with chat_container:
            with st.chat_message("assistant", avatar=None):
                st.markdown(response_text)
    else:
        with chat_container:
            with st.chat_message("assistant", avatar=None):
                message_placeholder = st.empty()
                message_placeholder.markdown("...") # 等待符號
                
                response = api_client.query_knowledge(prompt)
                
                if "error" in response:
                    full_response = f"System Error: {response['error']}"
                else:
                    answer = response.get("answer", "")
                    sources = list(set(response.get("source_documents", [])))
                    
                    # 來源顯示
                    source_text = f"\n\n<small style='color:grey'>Source: {', '.join(sources)}</small>" if sources else ""
                    full_response = answer + source_text

                message_placeholder.markdown(full_response, unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": full_response})