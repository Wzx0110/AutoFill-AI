import streamlit as st
import pandas as pd
import uuid
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

# === Session ID 管理  ===
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4()) # 初始化一個 UUID
    st.session_state.messages = []
    st.session_state.upload_status = None

# 初始化 Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "upload_status" not in st.session_state:
    st.session_state.upload_status = None
    
if "extraction_results" not in st.session_state:
    st.session_state.extraction_results = None

# --- Sidebar: 文件上傳區 ---
with st.sidebar:
    st.title("AutoFill AI")
    
    # 顯示當前 Session ID (除錯用，也可以隱藏)
    st.caption(f"Session: {st.session_state.session_id[-8:]}...")
    
    if st.button("New Chat / Clear Memory", type="secondary"):
        # 重置所有狀態
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.upload_status = None
        st.session_state.extraction_results = None
        st.rerun() # 重新整理頁面
    
    # === 功能切換 ===
    mode = st.selectbox("Select Mode", ["Chat Assistant", "Auto-Fill Extraction"])
    st.divider()

    # === 檔案上傳區 ===
    st.markdown("### Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload PDF(s)", 
        type=["pdf"], 
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        if st.button("Process Document(s)", type="primary"):
            with st.status("Processing...", expanded=True) as status:
                st.write("Uploading to server...")
                result = api_client.upload_reference(uploaded_files, st.session_state.session_id)
                
                if "error" in result:
                    status.update(label="Failed", state="error")
                    st.error(result['error'])
                else:
                    count = result.get('uploaded_count', 0)
                    st.write(f"Indexed {count} files successfully!")
                    status.update(label="System Ready", state="complete", expanded=False)
                    st.session_state.upload_status = "ready"
    
    st.markdown("---")
    if st.session_state.get("upload_status") == "ready":
        st.success("System Online")
    else:
        st.warning("Waiting for Data")


# --- 主畫面 ---
st.markdown("## AutoFill AI Workspace")
# === 聊天室 ===
if mode == "Chat Assistant":
    st.markdown("## Chat Workspace")
    
    chat_container = st.container()
    
    # 顯示歷史訊息
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar=None):
                st.markdown(message["content"])

    # 輸入框
    if prompt := st.chat_input("Ask a question about the document..."):
        # 顯示使用者輸入
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user", avatar=None):
                st.markdown(prompt)
                
        # 直接呼叫 API，後端會自己判斷是有文件還是沒文件
        with chat_container:
            with st.chat_message("assistant", avatar=None):
                message_placeholder = st.empty()
                message_placeholder.markdown("Thinking...")
                    
                response = api_client.query_knowledge(prompt, st.session_state.session_id)
                if "error" in response:
                        full_res = f"Error: {response['error']}"
                else:
                    ans = response.get("answer", "")
                    if isinstance(ans, list):
                        # 列表接成字串
                        ans = "".join([str(x) if isinstance(x, str) else str(x.get('text','')) for x in ans])
                    elif not isinstance(ans, str):
                        ans = str(ans)
                    src = list(set(response.get("source_documents", [])))
                    # 根據有沒有來源，顯示不同的小字
                    if src:
                        src_text = f"\n\n<small style='color:grey'>Ref: {', '.join(src)}</small>"
                    else:
                        src_text = f"\n\n<small style='color:grey'>(General Knowledge)</small>"
                        
                    full_res = ans + src_text 

                message_placeholder.markdown(full_res, unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": full_res})

# === 自動填表 ===
elif mode == "Auto-Fill Extraction":
    st.markdown("## Auto-Fill Engine")
    
    col1, col2 = st.columns([1, 1])

    # 左側：定義 Schema
    with col1:
        st.subheader("1. Target Form & Schema")
        
        # === 目標表單上傳區 ===
        st.info("Step A: 上傳你要填寫的「空白表格」(PDF/Word)")
        target_form = st.file_uploader("Upload Target Form", type=["pdf", "docx"], key="target_form")
        if target_form:
            st.session_state["uploaded_target_form_obj"] = target_form # 把物件存起來
        
        # 初始化 Session 中的欄位資料
        if "schema_df" not in st.session_state:
            # 預設範例
            st.session_state.schema_df = pd.DataFrame([
                {"key": "example_field", "description": "Example description...", "data_type": "string"}
            ])

        if target_form:
            if st.button("AI Analyze Form Structure", type="secondary"):
                with st.spinner("AI is reading the form structure..."):
                    res = api_client.analyze_form(target_form)
                    if "error" in res:
                        st.error(res["error"])
                    else:
                        fields = res.get("fields", [])
                        if fields:
                            st.session_state.schema_df = pd.DataFrame(fields)
                            st.success(f"Detected {len(fields)} fields!")
                        else:
                            st.warning("Could not detect any fields.")

        st.markdown("---")
        st.markdown("Step B: 確認或修改 AI 抓到的欄位")
        
        # 讓使用者編輯 AI 分析出來的結果
        edited_df = st.data_editor(
            st.session_state.schema_df, 
            num_rows="dynamic",
            width='stretch',
            column_config={
                "key": st.column_config.TextColumn("Field Key", required=True),
                "description": st.column_config.TextColumn("Question for AI", required=True),
            },
            hide_index=True,
            key="schema_editor" # 加上 key 避免重繪問題
        )

        st.markdown("---")
        st.markdown("Step C: 執行填寫")
        if st.button("Run Extraction (RAG + Web)", type="primary"):
             # 直接執行
            with st.spinner("AI is finding answers (Checking Docs -> Web)..."):
                fields_payload = edited_df.to_dict(orient="records")
                # 記得補上 data_type
                for f in fields_payload:
                     if "data_type" not in f: f["data_type"] = "string"

                # 呼叫後端 (傳入 session_id)
                result = api_client.extract_data(fields_payload, st.session_state.session_id)
                
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.session_state.extraction_results = result.get("results", [])
                    st.success("Extraction Complete!")
    
    # 右側：顯示結果
    with col2:
        st.subheader("2. Extraction Results")
        
        results = st.session_state.extraction_results
        
        if results:
            res_df = pd.DataFrame(results)
            display_df = res_df[["key", "value", "confidence", "source"]]
            
            st.dataframe(
                display_df, 
                width='stretch',
                column_config={
                    "key": "Field",
                    "value": "Extracted Value",
                    "confidence": "Confidence",
                    "source": "Source Doc"
                }
            )
            
            # 提供 JSON 下載
            st.download_button(
                label="Download JSON",
                data=res_df.to_json(orient="records", indent=2, force_ascii=False),
                file_name="extracted_data.json",
                mime="application/json"
            )
        else:
            st.info("No results yet. Define schema and click Run.")
    
    st.markdown("---")
    st.subheader("3. Download Filled Document")
    
    # 取得當前上傳的 Target Form
    target_form = st.session_state.get("uploaded_target_form_obj") # 這裡你要在上面 upload 時存起來
    
    # *注意*: Streamlit 的 file_uploader 物件如果頁面重整會消失
    # 為了簡化，我們這裡假設使用者還沒重新整理
    
    if st.button("Generate Filled File"):
        # 這裡需要一個小技巧：因為 target_form 是一個 Streamlit UploadedFile 物件
        # 我們需要確保它是可讀的
        if target_form:
            target_form.seek(0) # 重置游標
            
            with st.spinner("Generating document..."):
                file_content = api_client.generate_filled_file(
                    target_form, 
                    st.session_state.extraction_results
                )
                
                if isinstance(file_content, dict) and "error" in file_content:
                    st.error(file_content["error"])
                else:
                    # 提供下載按鈕
                    st.download_button(
                        label="Click to Download",
                        data=file_content,
                        file_name=f"filled_{target_form.name}",
                        mime="application/octet-stream"
                    )
        else:
            st.warning("Target form session expired. Please upload the form again.")
