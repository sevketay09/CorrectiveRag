import os  
import streamlit as st  
import tempfile  
from app.c_rag import chat_with_crag_ui, crag_main  
import shutil  
  
def main(embedding_model_name):  
    st.title("Corrective RAG Chat Interface")   
  
    uploaded_files = st.file_uploader("PDF dosya(lar)ı yükleyin", type="pdf", accept_multiple_files=True)  
  
    sources = []  
    temp_dir = tempfile.mkdtemp()  
      
    if uploaded_files:  
        for uploaded_file in uploaded_files:  
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)  
            with open(temp_file_path, "wb") as f:  
                f.write(uploaded_file.getbuffer())  
            sources.append(temp_file_path)  
  
    if not sources:  
        st.warning("Devam etmek için lütfen PDF dosyası yükleyin.")  
        return  
  
    st.write("Süreç başlıyor...")  
  
    custom_graph_crag = crag_main(sources, embedding_model_name=embedding_model_name, max_results_k=5)  
  
    if "chat_history" not in st.session_state:  
        st.session_state["chat_history"] = []  
  
    user_input = st.text_input("Siz:", key="user_input")  
    if st.button("Gönder"):  
        if user_input:  
            st.session_state.chat_history.append(("User", user_input))  
            with st.spinner("Agent yazıyor..."):  
                response = chat_with_crag_ui(custom_graph_crag, user_input)  
            st.session_state.chat_history.append(("Agent", response))  
            st.write(f"**Siz:** {user_input}")  
            st.write(f"**Agent:** {response}")  
  
    if st.session_state.chat_history:  
        st.write("## Sohbet Geçmişi")  
        for speaker, text in st.session_state.chat_history:  
            st.write(f"**{speaker}:** {text}")  
  
    shutil.rmtree(temp_dir)  
  
if __name__ == "__main__":  
    embedding_model_name = "all-MiniLM-L6-v2"  
    main(embedding_model_name=embedding_model_name)  