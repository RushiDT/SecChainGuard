import requests
import streamlit as st

API_URL = "http://127.0.0.1:8001/analyze"  # backend

st.set_page_config("SecChainGuard – IoT Security Assistant")

st.title("🔐 SecChainGuard – IoT Security & Blockchain Assistant")

st.markdown(
    "Give me an IoT system, and I'll do **STRIDE + Blockchain AC + ML anomaly analysis**."
)

query = st.text_area(
    "Question / Task",
    value="Perform STRIDE threat modelling and design a blockchain-based access control scheme. Explain how ML-based anomaly detection fits in.",
    height=100,
)

system_description = st.text_area(
    "System Description (devices, actors, network)",
    value="Devices: smart lock on main door, IP camera in living room.\nNetwork: home WiFi + cloud backend.\nActors: homeowner, guest, cleaner.",
    height=140,
)

if st.button("Analyze"):
    if not query.strip() or not system_description.strip():
        st.error("Please fill both fields.")
    else:
        with st.spinner("Talking to your fine-tuned TinyLlama..."):
            try:
                resp = requests.post(
                    API_URL,
                    json={
                        "query": query,
                        "system_description": system_description,
                    },
                    timeout=300,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.markdown("### ✅ Analysis")
                    st.markdown(data.get("answer", ""))

                    st.markdown("---")
                    with st.expander("Retrieved Context (RAG)", expanded=False):
                        for ctx in data.get("retrieved_contexts", []):
                            st.code(ctx.strip())
                else:
                    st.error(f"API error {resp.status_code}: {resp.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")
