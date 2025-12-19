@st.cache_resource
def load_model():

    search_root = os.getcwd()      # Always repo root
    found_path = None

    for root, dirs, files in os.walk(search_root):
        if "model.pkl" in files:
            found_path = os.path.join(root, "model.pkl")
            break

    if found_path is None:
        st.error("❌ model.pkl NOT FOUND anywhere in this GitHub repo.")
        st.write("🔍 Searched directory:", search_root)
        st.stop()

    st.success(f"✅ model.pkl found at: {found_path}")

    with open(found_path, "rb") as f:
        return pickle.load(f)
