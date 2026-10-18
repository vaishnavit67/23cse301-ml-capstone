import streamlit as st

def render_about_page():
    st.title("ℹ️ About Capstone Project & Team")
    st.markdown("""
    Academic information, team member roles, review schedules, rubric alignment, and bonus opportunities for the **23CSE301 Machine Learning Capstone Project**.
    """)
    
    col1, col2 = st.columns([0.6, 0.4])
    
    with col1:
        st.subheader("🎓 Course & Institutional Context")
        st.markdown("""
        - **Course Code:** 23CSE301
        - **Course Name:** Machine Learning
        - **Program:** B.Tech. Computer Science and Engineering (III Year)
        - **Institution:** Amrita Vishwa Vidyapeetham
        - **Academic Year:** 2026–27
        - **Team Identification:** Team — 17
        """)
        
        st.subheader("👥 Team Members & Responsibilities")
        st.markdown("""
        | S.No. | Team Member | Register Number | Primary Ownership |
        | :---: | :--- | :--- | :--- |
        | 1 | **Himakesh Puthumbaku** | CB.SC.U4CSE24741 | Full Pipeline Integration, GUI Web App & Deployment, Clustering |
        | 2 | **Vaishnavi Tallada** | CB.SC.U4CSE24757 | Classification Track (Part A & Part B), EDA & Outlier Strategy |
        | 3 | **Vuyyala Manesh** | CB.SC.U4CSE24759 | Regression Track (10 Models, Hyperparameter Tuning & Cross-Val) |
        """)
        
        st.subheader("⭐ Bonus Mark Opportunities (Review 2 Rubric)")
        st.markdown("""
        The course syllabus explicitly defines bonus criteria for Review 2 (up to **+2 marks**):
        
        | Bonus Descriptor | Requirement | Implementation Status |
        | :--- | :--- | :---: |
        | **+1 Interactive GUI** | A working web-based interface (Streamlit recommended) that accepts user inputs and returns real predictions from the trained models. | **✅ Complete** |
        | **+1 Public Deployment** | The application is deployed and accessible via a public live URL (e.g. Streamlit Community Cloud or Hugging Face Spaces). | **✅ Prepared & Deployed** |
        | **+2 Both** | Both a working interactive GUI and a live public deployment. | **✅ Fully Satisfied (+2 Marks)** |
        """)
        
    with col2:
        st.subheader("🔗 Project Repository")
        st.markdown("""
        - **GitHub Repository:** [`vaishnavit67/23cse301-ml-capstone`](https://github.com/vaishnavit67/23cse301-ml-capstone)
        - **Branch:** `main`
        - **Jupyter Notebooks:**
          - [`Regression.ipynb`](https://github.com/vaishnavit67/23cse301-ml-capstone/blob/main/Notebooks/Regression.ipynb) (10 Algorithms)
          - [`Classification.ipynb`](https://github.com/vaishnavit67/23cse301-ml-capstone/blob/main/Notebooks/Classification.ipynb) (Part A & Part B)
          - [`Clustering.ipynb`](https://github.com/vaishnavit67/23cse301-ml-capstone/blob/main/Notebooks/Clustering.ipynb) (Unsupervised Track)
        """)
        
        st.subheader("🛠️ Technical Stack")
        st.markdown("""
        - **Frontend & Interface:** Streamlit (v1.66+)
        - **Interactive Plotting:** Plotly Express & Plotly Graph Objects
        - **Core Machine Learning:** Scikit-learn (v1.5+), SciPy
        - **Data Handling:** Pandas, NumPy
        - **Model Persistence:** Joblib serialization
        - **Deployment Target:** Streamlit Community Cloud / Hugging Face Spaces
        """)
        
        st.subheader("📜 Academic Integrity Citation")
        st.markdown("""
        In accordance with Section 8 of the guidelines, Generative AI tools were used for code scaffolding, debugging, and UI structuring. All feature engineering, analytical decisions, model training, metric interpretations, and conclusions represent our team's verified work.
        """)
