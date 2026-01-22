import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
st.set_page_config(page_title="Sentin-AI Forensics", layout="wide")

st.title("🛡️ Sentin-AI: Manual Behavioral Forensic Engine")
st.markdown("---")
st.sidebar.header("Forensic Parameters")
similarity_threshold = st.sidebar.slider("Similarity Threshold", 0.70, 1.00, 0.85)
st.sidebar.info("Tip: Lower the threshold to catch more 'loose' similarities.")
st.subheader("📝 Manual Exam Data Entry")
st.markdown("Double-click any cell to edit. You can add more rows at the bottom.")
default_data = pd.DataFrame({
    'student_id': ['S01', 'S02', 'S03'],
    'time_taken_min': [45, 42, 12],
    'answers': ["ABCD", "ABCD", "ABCC"],
    'wrong_answers_count': [1, 1, 0] 
})

user_df = st.data_editor(default_data, num_rows="dynamic", use_container_width=True)


def perform_forensics(df):
    if len(df) < 2:
        st.warning("Please enter at least 2 students to perform analysis.")
        return None
        
   
    if df['time_taken_min'].std() == 0:
        df['z_score'] = 0
    else:
        df['z_score'] = np.abs(stats.zscore(df['time_taken_min']))
    
    
    df['risk_index'] = (1 / (df['time_taken_min'] + 1)) * (10 - df['wrong_answers_count'])
    return df


if st.button('🚀 RUN FULL FORENSIC ANALYSIS'):
    df = perform_forensics(user_df)
    
    if df is not None:
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("### 🚩 High-Risk Outliers")
            flags = df[(df['z_score'] > 1.0) | (df['risk_index'] > 0.5)]
            if not flags.empty:
                st.error(f"Detected {len(flags)} students with anomalous behavior.")
                st.table(flags[['student_id', 'time_taken_min', 'risk_index']])
            else:
                st.success("No significant outliers detected.")

        with col2:
            st.write("### 📈 Time Distribution")
            fig2, ax2 = plt.subplots()
            sns.histplot(df['time_taken_min'], kde=True, color="purple", ax=ax2)
            st.pyplot(fig2)

        st.markdown("---")

        
        col3, col4 = st.columns([1, 1])

        with col3:
            st.write("### 🤝 Collusion Network")
            G = nx.Graph()
            num_students = len(df)
            
            for i in range(num_students):
                for j in range(i + 1, num_students):
                   
                    s1_ans, s2_ans = str(df.iloc[i]['answers']), str(df.iloc[j]['answers'])
                    set1, set2 = set(s1_ans), set(s2_ans)
                    
                   
                    intersection = len(set1.intersection(set2))
                    union = len(set1.union(set2))
                    similarity = intersection / union if union != 0 else 0
                    
                    if similarity >= similarity_threshold:
                        G.add_edge(df.iloc[i]['student_id'], df.iloc[j]['student_id'], weight=round(similarity, 2))
            
            fig, ax = plt.subplots(facecolor='#f0f2f6')
            if len(G.nodes) > 0:
                pos = nx.spring_layout(G)
                nx.draw(G, pos, with_labels=True, node_color='red', node_size=1000, font_weight='bold')
                labels = nx.get_edge_attributes(G, 'weight')
                nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
            else:
                plt.text(0.5, 0.5, "No Collusion Detected", ha='center')
            st.pyplot(fig)

        with col4:
            st.write("### 🌡️ Similarity Heatmap")
            num_students = len(df)
            sim_matrix = np.zeros((num_students, num_students))
            
            for i in range(num_students):
                for j in range(num_students):
                    s1_ans, s2_ans = str(df.iloc[i]['answers']), str(df.iloc[j]['answers'])
                    set1, set2 = set(s1_ans), set(s2_ans)
                    u = len(set1.union(set2))
                    sim_matrix[i,j] = len(set1.intersection(set2)) / u if u != 0 else 0
            
            fig3, ax3 = plt.subplots()
            sns.heatmap(sim_matrix, annot=True, cmap="YlOrRd", 
                        xticklabels=df['student_id'], yticklabels=df['student_id'])
            st.pyplot(fig3)