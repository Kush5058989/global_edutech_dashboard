import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

st.title("📊 Global EduTech Dashboard")

# 📚 Detailed Subject Mapping
subject_mapping = {
    "K-12": ["Math", "Science", "English", "Social Studies", "EVS", "Computer Science"],
    "Test Prep": ["JEE", "NEET", "CAT", "SAT", "GMAT", "UPSC", "GRE", "IELTS"],
    "Higher Ed": ["Engineering", "Medicine", "Commerce", "Arts", "Law", "Business"],
    "Skill Development": ["Coding", "Digital Marketing", "Design", "Finance", "AI/ML"],
    "Language Learning": ["English", "Hindi", "Spanish", "French", "German", "Chinese"],
    "Professional Certification": ["Data Science", "Project Management", "Cybersecurity", "Cloud Computing"],
    "Online Learning": ["Various", "Tech", "Business", "STEM"],
    "Vocational": ["IT", "Healthcare", "Automotive", "Skilled Trades"],
    "Academic Publishing": ["Textbooks", "Research", "Journals"],
    "EdTech": ["LMS", "SIS", "Analytics", "Interactive Lessons"],
    "Corporate Training": ["Tech", "Business", "Compliance"],
    "Early Childhood": ["Early Learning", "Childcare"],
    "Gamified Learning": ["Multiple"],
}

def clean_currency(col):
    if pd.isnull(col):
        return None
    col = str(col).upper().replace(",", "").replace("+", "")
    multiplier = 1
    if "M" in col:
        multiplier = 1e6
        col = col.replace("M", "")
    elif "B" in col:
        multiplier = 1e9
        col = col.replace("B", "")
    elif "K" in col:
        multiplier = 1e3
        col = col.replace("K", "")
    col = col.replace("$", "").replace("£", "").replace("₹", "")  # Handle $, £, ₹
    try:
        return float(col) * multiplier
    except:
        return None

@st.cache_data
def load_data():
    # Load Indian dataset with raw string path
    df_india = pd.read_csv("edu_tech_data.csv")
    df_india['Country'] = 'India'  # Add Country column
    # Load international dataset with raw string path
    df_international = pd.read_csv("Updated_International_EdTech_Companies.csv")
    
    # Standardize column names (if needed)
    df_india = df_india.rename(columns={
        'Investors (Major)': 'Investors (Major)',  # Ensure consistency
        'Amount Invested (Approx)': 'Amount Invested (Approx)'
    })
    df_international = df_international.rename(columns={
        'Investors (Major)': 'Investors (Major)',
        'Amount Invested (Approx)': 'Amount Invested (Approx)'
    })
    
    # Clean currency columns
    for col in ['Revenue 2021', 'Revenue 2022', 'Revenue 2023', 'Amount Invested (Approx)']:
        if col in df_india.columns:
            df_india[col] = df_india[col].apply(clean_currency)
        if col in df_international.columns:
            df_international[col] = df_international[col].apply(clean_currency)
    
    # Merge datasets
    common_columns = list(set(df_india.columns).intersection(set(df_international.columns)))
    df = pd.concat([df_india[common_columns], df_international[common_columns]], ignore_index=True)
    
    # Deduplicate based on Company Name (case-insensitive) and Country
    df['Company Name'] = df['Company Name'].str.strip().str.title()  # Standardize case and remove whitespace
    df = df.drop_duplicates(subset=['Company Name', 'Country'], keep='first')
    
    return df

df = load_data()

# Initialize session state for filters
if 'filter_state' not in st.session_state:
    st.session_state.filter_state = {
        'companies': [],
        'focus_areas': [],
        'investors': [],
        'countries': []
    }

# Sidebar filters
st.sidebar.header("🎯 Custom Filters")
all_companies = df['Company Name'].dropna().unique()
all_focus_areas = df['Focus Area'].dropna().unique()
all_investors = df['Investors (Major)'].dropna().unique() if 'Investors (Major)' in df.columns else []
all_countries = df['Country'].dropna().unique()

selected_countries = st.sidebar.multiselect(
    "Filter by Country",
    options=all_countries,
    default=st.session_state.filter_state['countries'],
    key="countries"
)
selected_companies = st.sidebar.multiselect(
    "Filter by Company Name",
    options=all_companies,
    default=st.session_state.filter_state['companies'],
    key="companies"
)
selected_focus_areas = st.sidebar.multiselect(
    "Filter by Focus Area",
    options=all_focus_areas,
    default=st.session_state.filter_state['focus_areas'],
    key="focus_areas"
)
selected_investors = st.sidebar.multiselect(
    "Filter by Investor",
    options=all_investors,
    default=st.session_state.filter_state['investors'],
    key="investors"
)
selected_year = st.sidebar.selectbox("Select Revenue Year", ['Revenue 2021', 'Revenue 2022', 'Revenue 2023'])

# Update filter_state with current selections
st.session_state.filter_state['companies'] = selected_companies
st.session_state.filter_state['focus_areas'] = selected_focus_areas
st.session_state.filter_state['investors'] = selected_investors
st.session_state.filter_state['countries'] = selected_countries

# Reset Filters button with callback
def reset_filters():
    st.session_state.filter_state = {
        'companies': [],
        'focus_areas': [],
        'investors': [],
        'countries': []
    }
    # Force clear the multiselect widgets
    st.session_state['companies'] = []
    st.session_state['focus_areas'] = []
    st.session_state['investors'] = []
    st.session_state['countries'] = []

if st.sidebar.button("🔄 Reset Filters", on_click=reset_filters):
    pass  # Callback handles the reset

# Apply filters
filtered_df = df.copy()
if selected_countries:
    filtered_df = filtered_df[filtered_df['Country'].isin(selected_countries)]
if selected_companies:
    filtered_df = filtered_df[filtered_df['Company Name'].isin(selected_companies)]
if selected_focus_areas:
    filtered_df = filtered_df[filtered_df['Focus Area'].isin(selected_focus_areas)]
if selected_investors:
    filtered_df = filtered_df[filtered_df['Investors (Major)'].isin(selected_investors)]

# Calculate growth with edge case handling
def calculate_growth(df):
    if 'Revenue 2022' in df.columns and 'Revenue 2023' in df.columns:
        df['Growth (%)'] = df.apply(
            lambda row: ((row['Revenue 2023'] - row['Revenue 2022']) / row['Revenue 2022'] * 100)
            if pd.notnull(row['Revenue 2022']) and pd.notnull(row['Revenue 2023']) and row['Revenue 2022'] != 0
            else None, axis=1
        )
        df['🔥 Booming?'] = df['Growth (%)'].apply(lambda x: '🔥 Yes' if pd.notnull(x) and x > 30 else 'No')
    return df

filtered_df = calculate_growth(filtered_df)

# Create tabs
tab1, tab2, tab3 = st.tabs(["Data Explorer", "Visual Insights", "Summary & Export"])

# Tab 1: Data Explorer
with tab1:
    st.subheader("📘 Subject Explorer")
    selected_focus = st.selectbox("Choose a Focus Area", list(subject_mapping.keys()))
    if selected_focus:
        st.markdown(f"**Subjects under `{selected_focus}`:**")
        st.write(", ".join(subject_mapping[selected_focus]))
        matching_companies = filtered_df[filtered_df['Focus Area'].str.contains(selected_focus, case=False, na=False)][['Company Name', 'Subjects Offered', 'Country']].dropna()
        st.markdown(f"**Companies offering `{selected_focus}` subjects:**")
        st.dataframe(matching_companies)

    st.subheader("🏫 Company–Subject Explorer")
    if 'Subjects Offered' in filtered_df.columns:
        st.markdown("🔍 **Explore which subjects are offered by which companies. Use the search box to filter subjects like 'Math', 'Coding', etc.**")
        subject_keyword = st.text_input("Search for a Subject (e.g., Math, Coding, Science)", value="")
        subj_df = filtered_df[['Company Name', 'Subjects Offered', 'Country']].dropna()
        subj_df = subj_df[subj_df['Subjects Offered'].str.contains(subject_keyword, case=False, na=False)] if subject_keyword else subj_df
        st.dataframe(subj_df.rename(columns={'Company Name': '📌 Company', 'Subjects Offered': '📘 Subjects', 'Country': '🌍 Country'}))
    else:
        st.warning("⚠️ 'Subjects Offered' column not found in data.")

    st.subheader("🏛️ Company Comparison")
    company_comparison = st.multiselect("Compare Companies", df['Company Name'].unique(), key="company_comparison")
    comparison_df = filtered_df[filtered_df['Company Name'].isin(company_comparison)]
    st.write(comparison_df)

# Tab 2: Visual Insights
with tab2:
    st.subheader("💰 Total Investment by Focus Area")
    fig1 = px.bar(filtered_df.groupby("Focus Area")["Amount Invested (Approx)"].sum().reset_index(),
                  x="Focus Area", y="Amount Invested (Approx)", color="Focus Area",
                  title="Investment by Focus Area")
    st.plotly_chart(fig1)

    st.subheader("📈 Revenue Growth Over Years")
    rev_df = filtered_df[['Company Name', 'Revenue 2021', 'Revenue 2022', 'Revenue 2023']].melt(id_vars='Company Name')
    fig2 = px.line(rev_df, x="variable", y="value", color="Company Name", labels={'variable': 'Year', 'value': 'Revenue'})
    st.plotly_chart(fig2)

    st.subheader("🫧 Bubble Chart: Investment vs Revenue 2023")
    bubble_df = filtered_df.dropna(subset=['Amount Invested (Approx)', 'Revenue 2023', 'Growth (%)'])
    fig_bubble = px.scatter(
        bubble_df,
        x="Amount Invested (Approx)", 
        y="Revenue 2023",
        size="Growth (%)",
        color="Focus Area",
        hover_name="Company Name",
        title="Investment vs Revenue Bubble Chart"
    )
    st.plotly_chart(fig_bubble)

    st.subheader("📚 Subject Trend Analysis")
    if 'Subjects Offered' in filtered_df.columns:
        subject_counts = filtered_df['Subjects Offered'].dropna().str.split(',').explode().str.strip().value_counts().reset_index()
        subject_counts.columns = ['Subject', 'Number of Companies']
        fig_subjects = px.bar(subject_counts, x='Subject', y='Number of Companies', title='Popularity of Subjects')
        st.plotly_chart(fig_subjects)
    else:
        st.warning("⚠️ 'Subjects Offered' column not found in data.")

    st.subheader("🌐 Focus Area Trend Heatmap")
    heatmap_focus = st.multiselect("Select Focus Areas for Heatmap:", options=df['Focus Area'].dropna().unique(), key="heatmap_focus")
    if heatmap_focus:
        heatmap_df = filtered_df[filtered_df['Focus Area'].isin(heatmap_focus)]
        heatmap_data = heatmap_df.groupby("Focus Area")[['Revenue 2021', 'Revenue 2022', 'Revenue 2023']].sum().dropna()
        if not heatmap_data.empty:
            fig, ax = plt.subplots(figsize=(10, len(heatmap_data)*0.5 + 1))
            sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("⚠️ No data available for the selected focus areas and filters.")
    else:
        st.info("Please select one or more focus areas to generate the heatmap.")

    st.subheader("📈 Revenue Growth Over Years (Custom Selection)")
    company_selection = st.multiselect("Select Companies to View Revenue Trend:", options=df['Company Name'].dropna().unique(), key="revenue_trend")
    if company_selection:
        trend_df = filtered_df[filtered_df['Company Name'].isin(company_selection)]
        rev_df = trend_df[['Company Name', 'Revenue 2021', 'Revenue 2022', 'Revenue 2023']].melt(id_vars='Company Name')
        fig3 = px.line(rev_df, x="variable", y="value", color="Company Name",
                       labels={'variable': 'Year', 'value': 'Revenue'}, title="📈 Custom Company Revenue Trends")
        st.plotly_chart(fig3)
    else:
        st.info("Please select one or more companies to visualize revenue trends.")

# Tab 3: Summary & Export
with tab3:
    st.subheader("📄 Filtered Company Data")
    st.dataframe(filtered_df[['Company Name', 'Focus Area', 'Country', selected_year, 'Amount Invested (Approx)', '🔥 Booming?']].dropna())

    st.subheader("🏆 Top and Bottom Companies")
    ranking_metric = st.selectbox(
        "Select Metric for Ranking",
        options=['Revenue 2023', 'Growth (%)', 'Amount Invested (Approx)'],
        key="ranking_metric"
    )
    # Prepare ranking DataFrame
    ranking_df = filtered_df[['Company Name', 'Focus Area', 'Country', ranking_metric]].dropna(subset=[ranking_metric])
    if not ranking_df.empty:
        # Top 10 companies
        top_10 = ranking_df.sort_values(by=ranking_metric, ascending=False).head(10)
        st.markdown(f"**Top 10 Companies by {ranking_metric}**")
        st.dataframe(top_10)
        # Bottom 10 companies
        bottom_10 = ranking_df.sort_values(by=ranking_metric, ascending=True).head(10)
        st.markdown(f"**Bottom 10 Companies by {ranking_metric}**")
        st.dataframe(bottom_10)
    else:
        st.warning(f"⚠️ No data available for {ranking_metric} with the current filters.")

    st.subheader("🚀 Why Are These Companies Booming?")
    # User-configurable thresholds
    growth_threshold_high = st.slider("High Growth Threshold (%)", 0, 100, 50, key="high_growth_threshold")
    growth_threshold_niche = st.slider("Niche Growth Threshold (%)", 0, 100, 40, key="niche_growth_threshold")
    investment_threshold = st.number_input("Investment Threshold (USD)", value=1000000, min_value=0, key="investment_threshold")
    # Dynamically select default focus areas
    default_niche_areas = [area for area in all_focus_areas if any(keyword in str(area).lower() for keyword in ['stem', 'test prep', 'online learning', 'test preparation', 'online', 'e-learning'])]
    if not default_niche_areas and len(all_focus_areas) > 0:  # Fallback to first 3 focus areas
        default_niche_areas = list(all_focus_areas)[:3]
    niche_focus_areas = st.multiselect(
        "Select Niche Focus Areas",
        options=all_focus_areas,
        default=default_niche_areas if default_niche_areas else [],
        key="niche_focus_areas"
    )

    def booming_reason(row):
        if (pd.notnull(row['Growth (%)']) and pd.notnull(row['Amount Invested (Approx)']) and 
            row['Growth (%)'] > growth_threshold_high and row['Amount Invested (Approx)'] > investment_threshold):
            return "🔥 Heavy Funding + High Growth"
        elif (pd.notnull(row['Growth (%)']) and pd.notnull(row['Focus Area']) and 
              row['Growth (%)'] > growth_threshold_niche and row['Focus Area'] in niche_focus_areas):
            return "🚀 Niche + Rapid Scale"
        return None

    filtered_df['Boom Reason'] = filtered_df.apply(booming_reason, axis=1)
    booming_df = filtered_df[filtered_df['Boom Reason'].notna()][['Company Name', 'Focus Area', 'Country', 'Growth (%)', 'Boom Reason']]
    if not booming_df.empty:
        st.dataframe(booming_df)
        # Export booming companies
        booming_csv = booming_df.to_csv(index=False).encode('utf-8')
        st.download_button(label="📥 Download Booming Companies", data=booming_csv, file_name='booming_edtech_companies.csv', mime='text/csv')
    else:
        st.warning("⚠️ No companies meet the booming criteria with the current filters and thresholds.")

    st.subheader("📉 Predict Revenue 2023 from Investment")
    ml_df = df.dropna(subset=['Amount Invested (Approx)', 'Revenue 2023'])
    if len(ml_df) >= 2:
        X = ml_df[['Amount Invested (Approx)']]
        y = ml_df['Revenue 2023']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        investment_input = st.number_input("Enter Investment Amount (USD):", value=1000000, min_value=0, key="predict_investment")
        predicted = model.predict([[investment_input]])[0]
        st.success(f"📊 Predicted Revenue 2023: ${predicted:,.2f}")
    else:
        st.warning("⚠️ Not enough data in the dataset to train the model.")

    st.subheader("📌 Explore Specific Columns")
    selected_columns = st.multiselect("Select columns to view:", df.columns.tolist(), default=['Company Name', 'Focus Area', 'Country', 'Revenue 2023'], key="summary_columns")
    if selected_columns:
        st.dataframe(filtered_df[selected_columns])
        st.markdown("**📊 Quick Summary**")
        st.write(filtered_df[selected_columns].describe(include='all').transpose())
    else:
        st.info("Please select at least one column to view.")

    st.subheader("📤 Export Filtered Data")
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(label="📥 Download CSV", data=csv, file_name='filtered_global_edtech_data.csv', mime='text/csv')

    st.subheader("📢 Your Feedback")
    feedback = st.text_area("What can we improve?")
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback!")