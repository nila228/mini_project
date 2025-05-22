
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression

@st.cache_data
def load_data():
    df = pd.read_csv("crop_yield.csv")
    df.dropna(inplace=True)
    df['Year'] = pd.to_datetime(df['Year'], format='%Y')
    return df

df = load_data()

st.title("🌾 Agricultural Yield Forecast Dashboard")

crop_list = df['Crop'].unique()
region_list = df['Region'].unique()

crop = st.selectbox("Select a Crop", crop_list)
region = st.selectbox("Select a Region", region_list)

filtered_df = df[(df['Crop'] == crop) & (df['Region'] == region)]

st.subheader("📈 Yield Trend by Year")
st.line_chart(filtered_df.set_index('Year')['Yield'])

st.subheader("🌍 Heatmap: Yield by Region and Crop")
heatmap_data = df.pivot_table(values='Yield', index='Region', columns='Crop')
fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.heatmap(heatmap_data, cmap="YlGnBu", annot=True, fmt=".1f", ax=ax1)
st.pyplot(fig1)

st.subheader("🗃️ Yield Variance Over Years")
fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.boxplot(x=df['Year'].dt.year, y='Yield', data=df, ax=ax2)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
st.pyplot(fig2)

st.subheader("🔮 Yield Prediction (2025–2027)")
pred_crop = df[df['Crop'] == crop]
X = pred_crop['Year'].dt.year.values.reshape(-1, 1)
y = pred_crop['Yield'].values

model = LinearRegression()
model.fit(X, y)

future_years = np.array([[2025], [2026], [2027]])
predicted_yields = model.predict(future_years)

for year, yield_pred in zip([2025, 2026, 2027], predicted_yields):
    st.write(f"📅 {year}: {yield_pred:.2f} tons/hectare")

st.markdown("---")
st.info("To upload data to GCP, use `gsutil cp crop_yield.csv gs://your-bucket-name/` in terminal.")
