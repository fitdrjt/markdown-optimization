import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Markdown Optimizer", layout="wide")

# Sidebar untuk navigasi
page = st.sidebar.selectbox("Choose Page", ["🏠 Home", "📈 Project Simulation"])

@st.cache_data
# Memuat model dan aset lainnya
def load_assets():
    model = joblib.load("model_xgb.pkl")  # Memuat model yang sudah dilatih
    scaler = joblib.load("scaler.pkl")  # Memuat scaler untuk fitur numerik
    feature_columns = joblib.load("feature_columns.pkl")  # Kolom fitur yang digunakan
    brand_mean = joblib.load("brand_mean.pkl")  # Mean per brand untuk pengolahan data
    product_mean = joblib.load("product_mean.pkl")  # Mean per product untuk pengolahan data
    numeric_cols = joblib.load("numeric_cols.pkl")
    return model, scaler, feature_columns, brand_mean, product_mean, numeric_cols


st.title("🏷️ The Profit Behind the Price Cut 💰")

# Memuat data dan memanipulasi dataframe
def load_data():
    df = pd.read_csv('retail_markdown.csv')

    df['Discounted_Price_M1'] = df['Original_Price'] * (1 - df['Markdown_1'])
    df['Discounted_Price_M2'] = df['Original_Price'] * (1 - df['Markdown_2'])
    df['Discounted_Price_M3'] = df['Original_Price'] * (1 - df['Markdown_3'])
    df['Discounted_Price_M4'] = df['Original_Price'] * (1 - df['Markdown_4'])

    df['Price_per_Stock'] = df['Original_Price'] / (df['Stock_Level'] + 1)
    df['Price_Gap_Competitor'] = df['Original_Price'] - df['Competitor_Price']
    df['Price_Gap_Percent'] = df['Price_Gap_Competitor'] / df['Competitor_Price']

    # Melting kolom 
    markdown_columns = ['Markdown_1', 'Markdown_2', 'Markdown_3', 'Markdown_4']
    sales_columns = ['Sales_After_M1', 'Sales_After_M2', 'Sales_After_M3', 'Sales_After_M4']
    discounted_price_columns = ['Discounted_Price_M1', 'Discounted_Price_M2', 'Discounted_Price_M3', 'Discounted_Price_M4']

    df_sales_melted = pd.melt(
        df,
        id_vars=[col for col in df.columns if col not in sales_columns],
        value_vars=sales_columns,
        var_name='Sales_Month',
        value_name='Sales_After'
    )
    df_markdown_melted = pd.melt(
        df,
        id_vars=[col for col in df.columns if col not in markdown_columns],
        value_vars=markdown_columns,
        var_name='Markdown_Month',
        value_name='Markdown'
    )
    df_price_melted = pd.melt(
        df,
        id_vars=[col for col in df.columns if col not in discounted_price_columns],
        value_vars=discounted_price_columns,
        var_name='Price_Month',
        value_name='Final_Price'
    )

    df_melted = df_sales_melted.copy()
    df_melted['Markdown'] = df_markdown_melted['Markdown'].values
    df_melted['Final_Price'] = df_price_melted['Final_Price'].values
    df_melted['Markdown_Level'] = df_melted['Sales_Month'].str.extract(r'M(\d)').astype(int)
    df_melted.drop(columns=['Sales_Month'], inplace=True)

    return df_melted

# Fungsi untuk melakukan simulasi markdown berdasarkan kombinasi yang diinginkan
def simulate_optimal_discount_per_price(product_name, brand, season,
                                        original_df, model, encoder_dict,
                                        scaler, features_model, numeric_cols):

    # Filter subset sesuai dengan kombinasi produk
    subset = original_df[
        (original_df['Product_Name'] == product_name) &
        (original_df['Brand'] == brand) &
        (original_df['Season'] == season)
    ].copy()

    if subset.empty:
        raise ValueError("Data untuk kombinasi tersebut tidak ditemukan.")

    results = [] # Menginisiasi tempat penyimpanan hasil

    for price in subset['Original_Price'].unique():
        subset_price = subset[subset['Original_Price'] == price].copy()
        subset_price['Final_Price'] = subset_price['Original_Price'] * (1 - subset_price['Markdown'])
        subset_price['Revenue'] = subset_price['Final_Price'] * subset_price['Sales_After']

        # Menghitung Baseline (tanpa markdown)
        baseline_row = subset_price[np.isclose(subset_price['Final_Price'], subset_price['Original_Price'])]

        if not baseline_row.empty:
            revenue_baseline = (baseline_row['Final_Price'] * baseline_row['Sales_After']).values[0]
            sales_baseline = baseline_row['Sales_After'].values[0]
        else:
            revenue_baseline = subset_price['Revenue'].mean()
            sales_baseline = subset_price['Sales_After'].mean()

        # Encoding
        subset_price['Brand_encoded'] = subset_price['Brand'].map(encoder_dict['brand_mean'])
        subset_price['Product_Name_encoded'] = subset_price['Product_Name'].map(encoder_dict['product_mean'])
        subset_price = pd.get_dummies(subset_price, columns=['Category', 'Season', 'Promotion_Type'], drop_first=True)

        # Memastikan semua fitur tersedia
        for col in features_model:
            if col not in subset_price.columns:
                subset_price[col] = 0
        # Scaling
        df_sim = subset_price[features_model].copy()
        df_sim[numeric_cols] = scaler.transform(df_sim[numeric_cols])

        # Melakukan prediksi
        subset_price['Predicted_Sales'] = model.predict(df_sim)
        subset_price['Simulated_Revenue'] = subset_price['Final_Price'] * subset_price['Predicted_Sales']

        # Mengambil baris dengan revenue tertinggi
        best_row = subset_price.loc[subset_price['Simulated_Revenue'].idxmax()]

        results.append({
            'Original_Price': price,
            'Markdown_Optimal': best_row['Markdown'],
            'Final_Price': best_row['Final_Price'],
            'Predicted_Sales': best_row['Predicted_Sales'],
            'Simulated_Revenue': best_row['Simulated_Revenue'],
            'Revenue_Before_Optimization': revenue_baseline,
            'Sales_Before_Optimization': sales_baseline
        })

    # Membuat dataframe hasil
    df_result = pd.DataFrame(results)
    df_result['Revenue_Improvement'] = df_result['Simulated_Revenue'] - df_result['Revenue_Before_Optimization']
    df_result['Revenue_Improvement_Percentage'] = (df_result['Revenue_Improvement'] / df_result['Revenue_Before_Optimization']) * 100
    df_result['Sales_Improvement'] = df_result['Predicted_Sales'] - df_result['Sales_Before_Optimization']
    df_result['Sales_Improvement_Percentage'] = (df_result['Sales_Improvement'] / df_result['Sales_Before_Optimization']) * 100

    combo_label = f"{product_name} | {brand} | {season}"

     # Plot 1: Revenue Before vs After
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    sns.barplot(data=df_result, x='Original_Price', y='Revenue_Before_Optimization', color='gray', label='Before Optimization', ax=ax1)
    sns.barplot(data=df_result, x='Original_Price', y='Simulated_Revenue', color='green', label='After Optimization', ax=ax1)
    ax1.set_title(f'Revenue Before & After Markdown Optimization\n({combo_label})', pad=20)
    ax1.set_xlabel('Original Price', labelpad=15)
    ax1.set_ylabel('Revenue', labelpad=10)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
    ax1.legend()
    for patch, label in zip(ax1.patches[len(df_result):], df_result['Markdown_Optimal']):
        x = patch.get_x() + patch.get_width() / 2
        y = patch.get_height()
        ax1.text(x, y + 0.012 * df_result['Simulated_Revenue'].max(), f"{int(label*100)}%", ha='center', va='bottom', fontsize=9, rotation=90)
    plt.tight_layout()
    st.pyplot(fig1)

    # Plot 2: Revenue Improvement (%)
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    sns.barplot(data=df_result, x='Original_Price', y='Revenue_Improvement_Percentage', color='blue', ax=ax2)
    ax2.set_title(f'Revenue Increase (%) After Optimal Markdown\n({combo_label})', pad=20)
    ax2.set_xlabel('Original Price', labelpad=15)
    ax2.set_ylabel('Revenue Improvement (%)', labelpad=10)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)
    for patch, label in zip(ax2.patches, df_result['Markdown_Optimal']):
        x = patch.get_x() + patch.get_width() / 2
        y = patch.get_height()
        ax2.text(x, y + 0.012 * df_result['Revenue_Improvement_Percentage'].max(), f"{int(label*100)}%", ha='center', va='bottom', fontsize=9, rotation=90)
    plt.tight_layout()
    st.pyplot(fig2)

    # Plot 3: Sales Before vs After
    fig3, ax3 = plt.subplots(figsize=(14, 8))
    sns.barplot(data=df_result, x='Original_Price', y='Sales_Before_Optimization', color='gray', label='Before Optimization', ax=ax3)
    sns.barplot(data=df_result, x='Original_Price', y='Predicted_Sales', color='orange', label='After Optimization', ax=ax3)
    ax3.set_title(f'Sales Before & After Markdown Optimization\n({combo_label})', pad=20)
    ax3.set_xlabel('Original Price', labelpad=15)
    ax3.set_ylabel('Sales', labelpad=10)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=90)
    ax3.legend()
    for patch, label in zip(ax3.patches[len(df_result):], df_result['Markdown_Optimal']):
        x = patch.get_x() + patch.get_width() / 2
        y = patch.get_height()
        ax3.text(x, y + 0.012 * df_result['Predicted_Sales'].max(), f"{int(label*100)}%", ha='center', va='bottom', fontsize=9, rotation=90)
    plt.tight_layout()
    st.pyplot(fig3)

    return df_result
 


def simulate_detail_per_price(product_name, brand, season, original_df, model, encoder_dict,
                              scaler, features_model, numeric_cols):
    # Mengambil subset data berdasarkan kombinasi nama produk, brand, dan musim
    subset = original_df[
        (original_df['Product_Name'] == product_name) &
        (original_df['Brand'] == brand) &
        (original_df['Season'] == season)
    ].copy()

    if subset.empty:
        raise ValueError("Data untuk kombinasi tersebut tidak ditemukan.")

    all_details = [] # Menginisiasi pemyimpanan hasil simulasi

    # Iterasi setiap nilai original price yang unik
    for price in subset['Original_Price'].unique():
        base_subset = subset[subset['Original_Price'] == price].copy()

        # Iterasi setiap tingkat markdown yang tersedia untuk harga tersebut
        for markdown in sorted(base_subset['Markdown'].unique()):
            row = base_subset[base_subset['Markdown'] == markdown].copy()
            if row.empty:
                continue
            # Menghitung harga akhir setelah diskon
            row['Final_Price'] = row['Original_Price'] * (1 - row['Markdown'])

            # Encoding
            row['Brand_encoded'] = row['Brand'].map(encoder_dict['brand_mean'])
            row['Product_Name_encoded'] = row['Product_Name'].map(encoder_dict['product_mean'])
            row = pd.get_dummies(row, columns=['Category', 'Season', 'Promotion_Type'], drop_first=True)

            # Menambahkan kolom dummy yang belum ada (karena mungkin tidak muncul dalam subset)
            for col in features_model:
                if col not in row.columns:
                    row[col] = 0

            # Mengambil fitur yang diperlukan model
            df_sim = row[features_model].copy()

            df_sim[numeric_cols] = scaler.transform(df_sim[numeric_cols]) #Scaling
            row['Predicted_Sales'] = model.predict(df_sim) # Memprediksi Sales
            row['Simulated_Revenue'] = row['Final_Price'] * row['Predicted_Sales'] # Menghitung revenue
            row['Original_Price_Value'] = price
            row['Markdown_Value'] = markdown

            # Menyimpan hasil ke dalam list
            all_details.append(row[['Original_Price_Value', 'Markdown', 'Predicted_Sales', 'Simulated_Revenue']])

    df_detail = pd.concat(all_details, ignore_index=True)
    return df_detail


def get_simulation_details_for_price(df_result, sim_detail, price_val):

    # Mengecek apakah price_val tersedia dalam data
    if price_val not in df_result['Original_Price'].values:
        print(f"Tidak ditemukan data simulasi untuk Original Price = {price_val}")
        return

    # Mengambil baris hasil simulasi
    result = df_result[df_result['Original_Price'] == price_val].iloc[0]

    # Memformat output
    output = {
        'Original Price': f"₣  {result['Original_Price']:.2f}",
        'Markdown Optimal': f"{result['Markdown_Optimal'] * 100:.0f}%",
        'Discounted Price': f"₣  {result['Final_Price']:.2f}",
        'Predicted Sales': f"{int(result['Predicted_Sales'])} unit",
        'Predicted Revenue': f"₣  {result['Simulated_Revenue']:.2f}",
        'Revenue Improvement': f"₣  {result['Revenue_Improvement']:.2f} ({result['Revenue_Improvement_Percentage']:.0f}%)"
    }

    # Mencetak hasil simulasi
    for key, value in output.items():
        print(f"{key:<20}: {value}")

    # Mengambil data simulasi detail berdasarkan original price
    subset = sim_detail[sim_detail['Original_Price_Value'] == price_val]

    # Membuat plot
    if not subset.empty:
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Axis 1: Predicted Sales
        color1 = 'tab:blue'
        ax1.set_xlabel('Markdown')
        ax1.set_ylabel('Predicted Sales', color=color1)
        sns.lineplot(data=subset, x='Markdown', y='Predicted_Sales', marker='o', ax=ax1, label='Predicted Sales', color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(False)

        # Axis 2: Simulated Revenue
        ax2 = ax1.twinx()
        color2 = 'tab:green'
        ax2.set_ylabel('Simulated Revenue', color=color2)
        sns.lineplot(data=subset, x='Markdown', y='Simulated_Revenue', marker='o', ax=ax2, label='Simulated Revenue', color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.grid(False)

        # Garis optimal markdown
        ax1.axvline(x=result['Markdown_Optimal'], linestyle='--', color='red', label='Optimal Markdown')

        # Legend
        ax1.legend(loc='upper right', bbox_to_anchor=(1, 1))
        ax2.legend(loc='lower right', bbox_to_anchor=(1, 0.80))

        # Title
        plt.title(f'Sales and Revenue Response to Markdown\n(Original Price = {price_val})')
        plt.tight_layout()
        #plt.show()
        st.pyplot(fig)
    else:
        print(f"Tidak ditemukan data simulasi detail untuk Original Price = {price_val}")

    return output

# Load data dan aset lain
df = load_data()

model, scaler, feature_columns, brand_mean, product_mean, numeric_cols = load_assets()

encoder_dict = {
    'brand_mean': brand_mean,
    'product_mean': product_mean
}

product_options = df['Product_Name'].unique()
brand_options = df['Brand'].unique()
season_options = df['Season'].unique()

# Halaman 1 - Home
if page == "🏠 Home":

    st.markdown("""
    ### 📜 About the Project
    In retail, offering discounts is common. But here's the catch: bigger discounts don’t always mean bigger profits. Sometimes, we sell more but earn less.
    This project aimed to help retailers identify the optimal markdown strategy, one that doesn't just drive sales, but also improves overall revenue. 
    By analyzing historical sales and pricing data, machine learning models were used to simulate how different levels of discounts affect customer behavior and business outcomes.

    
    ### 📫 Contact:
    - Email: fitri.wdarojati@gmail.com  
    - GitHub: [github](https://github.com/fitdrjt)
    """)

# Halaman 2 - Simulasi Project
elif page == "📈 Project Simulation":
    st.title("📈 Markdown Optimization for Revenue Growth")
    st.markdown("Select a product combination to simulate markdown strategies and see how pricing impacts revenue.")

    # Input UI di main page
    col1, col2, col3 = st.columns(3)
    with col1:
        product_name = st.selectbox("Product Name", product_options, key="product_name")
    with col2:
        brand = st.selectbox("Brand", brand_options, key="brand")
    with col3:
        season = st.selectbox("Season", season_options, key="season")

    # Tombol Simulasi
    if st.button("🔍 Searching for Optimal Markdown"):
        with st.spinner("Running simulation..."):
            try:
                df_result = simulate_optimal_discount_per_price(
                    product_name=product_name,
                    brand=brand,
                    season=season,
                    original_df=df,
                    model=model,
                    encoder_dict=encoder_dict,
                    scaler=scaler,
                    features_model=feature_columns,
                    numeric_cols=numeric_cols
                )

                st.success("Simulation completed!")

                # Menyimpan hasil di session state
                st.session_state.df_result = df_result
                st.session_state.sim_detail = simulate_detail_per_price(
                    product_name=product_name,
                    brand=brand,
                    season=season,
                    original_df=df,
                    model=model,
                    encoder_dict=encoder_dict,
                    scaler=scaler,
                    features_model=feature_columns,
                    numeric_cols=numeric_cols
                )

            except Exception as e:
                st.error(f"Simulation failed: {e}")

# Menampilkan hasil jika sudah tersedia di session_state
if 'df_result' in st.session_state and not st.session_state.df_result.empty:
    st.markdown("---")
    st.subheader("📊 Optimal Markdown Simulation Result")
    st.dataframe(st.session_state.df_result)

    st.markdown("---")
    st.subheader("🔍 View Simulation Details by Original Price")

    price_val = st.selectbox(
        "Select Original Price",
        st.session_state.df_result['Original_Price'].unique(),
        key="price_val"
    )

    if st.button("Show Details"):
        output = get_simulation_details_for_price(
            df_result=st.session_state.df_result,
            sim_detail=st.session_state.sim_detail,
            price_val=price_val
        )

        st.markdown("### Detailed Result 📈")
        for key, value in output.items():
            st.markdown(f"**{key}**: {value}")
