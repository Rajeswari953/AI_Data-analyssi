import pandas as pd
import numpy as np
import streamlit as st
import altair as alt


@st.cache_data
def load_raw_data(path: str = "Online Retail.xlsx") -> pd.DataFrame:
    df = pd.read_excel(path)
    return df


def clean_for_overview(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (df_clean, df_returns) aligned with the notebook business analysis."""
    df_clean = df.copy()

    # Before-clean metrics use this same copy in notebook, but for the app
    # we only need cleaned and returns splits.

    # Remove exact duplicate rows
    df_clean = df_clean.drop_duplicates()

    # Drop rows with missing critical values (customer and description)
    df_clean = df_clean.dropna(subset=["CustomerID", "Description"])

    # Create Revenue (can be negative for returns before filtering)
    df_clean["Revenue"] = df_clean["Quantity"] * df_clean["UnitPrice"]

    # Identify cancelled invoices (InvoiceNo starting with 'C')
    cancel_mask = df_clean["InvoiceNo"].astype(str).str.startswith("C")

    # Identify returns (negative quantities)
    return_mask = df_clean["Quantity"] < 0

    # Save cancelled/returned transactions separately
    df_returns = df_clean[cancel_mask | return_mask].copy()

    # For main analysis, exclude cancelled invoices
    df_clean = df_clean[~cancel_mask]

    # Keep only positive quantities and prices
    df_clean = df_clean[df_clean["Quantity"] > 0]
    df_clean = df_clean[df_clean["UnitPrice"] > 0]

    # Recompute Revenue after filters (positive only)
    df_clean["Revenue"] = df_clean["Quantity"] * df_clean["UnitPrice"]

    return df_clean, df_returns


def clean_for_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """Return cleaned dataframe for RFM analysis (customer-level)."""
    df_clean = df.copy()
    df_clean = df_clean.drop_duplicates()
    df_clean = df_clean.dropna(subset=["CustomerID"])
    df_clean = df_clean[~df_clean["InvoiceNo"].astype(str).str.startswith("C")]
    df_clean = df_clean[df_clean["Quantity"] > 0]
    df_clean = df_clean[df_clean["UnitPrice"] > 0]
    df_clean["Revenue"] = df_clean["Quantity"] * df_clean["UnitPrice"]
    return df_clean


def compute_business_overview(df_clean: pd.DataFrame, df_returns: pd.DataFrame):
    # KPIs
    total_revenue = df_clean["Revenue"].sum()
    total_customers = df_clean["CustomerID"].nunique()
    total_orders = df_clean["InvoiceNo"].nunique()
    aov = total_revenue / total_orders if total_orders > 0 else np.nan

    # Return rate based on quantity
    total_qty_pos = df_clean["Quantity"].sum()
    total_qty_ret = np.abs(df_returns["Quantity"][df_returns["Quantity"] < 0].sum())
    denom = total_qty_pos + total_qty_ret
    return_rate = (total_qty_ret / denom) if denom > 0 else 0.0

    # Time-based metrics
    df_time = df_clean.copy()
    df_time["InvoiceDate_date"] = df_time["InvoiceDate"].dt.date
    df_time["InvoiceMonth"] = df_time["InvoiceDate"].dt.to_period("M").dt.to_timestamp()

    monthly_sales = (
        df_time.groupby("InvoiceMonth")
        .agg(
            monthly_revenue=("Revenue", "sum"),
            monthly_orders=("InvoiceNo", "nunique"),
        )
        .reset_index()
    )

    # Geographic distribution
    orders = (
        df_clean.groupby("InvoiceNo")
        .agg(
            OrderDate=("InvoiceDate", "min"),
            Country=("Country", "first"),
            CustomerID=("CustomerID", "first"),
            OrderRevenue=("Revenue", "sum"),
            OrderQuantity=("Quantity", "sum"),
        )
        .reset_index()
    )

    geo_summary = (
        orders.groupby("Country")
        .agg(
            total_revenue=("OrderRevenue", "sum"),
            total_orders=("InvoiceNo", "nunique"),
            total_quantity=("OrderQuantity", "sum"),
            avg_order_value=("OrderRevenue", "mean"),
            avg_items_per_order=("OrderQuantity", "mean"),
        )
        .reset_index()
        .sort_values("total_revenue", ascending=False)
    )

    # Product performance
    product_summary = (
        df_clean.groupby(["StockCode", "Description"], dropna=False)
        .agg(
            total_revenue=("Revenue", "sum"),
            total_quantity=("Quantity", "sum"),
            order_lines=("InvoiceNo", "count"),
            unique_invoices=("InvoiceNo", "nunique"),
        )
        .reset_index()
        .sort_values("total_revenue", ascending=False)
    )

    top_products_by_revenue = product_summary.head(10).copy()

    return {
        "total_revenue": total_revenue,
        "total_customers": total_customers,
        "aov": aov,
        "return_rate": return_rate,
        "monthly_sales": monthly_sales,
        "geo_summary": geo_summary,
        "top_products_by_revenue": top_products_by_revenue,
    }


def compute_rfm(df_clean_rfm: pd.DataFrame) -> pd.DataFrame:
    snapshot_date = df_clean_rfm["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = (
        df_clean_rfm.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("Revenue", "sum"),
        )
        .reset_index()
    )

    quantiles = rfm[["Recency", "Frequency", "Monetary"]].quantile(
        [0.2, 0.4, 0.6, 0.8]
    ).to_dict()

    def r_score(x):
        if x <= quantiles["Recency"][0.2]:
            return 5
        elif x <= quantiles["Recency"][0.4]:
            return 4
        elif x <= quantiles["Recency"][0.6]:
            return 3
        elif x <= quantiles["Recency"][0.8]:
            return 2
        else:
            return 1

    def fm_score(x, col):
        if x <= quantiles[col][0.2]:
            return 1
        elif x <= quantiles[col][0.4]:
            return 2
        elif x <= quantiles[col][0.6]:
            return 3
        elif x <= quantiles[col][0.8]:
            return 4
        else:
            return 5

    rfm["R_Score"] = rfm["Recency"].apply(r_score)
    rfm["F_Score"] = rfm["Frequency"].apply(lambda x: fm_score(x, "Frequency"))
    rfm["M_Score"] = rfm["Monetary"].apply(lambda x: fm_score(x, "Monetary"))

    rfm["RFM_Score"] = (
        rfm["R_Score"].astype(str)
        + rfm["F_Score"].astype(str)
        + rfm["M_Score"].astype(str)
    )

    # Simple segmentation based on R and F primarily
    def segment_customer(row):
        r, f, m = row["R_Score"], row["F_Score"], row["M_Score"]
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        if r >= 4 and f >= 3:
            return "Loyal Customers"
        if r >= 3 and f >= 3 and m >= 3:
            return "Potential Loyalist"
        if r <= 2 and f >= 4:
            return "At Risk"
        if r <= 2 and f <= 2 and m <= 2:
            return "Hibernating"
        return "Others"

    rfm["Segment"] = rfm.apply(segment_customer, axis=1)

    return rfm


def customer_profile(df_clean_rfm: pd.DataFrame, customer_id: float) -> dict:
    cust_df = df_clean_rfm[df_clean_rfm["CustomerID"] == customer_id].copy()
    if cust_df.empty:
        return {}

    country = cust_df["Country"].mode().iloc[0]
    first_purchase = cust_df["InvoiceDate"].min()
    last_purchase = cust_df["InvoiceDate"].max()
    total_revenue = cust_df["Revenue"].sum()
    txn_count = cust_df["InvoiceNo"].nunique()
    aov = total_revenue / txn_count if txn_count > 0 else np.nan

    # Top products
    prod_summary = (
        cust_df.groupby("Description", dropna=False)
        .agg(
            revenue=("Revenue", "sum"),
            quantity=("Quantity", "sum"),
        )
        .reset_index()
        .sort_values("revenue", ascending=False)
        .head(10)
    )

    # Simple product category from first word
    cust_df["Category"] = (
        cust_df["Description"]
        .astype(str)
        .str.upper()
        .str.strip()
        .str.split()
        .str[0]
    )
    category_summary = (
        cust_df.groupby("Category")
        .agg(
            revenue=("Revenue", "sum"),
            quantity=("Quantity", "sum"),
        )
        .reset_index()
        .sort_values("revenue", ascending=False)
        .head(10)
    )

    return {
        "country": country,
        "first_purchase": first_purchase,
        "last_purchase": last_purchase,
        "total_revenue": total_revenue,
        "txn_count": txn_count,
        "aov": aov,
        "top_products": prod_summary,
        "categories": category_summary,
    }


def page_overview(df_clean: pd.DataFrame, df_returns: pd.DataFrame):
    st.title("Business Overview")

    metrics = compute_business_overview(df_clean, df_returns)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue", f"£{metrics['total_revenue']:,.0f}")
    col2.metric("Unique Customers", f"{metrics['total_customers']:,}")
    col3.metric("Average Order Value", f"£{metrics['aov']:,.2f}")
    col4.metric("Return Rate", f"{metrics['return_rate'] * 100:.1f}%")

    st.markdown("---")

    st.subheader("Revenue and Orders Over Time")
    ms = metrics["monthly_sales"]
    ms_long = ms.melt(
        id_vars="InvoiceMonth",
        value_vars=["monthly_revenue", "monthly_orders"],
        var_name="Metric",
        value_name="Value",
    )
    chart = (
        alt.Chart(ms_long)
        .mark_line(point=True)
        .encode(
            x="InvoiceMonth:T",
            y="Value:Q",
            color="Metric:N",
            tooltip=["InvoiceMonth:T", "Metric:N", "Value:Q"],
        )
        .properties(height=350)
    )
    st.altair_chart(chart, use_container_width=True)

    st.markdown("---")

    st.subheader("Geographic Revenue by Country")
    geo = metrics["geo_summary"].copy()
    geo_top = geo.head(15)
    geo_chart = (
        alt.Chart(geo_top)
        .mark_bar()
        .encode(
            x=alt.X("total_revenue:Q", title="Total Revenue"),
            y=alt.Y("Country:N", sort="-x", title="Country"),
            tooltip=["Country:N", "total_revenue:Q", "total_orders:Q"],
        )
        .properties(height=400)
    )
    st.altair_chart(geo_chart, use_container_width=True)

    st.markdown("---")

    st.subheader("Top 10 Products by Revenue")
    top_products = metrics["top_products_by_revenue"].copy()
    top_products["Label"] = top_products["Description"].fillna(
        top_products["StockCode"]
    )
    prod_chart = (
        alt.Chart(top_products)
        .mark_bar()
        .encode(
            x=alt.X("total_revenue:Q", title="Total Revenue"),
            y=alt.Y("Label:N", sort="-x", title="Product"),
            tooltip=["Label:N", "total_revenue:Q", "total_quantity:Q"],
        )
        .properties(height=400)
    )
    st.altair_chart(prod_chart, use_container_width=True)


def page_rfm(df_clean_rfm: pd.DataFrame):
    st.title("Customer RFM Analysis")

    rfm = compute_rfm(df_clean_rfm)

    st.subheader("RFM Segmentation Overview")
    seg_counts = (
        rfm.groupby("Segment")["CustomerID"]
        .count()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    seg_chart = (
        alt.Chart(seg_counts)
        .mark_bar()
        .encode(
            x=alt.X("count:Q", title="Number of Customers"),
            y=alt.Y("Segment:N", sort="-x", title="Segment"),
            tooltip=["Segment:N", "count:Q"],
        )
        .properties(height=300)
    )
    st.altair_chart(seg_chart, use_container_width=True)

    st.markdown("---")

    st.subheader("Customer Lookup")
    customer_ids = sorted(rfm["CustomerID"].unique())
    customer_ids_display = [str(int(c)) for c in customer_ids]
    selected_str = st.selectbox(
        "Select a CustomerID", options=customer_ids_display, index=0
    )
    selected_id = float(selected_str)

    cust_rfm = rfm[rfm["CustomerID"] == selected_id].iloc[0]

    st.markdown(f"**Segment:** {cust_rfm['Segment']}  \n"
                f"**RFM Score:** {cust_rfm['RFM_Score']}  \n"
                f"**Recency (days):** {cust_rfm['Recency']}  \n"
                f"**Frequency:** {cust_rfm['Frequency']}  \n"
                f"**Monetary:** £{cust_rfm['Monetary']:,.2f}")

    st.markdown("---")
    st.subheader("Customer Profile")

    profile = customer_profile(df_clean_rfm, selected_id)
    if not profile:
        st.info("No detailed data available for this customer.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Country", profile["country"])
    col2.metric("First Purchase", profile["first_purchase"].date().isoformat())
    col3.metric("Last Purchase", profile["last_purchase"].date().isoformat())

    col4, col5, col6 = st.columns(3)
    col4.metric("Total Revenue", f"£{profile['total_revenue']:,.2f}")
    col5.metric("Transactions", f"{profile['txn_count']}")
    col6.metric("AOV", f"£{profile['aov']:,.2f}")

    st.markdown("#### Top Products Purchased")
    top_products = profile["top_products"]
    if not top_products.empty:
        prod_chart = (
            alt.Chart(top_products)
            .mark_bar()
            .encode(
                x=alt.X("revenue:Q", title="Revenue"),
                y=alt.Y("Description:N", sort="-x", title="Product"),
                tooltip=["Description:N", "revenue:Q", "quantity:Q"],
            )
            .properties(height=300)
        )
        st.altair_chart(prod_chart, use_container_width=True)
    else:
        st.write("No product data available for this customer.")

    st.markdown("#### Product Categories Purchased")
    categories = profile["categories"]
    if not categories.empty:
        cat_chart = (
            alt.Chart(categories)
            .mark_bar()
            .encode(
                x=alt.X("revenue:Q", title="Revenue"),
                y=alt.Y("Category:N", sort="-x", title="Category"),
                tooltip=["Category:N", "revenue:Q", "quantity:Q"],
            )
            .properties(height=300)
        )
        st.altair_chart(cat_chart, use_container_width=True)
    else:
        st.write("No category data available for this customer.")


def main():
    st.set_page_config(
        page_title="Online Retail Dashboard",
        layout="wide",
    )

    st.sidebar.title("Online Retail Dashboard")
    page = st.sidebar.radio("Navigate", ["Business Overview", "Customer RFM Analysis"])

    df_raw = load_raw_data()
    df_clean_overview, df_returns = clean_for_overview(df_raw)
    df_clean_rfm = clean_for_rfm(df_raw)

    if page == "Business Overview":
        page_overview(df_clean_overview, df_returns)
    else:
        page_rfm(df_clean_rfm)


if __name__ == "__main__":
    main()

