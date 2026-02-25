# Online Retail E‑commerce Analytics Dashboard

This project analyzes the **Online Retail** dataset and exposes the results through a **two‑page Streamlit dashboard**:

- **Page 1 – Business Overview**
  - Total revenue, unique customers, average order value (AOV), and return rate
  - Monthly revenue and order trends over one year
  - Revenue by country (geographic performance)
  - Top 10 products by revenue

- **Page 2 – Customer RFM Analysis**
  - RFM segmentation (Recency, Frequency, Monetary)
  - Customer lookup by `CustomerID`
  - Customer profile:
    - Country
    - First / last purchase dates
    - Total revenue, AOV, transaction count
    - Top products and product categories purchased

---

## Tech Stack

- Python
- Streamlit
- Pandas, NumPy
- Altair (for charts)
- OpenPyXL (to read the Excel dataset)

---

## How to Run Locally

1. Clone or download this repository.
2. Make sure `Online Retail.xlsx` is in the same folder as `streamlit_app.py`.
3. Install dependencies:

   
