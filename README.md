# SPER Allocation & Inventory Targeting Dashboard

A flat, upload-driven Streamlit dashboard for Camp + GiftBar SPER weekly files.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files supported

- Pivot-style SPER CSV/Excel exports with a `Row Labels` header row
- Multi-row `SPER Report` exports with store detail, WOC, in-transit, allocated dollars, GM%, SKU count, and in-stock rate

## Major tabs

- Executive Summary
- Regional Diagnostics
- Store Leaderboard
- Inventory Targeting
- Allocation Engine
- Inventory & In-Stock
- Margin & Receipts
- Outliers & Clusters
- Data Explorer

## Inventory Targeting Engine

The new targeting tab highlights the top 5 stores most likely to benefit from additional inventory. It uses a composite score built from:

- In-stock shortage pressure
- YTD sales YoY
- last-week sales momentum when available
- WOC / coverage
- inventory-to-sales load
- sales per SKU
- inventory productivity
- GM%
- allocated and in-transit supply pressure

The app also includes trendline overlays on key scatter charts so relationships are easier to interpret visually.
