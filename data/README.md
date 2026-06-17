# Data

This project uses the public **Olist Brazilian E-Commerce** dataset. The raw
data is **not** redistributed in this repository; download it from the original
source.

## Download

Kaggle: <https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce>

The dataset is a set of CSV files (orders, order items, products, customers,
reviews, etc.).

## Where to put the files

Place the downloaded CSV files here:

```
data/
  olist_datasets/        <-- put the raw Olist CSV files here
    olist_orders_dataset.csv
    olist_order_items_dataset.csv
    olist_products_dataset.csv
    olist_order_reviews_dataset.csv
    ...
  processed/             <-- created automatically by the data pipeline
```

## Building the processed panel

Run `notebooks/data_pipeline_final.ipynb`. It aggregates the raw data to the
product-week level, applies the filters (min 10 items, 2 distinct prices,
5 weeks per product), engineers the features, computes the relative price
`r = P / P0` (baseline from the training period only), creates the time-based
train/val/test split, and writes the outputs to `data/processed/`
(`panel.csv`, `split_weeks.csv`, and the model/log artifacts).

The processed panel is derived entirely from the raw Olist files, so it is not
checked into version control.
