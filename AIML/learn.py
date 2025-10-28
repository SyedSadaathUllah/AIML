import pandas as pd
import numpy as np

customers = pd.read_csv("customers.csv")
orders = pd.read_csv("orders.csv")
products = pd.read_csv("products.csv")

cust_orders = pd.merge(customers,orders,on="CustomerID",how="outer")
print(cust_orders)
prod_ord = pd.merge(cust_orders,products,how="outer")
print(prod_ord)