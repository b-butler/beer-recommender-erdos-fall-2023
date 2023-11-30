import tkinter as tk
from tkinter import ttk
from fuzzywuzzy import process
import pandas as pd
import numpy as np
import scipy as sp
import warnings
import joblib

import clean
import process as p
from lfm_resizable import LightFMResizable


OPTIMAL_MATRIX = {"method": "zero_one", "threshold": 2.5}

reviews = pd.read_parquet("raw-data.pq")
cleaned_df = clean.merge_similar_name_breweries(reviews)
cleaned_df = clean.merge_brewery_ids(cleaned_df)
cleaned_df = clean.remove_dup_beer_rows(cleaned_df)
cleaned_df = clean.remove_null_rows(cleaned_df)
cleaned_df = clean.remove_duplicate_reviews(cleaned_df)

cleaned_df["beer_id"] = cleaned_df["beer_beerid"].astype("category").cat.codes

beers = cleaned_df.copy().drop(
    [
        "review_time",
        "review_overall",
        "review_aroma",
        "review_appearance",
        "review_profilename",
        "review_palate",
        "review_taste",
    ],
    axis=1,
)  # drop all columns having to do with a specific review

beers = beers[
    ["beer_id", "beer_name", "brewery_name"]
]  # and then reorder the columns to prioritize the most important info
beers = beers.drop_duplicates()  # and then eliminate duplicates

# load the model and build the interaction matrix
model = joblib.load(
    "final-model.joblib"
)  # LightFMResizable object with the fitted model
int_matrix_trans = p.InteractionMatrixTransformer(cleaned_df)
matrix = int_matrix_trans.fit(**OPTIMAL_MATRIX)

selected_combinations = []
selected_beer_ids = []


# for when a brewery is selected from the dropdown menu
def on_brewery_select(event):
    selected_brewery = brewery_combobox.get()

    # Update the beer_combobox based on the selected brewery
    beers_for_brewery = beers[beers["brewery_name"] == selected_brewery][
        "beer_name"
    ].tolist()
    beer_combobox["values"] = beers_for_brewery


# determines which breweries to show depending on what is typed into the box
def on_brewery_search(event):
    query = brewery_combobox.get()

    # Suppress UserWarning related to the empty string query
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=".*Applied processor reduces input query to empty string.*",
        )

        # Perform fuzzy matching to find breweries that match the user's input
        matched_breweries = process.extract(
            query, beers["brewery_name"].unique(), limit=5
        )

    brewery_combobox["values"] = [brewery for brewery, _ in matched_breweries]
    brewery_combobox.set(query)  # Keep the user's input in the combobox


# adds a beer to the list of selected beers when you select it from the dropdown menu
def on_beer_select(event):
    selected_brewery = brewery_combobox.get()
    selected_beer = beer_combobox.get()

    # Add the selected combination to the list
    selected_combinations.append((selected_brewery, selected_beer))
    selected_beer_ids.append(
        beers.loc[
            (beers.brewery_name == selected_brewery)
            & (beers.beer_name == selected_beer),
            "beer_id",
        ].values[0]
    )
    s = "Selected Combinations:" + str(
        selected_combinations
    )  # + '\n' + "Selected Beer IDs:" + str(selected_beer_ids)
    selected_beers_label.config(text=s)


# filters which beers are shown based on what is written in the textbox
def on_beer_search(event):
    brewery = brewery_combobox.get()
    query = beer_combobox.get()

    # Suppress UserWarning related to the empty string query
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=".*Applied processor reduces input query to empty string.*",
        )

        # Perform fuzzy matching to find breweries that match the user's input
        matched_beers = process.extract(
            query,
            beers.loc[beers.brewery_name == brewery, "beer_name"].unique(),
            limit=10,
        )

    beer_combobox["values"] = [beer for beer, _ in matched_beers]
    beer_combobox.set(query)  # Keep the user's input in the combobox


# runs the model and prints your personalized recommendations
def get_beer_recs():
    new_row = np.zeros(matrix.shape[1])
    beer_ids = selected_beer_ids
    for beer_id in beer_ids:
        new_row[beer_id] = 1
    new_row = sp.sparse.csr_matrix(new_row)
    new_matrix = sp.sparse.vstack([matrix, new_row], format="csr")

    model.fit_partial(new_matrix)
    preds = model.predict(33363 * np.ones(65357), [i for i in range(65357)])

    for beer_id in beer_ids:
        preds[beer_id] = -100
    ind = np.argpartition(preds, -10)[-10:]
    recs = beers.loc[beers.beer_id.isin(ind), ["brewery_name", "beer_name"]]
    recs_label.config(text="Your recs are:\n" + recs.to_string(index=False))


root = tk.Tk()
root.title("Beer Recommender")

root.geometry("800x600")
instructions = "1. Find the brewery of the beer you have in mind by typing in all or part of the brewery's name in the first box and then select the desired brewery from the dropdown menu. \n2. Type in all of part of the beer name you have in mind and select the desired beer using the dropdown menu for the second box. \n3. Click 'Add Beer'. \n4. Repeat steps 1 and 2 as many times as desired -- we recommend adding at least 5 beers that you like. \n5. Click 'Get Beer Recommendations' to get your personalized beer recommendations!"
header_label = tk.Label(
    root, text="Beer Recommendation instructions:", justify="center"
)

header_label.grid(row=0, column=0, columnspan=2, sticky="w")
instructions_label = tk.Label(root, text=instructions, justify="left", wraplength=600)
instructions_label.grid(row=1, column=0, columnspan=2, sticky="w")


# Create an AutocompleteCombobox for selecting a brewery
brewery_label = tk.Label(root, text="Brewery name:")
brewery_combobox = ttk.Combobox(root, width=30)
brewery_label.grid(row=2, column=0, sticky="e")
brewery_combobox.grid(row=2, column=1, sticky="w", padx=5)

# Create a Combobox for selecting a beer (initially empty)
beer_label = tk.Label(root, text="Beer name:")
beer_combobox = ttk.Combobox(root, width=30)
beer_label.grid(row=3, column=0, sticky="e")
beer_combobox.grid(row=3, column=1, sticky="w", padx=5)

# Bind events to the brewery_combobox
brewery_combobox.bind("<FocusIn>", on_brewery_search)
brewery_combobox.bind("<KeyRelease>", on_brewery_search)
brewery_combobox.bind("<<ComboboxSelected>>", on_brewery_select)
beer_combobox.bind("<FocusIn>", on_beer_search)
beer_combobox.bind("<KeyRelease>", on_beer_search)
beer_combobox.bind("<<ComboboxSelected>>", on_beer_select)


# sub_btn = tk.Button(root, text='Add Beer', command = show_beers)
# sub_btn.grid(row=4, column = 0, columnspan=2)


# Label to display the text
selected_beers_label = tk.Label(root, text="", wraplength=700)
selected_beers_label.grid(row=5, pady=5, column=0, columnspan=2)

get_recs = tk.Button(root, text="Get Beer Recommendations", command=get_beer_recs)
get_recs.grid(row=6, column=0, columnspan=2)

recs_label = tk.Label(root, text="")
recs_label.grid(row=7, column=0, columnspan=2)

root.mainloop()
