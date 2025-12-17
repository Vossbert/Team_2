import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import altair as alt
    return alt, mo, pd


@app.cell
def _(mo):
    mo.md("""
    # TSAR 2025 Text Simplification Challenge

    This notebook explores the trial data from the TSAR 2025 shared task on text simplification.
    The task involves simplifying texts to different CEFR levels (Common European Framework of Reference for Languages).
    """)
    return


@app.cell
def _(pd):
    import os
    # Construct the absolute path to the data file
    current_dir = os.path.dirname(__file__)
    data_path = os.path.join(current_dir, "..", "data", "tsar2025_trialdata.jsonl")

    # Open the file and pass the file object to pandas
    with open(data_path, "r", encoding="utf-8") as f:
        tsar_data = pd.read_json(f, lines=True)
    return (tsar_data,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Dataset Overview

    The dataset contains **{len(tsar_data)}** examples with text simplification pairs.
    Each example has an original text and a simplified reference at different CEFR levels.
    """)
    return


@app.cell
def _(mo, tsar_data):
    mo.ui.data_explorer(tsar_data)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Interactive Text Explorer

    Select a CEFR level and text ID to compare original and simplified versions:
    """)
    return


@app.cell
def _(mo, tsar_data):
    cefr_selector = mo.ui.dropdown(
        options=sorted(tsar_data["target_cefr"].unique().tolist()),
        value="a2",
        label="CEFR Level"
    )
    cefr_selector
    return (cefr_selector,)


@app.cell
def _(cefr_selector, tsar_data):
    filtered_by_cefr = tsar_data[tsar_data["target_cefr"] == cefr_selector.value]
    return (filtered_by_cefr,)


@app.cell
def _(filtered_by_cefr, mo):
    text_selector = mo.ui.dropdown(
        options=filtered_by_cefr["text_id"].tolist(),
        value=filtered_by_cefr["text_id"].iloc[0],
        label="Text ID"
    )
    text_selector
    return (text_selector,)


@app.cell
def _(filtered_by_cefr, text_selector):
    selected_example = filtered_by_cefr[
        filtered_by_cefr["text_id"] == text_selector.value
    ].iloc[0].to_dict()
    return (selected_example,)


@app.cell
def _(mo, selected_example):
    mo.md(f"""
    ### Original Text

    {selected_example['original']}

    ---

    ### Simplified Text (Target: {selected_example['target_cefr'].upper()})

    {selected_example['reference']}
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Text Length Comparison

    Let's analyze how text simplification affects length:
    """)
    return


@app.cell
def _(tsar_data):
    length_stats = tsar_data.copy()
    length_stats['original_length'] = length_stats['original'].str.len()
    length_stats['reference_length'] = length_stats['reference'].str.len()
    length_stats['original_words'] = length_stats['original'].str.split().str.len()
    length_stats['reference_words'] = length_stats['reference'].str.split().str.len()
    length_stats['length_ratio'] = (length_stats['reference_length'] / length_stats['original_length'] * 100).round(1)
    length_stats['word_ratio'] = (length_stats['reference_words'] / length_stats['original_words'] * 100).round(1)
    return (length_stats,)


@app.cell
def _(alt, length_stats, pd):
    length_chart = alt.Chart(length_stats).mark_circle(size=100, opacity=0.7).encode(
        x=alt.X('original_words:Q', title='Original Word Count'),
        y=alt.Y('reference_words:Q', title='Simplified Word Count'),
        color=alt.Color('target_cefr:N', title='CEFR Level', scale=alt.Scale(scheme='category10')),
        tooltip=['text_id:N', 'original_words:Q', 'reference_words:Q', 'target_cefr:N']
    ).properties(
        width=600,
        height=400,
        title='Word Count: Original vs Simplified'
    )

    diagonal_line = alt.Chart(
        pd.DataFrame({"x": [0, 200], "y": [0, 200]})
    ).mark_line(color='gray', strokeDash=[5, 5]).encode(
        x='x:Q',
        y='y:Q'
    )

    length_chart + diagonal_line
    return diagonal_line, length_chart


@app.cell
def _(mo):
    mo.md("""
    ## CEFR Level Distribution

    How many examples do we have for each CEFR level?
    """)
    return


@app.cell
def _(alt, tsar_data):
    cefr_counts = tsar_data.groupby("target_cefr").size().reset_index(name='count').sort_values('target_cefr')

    cefr_chart = alt.Chart(cefr_counts).mark_bar().encode(
        x=alt.X('target_cefr:N', title='CEFR Level', sort=None),
        y=alt.Y('count:Q', title='Number of Examples'),
        color=alt.Color('target_cefr:N', legend=None, scale=alt.Scale(scheme='category10')),
        tooltip=['target_cefr:N', 'count:Q']
    ).properties(
        width=400,
        height=300,
        title='Distribution of CEFR Levels'
    )

    cefr_chart
    return cefr_chart, cefr_counts


@app.cell
def _(alt, length_stats):
    ratio_chart = alt.Chart(length_stats).mark_boxplot().encode(
        x=alt.X('target_cefr:N', title='CEFR Level'),
        y=alt.Y('word_ratio:Q', title='Word Count Ratio (%)', scale=alt.Scale(domain=[0, 150])),
        color=alt.Color('target_cefr:N', legend=None, scale=alt.Scale(scheme='category10'))
    ).properties(
        width=400,
        height=300,
        title='Simplification Ratio by CEFR Level'
    )

    ratio_chart
    return (ratio_chart,)


@app.cell
def _(length_stats, mo):
    summary_stats = length_stats.groupby('target_cefr').agg({
        'word_ratio': 'mean',
        'length_ratio': 'mean',
        'reference_words': 'mean'
    }).round(1).reset_index()
    summary_stats.columns = ['target_cefr', 'avg_word_ratio', 'avg_char_ratio', 'avg_simplified_words']
    summary_stats = summary_stats.sort_values('target_cefr')

    mo.md("""
    ## Summary Statistics

    Average simplification ratios by CEFR level:
    """)
    return (summary_stats,)


@app.cell
def _(summary_stats):
    summary_stats
    return


@app.cell
def _(mo):
    mo.md("""
    ## Key Observations

    - **A2 level**: Most simplified, typically reducing text length significantly
    - **B1 level**: Moderate simplification, maintaining more original structure
    - Points below the diagonal line indicate text reduction
    - Different CEFR levels show different compression strategies
    """)
    return


if __name__ == "__main__":
    app.run()
