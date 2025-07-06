## SANKEY.PY

import plotly.graph_objects as go
import pandas as pd

def _code_mapping(df, src, targ):
    """ Helper function to ensure origin states and destination states remain separate in the Sankey diagram. """
    # Assign unique IDs to ORIGIN_STATE and DEST_STATE separately
    src_labels = list(df[src].unique())  # Left side origin
    targ_labels = list(df[targ].unique())  # Right side destination

    # Create separate mappings for origin and dest
    src_map = {label: i for i, label in enumerate(src_labels)}
    targ_map = {label: i + len(src_map) for i, label in enumerate(targ_labels)}

    # Merge mappings (ensuring separate indexes)
    df[src] = df[src].map(src_map)
    df[targ] = df[targ].map(targ_map)

    # Create a combined ordered label list
    labels = src_labels + targ_labels

    return df, labels


def make_sankey(df, *cols, vals=None, **kwargs):
    """
    Creates a Sankey diagram dynamically from a df with multiple levels.
    """
    # Check
    assert len(cols) >= 2, "Sankey Diagram requires at least two columns."

    df = df.copy()
    df["values"] = df[vals] if vals else 1

    # Convert multi-layer hierarchy into (source → target) pairs
    stacked_df = pd.DataFrame()
    for i in range(len(cols) - 1):
        temp_df = df[[cols[i], cols[i + 1], "values"]].copy()
        temp_df.columns = ["src", "targ", "values"]
        stacked_df = pd.concat([stacked_df, temp_df], ignore_index=True)

    # Add duplicate links & apply mapping for nodes
    stacked_df = stacked_df.groupby(["src", "targ"]).agg({"values": "sum"}).reset_index()
    stacked_df, labels = _code_mapping(stacked_df, "src", "targ")


    # Define node & link structure
    link = {
        "source": stacked_df["src"],  # Left side (origin)
        "target": stacked_df["targ"],  # Right side (destination)
        "value": stacked_df["values"]
    }

    node = {
        "label": labels,
        "pad": 200,
        "thickness": 30,
        "line": {"color": "black", "width": 1}
    }

    # Create/return Sankey
    fig = go.Figure(data=[go.Sankey(link=link, node=node)])
    fig.update_layout(
        title_text="Flight Traffic Sankey Diagram (State to State Flow)<br>"
                   "<sup>Flight Distribution Origin State (left) → Destination State (right)</sup>",
        font_size=12,
        autosize=True
    )

    return fig