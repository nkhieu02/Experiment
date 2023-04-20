import os
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# Define the path to the folder containing the pickle files
folder_path = "./training_history/"

# Get a list of all the pickle files in the folder
pickle_files = [f for f in os.listdir(folder_path) if f.endswith(".pickle")]

# Create a dictionary of DataFrames with file names as keys
dfs = {}
for file in pickle_files:
    filepath = os.path.join(folder_path, file)
    df = pd.read_pickle(filepath)
    dfs[file] = df

# Create a Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1("Training History Viewer"),
    dcc.Dropdown(
        id="file-dropdown",
        options=[{"label": file, "value": file} for file in pickle_files],
        value=[pickle_files[0]],
        multi=True
    ),
    dcc.Graph(id="training-graph")
])

# Define the callback function to update the plot
@app.callback(Output("training-graph", "figure"), [Input("file-dropdown", "value")])
def update_figure(selected_files):
    # Create a figure object
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Add the lines for all the selected files to the figure
    for file in selected_files:
        df = dfs[file]
        fig.add_trace(go.Scatter(x=df.index, y=df["TrainLoss"], name=f"{file} - Train Loss"), secondary_y=False)
        fig.add_trace(go.Scatter(x=df.index, y=df["TestLoss"], name=f"{file} - Test Loss"), secondary_y=False)

    # Customize the layout of the figure
    fig.update_layout(title="Training History", xaxis_title="Time", yaxis_title="Train/Test Loss")

    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
