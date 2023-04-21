import os
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

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

# Get the unique model names, input dimensions, and output dimensions from the file names
models = sorted(list(set([f.split("_")[0] for f in pickle_files])))
input_dims = sorted(list(set([f.split("_")[2] for f in pickle_files])))
output_dims = sorted(list(set([f.split("_")[3] for f in pickle_files])))
ranks = sorted(list(set([f.split("_")[4] for f in pickle_files])))
nof_datas = sorted(list(set([f.split("_")[6] for f in pickle_files])))
epochs = list(range(30))
# Create a Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1("Training History Viewer"),
    html.Div([
        dcc.Dropdown(
            id="model-dropdown",
            options=[{"label": m, "value": m} for m in models],
            value=models[0],
            multi=True
        ),
        dcc.Dropdown(
            id="input-dim-dropdown",
            options=[{"label": d, "value": d} for d in input_dims],
            value=input_dims[0],
            multi=True
        ),
        dcc.Dropdown(
            id="output-dim-dropdown",
            options=[{"label": d, "value": d} for d in output_dims],
            value=output_dims[0],
            multi=True
        ),
        dcc.Dropdown(
            id="rank-dropdown",
            options=[{"label": d, "value": d} for d in ranks],
            value=ranks[0],
            multi=True
        ),
        dcc.Dropdown(
            id="nof_data-dropdown",
            options=[{"label": d, "value": d} for d in nof_datas],
            value=nof_datas[0],
            multi=True
        ),
        dcc.Dropdown(
            id="epoch-dropdown",
            options=[{"label": d, "value": d} for d in epochs],
            value=epochs[0],
        ),

        html.Button("Filter", id="filter-button")
    ]),
    dcc.Graph(id="training-graph")
])

# Define the callback function to update the plot
@app.callback(Output("training-graph", "figure"), [Input("filter-button", "n_clicks")],
              [State("model-dropdown", "value"), State("input-dim-dropdown", "value"), State("output-dim-dropdown", "value"),
               State("rank-dropdown", "value"), State("nof_data-dropdown", "value"), State("epoch-dropdown", "value")])
def update_figure(n_clicks, models, input_dims, output_dims, ranks, nof_datas, epoch):
    # Create a figure object
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Add the lines for all the selected files to the figure
    for file in pickle_files:
        filter_boolean = file.split("_")[0] in models and \
        file.split("_")[2] in input_dims and \
            file.split("_")[3] in output_dims and \
                  file.split("_")[4] in ranks and \
                    file.split("_")[6] in nof_datas
        if filter_boolean:
            df = dfs[file]
            df = df[epoch:]
            fig.add_trace(go.Scatter(x=df.index, y=df["TrainLoss"], name=f"{file} - Train Loss"), secondary_y=False)
            fig.add_trace(go.Scatter(x=df.index, y=df["TestLoss"], name=f"{file} - Test Loss"), secondary_y=False)

    # Customize the layout of the figure
    fig.update_layout(title="Training History", xaxis_title="Time", yaxis_title="Train/Test Loss")

    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
