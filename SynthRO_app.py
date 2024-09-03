import os

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objs as go
import plotly.io as pio
import scipy.stats as stats
from dash import dcc, html, dash_table, ctx
from dash.dependencies import Input, Output, State, ALL
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from multiprocessing import Pool
from itertools import repeat
from utils import GeneralUtils as utils, ResemblanceMetrics as ResMet, UtilityMetrics as UtiMet, PrivacyMetrics as PriMet

global real_data, synthetic_datasets, dict_features_type, num_features, cat_features, real_train_data, real_test_data, encoder_labels

real_data = pd.DataFrame()
synthetic_datasets = []
dict_features_type = {}
real_train_data = pd.DataFrame()
real_test_data = pd.DataFrame()

global path_user
path_user = os.getcwd().replace('\\', '/')
if not os.path.exists(os.path.join(path_user, 'data_figures')):
    os.makedirs(os.path.join(path_user, 'data_figures'))
if not os.path.exists(os.path.join(path_user, 'data_report')):
    os.makedirs(os.path.join(path_user, 'data_report'))

global n_cpu
n_cpu = os.cpu_count()

print("!! START !!")

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True,
                prevent_initial_callbacks="initial_duplicate")

# PAGE LAYOUT (static elements)
app.layout = html.Div([

    # Header application (top)
    dbc.NavbarSimple([
        html.H1("SynthRO", style={"color": "#ffffff"})
    ],
        sticky="top",
        color="primary",
        dark=True,
        style={"height": "10vh"}
    ),

    # Actual user position
    dcc.Location(id='user_position', refresh=False),

    # Page content
    html.Div(id='page_content', style={'minHeight': '82vh'}),

    # Navigation bar (bottom)
    dbc.NavbarSimple([
        dbc.NavItem(dbc.NavLink('Load Data', href='/page_1'), style={"marginLeft": "2vw", "marginRight": "5vw"}),

        dbc.DropdownMenu(
            [
                dbc.DropdownMenuItem('Univariate Resemblance Analysis', href='/page_2_ura'),
                dbc.DropdownMenuItem(divider=True),
                dbc.DropdownMenuItem('Multivariate Relationships Analysis', href='/page_2_mra'),
                dbc.DropdownMenuItem(divider=True),
                dbc.DropdownMenuItem('Data Labeling Analysis', href='/page_2_dla'),
            ],
            style={"marginRight": "5vw"},
            menu_variant="dark",
            direction='up',
            nav=True,
            in_navbar=True,
            disabled=True,
            id="nav2",
            label="Resemblance"),

        dbc.NavItem(dbc.NavLink('Utility', href='/page_3', id="nav3", disabled=True), style={"marginRight": "5vw"}),

        dbc.DropdownMenu(
            [
                dbc.DropdownMenuItem('Similarity Evaluation Analysis', href='/page_4_sea'),
                dbc.DropdownMenuItem(divider=True),
                dbc.DropdownMenuItem('Membership Inference Attack', href='/page_4_mia'),
                dbc.DropdownMenuItem(divider=True),
                dbc.DropdownMenuItem('Attribute Inference Attack', href='/page_4_aia'),
            ],
            style={"marginRight": "5vw"},
            menu_variant="dark",
            direction='up',
            nav=True,
            in_navbar=True,
            disabled=True,
            id="nav4",
            label="Privacy"),

        dbc.NavItem(dbc.NavLink('Benchmarking', href='/page_5', id="nav5", disabled=True),
                    style={"marginRight": "3vw"}),

        dbc.NavItem(dbc.Button(["Download Report"], id="button-report", color="primary", disabled=True),
                    id="item-report"),
        dcc.Loading([dcc.Download(id="download-report")], type="circle", fullscreen=True),

    ],
        style={"height": "8vh"},
        color="primary",
        dark=True,
        sticky="bottom",
        links_left=True
    )

])


@app.callback(Output('item-report', 'style'),
              Input('user_position', 'pathname'))
def update_download_visibility(pathname):
    if pathname == "/page_1" or pathname == "/":
        return {'display': 'none'}
    else:
        return {"display": "block", "position": "relative", "left": "20vw"}


@app.callback(Output('page_content', 'children'),
              Input('user_position', 'pathname'))
def display_page(pathname):
    if pathname == '/page_1':
        return page_1
    elif pathname == '/page_2_ura':
        return page_2_ura
    elif pathname == '/page_2_mra':
        return page_2_mra
    elif pathname == '/page_2_dla':
        return page_2_dla
    elif pathname == '/page_3':
        return page_3
    elif pathname == '/page_4_sea':
        return page_4_sea
    elif pathname == '/page_4_mia':
        return page_4_mia
    elif pathname == '/page_4_aia':
        return page_4_aia
    elif pathname == '/page_5':
        return page_5
    else:
        return page_1


# PAGE 1 CONTENTS (load data)
page_1 = html.Div([
    dbc.Container([

        # Upload real dataset
        dbc.Row([dbc.Col([
            html.H2("Real dataset", style={'margin-left': '1vw', 'margin-top': '1vw'}),
            dcc.Upload(
                id='upload-data-real',
                children=html.Div([
                    'Drag and drop the file here or ',
                    html.A('select a file')
                ]),
                style={
                    'width': '80vw',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin-left': '5vw',
                    'margin-right': '5vw'
                },
                multiple=False
            ),
            dcc.Loading(html.Div(id='output-data-upload-real', style={'margin-bottom': '1.5vw'})),
        ])]),

        dbc.Row([dbc.Col(html.Div([html.Br(), html.Br()]), width=True)]),

        # Upload synthetic datasets
        dbc.Row([dbc.Col([
            html.H2("Synthetic datasets", style={'margin-left': '1vw'}),
            dcc.Upload(
                id='upload-data-syn',
                children=html.Div([
                    'Drag and drop the files here or ',
                    html.A('select datasets (even more than one)')
                ]),
                style={
                    'width': '80vw',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin-left': '5vw',
                    'margin-right': '5vw'
                },
                multiple=True
            ),
            dcc.Loading(html.Div(id='output-data-upload-syn', style={'margin-bottom': '1.5vw'})),
        ])]),

        dbc.Row([dbc.Col(html.Div([html.Br(), html.Br()]), width=True)]),

        # Upload features type
        dbc.Row([dbc.Col([
            html.H2(["Features data types "], style={'margin-left': '1vw', 'display': 'inline-block'}),

            html.Span("ℹ", id="info-feature-type", style={'cursor': 'pointer'}),
            dbc.Tooltip(
                "The file must be structured into two columns named 'Feature' and 'Type': the first column should "
                "contain names, while the second should contain a string either 'numerical' or 'categorical'. "
                "Otherwise, select the type using the table below.",
                target="info-feature-type",
            ),

            dcc.Upload(
                id='upload-data-type',
                children=html.Div([
                    'Drag and drop the file here or ',
                    html.A('select a file')
                ]),
                style={
                    'width': '80vw',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin-left': '5vw',
                    'margin-right': '5vw'
                },
                multiple=False
            ),
            html.Div(id='output-data-upload-type', style={'margin-bottom': '1.5vw'}),
            html.Div(id='none-div', style={'display': 'none'})
        ])]),

        dbc.Row([dbc.Col(html.Div([html.Br(), html.Br()]), width=True)]),

    ], fluid=True)
])


@app.callback(Output('output-data-upload-real', 'children'),
              Input('upload-data-real', 'contents'))
def update_table_real(contents):
    global real_data, synthetic_datasets
    if contents is not None:
        real_data = utils.parse_contents(contents)

        global encoder_labels
        encoder_labels = LabelEncoder()

        for col in real_data.columns:
            if real_data[col].dtype == 'object':
                real_data[col] = encoder_labels.fit_transform(real_data[col])

        # correction different dataset sizes
        if synthetic_datasets:
            if real_data.shape[0] < synthetic_datasets[0].shape[0]:
                synthetic_datasets = [df.sample(n=len(real_data), random_state=80).reset_index(drop=True) for df in
                                      synthetic_datasets]
            elif real_data.shape[0] > synthetic_datasets[0].shape[0]:
                real_data = real_data.sample(n=synthetic_datasets[0].shape[0], random_state=80).reset_index(drop=True)

        children = [
            html.Div([
                dash_table.DataTable(
                    data=real_data.head(50).to_dict('records'),
                    columns=[{'name': col, 'id': col} for col in real_data.columns],
                    page_action='none',
                    style_table={'height': '300px',
                                 'width': '80vw',
                                 'margin-left': '5vw',
                                 'margin-right': '5vw',
                                 'overflowY': 'auto',
                                 'overflowX': 'auto'
                                 },
                    fixed_rows={'headers': True},
                    style_header={'text-align': 'center'},
                    style_cell={'minWidth': 100, 'maxWidth': 100, 'width': 100},
                )
            ])
        ]
        return children


@app.callback(Output('output-data-upload-syn', 'children'),
              Input('upload-data-syn', 'contents'))
def update_table_syn(contents):
    global synthetic_datasets, real_data
    if contents is not None:
        with Pool(round(n_cpu * 0.75)) as pool:
            synthetic_datasets = pool.map(utils.parse_contents, contents)

        # correction different synthetic dataset sizes
        synthetic_datasets = [
            df.sample(n=min(len(df) for df in synthetic_datasets), random_state=80).reset_index(drop=True) for df in
            synthetic_datasets]

        encoder_label = LabelEncoder()
        for syn_dataset in synthetic_datasets:
            for col in syn_dataset.columns:
                if syn_dataset[col].dtype == 'object':
                    syn_dataset[col] = encoder_label.fit_transform(syn_dataset[col])

        # correction different dataset sizes
        if not real_data.empty:
            if real_data.shape[0] < synthetic_datasets[0].shape[0]:
                synthetic_datasets = [df.sample(n=len(real_data), random_state=80).reset_index(drop=True) for df in
                                      synthetic_datasets]
            elif real_data.shape[0] > synthetic_datasets[0].shape[0]:
                real_data = real_data.sample(n=synthetic_datasets[0].shape[0], random_state=80).reset_index(drop=True)

        tabs = []
        for i, df in enumerate(synthetic_datasets):
            tab_content = dbc.Card([
                dbc.CardBody([
                    dash_table.DataTable(
                        data=df.head(50).to_dict('records'),
                        columns=[{'name': col, 'id': col} for col in df.columns],
                        page_action='none',
                        style_table={'height': '300px',
                                     'overflowY': 'auto',
                                     'overflowX': 'auto'
                                     },
                        fixed_rows={'headers': True},
                        style_header={'text-align': 'center'},
                        style_cell={'minWidth': 100, 'maxWidth': 100, 'width': 100},
                    )
                ])
            ], color="primary", outline=True)

            tabs.append(dbc.Tab(tab_content, label=f"Dataset {i + 1}", tab_id=f"tab{i + 1}"))

        children = [
            html.Div([dbc.Tabs(tabs)], style={'width': '80vw', 'margin-left': '5vw', 'margin-right': '5vw'})
        ]
        return children


@app.callback(Output('output-data-upload-type', 'children'),
              Input('upload-data-type', 'contents'),
              Input('output-data-upload-real', 'children'))
def update_table_type(contents, children):
    global dict_features_type, num_features, cat_features

    if ctx.triggered[0]['prop_id'] == 'upload-data-type.contents' and contents:
        df = utils.parse_contents(contents)
    elif ctx.triggered[0]['prop_id'] == 'output-data-upload-real.children' and children:
        df = pd.DataFrame({'Feature': real_data.columns.tolist(), 'Type': ''})
        for c in real_data.columns.tolist():
            if (real_data[c].dtype == 'int64' and len(real_data[c].unique()) <= 10) or real_data[c].dtype == 'object':
                df.loc[df['Feature'] == c, 'Type'] = 'categorical'
            else:
                df.loc[df['Feature'] == c, 'Type'] = 'numerical'
    else:
        raise PreventUpdate

    dict_features_type = df.set_index("Feature")["Type"].to_dict()
    num_features = [key for key, value in dict_features_type.items() if value == "numerical"]
    cat_features = [key for key, value in dict_features_type.items() if value == "categorical"]

    children = [
        html.Div([
            dash_table.DataTable(
                id="table-type-dropdown",
                data=df.to_dict('records'),
                columns=[{'name': "Feature", 'id': "Feature"},
                         {'name': "Type", 'id': "Type", 'presentation': 'dropdown'}],
                page_action='none',
                style_table={'height': '300px',
                             'width': '60vw',
                             'margin-left': '15vw',
                             'margin-right': '15vw',
                             'overflowY': 'auto',
                             'overflowX': 'auto'
                             },
                fixed_rows={'headers': True},
                style_header={'text-align': 'center'},
                editable=True,
                dropdown={
                    "Type": {
                        'options': [
                            {'label': 'numerical', 'value': 'numerical'},
                            {'label': 'categorical', 'value': 'categorical'}
                        ],
                        'clearable': False
                    },
                },
                css=[
                    {
                        "selector": ".Select-menu-outer",
                        "rule": 'display : block !important',
                    },
                    {
                        "selector": ".Select-value",
                        "rule": 'max-width : 95%;'
                    }
                ]
            )
        ])
    ]
    return children


@app.callback(Output('none-div', 'children'),
              Input('table-type-dropdown', 'data'),
              prevent_initial_call=True)
def update_dict_type(data):
    global dict_features_type, num_features, cat_features
    d = pd.DataFrame(data).set_index("Feature")["Type"].to_dict()
    dict_features_type = dict(zip(dict_features_type.keys(), d.values()))
    num_features = [key for key, value in dict_features_type.items() if value == "numerical"]
    cat_features = [key for key, value in dict_features_type.items() if value == "categorical"]
    return []


@app.callback(Output('nav2', 'disabled'),
              Output('nav3', 'disabled'),
              Output('nav4', 'disabled'),
              Output('nav5', 'disabled'),
              Input('output-data-upload-real', 'children'),
              Input('output-data-upload-syn', 'children'),
              Input('output-data-upload-type', 'children'))
def active_nav(r, s, t):
    if not real_data.empty and synthetic_datasets and dict_features_type:
        return False, False, False, False
    else:
        return True, True, True, True


# PAGE 2 CONTENTS (resemblance metrics)

# URA metrics
page_2_ura = html.Div([

    dbc.Container(
        [
            # header URA section
            dbc.Row(
                dbc.Col(html.H3("Univariate Resemblance Analysis",
                                style={'margin-left': '1vw', 'margin-top': '1vw', 'margin-bottom': '2vw'}),
                        width="auto")),

            # Dropdown URA numerical test
            dbc.Row(
                [
                    dbc.Col(html.Div(
                        [
                            dbc.Label("Choose statistical test", html_for="dropdown"),
                            dcc.Dropdown(
                                id="dropdown-test-num",
                                options=[
                                    {"label": "Kolmogorov–Smirnov test", "value": "ks_test"},
                                    {"label": "Student T-test", "value": "t_test"},
                                    {"label": "Mann Whitney U-test", "value": "u_test"},
                                ],
                                clearable=False
                            ),
                        ],
                        className="mb-3",
                    ), width={'size': 3, 'offset': 1}),

                    dbc.Col(html.Div(["Features type: ", html.I("numerical")]), width={'size': 'auto'}, align="center")
                ]
            ),

            # Table and graphs URA numerical test
            dbc.Row(
                [
                    dbc.Col(dcc.Loading(html.Div(id="output-table-pvalue-num")), width={'size': 3, 'offset': 1},
                            align="center"),
                    dbc.Col(html.Div(id="output-graph-num"), width={'size': 5, 'offset': 2}, align="center")
                ],
                style={'margin-bottom': '2vw'}),
            dcc.Store(id={"type": "data-report", "index": 0}),

            # Dropdown URA categorical test
            dbc.Row(
                [
                    dbc.Col(html.Div(
                        [
                            dbc.Label("Choose statistical test", html_for="dropdown"),
                            dcc.Dropdown(
                                id="dropdown-test-cat",
                                options=[
                                    {"label": "Chi-square test", "value": "chi_test"},
                                ],
                                clearable=False
                            ),
                        ],
                        className="mb-3",
                    ), width={'size': 3, 'offset': 1}),

                    dbc.Col(html.Div(["Features type: ", html.I("categorical")]), width={'size': 'auto'},
                            align="center")
                ]
            ),

            # Table and graphs URA categorical test
            dbc.Row(
                [
                    dbc.Col(dcc.Loading(html.Div(id="output-table-pvalue-cat")), width={'size': 3, 'offset': 1},
                            align="center"),
                    dbc.Col(html.Div(id="output-graph-cat"), width={'size': 5, 'offset': 2}, align="center")
                ],
                style={'margin-bottom': '2vw'}),
            dcc.Store(id={"type": "data-report", "index": 1}),

            # Dropdown URA metrics distance
            dbc.Row(
                [
                    dbc.Col(html.Div(
                        [
                            dbc.Label("Choose distance metric", html_for="dropdown"),
                            dcc.Dropdown(
                                id="dropdown-dist",
                                options=[
                                    {"label": "Cosine distance", "value": "cos_dist"},
                                    {"label": "Jensen-Shannon distance", "value": "js_dist"},
                                    {"label": "Wasserstein distance", "value": "w_dist"},
                                ],
                                clearable=False
                            ),
                        ],
                        className="mb-3",
                    ), width={'size': 3, 'offset': 1}),

                    dbc.Col(html.Div(["Features type: ", html.I("numerical")]), width={'size': 'auto'},
                            align="center")
                ]
            ),

            # Table URA distances
            dbc.Row(
                [
                    dbc.Col(dcc.Loading(html.Div(id="output-table-dist")), width={'size': 3, 'offset': 1},
                            align="center"),
                ],
            ),
            dcc.Store(id={"type": "data-report", "index": 2}),

            dbc.Row([dbc.Col(html.Div([html.Br(), html.Br()]), width=True)])

        ], fluid=True
    )
])


@app.callback(Output('output-table-pvalue-num', 'children'),
              Output({"type": "data-report", "index": 0}, 'data'),
              Input('dropdown-test-num', 'value'),
              prevent_initial_call=True)
def update_table_pvalue_num(value):
    if value is not None:

        if value == 'ks_test':
            with Pool(round(n_cpu * 0.75)) as pool:
                dict_pvalues = pool.starmap(ResMet.URA.ks_tests, zip(repeat(real_data[num_features]), synthetic_datasets))
        elif value == 't_test':
            with Pool(round(n_cpu * 0.75)) as pool:
                dict_pvalues = pool.starmap(ResMet.URA.student_t_tests, zip(repeat(real_data[num_features]), synthetic_datasets))
        elif value == 'u_test':
            with Pool(round(n_cpu * 0.75)) as pool:
                dict_pvalues = pool.starmap(ResMet.URA.mann_whitney_tests,
                                            zip(repeat(real_data[num_features]), synthetic_datasets))

        dfs = [pd.DataFrame(list(d.items()), columns=['Feature', 'p value']) for d in dict_pvalues]

        tabs = []
        for i, df in enumerate(dfs):
            tab_content = dbc.Card([
                dbc.CardBody([
                    dash_table.DataTable(
                        data=df.to_dict('records'),
                        columns=[{'name': "Feature", 'id': "Feature"}, {'name': "p value", 'id': "p value"}],
                        page_action='none',
                        style_table={'height': '300px',
                                     'overflowY': 'auto',
                                     },
                        fixed_rows={'headers': True},
                        style_header={'text-align': 'center'},
                        style_cell={'minWidth': '50%', 'maxWidth': '50%', 'width': '50%'},
                        style_data_conditional=[
                            {
                                'if': {
                                    'filter_query': '{p value} <= 0.05',
                                },
                                'backgroundColor': '#e74c3c',
                                'color': 'white'
                            },
                            {
                                'if': {
                                    'filter_query': '{p value} > 0.05',
                                },
                                'backgroundColor': '#18bc9c',
                                'color': 'white'
                            },
                            {
                                'if': {'state': 'active'},
                                'backgroundColor': '#003153',
                                'border': '1px solid blue'
                            }
                        ],
                        id={"type": "tbl-pvalue-num", "index": i}
                    )
                ])
            ], color="primary", outline=True)

            tabs.append(dbc.Tab(tab_content, label=f"Dataset {i + 1}", tab_id=f"tab{i + 1}"))

        children = [
            html.Div([dbc.Tabs(tabs, active_tab="tab1")])
        ]

        return children, [dict_pvalues, {'selected_opt': value}]

    else:
        return [], []


@app.callback(Output('output-table-pvalue-cat', 'children'),
              Output({"type": "data-report", "index": 1}, 'data'),
              Input('dropdown-test-cat', 'value'),
              prevent_initial_call=True)
def update_table_pvalue_cat(value):
    if value is not None:

        if value == 'chi_test':
            with Pool(round(n_cpu * 0.75)) as pool:
                dict_pvalues = pool.starmap(ResMet.URA.chi_squared_tests, zip(repeat(real_data[cat_features]), synthetic_datasets))

        dfs = [pd.DataFrame(list(d.items()), columns=['Feature', 'p value']) for d in dict_pvalues]

        tabs = []
        for i, df in enumerate(dfs):
            tab_content = dbc.Card([
                dbc.CardBody([
                    dash_table.DataTable(
                        data=df.to_dict('records'),
                        columns=[{'name': "Feature", 'id': "Feature"}, {'name': "p value", 'id': "p value"}],
                        page_action='none',
                        style_table={'height': '300px',
                                     'overflowY': 'auto',
                                     },
                        fixed_rows={'headers': True},
                        style_header={'text-align': 'center'},
                        style_cell={'minWidth': '50%', 'maxWidth': '50%', 'width': '50%'},
                        style_data_conditional=[
                            {
                                'if': {
                                    'filter_query': '{p value} > 0.05',
                                },
                                'backgroundColor': '#e74c3c',
                                'color': 'white'
                            },
                            {
                                'if': {
                                    'filter_query': '{p value} <= 0.05',
                                },
                                'backgroundColor': '#18bc9c',
                                'color': 'white'
                            },
                            {
                                'if': {'state': 'active'},
                                'backgroundColor': '#003153',
                                'border': '1px solid blue'
                            }
                        ],
                        id={"type": "tbl-pvalue-cat", "index": i}
                    )
                ])
            ], color="primary", outline=True)

            tabs.append(dbc.Tab(tab_content, label=f"Dataset {i + 1}", tab_id=f"tab{i + 1}"))

        children = [
            html.Div([dbc.Tabs(tabs, active_tab="tab1")])
        ]

        return children, [dict_pvalues, {'selected_opt': value}]

    else:
        return [], []


@app.callback(Output('output-table-dist', 'children'),
              Output({"type": "data-report", "index": 2}, 'data'),
              Input('dropdown-dist', 'value'),
              prevent_initial_call=True)
def update_table_dist(value):
    if value is not None:

        if value == 'cos_dist':
            with Pool(round(n_cpu * 0.75)) as pool:
                dict_distances = pool.starmap(ResMet.URA.cosine_distances,
                                              zip(repeat(real_data[num_features]), synthetic_datasets))
        elif value == 'js_dist':
            with Pool(round(n_cpu * 0.75)) as pool:
                dict_distances = pool.starmap(ResMet.URA.js_distances,
                                              zip(repeat(real_data[num_features]), synthetic_datasets))
        elif value == 'w_dist':
            with Pool(round(n_cpu * 0.75)) as pool:
                dict_distances = pool.starmap(ResMet.URA.wass_distances,
                                              zip(repeat(real_data[num_features]), synthetic_datasets))

        dfs = [pd.DataFrame(list(d.items()), columns=['Feature', 'Distance value']) for d in dict_distances]

        tabs = []
        for i, df in enumerate(dfs):
            tab_content = dbc.Card([
                dbc.CardBody([
                    dash_table.DataTable(
                        data=df.to_dict('records'),
                        columns=[{'name': "Feature", 'id': "Feature"},
                                 {'name': "Distance value", 'id': "Distance value"}],
                        page_action='none',
                        sort_action="native",
                        sort_mode="multi",
                        style_table={'height': '300px',
                                     'overflowY': 'auto',
                                     },
                        fixed_rows={'headers': True},
                        style_header={'text-align': 'center'},
                        style_cell={'minWidth': '50%', 'maxWidth': '50%', 'width': '50%'},
                        id='tbl-dist'
                    )
                ])
            ], color="primary", outline=True)

            tabs.append(dbc.Tab(tab_content, label=f"Dataset {i + 1}", tab_id=f"tab{i + 1}"))

        children = [
            html.Div([dbc.Tabs(tabs, active_tab="tab1")])
        ]

        return children, [dict_distances, {'selected_opt': value}]

    else:
        return [], []


def create_fig_ura(col_real, col_syn, feature_name, feature_type):
    if feature_type == "numerical":
        fig_real = ff.create_distplot([col_real], ['Original Data'], show_hist=False, show_rug=False, colors=['blue'])
        fig_syn = ff.create_distplot([col_syn], ['Synthetic Data'], show_hist=False, show_rug=False, colors=['red'])
        fig_real.update_traces(fill='tozeroy', fillcolor='rgba(0, 0, 255, 0.5)', selector=dict(type='scatter'))
        fig_syn.update_traces(fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.5)', selector=dict(type='scatter'))
        fig = fig_real.add_traces(fig_syn.data)

    else:
        plot_data = pd.DataFrame({
            'value': col_real + col_syn,
            'variable': ['Original Data'] * len(col_real) + ['Synthetic Data'] * len(col_syn)
        })
        fig = px.histogram(plot_data, x="value", color='variable', barmode='group', histnorm='percent',
                           color_discrete_map={'Original Data': 'blue', 'Synthetic Data': 'red'}, opacity=0.9)

    fig.update_layout(title_text=feature_name, showlegend=True)

    return fig


@app.callback(Output('output-graph-num', 'children'),
              Input({"type": "tbl-pvalue-num", "index": ALL}, 'active_cell'),
              prevent_initial_call=True)
def update_graphs(active_cell):
    if any(active_cell):

        selected_feature = num_features[active_cell[ctx.triggered_id.index]['row']]

        col_real = real_data[selected_feature].tolist()
        col_syn = synthetic_datasets[ctx.triggered_id.index][selected_feature].tolist()

        fig = create_fig_ura(col_real, col_syn, selected_feature, "numerical")

        children = [
            html.Div([
                dcc.Graph(figure=fig)
            ])
        ]

    else:
        children = [
            html.Div([
                dcc.Graph(
                    figure={
                        'data': [],
                        'layout': {
                            'annotations': [{
                                'x': 0.5,
                                'y': 0.5,
                                'xref': 'paper',
                                'yref': 'paper',
                                'text': 'Click on an adjacent table cell to view the corresponding chart,<br>which '
                                        'compares real data with synthetic data for the selected feature.',
                                'showarrow': False,
                                'font': {'size': 16},
                                'xanchor': 'center',
                                'yanchor': 'middle',
                                'bgcolor': 'rgba(255, 255, 255, 0.7)',
                                'bordercolor': 'rgba(0, 0, 0, 0.2)',
                                'borderwidth': 1,
                                'borderpad': 4,
                                'borderline': {'width': 2}
                            }],
                            'xaxis': {'visible': False},
                            'yaxis': {'visible': False},
                            'template': 'plotly_white'
                        }
                    },
                    config={'displayModeBar': False}
                )
            ])
        ]

    return children


@app.callback(Output('output-graph-cat', 'children'),
              Input({"type": "tbl-pvalue-cat", "index": ALL}, 'active_cell'),
              prevent_initial_call=True)
def update_graphs(active_cell):
    if any(active_cell):

        selected_feature = cat_features[active_cell[ctx.triggered_id.index]['row']]

        col_real = real_data[selected_feature].tolist()
        col_syn = synthetic_datasets[ctx.triggered_id.index][selected_feature].tolist()

        fig = create_fig_ura(col_real, col_syn, selected_feature, "categorical")

        children = [
            html.Div([
                dcc.Graph(figure=fig)
            ])
        ]

    else:
        children = [
            html.Div([
                dcc.Graph(
                    figure={
                        'data': [],
                        'layout': {
                            'annotations': [{
                                'x': 0.5,
                                'y': 0.5,
                                'xref': 'paper',
                                'yref': 'paper',
                                'text': 'Click on an adjacent table cell to view the corresponding chart,<br>which '
                                        'compares real data with synthetic data for the selected feature.',
                                'showarrow': False,
                                'font': {'size': 16},
                                'xanchor': 'center',
                                'yanchor': 'middle',
                                'bgcolor': 'rgba(255, 255, 255, 0.7)',
                                'bordercolor': 'rgba(0, 0, 0, 0.2)',
                                'borderwidth': 1,
                                'borderpad': 4,
                                'borderline': {'width': 2}
                            }],
                            'xaxis': {'visible': False},
                            'yaxis': {'visible': False},
                            'template': 'plotly_white'
                        }
                    },
                    config={'displayModeBar': False}
                )
            ])
        ]

    return children


# MRA metrics
page_2_mra = html.Div([

    dbc.Container(
        [
            # Header MRA section
            dbc.Row(
                dbc.Col(html.H3("Multivariate Relationship Analysis",
                                style={'margin-left': '1vw', 'margin-top': '1vw', 'margin-bottom': '1vw'}),
                        width="auto")),

            dbc.Row(
                dbc.Col(html.H4("Comparison correlation matrices",
                                style={'margin-left': '1.5vw', 'margin-top': '2vw', 'margin-bottom': '1.5vw'}),
                        width="auto")),

            # Dropdown numerical matrices/categorical matrices, Radio for graphical visualization
            dbc.Row(
                [
                    dbc.Col(html.Div(
                        [
                            dbc.Label("Choose correlation matrix type", html_for="dropdown"),
                            dcc.Dropdown(
                                id="dropdown-corr-mat",
                                options=[
                                    {"label": "Pairwise Pearson correlation matrices", "value": "corr_num"},
                                    {"label": "Normalized contingency tables", "value": "corr_cat"},
                                ],
                                clearable=False
                            ),
                        ],
                        className="mb-3",
                    ), width={'size': 3, 'offset': 1}),

                    dbc.Col(html.Div(["Features type: ", html.I(id="label-type")]),
                            width={'size': 2}, align="center"),

                    dbc.Col(html.Div(
                        [
                            dbc.RadioItems(
                                id="radios-mat",
                                className="btn-group",
                                inputClassName="btn-check",
                                labelClassName="btn btn-outline-primary",
                                labelCheckedClassName="active",
                                options=[
                                    {"label": "Real vs Syn", "value": "rs"},
                                    {"label": "Differences", "value": "diff"},
                                ],
                                value="rs",
                            )
                        ], className="btn-group"
                    ), width={'size': 3}, align="center")
                ]
            ),

            # Correlation matrices
            dbc.Row(
                [
                    dbc.Col(dcc.Loading(html.Div(id="output-corr-mat")), width={'size': 8, 'offset': 2}, align="center")
                ],
            ),
            dcc.Store(id={"type": "data-report", "index": 3}),

            dbc.Row(
                dbc.Col([html.H4(["Comparison Outliers ",
                                  html.Span("ℹ", id="info-outlier", style={'cursor': 'pointer'})],
                                 style={'margin-left': '1.5vw', 'margin-top': '7vw', 'margin-bottom': '0vw'},
                                 id="title-outlier"),
                         dbc.Tooltip(
                             "Unsupervised Outlier Detection using the Local Outlier Factor. It measures the "
                             "local deviation of the density of a given sample with respect to its neighbors.",
                             target="info-outlier",
                         )],
                        width="auto")),

            # Boxplot LOF scores
            dbc.Row(
                [
                    dbc.Col(dcc.Loading(html.Div(id="output-boxplot")), width={'size': 8, 'offset': 2}, align="center")
                ],
            ),
            dcc.Store(id={"type": "data-report", "index": 4}),

            dbc.Row(
                dbc.Col(html.H4("Comparison Principal Component Analysis",
                                style={'margin-left': '1.5vw', 'margin-top': '7vw', 'margin-bottom': '0vw'},
                                id="title-pca"),
                        width="auto")),

            # PCA
            dcc.Loading(dbc.Row(
                [
                    dbc.Col(html.Div(id="output-pca"), width={'size': 6, 'offset': 1}, align="center"),
                    dbc.Col(html.Div(id="output-table-pca"), width={'size': 3, 'offset': 1}, align="center")
                ],
            )),
            dcc.Store(id={"type": "data-report", "index": 5}),

            dbc.Row(
                dbc.Col(html.H4("Comparison UMAP visualization",
                                style={'margin-left': '1.5vw', 'margin-top': '7vw', 'margin-bottom': '1.5vw'}),
                        width="auto")),

            # Radio for UMAP visualization, input meta-parameters
            dbc.Form(
                dbc.Row(
                    [
                        dbc.Col(html.Div(
                            [
                                dbc.RadioItems(
                                    id="radios-umap",
                                    className="btn-group",
                                    inputClassName="btn-check",
                                    labelClassName="btn btn-outline-primary",
                                    labelCheckedClassName="active",
                                    options=[
                                        {"label": "Real vs Syn", "value": "rs"},
                                        {"label": "Together", "value": "tog"},
                                    ],
                                    value="rs",
                                )
                            ], className="radio-group"
                        ), width={'size': 'auto', 'offset': 1}, align="end"),

                        dbc.Col(html.Div(
                            [
                                html.P("Type the number of neighboring"),
                                dbc.Input(type="number", min=2, step=1,
                                          value=20, required="required", id="num-neighbors-input"),
                            ]
                        ), width={'size': 'auto', 'offset': 1}),

                        dbc.Col(html.Div(
                            [
                                html.P("Type min_dist parameter value"),
                                dbc.Input(type="number", min=0, max=1, step=0.05,
                                          value=0.1, required="required", id="min-dist-input"),
                            ]
                        ), width={'size': 'auto'}),

                        dbc.Col(html.Div(
                            [
                                dbc.Button("Run UMAP", color="info", id="run-umap"),
                            ]
                        ), width={'size': 'auto'}, align="end"),
                    ]
                )
            ),

            dbc.Row([dbc.Col(html.Div([html.Br()]), width=True)]),

            # Graphs UMAP
            dcc.Loading(dbc.Row(
                [
                    dbc.Col(html.Div(id="output-umap"), width={'size': 8, 'offset': 2}, align="center"),
                ],
            ), id={"type": "load-res", "index": 1}),
            dcc.Store(id={"type": "data-report", "index": 6}),

            dbc.Row([dbc.Col(html.Div([html.Br(), html.Br()]), width=True)])

        ], fluid=True
    )
])


@app.callback(Output('label-type', 'children'),
              Input('dropdown-corr-mat', 'value'))
def change_label_type(value):
    if value is not None:
        if value == "corr_num":
            return ["numerical"]
        elif value == "corr_cat":
            return ["categorical"]


@app.callback(Output('output-corr-mat', 'children'),
              Output({"type": "data-report", "index": 3}, 'data'),
              Input('dropdown-corr-mat', 'value'),
              Input('radios-mat', 'value'),
              prevent_initial_call=True)
def update_graphs(value, radio):
    if value is not None:

        if value == "corr_num":
            mat_real = ResMet.MRA.compute_correlations_matrices(real_data, value, num_features)

            with Pool(round(n_cpu * 0.75 / 3)) as pool:
                mats_syns = pool.starmap(ResMet.MRA.compute_correlations_matrices,
                                         zip(synthetic_datasets, repeat(value), repeat(num_features)))

        elif value == "corr_cat":
            mat_real = ResMet.MRA.compute_correlations_matrices(real_data, value, cat_features)

            with Pool(round(n_cpu * 0.75 / 3)) as pool:
                mats_syns = pool.starmap(ResMet.MRA.compute_correlations_matrices,
                                         zip(synthetic_datasets, repeat(value), repeat(cat_features)))

        if radio == "rs":
            fig1 = px.imshow(mat_real, aspect="auto")
            fig1.update_layout(title_text='Real Data')

            tabs = []
            figs = []
            for i, mat in enumerate(mats_syns):
                fig2 = px.imshow(mat, aspect="auto")
                fig2.update_layout(title_text='Synthetic Data')

                tab_content = dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(figure=fig2)
                    ])
                ], color="primary", outline=True)

                tabs.append(dbc.Tab(tab_content, label=f"Dataset {i + 1}", tab_id=f"tab{i + 1}"))

                figs.append([fig1, fig2])

            children = [
                html.Div([
                    dbc.Row([dbc.Col(html.Div([html.Br()]), width=True)]),
                    dbc.Row([
                        dbc.Col(dcc.Graph(figure=fig1), width=6, align="center"),
                        dbc.Col(dbc.Tabs(tabs, active_tab="tab1"), width=6)
                    ])
                ])
            ]

            return children, [figs, value, radio]

        else:  # "diff"
            mats_diff = [np.abs(mat_real - mat_syn) for mat_syn in mats_syns]

            tabs = []
            figs = []
            for i, mat in enumerate(mats_diff):
                fig = px.imshow(mat)
                figs.append(fig)

                tab_content = dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(figure=fig)
                    ])
                ], color="primary", outline=True)

                tabs.append(dbc.Tab(tab_content, label=f"Dataset {i + 1}", tab_id=f"tab{i + 1}"))

            children = [
                html.Div([
                    dbc.Row([dbc.Col(html.Div([html.Br()]), width=True)]),
                    dbc.Tabs(tabs, active_tab="tab1")
                ])
            ]

            return children, [figs, value, radio]

    else:
        children = [
            html.Div([
                dcc.Graph(
                    figure={
                        'data': [],
                        'layout': {
                            'annotations': [{
                                'x': 0.5,
                                'y': 0.5,
                                'xref': 'paper',
                                'yref': 'paper',
                                'text': 'Choose a correlation matrix type to view the corresponding chart,<br>which '
                                        'compares real data with synthetic data.',
                                'showarrow': False,
                                'font': {'size': 16},
                                'xanchor': 'center',
                                'yanchor': 'middle',
                                'bgcolor': 'rgba(255, 255, 255, 0.7)',
                                'bordercolor': 'rgba(0, 0, 0, 0.2)',
                                'borderwidth': 1,
                                'borderpad': 4,
                                'borderline': {'width': 2}
                            }],
                            'xaxis': {'visible': False},
                            'yaxis': {'visible': False},
                            'template': 'plotly_white'
                        }
                    },
                    config={'displayModeBar': False}
                )
            ])
        ]
        return children, []


@app.callback(Output('output-boxplot', 'children'),
              Output({"type": "data-report", "index": 4}, 'data'),
              Input('title-outlier', 'children'))
def update_graphs(children_in):
    if children_in is not None:

        neg_lof_score_real = ResMet.MRA.check_lof(real_data)

        with Pool(round(n_cpu * 0.75 / 3)) as pool:
            neg_lof_score_syns = pool.map(ResMet.MRA.check_lof, synthetic_datasets)

        tabs = []
        figs = []
        for i, score_syn in enumerate(neg_lof_score_syns):
            fig = go.Figure()

            fig.add_trace(go.Violin(x=neg_lof_score_real,
                                    name='Real',
                                    side='positive',
                                    line_color='blue'))

            fig.add_trace(go.Violin(x=score_syn,
                                    name='Synthetic',
                                    side='negative',
                                    line_color='red'))

            fig.update_traces(meanline_visible=True, y0=0)
            fig.update_layout(violinmode='overlay', xaxis_title="negative LOF score", yaxis=dict(showticklabels=False))

            tab_content = dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fig)
                ])
            ], color="primary", outline=True)

            tabs.append(dbc.Tab(tab_content, label=f"Dataset {i + 1}", tab_id=f"tab{i + 1}"))

            figs.append(fig)

        children = [
            html.Div([
                dbc.Row([dbc.Col(html.Div([html.Br()]), width=True)]),
                dbc.Tabs(tabs, active_tab="tab1")
            ])
        ]

        return children, fig


@app.callback(Output({"type": "data-report", "index": 5}, 'data'),
              Output('output-pca', 'children'),
              Input('title-pca', 'children'))
def update_graphs(children_in):
    if children_in is not None:

        real_var_ratio_cum = ResMet.MRA.do_pca(real_data)

        with Pool(round(n_cpu * 0.75 / 3)) as pool:
            syns_var_ratio_cum = pool.map(ResMet.MRA.do_pca, synthetic_datasets)

        components = list(range(1, len(real_var_ratio_cum) + 1))

        tabs = []
        figs = []
        for i, syn_var in enumerate(syns_var_ratio_cum):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=components, y=real_var_ratio_cum, mode='lines+markers', name="Real"))
            fig.add_trace(go.Scatter(x=components, y=syn_var, mode='lines+markers', name="Synthetic"))

            fig.update_layout(xaxis_title="Components", yaxis_title="Explained variance ratio (%)")
            fig.update_layout(xaxis=dict(tickmode='array', tickvals=components))

            tab_content = dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fig)
                ])
            ], color="primary", outline=True)

            tabs.append(dbc.Tab(tab_content, label=f"Dataset {i + 1}", tab_id=f"tab{i + 1}"))

            figs.append(fig)

        children = [
            html.Div([
                dbc.Row([dbc.Col(html.Div([html.Br()]), width=True)]),
                dbc.Tabs(tabs, active_tab="tab1", id="tabs-pca")
            ])
        ]

        diffs = np.round(abs(np.array(real_var_ratio_cum) - np.array(syns_var_ratio_cum)), 2)

        return [figs, {'x': components, 'diffs': diffs}], children


@app.callback(Output('output-table-pca', 'children'),
              Input('tabs-pca', 'active_tab'),
              State({"type": "data-report", "index": 5}, 'data'))
def update_table_pca(id_tab, data):
    if data is not None:
        components = data[1]['x']
        diff = data[1]['diffs'][int(id_tab[-1]) - 1]

        df = pd.DataFrame(list(zip(components, diff)), columns=['Component', 'Difference (%)'])

        children = [
            html.Div([
                dash_table.DataTable(
                    data=df.to_dict('records'),
                    columns=[{'name': "Component", 'id': "Component"},
                             {'name': "Difference (%)", 'id': "Difference (%)"}],
                    page_action='none',
                    style_table={'height': '40vh',
                                 'overflowY': 'auto',
                                 },
                    fixed_rows={'headers': True},
                    style_header={'text-align': 'center'},
                    style_cell={'minWidth': '50%', 'maxWidth': '50%', 'width': '50%'},
                )
            ])
        ]
        return children


@app.callback(Output('output-umap', 'children'),
              Output({"type": "data-report", "index": 6}, 'data'),
              [Input('run-umap', 'n_clicks')],
              [State('num-neighbors-input', 'value'),
               State('min-dist-input', 'value'),
               State('radios-umap', 'value')])
def run_code_on_click(n_clicks, num_neighbors, min_dist, radio):
    if n_clicks is not None and num_neighbors is not None and min_dist is not None:
        if radio == "rs":

            with Pool(round(n_cpu * 0.90 / 3)) as pool:
                embeddings = pool.starmap(ResMet.MRA.do_umap,
                                          zip([real_data] + synthetic_datasets, repeat(num_neighbors),
                                              repeat(min_dist)))

            embedding_real = embeddings[0]
            embeddings_syn = embeddings[1:]

            fig1 = go.Figure()
            fig1 = fig1.add_trace(go.Scatter(x=embedding_real[:, 0], y=embedding_real[:, 1],
                                             mode='markers', marker_color="blue"))
            fig1.update_layout(title_text='Real Data')

            tabs = []
            figs = []
            for i, emb_syn in enumerate(embeddings_syn):
                fig2 = go.Figure()
                fig2 = fig2.add_trace(go.Scatter(x=emb_syn[:, 0], y=emb_syn[:, 1],
                                                 mode='markers', marker_color="red"))
                fig2.update_layout(title_text='Synthetic Data')

                tab_content = dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(figure=fig2)
                    ])
                ], color="primary", outline=True)

                tabs.append(dbc.Tab(tab_content, label=f"Dataset {i + 1}", tab_id=f"tab{i + 1}"))

                figs.append([fig1, fig2])

            children = [
                html.Div([
                    dbc.Row([dbc.Col(html.Div([html.Br()]), width=True)]),
                    dbc.Row([
                        dbc.Col(dcc.Graph(figure=fig1), width=6, align="center"),
                        dbc.Col(dbc.Tabs(tabs, active_tab="tab1"), width=6)
                    ])
                ])
            ]

        else:

            concatenated_dfs = []
            for syn_data in synthetic_datasets:
                concatenated_dfs.append(pd.concat([real_data, syn_data], ignore_index=True))

            with Pool(round(n_cpu * 0.90 / 3)) as pool:
                embeddings = pool.starmap(ResMet.MRA.do_umap,
                                          zip(concatenated_dfs, repeat(num_neighbors), repeat(min_dist)))

            tabs = []
            figs = []
            for i, emb in enumerate(embeddings):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=emb[0:real_data.shape[0], 0], y=emb[0:real_data.shape[0], 1],
                                         mode='markers', name="Real"))
                fig.add_trace(go.Scatter(x=emb[real_data.shape[0]:, 0], y=emb[real_data.shape[0]:, 1],
                                         mode='markers', name="Synthetic"))

                tab_content = dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(figure=fig)
                    ])
                ], color="primary", outline=True)

                tabs.append(dbc.Tab(tab_content, label=f"Dataset {i + 1}", tab_id=f"tab{i + 1}"))

                figs.append(fig)

            children = [
                html.Div([
                    dbc.Row([dbc.Col(html.Div([html.Br()]), width=True)]),
                    dbc.Tabs(tabs, active_tab="tab1")
                ])
            ]

        return children, [figs, num_neighbors, min_dist, radio]

    else:
        return [], []


# DLA metrics
page_2_dla = html.Div([

    dbc.Container(
        [
            # Header DLA section
            dbc.Row(
                dbc.Col(html.H3("Data Labeling Analysis",
                                style={'margin-left': '1vw', 'margin-top': '1vw', 'margin-bottom': '1vw'}),
                        width="auto")),

            # Dropdown for classifiers selection
            dbc.Form([
                dbc.Row([
                    dbc.Col(html.Div([
                        dbc.Label("Select different classifiers", html_for="dropdown"),
                        dcc.Dropdown(
                            id="dropdown-dla",
                            options=[
                                {"label": "Random Forest", "value": "RF"},
                                {"label": "K-Nearest Neighbors", "value": "KNN"},
                                {"label": "Decision Tree", "value": "DT"},
                                {"label": "Support Vector Machines", "value": "SVM"},
                                {"label": "Multilayer Perceptron", "value": "MLP"},
                            ],
                            clearable=False,
                            multi=True
                        ),
                    ],
                        className="mb-3",
                    ), width={'size': 2, 'offset': 1}),

                    dbc.Col(html.Div([
                        dbc.Button("Run DLA", color="info", id="run-dla")
                    ]), style={'margin-left': '1.5vw', 'margin-top': '1vw'}, align="center")
                ]),
            ]),

            dcc.Loading(dbc.Row([
                dcc.Store(id={"type": "data-report", "index": 7}),

                dbc.Col([
                    dbc.Row([
                        html.H4("Classifier performance metrics",
                                style={'margin-left': '1.5vw', 'margin-top': '2vw', 'margin-bottom': '1vw'},
                                id="title-dla")
                    ]),
                    dbc.Row(id="sec-dla",
                            style={'margin-left': '2vw'}),
                ], width={'size': 5}),

                # Boxplot results
                dbc.Col([html.Div(id="output-boxplot-dla")], width={'size': 7}, align='center'),

            ])),

            dbc.Row([dbc.Col(html.Div([html.Br(), html.Br()]), width=True)])

        ], fluid=True
    )
])


@app.callback(Output('sec-dla', 'children'),
              Output('output-boxplot-dla', 'children'),
              Output({"type": "data-report", "index": 7}, 'data'),
              Input('run-dla', 'n_clicks'),
              State('dropdown-dla', 'value'),
              prevent_initial_call=True)
def fill_with_results(n_clicks, selected_models):
    if n_clicks is not None and selected_models is not None:

        with Pool(round(n_cpu * 0.75)) as pool:
            results_dla = pool.starmap(ResMet.DLA.classify_real_vs_synthetic_data,
                                       zip(repeat(selected_models), repeat(real_data.copy()), synthetic_datasets.copy(),
                                           repeat(num_features), repeat(cat_features)))

        tabs = []
        report_data = []
        for i, result in enumerate(results_dla):

            report_data.append(result.to_dict("list"))

            child = []
            for model in selected_models:
                prov_child = [
                    html.H5(model,
                            style={'margin-top': '2vw', 'margin-bottom': '1vw'}),
                    dbc.Row([
                        dbc.Col(html.Div([
                            html.B("Accuracy: "),
                            html.P(result[result['model'] == model]['accuracy'])
                        ])),
                        dbc.Col(html.Div([
                            html.B("Precision: "),
                            html.P(result[result['model'] == model]['precision'])
                        ])),
                        dbc.Col(html.Div([
                            html.B("Recall: "),
                            html.P(result[result['model'] == model]['recall'])
                        ])),
                        dbc.Col(html.Div([
                            html.B("F1-score: "),
                            html.P(result[result['model'] == model]['f1'])
                        ])),
                    ]),
                ]
                child = child + prov_child

            tab_content = dbc.Card([
                dbc.CardBody(child)
            ], color="primary", outline=True)

            tabs.append(dbc.Tab(tab_content, label=f"Dataset {i + 1}", tab_id=f"tab{i + 1}"))

        children_tbl = [
            html.Div([
                dbc.Row([dbc.Col(html.Div([html.Br()]), width=True)]),
                dbc.Tabs(tabs, active_tab="tab1")
            ])
        ]

        fig = go.Figure()
        for i, d in enumerate(report_data):
            df = pd.DataFrame(d)
            fig.add_trace(go.Box(
                y=df.values.flatten('F'),
                x=list(df.columns.repeat(len(df))),
                name=f"Dataset {i + 1}",
            ))

        fig.update_layout(
            yaxis_title='metrics values',
            boxmode='group'
        )

        children_fig = [
            dcc.Graph(figure=fig)
        ]

        return children_tbl, children_fig, [fig, report_data]
    else:
        raise PreventUpdate


# PAGE 3 CONTENTS (utility metrics)
page_3 = html.Div([

    dbc.Container(
        [
            # Header Utility section
            dbc.Row(
                dbc.Col([html.H4(["Utility Evaluation "],
                                 style={'margin-left': '2vw', 'margin-top': '1vw', 'margin-bottom': '2vw'},
                                 )],
                        width="auto")),

            # Upload train and test datasets
            dbc.Row([

                dbc.Col([
                    html.P(["Import datasets: ",
                            html.Span("ℹ", id="info-utl", style={'cursor': 'pointer'})]),
                    dbc.Tooltip(
                        "Import the training and testing datasets, otherwise, a random split will be performed on "
                        "the existing real dataset already imported.",
                        target="info-utl",
                    )
                ], width={'size': 2}),

                dbc.Col([
                    dcc.Upload([dbc.Button(["Upload Train"], id="button-train", color="primary")],
                               id="upload-data-train-utl",
                               multiple=False)
                ], width={'size': 'auto'}),

                dbc.Col([
                    dcc.Upload([dbc.Button(["Upload Test"], id="button-test", color="primary")],
                               id="upload-data-test-utl",
                               multiple=False),
                ], width={'size': 'auto'}),

            ], style={'margin-bottom': '1vw', 'margin-left': '4vw'}),

            dbc.Modal(
                [
                    dbc.ModalHeader("Error"),
                    dbc.ModalBody("The uploaded train dataset does not have the same columns as the real dataset. "
                                  "Please load another file."),
                    dbc.ModalFooter(dbc.Button("Close", id="close1", className="ml-auto")),
                ],
                id="error-utl-train",
                centered=True,
                is_open=False,
            ),

            dbc.Modal(
                [
                    dbc.ModalHeader("Error"),
                    dbc.ModalBody("The uploaded test dataset does not have the same columns as the real dataset. "
                                  "Please load another file."),
                    dbc.ModalFooter(dbc.Button("Close", id="close2", className="ml-auto")),
                ],
                id="error-utl-test",
                centered=True,
                is_open=False,
            ),

            dbc.Form([
                # Dropdown for target class
                dbc.Row([
                    dbc.Col([
                        html.P("Select target class:")
                    ], width={'size': 2}),

                    dbc.Col([
                        dbc.Select(
                            id="dropdown-target",
                            options=[],
                            required='required'
                        )
                    ], width={'size': 2}),
                ], style={'margin-bottom': '1vw', 'margin-left': '4vw'}),

                # Dropdown for classifier
                dbc.Row([
                    dbc.Col([
                        html.P("Select classifier method:")
                    ], width={'size': 2}),

                    dbc.Col([
                        dbc.Select(
                            id="dropdown-classifier",
                            options=[
                                {"label": "Random Forest", "value": "RF"},
                                {"label": "K-Nearest Neighbors", "value": "KNN"},
                                {"label": "Decision Tree", "value": "DT"},
                                {"label": "Support Vector Machines", "value": "SVM"},
                                {"label": "Multilayer Perceptron", "value": "MLP"},
                            ],
                            required='required'
                        )
                    ], width={'size': 2}),
                ], style={'margin-bottom': '1vw', 'margin-left': '4vw'}),

                dbc.Row([
                    dbc.Col([dbc.Button("Evaluate Utility", color="info", id="run-utl")],
                            width={'size': 'auto'})
                ], style={'margin-bottom': '6vw', 'margin-left': '4vw'})
            ]),

            # Results section
            dbc.Row([
                dbc.Col(html.H4("Train on Real and Test on Real results"), width=5),
                dbc.Col(html.H4("Train on Synthetic and Test on Real results"), width={'size': 5, 'offset': 1}),
            ], style={'margin-bottom': '1vw', 'margin-left': '4vw'}),

            dcc.Loading([
                dbc.Row([
                    dbc.Col(id="output-utl-trtr", width=5),
                    dbc.Col(id="output-utl-tstr", width={'size': 5, 'offset': 1}),
                ], style={'margin-bottom': '0vw', 'margin-left': '4vw'}, align="center")
            ]),
            dcc.Store(id={"type": "data-report", "index": 8}),

            dbc.Row([dbc.Col(html.Div([html.Br(), html.Br()]), width=True)])

        ], fluid=True
    )
])


@app.callback(Output('dropdown-target', 'options'),
              Input('dropdown-classifier', 'options'))
def update_option(op):
    if not real_data.empty:
        return [{'label': col, 'value': col} for col in cat_features]
    else:
        return []


@app.callback(Output('error-utl-train', 'is_open'),
              Output('button-train', 'color'),
              Input('upload-data-train-utl', 'contents'),
              Input('close1', 'n_clicks'))
def upload_train_dataset(contents, n_clicks):
    global real_train_data

    ctx = dash.callback_context
    if not ctx.triggered or ctx.triggered[0]['prop_id'] == 'close1.n_clicks':
        if not real_train_data.empty:
            return False, 'success'
        return False, 'primary'

    if contents is not None:
        real_train_data = utils.parse_contents(contents)

        if list(real_data.columns) == list(real_train_data.columns):
            for col in real_data.columns:
                if real_train_data[col].dtype == 'object':
                    real_train_data[col] = encoder_labels.transform(real_train_data[col])
            return False, 'success'
        else:
            real_train_data = pd.DataFrame()
            return True, 'primary'


@app.callback(Output('error-utl-test', 'is_open'),
              Output('button-test', 'color'),
              Input('upload-data-test-utl', 'contents'),
              Input('close2', 'n_clicks'))
def upload_test_dataset(contents, n_clicks):
    global real_test_data

    ctx = dash.callback_context
    if not ctx.triggered or ctx.triggered[0]['prop_id'] == 'close2.n_clicks':
        if not real_test_data.empty:
            return False, 'success'
        return False, 'primary'

    if contents is not None:
        real_test_data = utils.parse_contents(contents)

        if list(real_data.columns) == list(real_test_data.columns):
            for col in real_data.columns:
                if real_test_data[col].dtype == 'object':
                    real_test_data[col] = encoder_labels.transform(real_test_data[col])
            return False, 'success'
        else:
            real_test_data = pd.DataFrame()
            return True, 'primary'


@app.callback(Output('output-utl-trtr', 'children'),
              Output('output-utl-tstr', 'children'),
              Output({"type": "data-report", "index": 8}, 'data'),
              Input('run-utl', 'n_clicks'),
              [State('dropdown-target', 'value'),
               State('dropdown-classifier', 'value')])
def run_code_on_click(n_clicks, target, classifier):
    if n_clicks is not None and target is not None and classifier is not None:
        if not real_train_data.empty and not real_test_data.empty:
            train_data_r = real_train_data.drop(columns=target)
            test_data_r = real_test_data.drop(columns=target)
            train_labels_r = real_train_data[target]
            test_labels_r = real_test_data[target]
        else:
            train_data_r, test_data_r, train_labels_r, test_labels_r = train_test_split(
                real_data.drop(columns=target), real_data[target], test_size=0.3, random_state=9)

        num = num_features.copy()
        cat = cat_features.copy()
        if target in num:
            num.remove(target)

        if target in cat:
            cat.remove(target)

        train_data_s = []
        train_labels_s = []
        for syn_data in synthetic_datasets:
            train_data, _, train_labels, _ = train_test_split(syn_data.drop(columns=target),
                                                              syn_data[target], test_size=0.2, random_state=19)
            train_data_s.append(train_data)
            train_labels_s.append(train_labels)

        with Pool(round(n_cpu * 0.75)) as pool:
            results_utl = pool.starmap(UtiMet.train_test_model,
                                       zip(repeat(classifier), [train_data_r] + train_data_s, repeat(test_data_r),
                                           [train_labels_r] + train_labels_s, repeat(test_labels_r),
                                           repeat(num), repeat(cat)))

        result_trtr = results_utl[0]
        results_tstr = results_utl[1:]

        fig_trtr = go.Figure(data=go.Heatmap(z=result_trtr[1][::-1], text=result_trtr[1][::-1],
                                             x=list(map(str, np.unique(real_data[target]))),
                                             y=list(map(str, np.unique(real_data[target])))[::-1],
                                             texttemplate="%{text}", colorscale='viridis', showscale=False))
        fig_trtr.update_layout(xaxis_title="Predicted Label", yaxis_title="True Label", title="Confusion Matrix")

        children_trtr = [
            dbc.Row([
                dbc.Col([html.Div([
                    html.B("Accuracy: "),
                    html.P(result_trtr[0]['accuracy'])
                ])], width={'size': 'auto'}),
                dbc.Col([html.Div([
                    html.B("Precision: "),
                    html.P(result_trtr[0]['precision'])
                ])], width={'size': 'auto', 'offset': 1}),
                dbc.Col([html.Div([
                    html.B("Recall: "),
                    html.P(result_trtr[0]['recall'])
                ])], width={'size': 'auto', 'offset': 1}),
                dbc.Col([html.Div([
                    html.B("F1-score: "),
                    html.P(result_trtr[0]['f1'])
                ])], width={'size': 'auto', 'offset': 1}),
            ]),
            dbc.Row([dbc.Col([
                dcc.Graph(figure=fig_trtr)
            ], width=9)], justify='center'),
        ]

        tabs = []
        figs_tstr = []
        results_tstr_report = []
        for i, res_tstr in enumerate(results_tstr):
            fig_tstr = go.Figure(data=go.Heatmap(z=res_tstr[1][::-1], text=res_tstr[1][::-1],
                                                 x=list(map(str, np.unique(real_data[target]))),
                                                 y=list(map(str, np.unique(real_data[target])))[::-1],
                                                 texttemplate="%{text}", colorscale='viridis', showscale=False))
            fig_tstr.update_layout(xaxis_title="Predicted Label", yaxis_title="True Label", title="Confusion Matrix")

            child = [
                dbc.Row([
                    dbc.Col([html.Div([
                        html.B("Accuracy: "),
                        html.P(res_tstr[0]['accuracy'])
                    ])], width={'size': 'auto'}),
                    dbc.Col([html.Div([
                        html.B("Precision: "),
                        html.P(res_tstr[0]['precision'])
                    ])], width={'size': 'auto', 'offset': 1}),
                    dbc.Col([html.Div([
                        html.B("Recall: "),
                        html.P(res_tstr[0]['recall'])
                    ])], width={'size': 'auto', 'offset': 1}),
                    dbc.Col([html.Div([
                        html.B("F1-score: "),
                        html.P(res_tstr[0]['f1'])
                    ])], width={'size': 'auto', 'offset': 1}),
                ]),
                dbc.Row([dbc.Col([
                    dcc.Graph(figure=fig_tstr)
                ], width=9)], justify='center'),
            ]

            tab_content = dbc.Card([
                dbc.CardBody(child)
            ], color="primary", outline=True)

            tabs.append(dbc.Tab(tab_content, label=f"Dataset {i + 1}", tab_id=f"tab{i + 1}"))

            figs_tstr.append(fig_tstr)
            results_tstr_report.append(res_tstr[0].to_dict("list"))

        children_tstr = [
            dbc.Tabs(tabs, active_tab="tab1")
        ]

        return children_trtr, children_tstr, [fig_trtr, figs_tstr, result_trtr[0].to_dict("list"), results_tstr_report]

    else:
        raise PreventUpdate


# PAGE 4 CONTENTS (privacy metrics)

# SEA metrics
page_4_sea = html.Div([

    dbc.Container(
        [
            # Header SEA section
            dbc.Row(
                dbc.Col(html.H4("Similarity Evaluation Analysis",
                                style={'margin-left': '2vw', 'margin-top': '1vw', 'margin-bottom': '2vw'}),
                        width="auto")),

            # Dropdown distance selection
            dbc.Row(
                [
                    dbc.Col(html.Div(
                        [
                            dbc.Label("Choose metric similarity", html_for="dropdown"),
                            dcc.Dropdown(
                                id="dropdown-sea",
                                options=[
                                    {"label": "Cosine similarity", "value": "cos"},
                                    {"label": "Euclidean distance", "value": "euc"},
                                    {"label": "Hausdorff distance", "value": "hau"},
                                ],
                                clearable=False
                            ),
                        ],
                        className="mb-3",
                    ), width={'size': 3, 'offset': 1}),

                ]
            ),

            # Graph distances
            dcc.Loading(dbc.Row(id="output-graph-sea"), id={"type": "load-res", "index": 2}),
            dcc.Store(id={"type": "data-report", "index": 9}),

            dbc.Row([dbc.Col(html.Div([html.Br(), html.Br()]), width=True)])

        ], fluid=True
    )
])


@app.callback(Output('output-graph-sea', 'children'),
              Output({"type": "data-report", "index": 9}, 'data'),
              Input('dropdown-sea', 'value'))
def update_graph(value):
    if value is not None:

        if value == "euc":
            with Pool(round(n_cpu * 0.75)) as pool:
                mats_dist = pool.starmap(PriMet.SEA.pairwise_euclidean_distance,
                                         zip(repeat(real_data), synthetic_datasets))
        elif value == "cos":
            with Pool(round(n_cpu * 0.75)) as pool:
                mats_dist = pool.starmap(PriMet.SEA.str_similarity,
                                         zip(repeat(real_data), synthetic_datasets))
        elif value == "hau":
            with Pool(round(n_cpu * 0.75)) as pool:
                dists = pool.starmap(PriMet.SEA.hausdorff_distance,
                                     zip(repeat(real_data), synthetic_datasets))

        if value == "euc" or value == "cos":

            tabs = []
            figs = []
            for i, mat in enumerate(mats_dist):
                fig = ff.create_distplot([mat.flatten()], ['Paired distance values'],
                                         show_hist=False, show_rug=False, colors=['blue'])
                fig.update_traces(fill='tozeroy', fillcolor='rgba(0, 0, 255, 0.5)', selector=dict(type='scatter'))

                tab_content = dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(figure=fig)
                    ])
                ], color="primary", outline=True)

                tabs.append(dbc.Tab(tab_content, label=f"Dataset {i + 1}", tab_id=f"tab{i + 1}"))

                figs.append(fig)

            children = [
                dbc.Col([
                    dbc.Tabs(tabs, active_tab="tab1")
                ], width={'size': 10, 'offset': 1}, align="center")
            ]

        else:

            tabs = []
            for i, dist in enumerate(dists):
                tab_content = dbc.Card([
                    dbc.CardBody([
                        html.Div([html.B("Hausdorff distance between synthetic and real datasets: "), html.P(dist)])
                    ])
                ], color="primary", outline=True)

                tabs.append(dbc.Tab(tab_content, label=f"Dataset {i + 1}", tab_id=f"tab{i + 1}"))

            children = [
                dbc.Col([
                    dbc.Tabs(tabs, active_tab="tab1")
                ], width={'size': 6, 'offset': 1}, align="center")
            ]

            figs = dists

        return children, [figs, value]

    else:
        raise PreventUpdate


# MIA metrics
page_4_mia = html.Div([
    dbc.Container(
        [

            # Header MIA section
            dbc.Row([
                dbc.Col([html.H4(["Membership Inference Attack"],
                                 style={'margin-left': '2vw', 'margin-top': '1vw', 'margin-bottom': '2vw'},
                                 )],
                        width="auto")
            ]),

            dbc.Row([

                # Attack schema
                dbc.Col([
                    dbc.Toast(
                        [
                            html.Img(
                                src="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10306449/bin/10-1055-s-0042-1760247-i22020006-2.jpg",
                                style={'width': '100%', 'height': 'auto',
                                       'borderRadius': '10px', 'boxShadow': '2px 2px 5px grey'
                                       }
                            )
                        ],
                        style={'width': '85%'},
                        header="MIA schema",
                    )
                ], width={'size': 6}),

                dbc.Col([

                    # Upload training set
                    dbc.Row([
                        dbc.Col([
                            html.P("Import training dataset:")
                        ], width={'size': 'auto'}),

                        dbc.Col([
                            dcc.Upload([dbc.Button(["Upload Train"], id="button-train-pr", color="primary")],
                                       id="upload-data-train-pr",
                                       multiple=False)
                        ], width={'size': 'auto'}),
                    ], style={'margin-bottom': '2vw'}),

                    dbc.Modal(
                        [
                            dbc.ModalHeader("Error"),
                            dbc.ModalBody(
                                "The uploaded train dataset does not have the same columns as the real dataset. "
                                "Please load another file."),
                            dbc.ModalFooter(dbc.Button("Close", id="close3", className="ml-auto")),
                        ],
                        id="error-pr-train",
                        centered=True,
                        is_open=False,
                    ),

                    dbc.Form([
                        # Slider subset
                        dbc.Row([
                            dbc.Col([
                                html.P("Enter subset size:")
                            ], width={'size': 'auto'}),

                            dbc.Col([
                                dcc.Slider(10, 100, 5,
                                           tooltip={"placement": "bottom", "always_visible": True},
                                           marks={
                                               10: '10%',
                                               25: '25%',
                                               50: '50%',
                                               75: '75%',
                                               100: '100%'
                                           },
                                           value=40,
                                           id="slider-subset")
                            ], width={'size': True}),
                        ], style={'margin-bottom': '2vw'}),

                        # Slider similarity
                        dbc.Row([
                            dbc.Col([
                                html.P("Enter similarity threshold:")
                            ], width={'size': 'auto'}),

                            dbc.Col([
                                dcc.Slider(0, 1, 0.05,
                                           tooltip={"placement": "bottom", "always_visible": True},
                                           marks={
                                               0: 'Low',
                                               1: 'High'
                                           },
                                           value=0.6,
                                           id="slider-sim")
                            ], width={'size': True}),
                        ], style={'margin-bottom': '2vw'}),

                        dbc.Row([
                            dbc.Col([dbc.Button("Simulate Attack", color="info", id="run-mia")],
                                    width={'size': 'auto'})
                        ], style={'margin-bottom': '1vw'})
                    ]),

                    # Graphs performance attacker
                    dcc.Loading(dbc.Row(html.Div(id="output-mia"))),
                    dcc.Store(id={"type": "data-report", "index": 10})

                ], width={'size': 5}, align='center')

            ], style={'margin-left': '4vw'}),

            dbc.Row([dbc.Col(html.Div([html.Br(), html.Br()]), width=True)])

        ], fluid=True
    )
])


@app.callback(Output('error-pr-train', 'is_open'),
              Output('button-train-pr', 'color'),
              Input('upload-data-train-pr', 'contents'),
              Input('close3', 'n_clicks'), )
def upload_train_dataset(contents, n_clicks):
    global real_train_data

    ctx = dash.callback_context
    if not ctx.triggered or ctx.triggered[0]['prop_id'] == 'close3.n_clicks':
        if not real_train_data.empty:
            return False, 'success'
        return False, 'primary'

    if contents is not None:
        real_train_data = utils.parse_contents(contents)

        if list(real_data.columns) == list(real_train_data.columns):
            return False, 'success'
        else:
            real_train_data = pd.DataFrame()
            return True, 'primary'


@app.callback(Output('run-mia', 'disabled'),
              Input('button-train-pr', 'color'))
def enable_submit(color):
    if color == 'success':
        return False
    else:
        return True


@app.callback(Output('output-mia', 'children'),
              Output({"type": "data-report", "index": 10}, 'data'),
              Input('run-mia', 'n_clicks'),
              State('slider-subset', 'value'),
              State('slider-sim', 'value'),
              prevent_initial_call=True)
def run_code_on_click(click, prop_subset, t_similarity):
    if click:

        real_subset = real_data.sample(frac=prop_subset / 100, random_state=42).reset_index(drop=True)

        label_membership_train = [tuple(row) in set(real_train_data.itertuples(index=False))
                                  for row in real_subset.itertuples(index=False)]

        with Pool(round(n_cpu * 0.75)) as pool:
            results_mia = pool.starmap(PriMet.MIA.simulate_mia,
                                       zip(repeat(real_subset), repeat(label_membership_train), synthetic_datasets,
                                           repeat(t_similarity)))

        tabs = []
        figs = []
        for i, res in enumerate(results_mia):
            acc = res[1]
            prec = res[0]

            fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]])
            fig.add_trace(go.Pie(labels=['Correct', 'Wrong'], values=[acc, 1 - acc], name="accuracy"), 1, 1)
            fig.add_trace(go.Pie(labels=['Correct', 'Wrong'], values=[prec, 1 - prec], name="precision"), 1, 2)

            fig.update_traces(hole=.4, marker=dict(colors=['lightgreen', 'black']))

            fig.update_layout(annotations=[dict(text='Accuracy', xref="x domain", yref="y domain", x=0.14, y=0.5,
                                                font_size=14, showarrow=False),
                                           dict(text='Precision', xref="x domain", yref="y domain", x=0.86, y=0.5,
                                                font_size=14, showarrow=False)],
                              showlegend=False,
                              title="Attacker performance")

            tab_content = dbc.Card([
                dbc.CardBody([
                    dcc.Graph(figure=fig)
                ])
            ], color="primary", outline=True)

            tabs.append(dbc.Tab(tab_content, label=f"Dataset {i + 1}", tab_id=f"tab{i + 1}"))

            figs.append(fig)

        children = [
            dbc.Tabs(tabs, active_tab="tab1")
        ]

        return children, [figs, prop_subset, t_similarity]

    else:
        raise PreventUpdate


# AIA metrics
page_4_aia = html.Div([
    dbc.Container(
        [

            # Header AIA section
            dbc.Row([
                dbc.Col([html.H4(["Attribute Inference Attack"],
                                 style={'margin-left': '2vw', 'margin-top': '1vw', 'margin-bottom': '2vw'},
                                 )],
                        width="auto")
            ]),

            dbc.Row([

                # Attack schema
                dbc.Col([
                    dbc.Toast(
                        [
                            html.Img(
                                src="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10306449/bin/10-1055-s-0042-1760247-i22020006-3.jpg",
                                style={'width': '100%', 'height': 'auto',
                                       'borderRadius': '10px', 'boxShadow': '2px 2px 5px grey'
                                       }
                            )
                        ],
                        style={'width': '85%'},
                        header="AIA schema",
                    )
                ], width={'size': 6}),

                dbc.Col([

                    dbc.Form([
                        # Slider subset size
                        dbc.Row([
                            dbc.Col([
                                html.P("Enter subset size:")
                            ], width={'size': 'auto'}),

                            dbc.Col([
                                dcc.Slider(10, 100, 5,
                                           tooltip={"placement": "bottom", "always_visible": True},
                                           marks={
                                               10: '10%',
                                               25: '25%',
                                               50: '50%',
                                               75: '75%',
                                               100: '100%'
                                           },
                                           value=40,
                                           id="slider-subset-aia")
                            ], width={'size': True}),
                        ], style={'margin-bottom': '2vw'}),

                        # Dropdown for attributes selection
                        dbc.Row([
                            dbc.Col([
                                html.P("Attacker attributes:")
                            ], width={'size': 'auto'}),

                            dbc.Col([
                                dcc.Dropdown(
                                    options=[],
                                    multi=True,
                                    id="dropdown-aia"
                                )
                            ], width={'size': True}),
                        ], style={'margin-bottom': '2vw'}),

                        dbc.Row([
                            dbc.Col([dbc.Button("Simulate Attack", color="info", id="run-aia")],
                                    width={'size': 'auto'})
                        ], style={'margin-bottom': '4vw'})
                    ]),

                    # Performance attacker
                    dcc.Loading(dbc.Row(html.Div(id="output-aia"))),
                    dcc.Store(id={"type": "data-report", "index": 11})

                ], width={'size': 5}, align='center')

            ], style={'margin-left': '4vw'}),

            dbc.Row([dbc.Col(html.Div([html.Br(), html.Br()]), width=True)])

        ], fluid=True
    )
])


@app.callback(Output('dropdown-aia', 'options'),
              Input('slider-subset-aia', 'marks'))
def update_options(m):
    if not real_data.empty:
        return [{'label': col, 'value': col} for col in real_data.columns]
    else:
        return []


@app.callback(Output('run-aia', 'disabled'),
              Input('dropdown-aia', 'value'))
def enable_submit(value):
    if value is not None and value:
        return False
    else:
        return True


@app.callback(Output('output-aia', 'children'),
              Output({"type": "data-report", "index": 11}, 'data'),
              Input('run-aia', 'n_clicks'),
              State('slider-subset-aia', 'value'),
              State('dropdown-aia', 'value'))
def run_code_on_click(click, prop_subset, attributes):
    if click:

        real_subset = real_data.sample(frac=prop_subset / 100, random_state=24).reset_index(drop=True)
        targets = [col for col in real_data.columns if col not in attributes]

        with Pool(round(n_cpu * 0.75)) as pool:
            results_aia = pool.starmap(PriMet.AIA.simulate_aia,
                                       zip(repeat(real_subset), synthetic_datasets,
                                           repeat(attributes), repeat(targets), repeat(dict_features_type)))

        tabs = []
        dfs_acc = []
        dfs_rmse = []
        for i, res in enumerate(results_aia):
            df_acc = res[res['Metric name'] == 'acc'][['Target name', 'Value']].to_dict('list')
            df_rmse = res[res['Metric name'] == 'rmse'][['Target name', 'IQR target', 'Value']].to_dict('list')

            dfs_acc.append(df_acc)
            dfs_rmse.append(df_rmse)

            tabs.append(dbc.Tab(label=f"Dataset {i + 1}", tab_id=f"tab{i + 1}"))

        children = dbc.Card(
            [
                dbc.CardHeader(
                    dbc.Tabs(tabs, id="tabs-aia", active_tab="tab1")
                ),
                dbc.CardBody(id="card-body-aia"),
            ], color="primary", outline=True
        )

        return children, [dfs_acc, dfs_rmse, prop_subset, attributes]

    else:
        raise PreventUpdate


@app.callback(Output("card-body-aia", "children"),
              Input("tabs-aia", "active_tab"),
              State({"type": "data-report", "index": 11}, 'data'))
def tab_content(id_tab, data):
    df_acc = pd.DataFrame(data[0][int(id_tab[-1]) - 1])
    df_rmse = pd.DataFrame(data[1][int(id_tab[-1]) - 1])

    card_content = dbc.Tabs([
        dbc.Tab([
            dbc.Card([
                dbc.CardBody([
                    dash_table.DataTable(
                        data=df_acc.to_dict('records'),
                        columns=[{'name': "Target name", 'id': "Target name"},
                                 {'name': "Value", 'id': "Value"}],
                        page_action='none',
                        style_table={'height': '300px', 'overflowY': 'auto'},
                        fixed_rows={'headers': True},
                        style_header={'text-align': 'center'},
                        style_cell={'minWidth': '50%', 'maxWidth': '50%', 'width': '50%'},
                    )
                ])
            ], color="dark", outline=True)
        ], label="Accuracy", tab_id="tab1-in"),
        dbc.Tab([
            dbc.Card([
                dbc.CardBody([
                    dash_table.DataTable(
                        data=df_rmse.to_dict('records'),
                        columns=[{'name': "Target name", 'id': "Target name"},
                                 {'name': "IQR target", 'id': "IQR target"},
                                 {'name': "Value", 'id': "Value"}],
                        page_action='none',
                        style_table={'height': '300px', 'overflowY': 'auto'},
                        fixed_rows={'headers': True},
                        style_header={'text-align': 'center'},
                        style_cell={'minWidth': '33%', 'maxWidth': '33%', 'width': '33%'},
                    )
                ])
            ], color="dark", outline=True)
        ], label="RMSE", tab_id="tab2-in")
    ], active_tab="tab1-in")

    return card_content


# PAGE 5 CONTENTS (benchmarking)
page_5 = html.Div([

    dbc.Container(
        [
            # Header Benchmarking section
            dbc.Row(
                dbc.Col([html.H4(["Benchmarking Section"],
                                 style={'margin-left': '2vw', 'margin-top': '1vw', 'margin-bottom': '2vw'},
                                 )],
                        width="auto")),

            # Metrics selection section
            dbc.Form([
                dbc.Row([

                    # Resemblance section
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Resemblance"),
                            dbc.CardBody([
                                dbc.Tabs([

                                    # Tab URA
                                    dbc.Tab([
                                        dbc.Card([dbc.CardBody([
                                            dbc.Checklist(
                                                id="list-ura",
                                                options=[
                                                    {"label": "Numerical statistical test", "value": "num-test"},
                                                    {"label": "Categorical statistical test", "value": "cat-test"},
                                                    {"label": "Distance metrics", "value": "dist"},
                                                ],
                                                label_checked_style={"color": "red"},
                                                input_checked_style={"backgroundColor": "#fa7268",
                                                                     "borderColor": "#ea6258"},
                                            ),
                                        ])], color="light", outline=True)
                                    ], label="URA", tab_id="tab-ura",
                                        label_style={"color": "#FFFFFF"}, active_label_style={"color": "#000000"}),

                                    # Tab MRA
                                    dbc.Tab([
                                        dbc.Card([dbc.CardBody([
                                            dbc.Checklist(
                                                id="list-mra",
                                                options=[
                                                    {"label": "Correlation matrices", "value": "corr_num"},
                                                    {"label": "Contingency tables", "value": "corr_cat"},
                                                    {"label": "Principal Component Analysis", "value": "pca"},
                                                ],
                                                label_checked_style={"color": "red"},
                                                input_checked_style={"backgroundColor": "#fa7268",
                                                                     "borderColor": "#ea6258"},
                                            ),
                                        ])], color="light", outline=True)
                                    ], label="MRA", tab_id="tab-mra",
                                        label_style={"color": "#FFFFFF"}, active_label_style={"color": "#000000"}),

                                    # Tab DLA
                                    dbc.Tab([
                                        dbc.Card([dbc.CardBody([
                                            dbc.Checklist(
                                                id="list-dla",
                                                options=[
                                                    {"label": "Random Forest", "value": "RF"},
                                                    {"label": "K-Nearest Neighbors", "value": "KNN"},
                                                    {"label": "Decision Tree", "value": "DT"},
                                                    {"label": "Support Vector Machines", "value": "SVM"},
                                                    {"label": "Multilayer Perceptron", "value": "MLP"}
                                                ],
                                                label_checked_style={"color": "red"},
                                                input_checked_style={"backgroundColor": "#fa7268",
                                                                     "borderColor": "#ea6258"},
                                            ),
                                        ])], color="light", outline=True)
                                    ], label="DLA", tab_id="tab-dla",
                                        label_style={"color": "#FFFFFF"}, active_label_style={"color": "#000000"}),

                                ], active_tab="tab-ura"),
                            ]),
                        ], color="info", inverse=True)
                    ], width={'size': 4}),

                    # Utility section
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Utility"),
                            dbc.CardBody([
                                html.Div([
                                    dbc.Card([dbc.CardBody([
                                        dbc.Checklist(
                                            id="list-utl",
                                            options=[
                                                {"label": "Run TRTR and TSTR analyses", "value": "run_utl"},
                                            ],
                                            label_checked_style={"color": "red"},
                                            input_checked_style={"backgroundColor": "#fa7268",
                                                                 "borderColor": "#ea6258"},
                                        ),
                                    ])], color="light", outline=True)
                                ]),
                            ]),
                        ], color="secondary", inverse=True)
                    ], width={'size': 4}),

                    # Privacy section
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Privacy"),
                            dbc.CardBody([
                                dbc.Tabs([
                                    # Tab SEA
                                    dbc.Tab([dbc.Card([dbc.CardBody([
                                        dbc.RadioItems(
                                            id="list-sea",
                                            options=[
                                                {"label": "Cosine similarity", "value": "cos"},
                                                {"label": "Euclidean distance", "value": "euc"},
                                                {"label": "Hausdorff distance", "value": "hau"},
                                            ],
                                            label_checked_style={"color": "red"},
                                            input_checked_style={"backgroundColor": "#fa7268",
                                                                 "borderColor": "#ea6258"},
                                            switch=True
                                        ),
                                    ])], color="light", outline=True)
                                    ], label="SEA", tab_id="tab-sea",
                                        label_style={"color": "#FFFFFF"}, active_label_style={"color": "#000000"}),

                                    # Tab MIA
                                    dbc.Tab([dbc.Card([dbc.CardBody([
                                        dbc.Checklist(
                                            id="list-mia",
                                            options=[
                                                {"label": "Run MIA simulation", "value": "run_mia"},
                                            ],
                                            label_checked_style={"color": "red"},
                                            input_checked_style={"backgroundColor": "#fa7268",
                                                                 "borderColor": "#ea6258"},
                                        ),
                                    ])], color="light", outline=True)
                                    ], label="MIA", tab_id="tab-mia",
                                        label_style={"color": "#FFFFFF"}, active_label_style={"color": "#000000"}),

                                    # Tab AIA
                                    dbc.Tab([dbc.Card([dbc.CardBody([
                                        dbc.Checklist(
                                            id="list-aia",
                                            options=[
                                                {"label": "Run AIA simulation", "value": "run_aia"},
                                            ],
                                            label_checked_style={"color": "red"},
                                            input_checked_style={"backgroundColor": "#fa7268",
                                                                 "borderColor": "#ea6258"},
                                        ),
                                    ])], color="light", outline=True)
                                    ], label="AIA", tab_id="tab-aia",
                                        label_style={"color": "#FFFFFF"}, active_label_style={"color": "#000000"}),
                                ], active_tab="tab-sea"),
                            ]),
                        ], color="success", inverse=True)
                    ], width={'size': 4}),

                ], style={'margin-bottom': '2vw'}),

                dbc.Row([dbc.Col([
                    dbc.Button("Next", color="light", id="next-bm")
                ], width={'size': 'auto'})], style={'margin-bottom': '4vw', 'margin-left': '2vw'}),

            ]),

            dbc.Modal(
                [
                    dbc.ModalHeader("Additional data required"),
                    dbc.ModalBody("Selected metrics may require additional details to be specified. Please input the "
                                  "requested information into the appropriate fields before proceeding."),
                ],
                id="modal-data-load-bm",
                centered=True,
                is_open=False,
            ),

            # Input necessary data section
            html.Div(id="data-load-bm"),

            # Ranking result
            dcc.Loading(html.Div(id="result-bm"), fullscreen=True, type='cube'),
            dcc.Store(id="data-ranking"),
            dcc.Store(id={"type": "data-report", "index": 12}),

            # Slider use case
            html.Div([dbc.Row([
                dbc.Col(id="use-case-sliders", width=6),
                dbc.Col(id="use-case-results", width=6)
            ], align="center")]),
            dcc.Store(id={"type": "data-report", "index": 13}),

            dbc.Row([dbc.Col(html.Div([html.Br(), html.Br()]), width=True)])

        ], fluid=True
    )
])


@app.callback(Output("data-load-bm", "children", allow_duplicate=True),
              Output("modal-data-load-bm", "is_open"),
              Output("result-bm", "children", allow_duplicate=True),
              Output("use-case-sliders", "children", allow_duplicate=True),
              Output("use-case-results", "children", allow_duplicate=True),
              Input("next-bm", "n_clicks"),
              [State("list-ura", "value"),
               State("list-mra", "value"),
               State("list-dla", "value"),
               State("list-utl", "value"),
               State("list-sea", "value"),
               State("list-mia", "value"),
               State("list-aia", "value"), ],
              prevent_initial_call=True)
def run_code_on_click(click, list_ura, list_mra, list_dla, list_utl, list_sea, list_mia, list_aia):
    if click:

        content_resemblance, content_utility, content_privacy = [], [], []

        # Data input URA section
        if list_ura:
            content_resemblance.append(html.Hr())
            for item in list_ura:

                if item == "num-test":
                    content_resemblance.append(
                        html.Div([
                            dbc.Row([
                                dbc.Col([html.H5("Univariate Resemblance Analysis (numerical test)",
                                                 style={"font-weight": "bold"})],
                                        width="auto")
                            ], style={'margin-bottom': '2vw', 'margin-left': '3.5vw'}),

                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Choose statistical test", html_for="dropdown"),
                                    dcc.Dropdown(
                                        id={"type": "opt-ura-bm", "index": 0},
                                        options=[
                                            {"label": "Kolmogorov–Smirnov test", "value": "ks_test"},
                                            {"label": "Student T-test", "value": "t_test"},
                                            {"label": "Mann Whitney U-test", "value": "u_test"},
                                        ],
                                        value="ks_test",
                                        clearable=False
                                    ),
                                ])], style={'margin-bottom': '1vw', 'margin-left': '4vw'}),

                        ], style={'margin-bottom': '3vw', 'margin-top': '3vw', 'margin-left': '6vw',
                                  'margin-right': '6vw'})
                    )
                    content_resemblance.append(html.Hr())

                elif item == "cat-test":
                    content_resemblance.append(
                        html.Div([
                            dbc.Row([
                                dbc.Col([html.H5("Univariate Resemblance Analysis (categorical test)",
                                                 style={"font-weight": "bold"})],
                                        width="auto")
                            ], style={'margin-bottom': '2vw', 'margin-left': '3.5vw'}),

                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Choose statistical test", html_for="dropdown"),
                                    dcc.Dropdown(
                                        id={"type": "opt-ura-bm", "index": 1},
                                        options=[
                                            {"label": "Chi-square test", "value": "chi_test"},
                                        ],
                                        value="chi_test",
                                        clearable=False
                                    ),
                                ])], style={'margin-bottom': '1vw', 'margin-left': '4vw'}),

                        ], style={'margin-bottom': '3vw', 'margin-top': '3vw', 'margin-left': '6vw',
                                  'margin-right': '6vw'})
                    )
                    content_resemblance.append(html.Hr())

                elif item == "dist":
                    content_resemblance.append(
                        html.Div([
                            dbc.Row([
                                dbc.Col([html.H5("Univariate Resemblance Analysis (distance metric)",
                                                 style={"font-weight": "bold"})],
                                        width="auto")
                            ], style={'margin-bottom': '2vw', 'margin-left': '3.5vw'}),

                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Choose distance metric", html_for="dropdown"),
                                    dcc.Dropdown(
                                        id={"type": "opt-ura-bm", "index": 2},
                                        options=[
                                            {"label": "Cosine distance", "value": "cos_dist"},
                                            {"label": "Jensen-Shannon distance", "value": "js_dist"},
                                            {"label": "Wasserstein distance", "value": "w_dist"},
                                        ],
                                        value="cos_dist",
                                        clearable=False
                                    ),
                                ])], style={'margin-bottom': '1vw', 'margin-left': '4vw'}),

                        ], style={'margin-bottom': '3vw', 'margin-top': '3vw', 'margin-left': '6vw',
                                  'margin-right': '6vw'})
                    )
                    content_resemblance.append(html.Hr())

        # Data input MRA and DLA are not required

        # Data input Utility section
        if list_utl:
            content_utility.append(html.Hr())
            content_utility.append(
                html.Div([
                    dbc.Row([
                        dbc.Col([html.H5("Train on Real vs. Train on Synthetic", style={"font-weight": "bold"})],
                                width="auto")
                    ], style={'margin-bottom': '2vw', 'margin-left': '3.5vw'}),

                    dbc.Row([

                        dbc.Col([
                            html.P(["Import datasets: ",
                                    html.Span("ℹ", id="info-utl-bm", style={'cursor': 'pointer'})]),
                            dbc.Tooltip(
                                "Import the training and testing datasets, otherwise, a random split will be performed on "
                                "the existing real dataset already imported.",
                                target="info-utl-bm",
                            )
                        ], width={'size': 2}),

                        dbc.Col([
                            dcc.Upload([dbc.Button(["Upload Train"], id="button-train-utl-bm", color="primary")],
                                       id="upload-train-utl-bm",
                                       multiple=False)
                        ], width={'size': 'auto'}),

                        dbc.Col([
                            dcc.Upload([dbc.Button(["Upload Test"], id="button-test-utl-bm", color="primary")],
                                       id="upload-test-utl-bm",
                                       multiple=False),
                        ], width={'size': 'auto'}),

                    ], style={'margin-bottom': '1vw', 'margin-left': '4vw'}),

                    dbc.Row([
                        dbc.Col([
                            html.P("Select target class:")
                        ], width={'size': 2}),

                        dbc.Col([
                            dbc.Select(
                                id={"type": "opt-utl-bm", "index": 0},
                                options=[],
                                required='required'
                            )
                        ], width={'size': 2}),
                    ], style={'margin-bottom': '1vw', 'margin-left': '4vw'}),

                    dbc.Row([
                        dbc.Col([
                            html.P("Select classifier method:")
                        ], width={'size': 2}),

                        dbc.Col([
                            dbc.Select(
                                id={"type": "opt-utl-bm", "index": 1},
                                options=[
                                    {"label": "Random Forest", "value": "RF"},
                                    {"label": "K-Nearest Neighbors", "value": "KNN"},
                                    {"label": "Decision Tree", "value": "DT"},
                                    {"label": "Support Vector Machines", "value": "SVM"},
                                    {"label": "Multilayer Perceptron", "value": "MLP"},
                                ],
                                required='required'
                            )
                        ], width={'size': 2}),
                    ], style={'margin-bottom': '1vw', 'margin-left': '4vw'}),

                ], style={'margin-bottom': '3vw', 'margin-top': '3vw', 'margin-left': '6vw',
                          'margin-right': '6vw'})
            )
            content_utility.append(html.Hr())

        # Data input SEA is not required

        # Data input MIA section
        if list_mia:
            content_privacy.append(html.Hr())
            content_privacy.append(
                html.Div([

                    dbc.Row([
                        dbc.Col([html.H5("Membership Inference Attack", style={"font-weight": "bold"})], width="auto")
                    ], style={'margin-bottom': '2vw', 'margin-left': '3.5vw'}),

                    # Upload training set
                    dbc.Row([
                        dbc.Col([
                            html.P("Import training dataset:")
                        ], width={'size': 'auto'}),

                        dbc.Col([
                            dcc.Upload([dbc.Button(["Upload Train"], id="button-train-pr-bm", color="primary")],
                                       id="upload-train-pr-bm",
                                       multiple=False)
                        ], width={'size': 'auto'}),
                    ], style={'margin-bottom': '1vw', 'margin-left': '4vw'}),

                    # Slider subset
                    dbc.Row([
                        dbc.Col([
                            html.P("Enter subset size:")
                        ], width={'size': 'auto'}),

                        dbc.Col([
                            dcc.Slider(10, 100, 5,
                                       tooltip={"placement": "bottom", "always_visible": True},
                                       marks={
                                           10: '10%',
                                           25: '25%',
                                           50: '50%',
                                           75: '75%',
                                           100: '100%'
                                       },
                                       value=40,
                                       id={"type": "opt-mia-bm", "index": 0})
                        ], width={'size': True}),
                    ], style={'margin-bottom': '1vw', 'margin-left': '4vw'}),

                    # Slider similarity
                    dbc.Row([
                        dbc.Col([
                            html.P("Enter similarity threshold:")
                        ], width={'size': 'auto'}),

                        dbc.Col([
                            dcc.Slider(0, 1, 0.05,
                                       tooltip={"placement": "bottom", "always_visible": True},
                                       marks={
                                           0: 'Low',
                                           1: 'High'
                                       },
                                       value=0.6,
                                       id={"type": "opt-mia-bm", "index": 1})
                        ], width={'size': True}),
                    ], style={'margin-bottom': '1vw', 'margin-left': '4vw'}),

                ], style={'margin-bottom': '3vw', 'margin-top': '3vw', 'margin-left': '6vw',
                          'margin-right': '6vw'})
            )
            content_privacy.append(html.Hr())

        # Data input AIA section
        if list_aia:
            content_privacy.append(html.Hr())
            content_privacy.append(
                html.Div([

                    dbc.Row([
                        dbc.Col([html.H5("Attribute Inference Attack", style={"font-weight": "bold"})], width="auto")
                    ], style={'margin-bottom': '2vw', 'margin-left': '3.5vw'}),

                    # Slider subset size
                    dbc.Row([
                        dbc.Col([
                            html.P("Enter subset size:")
                        ], width={'size': 'auto'}),

                        dbc.Col([
                            dcc.Slider(10, 100, 5,
                                       tooltip={"placement": "bottom", "always_visible": True},
                                       marks={
                                           10: '10%',
                                           25: '25%',
                                           50: '50%',
                                           75: '75%',
                                           100: '100%'
                                       },
                                       value=40,
                                       id={"type": "opt-aia-bm", "index": 0})
                        ], width={'size': True}),
                    ], style={'margin-bottom': '1vw', 'margin-left': '4vw'}),

                    # Dropdown for attributes selection
                    dbc.Row([
                        dbc.Col([
                            html.P("Attacker attributes:")
                        ], width={'size': 'auto'}),

                        dbc.Col([
                            dcc.Dropdown(
                                options=[],
                                multi=True,
                                id={"type": "opt-aia-bm", "index": 1}
                            )
                        ], width={'size': True}),
                    ], style={'margin-bottom': '1vw', 'margin-left': '4vw'}),

                ], style={'margin-bottom': '3vw', 'margin-top': '3vw', 'margin-left': '6vw',
                          'margin-right': '6vw'})
            )
            content_privacy.append(html.Hr())

        if content_resemblance or content_utility or content_privacy:
            children = [
                dbc.Row([dbc.Col([
                    dbc.Accordion(
                        [
                            dbc.AccordionItem(
                                content_resemblance, title="Resemblance"
                            ),
                            dbc.AccordionItem(
                                content_utility, title="Utility"
                            ),
                            dbc.AccordionItem(
                                content_privacy, title="Privacy"
                            ),
                        ],
                        always_open=True,
                        start_collapsed=True,
                        flush=True,
                        id="accordion-bm"
                    )
                ])], style={'margin-bottom': '2vw', 'margin-left': '5vw', 'margin-right': '5vw'}),

                dbc.Row([dbc.Col([
                    dbc.Button("Compute Ranking", color="primary", id="next2-bm"),
                    # dbc.Tooltip(
                    #     "Please make sure you have entered all the required data.",
                    #     target="next2-bm",
                    # )
                ], width={'size': 'auto'})], style={'margin-bottom': '4vw', 'margin-left': '2vw'})
            ]
            return children, True, [], [], []
        elif all([not list_ura, not list_mra, not list_dla, not list_utl, not list_sea, not list_mia, not list_aia]):
            raise PreventUpdate
        else:
            return dbc.Button("Compute Ranking", color="light", id="next2-bm", n_clicks=1), False, [], [], []


@app.callback(Output({"type": "opt-utl-bm", "index": 0}, 'options'),
              Input({"type": "opt-utl-bm", "index": 1}, 'options'))
def update_option(op):
    if not real_data.empty:
        return [{'label': col, 'value': col} for col in cat_features]
    else:
        return []


@app.callback(Output({"type": "opt-aia-bm", "index": 1}, 'options'),
              Input({"type": "opt-aia-bm", "index": 0}, 'marks'))
def update_options(m):
    if not real_data.empty:
        return [{'label': col, 'value': col} for col in real_data.columns]
    else:
        return []


@app.callback(Output('button-train-utl-bm', 'color'),
              Input('upload-train-utl-bm', 'contents'))
def upload_train_dataset(contents):
    global real_train_data

    ctx = dash.callback_context
    if not ctx.triggered:
        if not real_train_data.empty:
            return 'success'
        return 'primary'

    if contents is not None:
        real_train_data = utils.parse_contents(contents)

        if list(real_data.columns) == list(real_train_data.columns):
            for col in real_data.columns:
                if real_train_data[col].dtype == 'object':
                    real_train_data[col] = encoder_labels.transform(real_train_data[col])
            return 'success'
        else:
            real_train_data = pd.DataFrame()
            return 'danger'


@app.callback(Output('button-test-utl-bm', 'color'),
              Input('upload-test-utl-bm', 'contents'))
def upload_test_dataset(contents):
    global real_test_data

    ctx = dash.callback_context
    if not ctx.triggered:
        if not real_train_data.empty:
            return 'success'
        return 'primary'

    if contents is not None:
        real_test_data = utils.parse_contents(contents)

        if list(real_data.columns) == list(real_test_data.columns):
            for col in real_data.columns:
                if real_test_data[col].dtype == 'object':
                    real_test_data[col] = encoder_labels.transform(real_test_data[col])
            return 'success'
        else:
            real_test_data = pd.DataFrame()
            return 'danger'


@app.callback(Output('button-train-pr-bm', 'color'),
              Input('upload-train-pr-bm', 'contents'))
def upload_train_dataset(contents):
    global real_train_data

    ctx = dash.callback_context
    if not ctx.triggered:
        if not real_train_data.empty:
            return 'success'
        return 'primary'

    if contents is not None:
        real_train_data = utils.parse_contents(contents)

        if list(real_data.columns) == list(real_train_data.columns):
            for col in real_data.columns:
                if real_train_data[col].dtype == 'object':
                    real_train_data[col] = encoder_labels.transform(real_train_data[col])
            return 'success'
        else:
            real_train_data = pd.DataFrame()
            return 'danger'


def compute_cumulative_rank(rank_vectors):
    total_vectors = len(rank_vectors)
    total_methods = len(rank_vectors[0])

    cumulative_counts = np.zeros((total_methods, total_methods))
    for ranks in rank_vectors:
        for i, rank in enumerate(ranks):
            cumulative_counts[i, rank - 1:] += 1

    cumulative_perc_dict = {}
    for i in range(total_methods):
        cumulative_perc_dict[i + 1] = list(cumulative_counts[i] / total_vectors)

    return cumulative_perc_dict


@app.callback(Output("result-bm", "children"),
              Output("data-load-bm", "children"),
              Output("data-ranking", "data"),
              Output({"type": "data-report", "index": 12}, 'data'),
              Input("next2-bm", "n_clicks"),
              [State("list-ura", "value"),
               State("list-mra", "value"),
               State("list-dla", "value"),
               State("list-utl", "value"),
               State("list-sea", "value"),
               State("list-mia", "value"),
               State("list-aia", "value"),
               State({"type": "opt-ura-bm", "index": ALL}, "value"),
               State({"type": "opt-utl-bm", "index": ALL}, "value"),
               State({"type": "opt-mia-bm", "index": ALL}, "value"),
               State({"type": "opt-aia-bm", "index": ALL}, "value")],
              prevent_initial_call=True)
def run_code_on_click(click, list_ura, list_mra, list_dla, list_utl, list_sea, list_mia, list_aia,
                      opt_ura, opt_utl, opt_mia, opt_aia):
    if click:

        if all([not list_ura, not list_mra, not list_dla, not list_utl, not list_sea, not list_mia, not list_aia]):
            raise PreventUpdate
        elif list_utl:
            if any(op is None for op in opt_utl):
                raise PreventUpdate
        elif list_mia:
            if real_train_data.empty:
                raise PreventUpdate
        elif list_aia:
            if any(op is None for op in opt_aia):
                raise PreventUpdate

        content_resemblance, content_utility, content_privacy, content_overall = [], [], [], []
        ranking_resemblance, ranking_utility, ranking_privacy = [], [], []
        data_ranking_resemblance, data_ranking_utility, data_ranking_privacy, data_all_ranking = [], [], [], []
        data_report_tbl, data_report_fig = {}, []

        # URA ranking calculation
        if list_ura:
            ranking_list_ura = []

            helping_var = len(list_ura)
            for item in list_ura:

                if item == "num-test":
                    value = opt_ura[len(list_ura) - helping_var]

                    if value == 'ks_test':
                        with Pool(round(n_cpu * 0.25)) as pool:
                            dict_pvalues = pool.starmap(ResMet.URA.ks_tests,
                                                        zip(repeat(real_data[num_features]), synthetic_datasets))
                    elif value == 't_test':
                        with Pool(round(n_cpu * 0.25)) as pool:
                            dict_pvalues = pool.starmap(ResMet.URA.student_t_tests,
                                                        zip(repeat(real_data[num_features]), synthetic_datasets))
                    elif value == 'u_test':
                        with Pool(round(n_cpu * 0.25)) as pool:
                            dict_pvalues = pool.starmap(ResMet.URA.mann_whitney_tests,
                                                        zip(repeat(real_data[num_features]), synthetic_datasets))

                    threshold = 0.05
                    dict_check_threshold = [{key: value < threshold for key, value in d.items()} for d in dict_pvalues]
                    sum_of_rejected = [sum(d.values()) for d in dict_check_threshold]
                    ranking_list = stats.rankdata(sum_of_rejected, method='min')
                    ranking_list_ura.append({"Numerical statistical test": ranking_list})
                    ranking_resemblance.append(ranking_list)

                elif item == "cat-test":
                    value = opt_ura[len(list_ura) - helping_var]

                    if value == 'chi_test':
                        with Pool(round(n_cpu * 0.25)) as pool:
                            dict_pvalues = pool.starmap(ResMet.URA.chi_squared_tests,
                                                        zip(repeat(real_data[cat_features]), synthetic_datasets))

                    threshold = 0.05
                    dict_check_threshold = [{key: value > threshold for key, value in d.items()} for d in dict_pvalues]
                    sum_of_accepted = [sum(d.values()) for d in dict_check_threshold]
                    ranking_list = stats.rankdata(sum_of_accepted, method='min')
                    ranking_list_ura.append({"Categorical statistical test": ranking_list})
                    ranking_resemblance.append(ranking_list)

                elif item == "dist":
                    value = opt_ura[len(list_ura) - helping_var]

                    if value == 'cos_dist':
                        with Pool(round(n_cpu * 0.25)) as pool:
                            dict_distances = pool.starmap(ResMet.URA.cosine_distances,
                                                          zip(repeat(real_data[num_features]), synthetic_datasets))
                    elif value == 'js_dist':
                        with Pool(round(n_cpu * 0.25)) as pool:
                            dict_distances = pool.starmap(ResMet.URA.js_distances,
                                                          zip(repeat(real_data[num_features]), synthetic_datasets))
                    elif value == 'w_dist':
                        with Pool(round(n_cpu * 0.25)) as pool:
                            dict_distances = pool.starmap(ResMet.URA.wass_distances,
                                                          zip(repeat(real_data[num_features]), synthetic_datasets))

                    mean_list = [sum(d.values()) / len(d) for d in dict_distances]
                    ranking_list = stats.rankdata(mean_list, method='min')
                    ranking_list_ura.append({"Distance metric": ranking_list})
                    ranking_resemblance.append(ranking_list)

                helping_var -= 1

            # Graphical output ranking URA
            values = [list(d.keys()) + [f"Rank {num}" for num in list(d.values())[0]]
                      for d in ranking_list_ura]
            df_rank = pd.DataFrame(values)
            df_rank.columns = [''] + [f'Dataset {i}' for i in range(1, len(df_rank.columns))]

            content_resemblance.append(
                html.Div([
                    dbc.Row([
                        dbc.Col([html.H5("Univariate Resemblance Analysis",
                                         style={"font-weight": "bold"})],
                                width="auto")
                    ], style={'margin-bottom': '2vw', 'margin-left': '3.5vw'}),

                    dbc.Row([
                        dbc.Col([
                            dash_table.DataTable(
                                columns=[{'name': col, 'id': col} for col in df_rank.columns],
                                data=df_rank.to_dict('records'),
                                page_action='none',
                                style_table={'height': 'auto',
                                             'width': 'auto',
                                             'overflowY': 'auto',
                                             'overflowX': 'auto'
                                             },
                                fixed_rows={'headers': True},
                                style_header={'text-align': 'center'},
                                style_cell={'minWidth': 100, 'maxWidth': 100, 'width': 100}
                            )
                        ])], style={'margin-bottom': '1vw', 'margin-left': '4vw'}),

                ], style={'margin-bottom': '3vw', 'margin-top': '3vw', 'margin-left': '5vw',
                          'margin-right': '5vw'})
            )
            content_resemblance.append(html.Hr())

            data_ranking_resemblance = data_ranking_resemblance + ranking_list_ura

            data_report_tbl['Univariate Resemblance Analysis'] = values

        # MRA ranking calculation
        if list_mra:
            ranking_list_mra = []
            for item in list_mra:

                if item == "corr_num":
                    mat_real = ResMet.MRA.compute_correlations_matrices(real_data, "corr_num", num_features)

                    with Pool(round(n_cpu * 0.25)) as pool:
                        mats_syns = pool.starmap(ResMet.MRA.compute_correlations_matrices,
                                                 zip(synthetic_datasets, repeat("corr_num"), repeat(num_features)))

                    mats_diff = [np.abs(mat_real - mat_syn) for mat_syn in mats_syns]
                    mean_diffs = [round(np.nanmean(mat_diff.values[np.triu_indices(len(mat_diff), k=1)]), 4) for mat_diff
                                  in mats_diff]
                    ranking_list = stats.rankdata(mean_diffs, method='min')
                    ranking_list_mra.append({"Correlation matrices": ranking_list})
                    ranking_resemblance.append(ranking_list)

                elif item == "corr_cat":
                    mat_real = ResMet.MRA.compute_correlations_matrices(real_data, "corr_cat", cat_features)

                    with Pool(round(n_cpu * 0.25)) as pool:
                        mats_syns = pool.starmap(ResMet.MRA.compute_correlations_matrices,
                                                 zip(synthetic_datasets, repeat("corr_cat"), repeat(cat_features)))

                    mats_diff = [np.abs(mat_real - mat_syn) for mat_syn in mats_syns]
                    mean_diffs = [np.nanmean(mat_diff.values[np.triu_indices(len(mat_diff), k=1)]) for mat_diff in
                                  mats_diff]
                    ranking_list = stats.rankdata(mean_diffs, method='min')
                    ranking_list_mra.append({"Contingency tables": ranking_list})
                    ranking_resemblance.append(ranking_list)

                elif item == "pca":
                    real_var_ratio_cum = ResMet.MRA.do_pca(real_data)

                    with Pool(round(n_cpu * 0.25)) as pool:
                        syns_var_ratio_cum = pool.map(ResMet.MRA.do_pca, synthetic_datasets)

                    rmse_pca = [round(mean_squared_error(real_var_ratio_cum, s, squared=False)) for s in syns_var_ratio_cum]
                    ranking_list = stats.rankdata(rmse_pca, method='min')
                    ranking_list_mra.append({"Principal Component Analysis": ranking_list})
                    ranking_resemblance.append(ranking_list)

            # Graphical output ranking MRA
            values = [list(d.keys()) + [f"Rank {num}" for num in list(d.values())[0]]
                      for d in ranking_list_mra]
            df_rank = pd.DataFrame(values)
            df_rank.columns = [''] + [f'Dataset {i}' for i in range(1, len(df_rank.columns))]

            content_resemblance.append(
                html.Div([
                    dbc.Row([
                        dbc.Col([html.H5("Multivariate Relationships Analysis",
                                         style={"font-weight": "bold"})],
                                width="auto")
                    ], style={'margin-bottom': '2vw', 'margin-left': '3.5vw'}),

                    dbc.Row([
                        dbc.Col([
                            dash_table.DataTable(
                                columns=[{'name': col, 'id': col} for col in df_rank.columns],
                                data=df_rank.to_dict('records'),
                                page_action='none',
                                style_table={'height': 'auto',
                                             'width': 'auto',
                                             'overflowY': 'auto',
                                             'overflowX': 'auto'
                                             },
                                fixed_rows={'headers': True},
                                style_header={'text-align': 'center'},
                                style_cell={'minWidth': 100, 'maxWidth': 100, 'width': 100}
                            )
                        ])], style={'margin-bottom': '1vw', 'margin-left': '4vw'}),

                ], style={'margin-bottom': '3vw', 'margin-top': '3vw', 'margin-left': '5vw',
                          'margin-right': '5vw'})
            )
            content_resemblance.append(html.Hr())

            data_ranking_resemblance = data_ranking_resemblance + ranking_list_mra

            data_report_tbl['Multivariate Relationships Analysis'] = values

        # DLA ranking calculation
        if list_dla:
            with Pool(round(n_cpu * 0.25)) as pool:
                results_dla = pool.starmap(ResMet.DLA.classify_real_vs_synthetic_data,
                                           zip(repeat(list_dla), repeat(real_data.copy()),
                                               synthetic_datasets.copy(),
                                               repeat(num_features), repeat(cat_features)))

            means_f1 = [np.mean(r['f1']) for r in results_dla]
            ranking_list_dla = stats.rankdata(means_f1, method='min')
            ranking_resemblance.append(ranking_list_dla)

            # Graphical output ranking DLA
            values = [f"Rank {num}" for num in ranking_list_dla]
            df_rank = pd.DataFrame([values])
            df_rank.columns = [f'Dataset {i}' for i in range(1, len(df_rank.columns) + 1)]

            content_resemblance.append(
                html.Div([
                    dbc.Row([
                        dbc.Col([html.H5("Data Labeling Analysis",
                                         style={"font-weight": "bold"})],
                                width="auto")
                    ], style={'margin-bottom': '2vw', 'margin-left': '3.5vw'}),

                    dbc.Row([
                        dbc.Col([
                            dash_table.DataTable(
                                columns=[{'name': col, 'id': col} for col in df_rank.columns],
                                data=df_rank.to_dict('records'),
                                page_action='none',
                                style_table={'height': 'auto',
                                             'width': 'auto',
                                             'overflowY': 'auto',
                                             'overflowX': 'auto'
                                             },
                                fixed_rows={'headers': True},
                                style_header={'text-align': 'center'},
                                style_cell={'minWidth': 100, 'maxWidth': 100, 'width': 100}
                            )
                        ])], style={'margin-bottom': '1vw', 'margin-left': '4vw'}),

                ], style={'margin-bottom': '3vw', 'margin-top': '3vw', 'margin-left': '5vw',
                          'margin-right': '5vw'})
            )
            content_resemblance.append(html.Hr())

            data_ranking_resemblance.append({"Data labeling analysis": ranking_list_dla})

            data_report_tbl['Data Labeling Analysis'] = [values]

        # Ranking comparison figure (resemblance)
        if len(synthetic_datasets) > 1 and len(ranking_resemblance) > 1:
            cumulative_resemblance = compute_cumulative_rank(ranking_resemblance)

            fig_rank_resemblance = go.Figure()
            x_values = list(range(1, len(synthetic_datasets) + 1))
            for i, cumulative_values in cumulative_resemblance.items():
                fig_rank_resemblance.add_trace(go.Scatter(x=x_values, y=cumulative_values,
                                                          mode='lines+markers', fill='tozeroy',
                                                          name=f'Dataset {i}'))

            fig_rank_resemblance.update_layout(
                title='Cumulative Distribution Function of ranking',
                xaxis=dict(title='Rank', tickvals=x_values, tickmode='array'),
                legend=dict(title='Synthetic datasets'),
            )

            auc_ranking = {d: round(auc(range(1, len(v) + 1), v), 2) for d, v in cumulative_resemblance.items()}
            df_auc_ranking = pd.DataFrame(list(auc_ranking.items()), columns=['Synthetic dataset', 'Ranking curve AUC'])

            content_resemblance.append(
                html.Div([
                    dbc.Row([
                        dbc.Col([html.H5("Synthetic datasets ranking comparison (resemblance)",
                                         style={"font-weight": "bold"})],
                                width="auto")
                    ], style={'margin-bottom': '2vw', 'margin-left': '3.5vw'}),

                    dbc.Row([

                        dbc.Col([
                            dcc.Graph(figure=fig_rank_resemblance)
                        ], width={'size': 8}),

                        dbc.Col([
                            dash_table.DataTable(
                                columns=[{'name': col, 'id': col} for col in df_auc_ranking.columns],
                                data=df_auc_ranking.to_dict('records'),
                                page_action='none',
                                sort_action='native',
                                style_table={'height': 'auto',
                                             'width': 'auto',
                                             'overflowY': 'auto',
                                             'overflowX': 'auto'
                                             },
                                fixed_rows={'headers': True},
                                style_header={'text-align': 'center'},
                                style_cell={'minWidth': 100, 'maxWidth': 100, 'width': 100}
                            )
                        ], width={'size': 4}),

                    ], align="center", style={'margin-bottom': '1vw', 'margin-left': '4vw'}),

                ], style={'margin-bottom': '3vw', 'margin-top': '3vw', 'margin-left': '4.5vw',
                          'margin-right': '5vw'})
            )

            data_report_fig.append([fig_rank_resemblance, auc_ranking, 'Resemblance ranking comparison'])

        # UTL ranking calculation
        if list_utl:
            target = opt_utl[0]
            classifier = opt_utl[1]

            if not real_train_data.empty and not real_test_data.empty:
                train_data_r = real_train_data.drop(columns=target)
                test_data_r = real_test_data.drop(columns=target)
                train_labels_r = real_train_data[target]
                test_labels_r = real_test_data[target]
            else:
                train_data_r, test_data_r, train_labels_r, test_labels_r = train_test_split(
                    real_data.drop(columns=target), real_data[target], test_size=0.3, random_state=9)

            num = num_features.copy()
            cat = cat_features.copy()
            if target in num:
                num.remove(target)
            if target in cat:
                cat.remove(target)

            train_data_s = []
            train_labels_s = []
            for syn_data in synthetic_datasets:
                train_data, _, train_labels, _ = train_test_split(syn_data.drop(columns=target),
                                                                  syn_data[target], test_size=0.2, random_state=19)
                train_data_s.append(train_data)
                train_labels_s.append(train_labels)

            with Pool(round(n_cpu * 0.25)) as pool:
                results_utl = pool.starmap(UtiMet.train_test_model,
                                           zip(repeat(classifier), [train_data_r] + train_data_s, repeat(test_data_r),
                                               [train_labels_r] + train_labels_s, repeat(test_labels_r),
                                               repeat(num), repeat(cat)))

            result_trtr = results_utl[0]
            results_tstr = results_utl[1:]

            means_diff_f1 = [np.mean(np.abs(result_trtr[0]['f1'] - result_tstr[0]['f1'])) for result_tstr in
                             results_tstr]
            ranking_list_utl = stats.rankdata(means_diff_f1, method='min')
            ranking_utility.append(ranking_list_utl)

            # Graphical output ranking UTL
            values = [f"Rank {num}" for num in ranking_list_utl]
            df_rank = pd.DataFrame([values])
            df_rank.columns = [f'Dataset {i}' for i in range(1, len(df_rank.columns) + 1)]

            content_utility.append(
                html.Div([
                    dbc.Row([
                        dbc.Col([html.H5("Train on Real vs. Train on Synthetic",
                                         style={"font-weight": "bold"})],
                                width="auto")
                    ], style={'margin-bottom': '2vw', 'margin-left': '3.5vw'}),

                    dbc.Row([
                        dbc.Col([
                            dash_table.DataTable(
                                columns=[{'name': col, 'id': col} for col in df_rank.columns],
                                data=df_rank.to_dict('records'),
                                page_action='none',
                                style_table={'height': 'auto',
                                             'width': 'auto',
                                             'overflowY': 'auto',
                                             'overflowX': 'auto'
                                             },
                                fixed_rows={'headers': True},
                                style_header={'text-align': 'center'},
                                style_cell={'minWidth': 100, 'maxWidth': 100, 'width': 100}
                            )
                        ])], style={'margin-bottom': '1vw', 'margin-left': '4vw'}),

                ], style={'margin-bottom': '3vw', 'margin-top': '3vw', 'margin-left': '5vw',
                          'margin-right': '5vw'})
            )
            content_utility.append(html.Hr())

            data_ranking_utility.append({"TRTR-TSTR analysis": ranking_list_utl})

            data_report_tbl['TRTR-TSTR Analysis'] = [values]

        # SEA ranking calculation
        if list_sea:
            if list_sea == "euc":
                with Pool(round(n_cpu * 0.40)) as pool:
                    mats_dist = pool.starmap(PriMet.SEA.pairwise_euclidean_distance,
                                             zip(repeat(real_data), synthetic_datasets))
                    dists = [np.mean(array, axis=None) for array in mats_dist]
            elif list_sea == "cos":
                with Pool(round(n_cpu * 0.40)) as pool:
                    mats_dist = pool.starmap(PriMet.SEA.str_similarity,
                                             zip(repeat(real_data), synthetic_datasets))
                    dists = [np.mean(1 - array, axis=None) for array in mats_dist]
            elif list_sea == "hau":
                with Pool(round(n_cpu * 0.40)) as pool:
                    dists = pool.starmap(PriMet.SEA.hausdorff_distance,
                                         zip(repeat(real_data), synthetic_datasets))

            ranking_list_sea = len(synthetic_datasets) + 1 - stats.rankdata(dists, method='max')
            ranking_privacy.append(ranking_list_sea)

            # Graphical output ranking SEA
            values = [f"Rank {num}" for num in ranking_list_sea]
            df_rank = pd.DataFrame([values])
            df_rank.columns = [f'Dataset {i}' for i in range(1, len(df_rank.columns) + 1)]

            content_privacy.append(
                html.Div([
                    dbc.Row([
                        dbc.Col([html.H5("Similarity Evaluation Analysis",
                                         style={"font-weight": "bold"})],
                                width="auto")
                    ], style={'margin-bottom': '2vw', 'margin-left': '3.5vw'}),

                    dbc.Row([
                        dbc.Col([
                            dash_table.DataTable(
                                columns=[{'name': col, 'id': col} for col in df_rank.columns],
                                data=df_rank.to_dict('records'),
                                page_action='none',
                                style_table={'height': 'auto',
                                             'width': 'auto',
                                             'overflowY': 'auto',
                                             'overflowX': 'auto'
                                             },
                                fixed_rows={'headers': True},
                                style_header={'text-align': 'center'},
                                style_cell={'minWidth': 100, 'maxWidth': 100, 'width': 100}
                            )
                        ])], style={'margin-bottom': '1vw', 'margin-left': '4vw'}),

                ], style={'margin-bottom': '3vw', 'margin-top': '3vw', 'margin-left': '5vw',
                          'margin-right': '5vw'})
            )
            content_privacy.append(html.Hr())

            data_ranking_privacy.append({"Similarity evaluation analysis": ranking_list_sea})

            data_report_tbl['Similarity Evaluation Analysis'] = [values]

        # MIA ranking calculation
        if list_mia:
            prop_subset = opt_mia[0]
            t_similarity = opt_mia[1]
            real_subset = real_data.sample(frac=prop_subset / 100, random_state=42).reset_index(drop=True)

            label_membership_train = [tuple(row) in set(real_train_data.itertuples(index=False))
                                      for row in real_subset.itertuples(index=False)]

            with Pool(round(n_cpu * 0.40)) as pool:
                results_mia = pool.starmap(PriMet.MIA.simulate_mia,
                                           zip(repeat(real_subset), repeat(label_membership_train), synthetic_datasets,
                                               repeat(t_similarity)))

            acc_values = [res[1] for res in results_mia]
            ranking_list_mia = stats.rankdata(acc_values, method='min')
            ranking_privacy.append(ranking_list_mia)

            # Graphical output ranking MIA
            values = [f"Rank {num}" for num in ranking_list_mia]
            df_rank = pd.DataFrame([values])
            df_rank.columns = [f'Dataset {i}' for i in range(1, len(df_rank.columns) + 1)]

            content_privacy.append(
                html.Div([
                    dbc.Row([
                        dbc.Col([html.H5("Membership Inference Analysis",
                                         style={"font-weight": "bold"})],
                                width="auto")
                    ], style={'margin-bottom': '2vw', 'margin-left': '3.5vw'}),

                    dbc.Row([
                        dbc.Col([
                            dash_table.DataTable(
                                columns=[{'name': col, 'id': col} for col in df_rank.columns],
                                data=df_rank.to_dict('records'),
                                page_action='none',
                                style_table={'height': 'auto',
                                             'width': 'auto',
                                             'overflowY': 'auto',
                                             'overflowX': 'auto'
                                             },
                                fixed_rows={'headers': True},
                                style_header={'text-align': 'center'},
                                style_cell={'minWidth': 100, 'maxWidth': 100, 'width': 100}
                            )
                        ])], style={'margin-bottom': '1vw', 'margin-left': '4vw'}),

                ], style={'margin-bottom': '3vw', 'margin-top': '3vw', 'margin-left': '5vw',
                          'margin-right': '5vw'})
            )
            content_privacy.append(html.Hr())

            data_ranking_privacy.append({"Membership inference attack": ranking_list_mia})

            data_report_tbl['Membership Inference Attack'] = [values]

        # AIA ranking calculation
        if list_aia:
            prop_subset = opt_aia[0]
            attributes = opt_aia[1]
            real_subset = real_data.sample(frac=prop_subset / 100, random_state=24).reset_index(drop=True)
            targets = [col for col in real_data.columns if col not in attributes]

            with Pool(round(n_cpu * 0.40)) as pool:
                results_aia = pool.starmap(PriMet.AIA.simulate_aia,
                                           zip(repeat(real_subset), synthetic_datasets,
                                               repeat(attributes), repeat(targets), repeat(dict_features_type)))

            acc_means = [np.mean(res[res['Metric name'] == 'acc'][['Value']]) for res in results_aia]
            rank_acc = stats.rankdata(acc_means, method='min')

            rmse_means = []
            for res in results_aia:
                rmse = res[res['Metric name'] == 'rmse'][['Value']]
                normalized_rmse = np.interp(rmse, (np.min(rmse), np.max(rmse)), (0, 1))
                rmse_means.append(np.mean(normalized_rmse))

            rank_rmse = len(synthetic_datasets) + 1 - stats.rankdata(rmse_means, method='max')

            helping_list = rank_acc + rank_rmse
            ranking_list_aia = stats.rankdata(helping_list, method='min')
            ranking_privacy.append(ranking_list_aia)

            # Graphical output ranking AIA
            values = [f"Rank {num}" for num in ranking_list_aia]
            df_rank = pd.DataFrame([values])
            df_rank.columns = [f'Dataset {i}' for i in range(1, len(df_rank.columns) + 1)]

            content_privacy.append(
                html.Div([
                    dbc.Row([
                        dbc.Col([html.H5("Attribute Inference Analysis",
                                         style={"font-weight": "bold"})],
                                width="auto")
                    ], style={'margin-bottom': '2vw', 'margin-left': '3.5vw'}),

                    dbc.Row([
                        dbc.Col([
                            dash_table.DataTable(
                                columns=[{'name': col, 'id': col} for col in df_rank.columns],
                                data=df_rank.to_dict('records'),
                                page_action='none',
                                style_table={'height': 'auto',
                                             'width': 'auto',
                                             'overflowY': 'auto',
                                             'overflowX': 'auto'
                                             },
                                fixed_rows={'headers': True},
                                style_header={'text-align': 'center'},
                                style_cell={'minWidth': 100, 'maxWidth': 100, 'width': 100}
                            )
                        ])], style={'margin-bottom': '1vw', 'margin-left': '4vw'}),

                ], style={'margin-bottom': '3vw', 'margin-top': '3vw', 'margin-left': '5vw',
                          'margin-right': '5vw'})
            )
            content_privacy.append(html.Hr())

            data_ranking_privacy.append({"Attribute inference attack": ranking_list_aia})

            data_report_tbl['Attribute Inference Attack'] = [values]

        # Ranking comparison figure (privacy)
        if len(synthetic_datasets) > 1 and len(ranking_privacy) > 1:
            cumulative_privacy = compute_cumulative_rank(ranking_privacy)

            fig_rank_privacy = go.Figure()
            x_values = list(range(1, len(synthetic_datasets) + 1))
            for i, cumulative_values in cumulative_privacy.items():
                fig_rank_privacy.add_trace(go.Scatter(x=x_values, y=cumulative_values,
                                                      mode='lines+markers', fill='tozeroy',
                                                      name=f'Dataset {i}'))

            fig_rank_privacy.update_layout(
                title='Cumulative Distribution Function of ranking',
                xaxis=dict(title='Rank', tickvals=x_values, tickmode='array'),
                legend=dict(title='Synthetic datasets'),
            )

            auc_ranking = {d: round(auc(range(1, len(v) + 1), v), 2) for d, v in cumulative_privacy.items()}
            df_auc_ranking = pd.DataFrame(list(auc_ranking.items()), columns=['Synthetic dataset', 'Ranking curve AUC'])

            content_privacy.append(
                html.Div([
                    dbc.Row([
                        dbc.Col([html.H5("Synthetic datasets ranking comparison (privacy)",
                                         style={"font-weight": "bold"})],
                                width="auto")
                    ], style={'margin-bottom': '2vw', 'margin-left': '3.5vw'}),

                    dbc.Row([

                        dbc.Col([
                            dcc.Graph(figure=fig_rank_privacy)
                        ], width={'size': 8}),

                        dbc.Col([
                            dash_table.DataTable(
                                columns=[{'name': col, 'id': col} for col in df_auc_ranking.columns],
                                data=df_auc_ranking.to_dict('records'),
                                page_action='none',
                                sort_action='native',
                                style_table={'height': 'auto',
                                             'width': 'auto',
                                             'overflowY': 'auto',
                                             'overflowX': 'auto'},
                                fixed_rows={'headers': True},
                                style_header={'text-align': 'center'},
                                style_cell={'minWidth': 100, 'maxWidth': 100, 'width': 100}
                            )
                        ], width={'size': 4}),

                    ], align="center", style={'margin-bottom': '1vw', 'margin-left': '4vw'}),

                ], style={'margin-bottom': '3vw', 'margin-top': '3vw', 'margin-left': '4.5vw',
                          'margin-right': '5vw'})
            )

            data_report_fig.append([fig_rank_privacy, auc_ranking, 'Privacy ranking comparison'])

        # Ranking comparison figure (all metrics)
        ranking_overall = ranking_resemblance + ranking_utility + ranking_privacy
        if len(synthetic_datasets) > 1 and len(ranking_overall) > 1:
            cumulative_overall = compute_cumulative_rank(ranking_overall)

            fig_rank_overall = go.Figure()
            x_values = list(range(1, len(synthetic_datasets) + 1))
            for i, cumulative_values in cumulative_overall.items():
                fig_rank_overall.add_trace(go.Scatter(x=x_values, y=cumulative_values,
                                                      mode='lines+markers', fill='tozeroy',
                                                      name=f'Dataset {i}'))

            fig_rank_overall.update_layout(
                title='Cumulative Distribution Function of ranking',
                xaxis=dict(title='Rank', tickvals=x_values, tickmode='array'),
                legend=dict(title='Synthetic datasets'),
            )

            auc_ranking = {d: round(auc(range(1, len(v) + 1), v), 2) for d, v in cumulative_overall.items()}
            df_auc_ranking = pd.DataFrame(list(auc_ranking.items()), columns=['Synthetic dataset', 'Ranking curve AUC'])

            content_overall.append(
                html.Div([
                    dbc.Row([
                        dbc.Col([html.H5("Synthetic datasets ranking comparison (all metrics)",
                                         style={"font-weight": "bold"})],
                                width="auto")
                    ], style={'margin-bottom': '1vw', 'margin-left': '3.5vw'}),

                    dbc.Row([

                        dbc.Col([
                            dcc.Graph(figure=fig_rank_overall)
                        ], width={'size': 8}),

                        dbc.Col([
                            dash_table.DataTable(
                                columns=[{'name': col, 'id': col} for col in df_auc_ranking.columns],
                                data=df_auc_ranking.to_dict('records'),
                                page_action='none',
                                sort_action='native',
                                style_table={'height': 'auto',
                                             'width': 'auto',
                                             'overflowY': 'auto',
                                             'overflowX': 'auto'},
                                fixed_rows={'headers': True},
                                style_header={'text-align': 'center'},
                                style_cell={'minWidth': 100, 'maxWidth': 100, 'width': 100}
                            )
                        ], width={'size': 4}),

                    ], align="center", style={'margin-bottom': '1vw', 'margin-left': '4vw'}),

                ])
            )

            data_report_fig.append([fig_rank_overall, auc_ranking, 'All metrics ranking comparison'])

        children = [
            dbc.Row([dbc.Col([
                dbc.Accordion(
                    [
                        dbc.AccordionItem(
                            content_resemblance, title="Resemblance Results"
                        ),
                        dbc.AccordionItem(
                            content_utility, title="Utility Results"
                        ),
                        dbc.AccordionItem(
                            content_privacy, title="Privacy Results"
                        ),
                    ],
                    always_open=True,
                    start_collapsed=True,
                    flush=True,
                )
            ])], style={'margin-bottom': '4vw', 'margin-left': '5vw', 'margin-right': '5vw'}),

            dbc.Row([dbc.Col(
                content_overall
            )], style={'margin-bottom': '2vw', 'margin-left': '5vw', 'margin-right': '5vw'}),
        ]

        if len(ranking_resemblance) > 0:
            data_all_ranking.append({"Resemblance metrics": data_ranking_resemblance})
        if len(ranking_utility) > 0:
            data_all_ranking.append({"Utility metrics": data_ranking_utility})
        if len(ranking_privacy) > 0:
            data_all_ranking.append({"Privacy metrics": data_ranking_privacy})

        return children, [], data_all_ranking, [data_report_fig, data_report_tbl]

    else:
        raise PreventUpdate


@app.callback(Output("use-case-sliders", "children"),
              Input("data-ranking", "data"),
              prevent_initial_call=True)
def create_sliders(data):
    children = [html.Div([
        dbc.Row([

            dbc.Col([
                html.H5(["Metrics weights", html.Span("ℹ", id="info-weights", style={'cursor': 'pointer'})],
                        style={"font-weight": "bold"}),
            ], width={'size': 'auto'}),

            dbc.Tooltip(
                "Associate different weights (importance) to various groups of metrics.",
                target="info-weights",
            ),

            dbc.Col([
                dbc.Button("Compute Final Score", color="primary", id="next3-bm")
            ], width={'size': 'auto'})

        ], align='center'),
        html.Br(),
        dbc.Row([
            dbc.Col([
                dbc.Button("Reset default weights", id="reset-weights", color="light")
            ], width={'size': 'auto'})
        ])
    ], style={'margin-top': '2vw', 'margin-bottom': '2vw', 'margin-left': '5vw', 'margin-right': '5vw'})]

    labels_1 = [list(d.keys())[0] for d in data]
    initial_value_1 = 1 / len(labels_1)
    for i, dict_1_level in enumerate(data):  # first level metrics (resemblance, utility, privacy)

        # Sliders first level
        children.append(
            html.Div([
                dbc.Row([
                    dbc.Col([
                        html.P(html.B(labels_1[i]))
                    ], width={'size': 'auto'}),
                ], style={'margin-bottom': '0.5vw', 'margin-left': '5vw', 'margin-right': '5vw'}),

                dbc.Row([

                    dbc.Col([
                        dcc.Slider(0, 1, 0.01,
                                   tooltip={"placement": "bottom", "always_visible": True},
                                   marks={
                                       0: '0', 0.25: '0.25', 0.5: '0.5', 0.75: '0.75', 1: '1'
                                   },
                                   value=initial_value_1,
                                   id={"type": "slider-bm", "index": i})
                    ], width={'size': 8}, style={'padding-left': '0vw', 'padding-right': '0vw'}),

                    dbc.Col([
                        dbc.Checkbox(
                            id={"type": "block-sliders", "index": i},
                            label="lock",
                            value=False
                        )
                    ])

                ], align='center', style={'margin-bottom': '2vw', 'margin-left': '5vw', 'margin-right': '5vw'}),
            ])
        )

        labels_2 = [list(d.keys())[0] for d in dict_1_level[labels_1[i]]]
        initial_value_2 = 1 / len(labels_2)
        content_accordion = []
        for j, dict_2_level in enumerate(dict_1_level[labels_1[i]]):  # second level metrics

            # Sliders second level
            content_accordion.append(
                html.Div([
                    dbc.Row([
                        dbc.Col([
                            html.P(labels_2[j])
                        ], width={'size': 'auto'}),
                    ], style={'margin-bottom': '0.5vw'}),

                    dbc.Row([

                        dbc.Col([
                            dcc.Slider(0, 1, 0.01,
                                       id={"type": "slider-bm-" + labels_1[i][:3], "index": j},
                                       tooltip={"placement": "bottom", "always_visible": True},
                                       marks={
                                           0: '0', 0.25: '0.25', 0.5: '0.5', 0.75: '0.75', 1: '1'
                                       },
                                       value=initial_value_2,
                                       )
                        ], width={'size': 8}, style={'padding-left': '0vw', 'padding-right': '0vw'}),

                        dbc.Col([
                            dbc.Checkbox(
                                id={"type": "block-sliders-" + labels_1[i][:3], "index": j},
                                label="lock",
                                value=False
                            )
                        ])

                    ], align='center', style={'margin-bottom': '2vw'})
                ])
            )

        children.append(
            html.Div([
                dbc.Row([
                    dbc.Accordion(
                        [
                            dbc.AccordionItem(
                                content_accordion, title="See all metrics"
                            ),
                        ],
                        always_open=True,
                        start_collapsed=True,
                        # flush=True,
                    )
                ], style={'margin-bottom': '2vw', 'margin-left': '5vw', 'margin-right': '5vw'})
            ])
        )

    return children


@app.callback(Output({"type": "slider-bm", "index": ALL}, "value", allow_duplicate=True),
              Output({"type": "slider-bm-Res", "index": ALL}, "value", allow_duplicate=True),
              Output({"type": "slider-bm-Uti", "index": ALL}, "value", allow_duplicate=True),
              Output({"type": "slider-bm-Pri", "index": ALL}, "value", allow_duplicate=True),
              Input("reset-weights", "n_clicks"),
              State({"type": "slider-bm", "index": ALL}, "value"),
              State({"type": "slider-bm-Res", "index": ALL}, "value"),
              State({"type": "slider-bm-Uti", "index": ALL}, "value"),
              State({"type": "slider-bm-Pri", "index": ALL}, "value"),
              prevent_initial_callback=True)
def reset_values(click, values0, values1, values2, values3):
    if click:
        return [[1 / len(v)] * len(v) if v else [] for v in [values0, values1, values2, values3]]

    else:
        raise PreventUpdate


@app.callback(Output({"type": "block-sliders", "index": ALL}, "disabled"),
              Input({"type": "block-sliders", "index": ALL}, "value"))
def disabled_blocks(values):
    if sum(values) >= len(values) - 2:
        return [not v for v in values]
    else:
        return [False] * len(values)


@app.callback(Output({"type": "block-sliders-Res", "index": ALL}, "disabled"),
              Input({"type": "block-sliders-Res", "index": ALL}, "value"))
def disabled_blocks(values):
    if sum(values) >= len(values) - 2:
        return [not v for v in values]
    else:
        return [False] * len(values)


@app.callback(Output({"type": "block-sliders-Uti", "index": ALL}, "disabled"),
              Input({"type": "block-sliders-Uti", "index": ALL}, "value"))
def disabled_blocks(values):
    if sum(values) >= len(values) - 2:
        return [not v for v in values]
    else:
        return [False] * len(values)


@app.callback(Output({"type": "block-sliders-Pri", "index": ALL}, "disabled"),
              Input({"type": "block-sliders-Pri", "index": ALL}, "value"))
def disabled_blocks(values):
    if sum(values) >= len(values) - 2:
        return [not v for v in values]
    else:
        return [False] * len(values)


@app.callback(Output({"type": "slider-bm", "index": ALL}, "disabled"),
              Input({"type": "block-sliders", "index": ALL}, "value"),
              prevent_initial_call=True)
def disabled_sliders(values):
    return values


@app.callback(Output({"type": "slider-bm-Res", "index": ALL}, "disabled"),
              Input({"type": "block-sliders-Res", "index": ALL}, "value"),
              prevent_initial_call=True)
def disabled_sliders(values):
    return values


@app.callback(Output({"type": "slider-bm-Uti", "index": ALL}, "disabled"),
              Input({"type": "block-sliders-Uti", "index": ALL}, "value"),
              prevent_initial_call=True)
def disabled_sliders(values):
    return values


@app.callback(Output({"type": "slider-bm-Pri", "index": ALL}, "disabled"),
              Input({"type": "block-sliders-Pri", "index": ALL}, "value"),
              prevent_initial_call=True)
def disabled_sliders(values):
    return values


def auto_adjust_value(trigger_id, values, blocks):

    max_value = 1 - sum(v for v, b in zip(values, blocks) if b)
    if values[trigger_id] >= max_value:

        values[trigger_id] = max_value

        for i, v in enumerate(values):
            if not i == trigger_id and not blocks[i]:
                values[i] = 0

        return values

    diff_value = 1 - sum(values)
    avl_sliders = (len(values) - sum(blocks) - 1)
    upd_value = diff_value / avl_sliders

    sorted_indexed_values = sorted(list(enumerate(values)), key=lambda x: x[1])
    sorted_indices = [index for index, value in sorted_indexed_values]

    k = 0
    for i in sorted_indices:
        if not i == trigger_id and not blocks[i]:
            if values[i] + upd_value < 0:
                k += 1
                upd_value += (values[i] + upd_value) / (avl_sliders - k)
                values[i] = 0
            else:
                values[i] += upd_value

    return values


@app.callback(Output({"type": "slider-bm", "index": ALL}, "value"),
              Input({"type": "slider-bm", "index": ALL}, "value"),
              State({"type": "block-sliders", "index": ALL}, "value"),
              prevent_initial_call=True)
def synchronizing_top_sliders(values, blocks):
    if len(values) == 1:  # single slider
        return [1]

    trigger_id = int(ctx.triggered[0]["prop_id"].split(".")[0].split(",")[0].split(":")[1])
    return auto_adjust_value(trigger_id, values, blocks)


@app.callback(Output({"type": "slider-bm-Res", "index": ALL}, "value"),
              Input({"type": "slider-bm-Res", "index": ALL}, "value"),
              State({"type": "block-sliders-Res", "index": ALL}, "value"),
              prevent_initial_call=True)
def synchronizing_res_sliders(values, blocks):
    if len(values) == 1:  # single slider
        return [1]

    trigger_id = int(ctx.triggered[0]["prop_id"].split(".")[0].split(",")[0].split(":")[1])
    return auto_adjust_value(trigger_id, values, blocks)


@app.callback(Output({"type": "slider-bm-Uti", "index": ALL}, "value"),
              Input({"type": "slider-bm-Uti", "index": ALL}, "value"),
              State({"type": "block-sliders-Uti", "index": ALL}, "value"),
              prevent_initial_call=True)
def synchronizing_uti_sliders(values, blocks):
    if len(values) == 1:  # single slider
        return [1]

    trigger_id = int(ctx.triggered[0]["prop_id"].split(".")[0].split(",")[0].split(":")[1])
    return auto_adjust_value(trigger_id, values, blocks)


@app.callback(Output({"type": "slider-bm-Pri", "index": ALL}, "value"),
              Input({"type": "slider-bm-Pri", "index": ALL}, "value"),
              State({"type": "block-sliders-Pri", "index": ALL}, "value"),
              prevent_initial_call=True)
def synchronizing_pri_sliders(values, blocks):
    if len(values) == 1:  # single slider
        return [1]

    trigger_id = int(ctx.triggered[0]["prop_id"].split(".")[0].split(",")[0].split(":")[1])
    return auto_adjust_value(trigger_id, values, blocks)


@app.callback(Output("use-case-results", "children"),
              Output({"type": "data-report", "index": 13}, 'data'),
              Input("next3-bm", "n_clicks"),
              State("data-ranking", "data"),
              State({"type": "slider-bm", "index": ALL}, "value"),
              State({"type": "slider-bm-Res", "index": ALL}, "value"),
              State({"type": "slider-bm-Uti", "index": ALL}, "value"),
              State({"type": "slider-bm-Pri", "index": ALL}, "value"),
              prevent_initial_call=True)
def display_weighted_ranking(click, data_ranking, weights_top, weights_res, weights_uti, weights_pri):
    if click:

        weights = []
        for w in weights_top:
            if weights_res:
                weights += [x * w for x in weights_res]
                weights_res = []
            elif weights_uti:
                weights += [x * w for x in weights_uti]
                weights_uti = []
            elif weights_pri:
                weights += [x * w for x in weights_pri]
                weights_pri = []

        ranking_lists = [d for data in data_ranking for value in data.values() for d in value]

        ranking_lists_w = [[x * w for x in list(d.values())[0]] for d, w in zip(ranking_lists, weights)]
        final_score_lists = [round(sum(v), 3) for v in zip(*ranking_lists_w)]

        labels = [f'Synthetic dataset {i}' for i in range(1, len(final_score_lists) + 1)]
        max_rank = len(final_score_lists)
        fig = go.Figure(data=[go.Bar(x=labels, y=[max_rank - v for v in final_score_lists],
                                     text=final_score_lists, textposition='outside')])
        fig.update_layout(title="Final weighted scores", yaxis=dict(showticklabels=False))
        fig.update_traces(textposition="outside", cliponaxis=False)

        children = [
            html.Div([
                dcc.Graph(figure=fig),
                dbc.Button(
                    "Info final score",
                    id="info-final-score",
                    color="warning"
                ),
                dbc.Popover(
                    [
                        dbc.PopoverHeader("Final score"),
                        dbc.PopoverBody("The final score is calculated based on the various rankings displayed above, "
                                        "weighted with the values chosen for each metric category. It is, thus, the "
                                        "weighted average of the various rankings: the lower the value, the better "
                                        "the synthetic dataset quality."),
                    ],
                    target="info-final-score",
                    trigger="click"
                )
            ])
        ]

        weights_report = dict(
            zip([list(d.keys())[0] for data in data_ranking for value in data.values() for d in value], weights))

        return children, [fig, weights_report]

    else:
        raise PreventUpdate


# REPORT SECTION
@app.callback(Output("download-report", "data"),
              Input("button-report", "n_clicks"),
              [State("user_position", "pathname"),
               State({"type": "data-report", "index": ALL}, "data")],
              prevent_initial_call=True)
def generate_and_download_report(n_clicks, pathname, data_report):
    if n_clicks is None:
        raise PreventUpdate

    if pathname == '/page_2_ura':
        data_report = [element for element in data_report if element]
        data_tbl = [data_report[i][0] for i in range(len(data_report))]
        data_opt = [data_report[i][1]['selected_opt'] for i in range(len(data_report))]

        dict_label_ura = {
            'ks_test': ['Numerical test results', 'Kolmogorov–Smirnov test'],
            't_test': ['Numerical test results', 'Student t-test'],
            'u_test': ['Numerical test results', 'Mann–Whitney U test'],
            'chi_test': ['Categorical test results', 'Chi-square test'],
            'cos_dist': ['Distance metric results', 'Cosine distance'],
            'js_dist': ['Distance metric results', 'Jensen-Shannon distance'],
            'w_dist': ['Distance metric results', 'Wasserstein distance']
        }

        html_string = '''
                <html>
                    <head>
                        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">
                        <style>
                            body{ margin:0 100; background:whitesmoke; }
                        </style>
                    </head>
                    <body>
                        <h1>Univariate Resemblance Analysis</h1> '''

        for i, dicts in enumerate(data_tbl):
            opt = data_opt[i]

            html_string += '''
            <h3>''' + dict_label_ura[opt][0] + '''</h3>
            <h4>Selected option: ''' + dict_label_ura[opt][1] + '''</h4></br>'''

            for j, d in enumerate(dicts):
                df_tbl = pd.DataFrame(list(d.items()), columns=['Feature', 'Metric value'])
                df_tbl = df_tbl.to_html(index=False).replace('<table border="1" class="dataframe">',
                                                             '<table class="table table-striped">')

                html_string += '''
                <h5><b>Synthetic dataset #''' + str(j + 1) + '''</b></h5>
                ''' + df_tbl + '''
                </br>
                '''

        html_string += '''
            </body>
        </html>'''

        utils.convert_html_to_pdf('data_report/Report Resemblance URA.html',
                                  html_string,
                                  'data_report/Report Resemblance URA.pdf')

        return dcc.send_file('data_report/Report Resemblance URA.pdf')

    elif pathname == '/page_2_mra':
        data_mra_corr = data_report[0]
        data_mra_lof = data_report[1]
        data_mra_pca = data_report[2]
        data_mra_umap = data_report[3]

        html_string = '''
                <html>
                    <head>
                        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">
                        <style>body{ margin:0 100; background:whitesmoke; }</style>
                    </head>
                    <body>
                        <h1>Multivariate Relationship Analysis</h1> '''

        if data_mra_corr:
            if data_mra_corr[1] == 'corr_num':
                corr_type = 'Pairwise Pearson correlation matrices'
            else:
                corr_type = 'Normalized contingency tables'

            html_string += '''
            <h3>Correlation matrices results</h3>
            <h4 style="display: list-item;">Matrix type selected: ''' + corr_type + '''</h4>'''

            if data_mra_corr[2] == 'rs':
                vis_corr_type = 'Real data vs. Synthetic data'

                html_string += '''
                <h4 style="display: list-item;">Visualization mode: ''' + vis_corr_type + '''</h4>'''

                for i, fig in enumerate(data_mra_corr[0]):
                    path_corr_r = os.path.join(path_user, 'data_figures', 'mat_corr_r.html').replace("\\", '/')
                    pio.write_html(fig[0], file=path_corr_r, auto_open=False)
                    fig_name = 'mat_corr_s{}.html'.format(i + 1)
                    path_corr_s = os.path.join(path_user, 'data_figures', fig_name).replace("\\", '/')
                    pio.write_html(fig[1], file=path_corr_s, auto_open=False)

                    html_string += '''
                        <h5><b>Synthetic dataset #''' + str(i + 1) + '''</b></h5>
                        <div style="display: flex;">
                            <div style="width: 50%;">
                                <iframe width="350" height="350" frameborder="0" seamless="seamless" scrolling="no" src="{}"></iframe>
                            </div>
                            <div style="width: 50%;">
                                <iframe width="350" height="350" frameborder="0" seamless="seamless" scrolling="no" src="{}"></iframe>
                            </div>
                        </div></br>
                    '''.format(path_corr_r, path_corr_s)
            else:
                vis_corr_type = 'Differences between Real data and Synthetic data'

                html_string += '''
                <h4 style="display: list-item;">Visualization mode: ''' + vis_corr_type + '''</h4>'''

                for i, fig in enumerate(data_mra_corr[0]):
                    fig_name = 'mat_corr_d{}.html'.format(i + 1)
                    path_corr_diff = os.path.join(path_user, 'data_figures', fig_name).replace("\\", '/')
                    pio.write_html(fig, file=path_corr_diff, auto_open=False)

                    html_string += '''
                        <h5><b>Synthetic dataset #''' + str(i + 1) + '''</b></h5>
                        <iframe width="800" height="500" frameborder="0" seamless="seamless" scrolling="no" src="{}">
                        </iframe></br>
                    '''.format(path_corr_diff)

        if data_mra_lof:

            html_string += '''<h3>Local Outlier Factor results</h3>'''

            for i, fig in enumerate(data_mra_lof):
                fig_name = 'box_lof{}.html'.format(i + 1)
                path_lof = os.path.join(path_user, 'data_figures', fig_name).replace("\\", '/')
                pio.write_html(data_mra_lof, file=path_lof, auto_open=False)

                html_string += '''
                    <h5><b>Synthetic dataset #''' + str(i + 1) + '''</b></h5>
                    <iframe width="800" height="500" frameborder="0" seamless="seamless" scrolling="no" 
                    src="{}"></iframe></br>
                '''.format(path_lof)

        if data_mra_pca:

            html_string += '''<h3>Principal Components Analysis results</h3>'''

            components = data_mra_pca[1]['x']

            for i, (fig, diff) in enumerate(zip(data_mra_pca[0], data_mra_pca[1]['diffs'])):
                df_pca = pd.DataFrame(list(zip(components, diff)), columns=['Component', 'Difference (%)'])

                df_pca = df_pca.to_html(index=False). \
                    replace('<table border="1" class="dataframe">', '<table class="table table-striped">')

                fig_name = 'pca{}.html'.format(i + 1)
                path_pca = os.path.join(path_user, 'data_figures', fig_name).replace("\\", '/')
                pio.write_html(fig, file=path_pca, auto_open=False)

                html_string += '''
                        <h5><b>Synthetic dataset #''' + str(i + 1) + '''</b></h5>
                        <iframe width="800" height="500" frameborder="0" seamless="seamless" scrolling="no" 
                        src="''' + path_pca + '''"></iframe>
                        </br>
                        ''' + df_pca + '''
                        </br>
                '''

        if data_mra_umap:

            if data_mra_umap[3] == 'rs':
                vis_type = 'Real data vs. Synthetic data'
            else:
                vis_type = 'Real data together with Synthetic data'

            html_string += '''
                    <h3>UMAP projections results</h3>
                    <h4 style="display: list-item;">Number of neighbors: ''' + str(data_mra_umap[1]) + '''</h4>
                    <h4 style="display: list-item;">Minimum distance: ''' + str(data_mra_umap[2]) + '''</h4>
                    <h4 style="display: list-item;">Visualization mode: ''' + vis_type + '''</h4> 
            '''

            if data_mra_umap[3] == 'rs':
                for i, figs in enumerate(data_mra_umap[0]):
                    path_umap_r = os.path.join(path_user, 'data_figures', 'umap_r.html').replace("\\", '/')
                    pio.write_html(figs[0], file=path_umap_r, auto_open=False)
                    fig_name = 'umap_s{}.html'.format(i + 1)
                    path_umap_s = os.path.join(path_user, 'data_figures', fig_name).replace("\\", '/')
                    pio.write_html(figs[1], file=path_umap_s, auto_open=False)

                    html_string += '''
                        <h5><b>Synthetic dataset #''' + str(i + 1) + '''</b></h5>
                        <div style="display: flex;">
                            <div style="width: 50%;">
                                <iframe width="350" height="350" frameborder="0" seamless="seamless" scrolling="no" src="{}"></iframe>
                            </div>
                            <div style="width: 50%;">
                                <iframe width="350" height="350" frameborder="0" seamless="seamless" scrolling="no" src="{}"></iframe>
                            </div>
                        </div></br>
                    '''.format(path_umap_r, path_umap_s)

            else:
                for i, fig in enumerate(data_mra_umap[0]):
                    fig_name = 'umap{}.html'.format(i + 1)
                    path_umap = os.path.join(path_user, 'data_figures', fig_name).replace("\\", '/')
                    pio.write_html(fig, file=path_umap, auto_open=False)

                    html_string += '''               
                            <iframe width="800" height="500" frameborder="0" seamless="seamless" scrolling="no" 
                            src="{}"></iframe>
                            </br>
                    '''.format(path_umap)

        html_string += '''
            </body>
        </html>'''

        utils.convert_html_to_pdf('data_report/Report Resemblance MRA.html',
                                  html_string,
                                  'data_report/Report Resemblance MRA.pdf')

        return dcc.send_file('data_report/Report Resemblance MRA.pdf')

    elif pathname == '/page_2_dla':
        fig_dla = data_report[0][0]
        data_dla = data_report[0][1]

        html_string = '''
                <html>
                    <head>
                        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">
                        <style>
                            body{ margin:0 100; background:whitesmoke; }
                        </style>
                    </head>
                    <body>
                        <h1>Data Labeling Analysis</h1>

                        <h3>Performance metrics results</h3>'''

        for i, d in enumerate(data_dla):
            df_dla = pd.DataFrame(d)
            df_dla.columns = ['Name model', 'Accuracy', 'Precision', 'Recall', 'F1 score']
            df_dla = df_dla.to_html(index=False).replace('<table border="1" class="dataframe">',
                                                         '<table class="table table-striped">')

            html_string += '''
            <h5><b>Synthetic dataset #''' + str(i + 1) + '''</b></h5>
            ''' + df_dla + '''
            </br>
            '''

        path_dla = os.path.join(path_user, 'data_figures', 'dla.html').replace("\\", '/')
        pio.write_html(fig_dla, file=path_dla, auto_open=False)

        html_string += '''
                <h3>Summary plot</h3>
                <iframe width="800" height="500" frameborder="0" seamless="seamless" scrolling="no" 
                        src="''' + path_dla + '''"></iframe>
            </body>
        </html>'''

        utils.convert_html_to_pdf('data_report/Report Resemblance DLA.html',
                                  html_string,
                                  'data_report/Report Resemblance DLA.pdf')

        return dcc.send_file('data_report/Report Resemblance DLA.pdf')

    elif pathname == '/page_3':
        fig_trtr = data_report[0][0]
        figs_tstr = data_report[0][1]
        data_trtr = data_report[0][2]
        data_tstr = data_report[0][3]

        df_trtr = pd.DataFrame(data_trtr)
        df_trtr.columns = ['Name model', 'Accuracy', 'Precision', 'Recall', 'F1 score']
        df_trtr = df_trtr.to_html(index=False).replace('<table border="1" class="dataframe">',
                                                       '<table class="table table-striped">')

        path_trtr = os.path.join(path_user, 'data_figures', 'mat_trtr.html').replace("\\", '/')
        pio.write_html(fig_trtr, file=path_trtr, auto_open=False)

        html_string = '''
                <html>
                    <head>
                        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">
                        <style>
                            body{ margin:0 100; background:whitesmoke; }
                        </style>
                    </head>
                    <body>
                        <h1>Utility Evaluation</h1>

                        <h3>Train on Real, Test on Real performance metrics results</h3>
                        ''' + df_trtr + '''
                        <iframe width="500" height="300" frameborder="0" seamless="seamless" scrolling="no" 
                        src="''' + path_trtr + '''"></iframe>
                        </br>
                        
                        <h3>Train on Synthetic, Test on Real performance metrics results</h3>
        '''

        for i, (fig, d) in enumerate(zip(figs_tstr, data_tstr)):
            df_tstr = pd.DataFrame(d)
            df_tstr.columns = ['Name model', 'Accuracy', 'Precision', 'Recall', 'F1 score']
            df_tstr = df_tstr.to_html(index=False).replace('<table border="1" class="dataframe">',
                                                           '<table class="table table-striped">')

            fig_name = 'mat_tstr{}.html'.format(i + 1)
            path_tstr = os.path.join(path_user, 'data_figures', fig_name).replace("\\", '/')
            pio.write_html(fig, file=path_tstr, auto_open=False)

            html_string += '''
            <h5><b>Synthetic dataset #''' + str(i + 1) + '''</b></h5>
            ''' + df_tstr + '''
            <iframe width="500" height="300" frameborder="0" seamless="seamless" scrolling="no" 
            src="''' + path_tstr + '''"></iframe>
            </br>
            '''

        # name_model = df_trtr['Name model'].iloc[0]
        # if name_model == 'RF':
        #     df_trtr['Name model'] = 'Random Forest'
        #     df_tstr['Name model'] = 'Random Forest'
        # elif name_model == 'KNN':
        #     df_trtr['Name model'] = 'K-Nearest Neighbors'
        #     df_tstr['Name model'] = 'K-Nearest Neighbors'
        # elif name_model == 'DT':
        #     df_trtr['Name model'] = 'Decision Tree'
        #     df_tstr['Name model'] = 'Decision Tree'
        # elif name_model == 'SVM':
        #     df_trtr['Name model'] = 'Support Vector Machine'
        #     df_tstr['Name model'] = 'Support Vector Machine'
        # elif name_model == 'MLP':
        #     df_trtr['Name model'] = 'Multilayer Perceptron'
        #     df_tstr['Name model'] = 'Multilayer Perceptron'

        html_string += '''
            </body>
        </html>'''

        utils.convert_html_to_pdf('data_report/Report Utility.html',
                                  html_string,
                                  'data_report/Report Utility.pdf')

        return dcc.send_file('data_report/Report Utility.pdf')

    elif pathname == '/page_4_sea':
        data_sea = data_report[0][0]
        selected_dist = data_report[0][1]

        html_dist = ''''''
        for i, d in enumerate(data_sea):
            if selected_dist == 'cos':
                selected_dist_label = 'Cosine similarity'
                fig_name = 'fig_sea{}.html'.format(i + 1)
                path = os.path.join(path_user, 'data_figures', fig_name).replace("\\", '/')
                pio.write_html(d, file=path, auto_open=False)
                html_dist += '''
                    <h5><b>Synthetic dataset #''' + str(i + 1) + '''</b></h5>
                    <iframe width="800" height="500" frameborder="0" seamless="seamless" scrolling="no" 
                    src="''' + path + '''"></iframe>
                    </br>
                '''
            elif selected_dist == 'euc':
                selected_dist_label = 'Euclidean distance'
                fig_name = 'fig_sea{}.html'.format(i + 1)
                path = os.path.join(path_user, 'data_figures', fig_name).replace("\\", '/')
                pio.write_html(d, file=path, auto_open=False)
                html_dist += '''
                    <h5><b>Synthetic dataset #''' + str(i + 1) + '''</b></h5>
                    <iframe width="800" height="500" frameborder="0" seamless="seamless" scrolling="no" 
                    src="''' + path + '''"></iframe>
                    </br>
                '''
            elif selected_dist == 'hau':
                selected_dist_label = 'Hausdorff distance'
                html_dist += '''
                    <h5><b>Synthetic dataset #''' + str(i + 1) + '''</b></h5>
                    <h5>Distance metric value: ''' + str(d) + '''</h5>
                    </br>
                '''

        html_string = '''
        <html>
            <head>
                <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">
                <style>
                    body{ margin:0 100; background:whitesmoke; }
                </style>
            </head>
            <body>
                <h1>Similarity Evaluation Analysis</h1>

                <h3>Paired distances results</h3>
                <h4>Distance metric selected: ''' + selected_dist_label + '''</h4>
                ''' + html_dist + '''

            </body>
        </html>'''

        utils.convert_html_to_pdf('data_report/Report Privacy SEA.html',
                                  html_string,
                                  'data_report/Report Privacy SEA.pdf')

        return dcc.send_file('data_report/Report Privacy SEA.pdf')

    elif pathname == '/page_4_mia':
        figs_mia = data_report[0][0]
        prop_subset = data_report[0][1]
        t_sim = data_report[0][2]

        html_fig = ''''''
        for i, fig in enumerate(figs_mia):
            fig_name = 'pie_mia.html'.format(i + 1)
            path = os.path.join(path_user, 'data_figures', fig_name).replace("\\", '/')
            pio.write_html(fig, file=path, auto_open=False)

            html_fig += '''
            <h5><b>Synthetic dataset #''' + str(i + 1) + '''</b></h5>
            <iframe width="800" height="400" frameborder="0" seamless="seamless" scrolling="no" 
                src="''' + path + '''"></iframe>
            '''

        html_string = '''
        <html>
            <head>
                <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">
                <style>body{ margin:0 100; background:whitesmoke; }</style>
            </head>
            <body>
                <h1>Membership Inference Attack</h1>

                <h3>Attacker information</h3>
                <h4>Real subset size: ''' + str(prop_subset) + '''% of the real dataset</h4>
                <h4>Similarity threshold: ''' + str(t_sim) + ''' (range 0-1)</h4>
                </br>

                <h3>Performance attacker results</h3>
                ''' + html_fig + '''
                </br>

            </body>
        </html>'''

        utils.convert_html_to_pdf('data_report/Report Privacy MIA.html',
                                  html_string,
                                  'data_report/Report Privacy MIA.pdf')

        return dcc.send_file('data_report/Report Privacy MIA.pdf')

    elif pathname == '/page_4_aia':
        data_aia_acc = data_report[0][0]
        data_aia_rmse = data_report[0][1]
        prop_subset = data_report[0][2]
        attributes = data_report[0][3]

        html_res = ''''''
        for i, (d1, d2) in enumerate(zip(data_aia_acc, data_aia_rmse)):
            df_acc = pd.DataFrame(d1)
            df_rmse = pd.DataFrame(d2)
            df_acc.columns = ['Attribute re-identified', 'Attacker accuracy']
            df_rmse.columns = ['Attribute re-identified', 'Interquartile range', 'Attacker RMSE']

            df_acc = df_acc.to_html(index=False).replace('<table border="1" class="dataframe">',
                                                         '<table class="table table-striped">')
            df_rmse = df_rmse.to_html(index=False).replace('<table border="1" class="dataframe">',
                                                           '<table class="table table-striped">')

            html_res += '''
                <h5><b>Synthetic dataset #''' + str(i + 1) + '''</b></h5>
                <h5>Categorical attributes results</h5>
                ''' + df_acc + '''
                </br>
                <h5>Numerical attributes results</h5>
                ''' + df_rmse + '''
                </br>
            '''

        html_string = '''
        <html>
            <head>
                <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">
                <style>body{ margin:0 100; background:whitesmoke; }</style>
            </head>
            <body>
                <h1>Attribute Inference Attack</h1>

                <h3>Attacker information</h3>
                <h4>Real subset size: ''' + str(prop_subset) + '''% of the real dataset</h4>
                <h4>Available attributes: ''' + str(attributes) + '''</h4>
                </br>

                <h3>Performance attacker results</h3>
                </br>
                ''' + html_res + '''
            </body>
        </html>'''

        utils.convert_html_to_pdf('data_report/Report Privacy AIA.html',
                                  html_string,
                                  'data_report/Report Privacy AIA.pdf')

        return dcc.send_file('data_report/Report Privacy AIA.pdf')

    elif pathname == '/page_5':
        data_ranking_1 = data_report[0]
        data_ranking_2 = data_report[1]

        html_string = '''
        <html>
            <head>
                <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.1/css/bootstrap.min.css">
                <style>
                    body{ margin:0 10; background:whitesmoke; }
                </style>
            </head>
            <body>
                <h1>Benchmarking Report</h1>
                <h3>Ranking lists (single metric)</h3>
        '''

        for label, d in data_ranking_1[1].items():

            df_tbl = pd.DataFrame(d)

            if len(df_tbl.columns) - len(synthetic_datasets) == 1:
                df_tbl.columns = [''] + [f'Synthetic dataset {i}' for i in range(1, len(df_tbl.columns))]
            else:
                df_tbl.columns = [f'Synthetic dataset {i}' for i in range(1, len(df_tbl.columns) + 1)]

            df_tbl = df_tbl.to_html(index=False).replace('<table border="1" class="dataframe">',
                                                         '<table class="table table-striped">')

            html_string += '''
            <h4><b>''' + label + '''</b></h4>
            ''' + df_tbl + '''
            </br>
            '''

        for i, element in enumerate(data_ranking_1[0]):
            fig_name = 'cdf{}.html'.format(i + 1)
            path_cdf = os.path.join(path_user, 'data_figures', fig_name).replace("\\", '/')
            pio.write_html(element[0], file=path_cdf, auto_open=False)

            df_auc = pd.DataFrame(list(element[1].items()), columns=['Synthetic dataset', 'Ranking curve AUC'])
            df_auc = df_auc.to_html(index=False).replace('<table border="1" class="dataframe">',
                                                         '<table class="table table-striped">')

            html_string += '''
                <h3>''' + element[2] + '''</h3>
                <div style="width: 100%;">
                    <iframe width="800" height="500" frameborder="0" seamless="seamless" scrolling="no" src="{}"></iframe>
                </div>
                <div style="width: 50%;">
                    {}
                </div>
                </br>
            '''.format(path_cdf, df_auc)

        if data_ranking_2:
            path_rank_w = os.path.join(path_user, 'data_figures', 'rank_w.html').replace("\\", '/')
            pio.write_html(data_ranking_2[0], file=path_rank_w, auto_open=False)

            df_rank_w = pd.DataFrame(list(data_ranking_2[1].items()), columns=['Metric name', 'Weight'])
            df_rank_w = df_rank_w.to_html(index=False).replace('<table border="1" class="dataframe">',
                                                               '<table class="table table-striped">')

            html_string += '''
            <h3>Final score weighted</h3>
            <div style="width: 100%;">
                <iframe width="800" height="450" frameborder="0" seamless="seamless" scrolling="no" src="{}"></iframe>
            </div>
            <div style="width: 50%;">
                {}
            </div>
            </br>
            '''.format(path_rank_w, df_rank_w)

        html_string += '''
                    </body>
                </html>'''

        utils.convert_html_to_pdf('data_report/Report Benchmarking.html',
                                  html_string,
                                  'data_report/Report Benchmarking.pdf')

        return dcc.send_file('data_report/Report Benchmarking.pdf')


@app.callback(Output("button-report", "disabled", allow_duplicate=True),
              Input("user_position", "pathname"),
              prevent_initial_call=True)
def download_disabled_change_position(p):
    return True


@app.callback(Output("button-report", "disabled"),
              Input({"type": "data-report", "index": ALL}, "data"),
              prevent_initial_call=True)
def update_download_disabled(data):
    return all(not obj for obj in data)


if __name__ == '__main__':
    app.run_server(debug=False, host='127.0.0.1', port=8080)
