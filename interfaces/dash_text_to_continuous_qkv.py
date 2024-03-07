# Import packages
from dash import Dash, html, dash_table, dcc, callback, Output, Input, State
import functools
import pandas as pd
import soundfile as sf
import plotly.express as px
import dash_bootstrap_components as dbc

from interfaces.diffusion_helper_qkv import *






# Initialize the app - incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.DARKLY]
app = Dash(__name__, external_stylesheets=external_stylesheets)

# App layout
app.layout = dbc.Container([
    dbc.Row([
        html.Div('   ', className="text-primary text-center pt-5")
    ]),
    dbc.Row([
        html.Div('Morhing Audio w/ Text', className="text-center fs-3")
    ]),

    dbc.Row([
        dbc.Col([

            # html.Div(
                # [
                    dbc.Input(id="source_text", placeholder="", type="text"),
                    html.Br(),
                    dcc.Graph(id="source_spectrogram"),
                    html.Br(),
                    # html.Audio(
                    #     id="source_audio",
                    #     autoPlay=False,
                    #     controls=False,
                    #     style={'width': '100%'}
                    # ),
                    # html.Br(),
                    dbc.Button("Generate", id="source-generate-button", n_clicks=0, color="primary", className="me-1")
            #     ]
            # )
            
        ], width=3),

        dbc.Col([
            html.Div(' ', className="text-center")
        ], width=6),
    ]),

], fluid=True)


@app.callback(
    Output("source_spectrogram", "figure"),Input("source-generate-button", 'n_clicks'), State('source_text', 'value')#, prevent_initial_call=True
)#,Output("source_audio", "src")
def get_sample(n_clicks, source_text):
    latent_diffusion = get_model('audioldm_16k_crossattn_t5')
    # if source_text is not None:
        
    print('getting sample ==>', source_text)
    wav, img = sample_diffusion_attention_edit(latent_diffusion=latent_diffusion, source_text=None, target_text=source_text)

    sf.write('assets/source_audio.wav', wav, 16000)

    print(img)
    fig = px.imshow(img)
    print(fig)
    return fig# 'assets/source_audio.wav'
    # else:
    #     print('returning None')
    #     return None#, None

# latent_diffusion, source_text, target_text, batch_size=1, ddim_steps=20, \
#                                     guidance_scale=3.0, random_seed=42, \
#                                     interpolation_level=0.5,\
#                                     source_selected_word_list=None, source_value_list=None, \
#                                     target_selected_word_list=None, target_value_list=None,\
#                                     interpolate_terms=['q','k','v']


# # Add controls to build the interaction
# @callback(
#     Output(component_id='my-first-graph-final', component_property='figure'),
#     Input(component_id='radio-buttons-final', component_property='value')
# )
# def update_graph(col_chosen):
#     fig = px.histogram(df, x='continent', y=col_chosen, histfunc='avg')
#     return fig

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
