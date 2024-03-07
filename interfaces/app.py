from multipage import MultiPage
from text_to_continuous_qkv import main as text_to_continuous_qkv
# from text_to_continuous_simple import main as text_to_continuous_simple

app = MultiPage()
app.add_app("audioldm", text_to_continuous_qkv)
# app.add_app("audioldm-simple", text_to_continuous_simple)
app.run()
