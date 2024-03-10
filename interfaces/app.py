from multipage import MultiPage
from text_to_continuous_morph_qkv import main as text_to_continuous_morph_qkv
from text_to_continuous_qkv import main as text_to_continuous_qkv

app = MultiPage()
app.add_app("audioldm-morph", text_to_continuous_morph_qkv)
app.add_app("audioldm-reweight", text_to_continuous_qkv)
app.run()
