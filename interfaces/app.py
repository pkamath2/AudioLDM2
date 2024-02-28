from multipage import MultiPage
from text_to_continuous import main as text_to_continous

app = MultiPage()
app.add_app("audioldm", text_to_continous)
app.run()
